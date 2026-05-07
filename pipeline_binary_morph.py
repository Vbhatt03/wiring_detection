#!/usr/bin/env python3
"""
Direct Binary Segmentation Pipeline for Wire Detection

Instead of Frangi, uses the Otsu binary directly with morphological operations:
1. Grid line removal via morphological opening
2. Scale-based separation: thin segments vs thick routes via width filtering
3. Connected components filtering
4. Skeletonization

Generates intermediate debug images at each step.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from pathlib import Path

from src.skeleton_graph import prune_spurs


class BinaryMorphPipeline:
    """Full pipeline with debug visualization."""
    
    def __init__(self, input_image_path: str, output_dir: str = "real_images"):
        self.input_path = Path(input_image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.debug_dir = self.output_dir / "debug"
        self.debug_dir.mkdir(exist_ok=True)
        
        # Load image
        self.img_color = cv2.imread(str(self.input_path))
        if self.img_color is None:
            raise FileNotFoundError(f"Could not load image: {self.input_path}")
        
        self.gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.gray.shape
        
        print(f"Loaded image: {self.input_path}")
        print(f"Size: {self.w} × {self.h}")
        
        self.step_num = 0
    
    def save_debug(self, image: np.ndarray, name: str, description: str = ""):
        """Save intermediate debug image."""
        self.step_num += 1
        filename = f"{self.step_num:02d}_{name}.png"
        path = self.debug_dir / filename
        
        # Convert bool to uint8 if needed
        if image.dtype == bool:
            image = image.astype(np.uint8) * 255
        
        cv2.imwrite(str(path), image)
        print(f"  [{self.step_num:02d}] {name:<40} {description}")
        return path
    
    # ===== STEP 1: Binarization =====
    
    def binarize(self) -> np.ndarray:
        """Convert to Otsu binary."""
        print("\n[STEP 1] Binarization...")
        
        _, binary = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        self.save_debug(self.gray, "00_original_gray", "Original grayscale")
        self.save_debug(binary, "01_binary_otsu", f"Otsu binary (white pixels)")
        
        return binary
    
    # ===== STEP 2: Grid Line Removal =====
    
    def remove_grid_lines(self, binary: np.ndarray, L: int = 150) -> np.ndarray:
        """Remove grid lines via morphological opening."""
        print("\n[STEP 2] Grid line removal (L={})...".format(L))
        
        # Horizontal opening
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
        h_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        self.save_debug(h_opened, "02_h_opened", "Horizontal opening")
        
        # Filter to thin only
        h_eroded = cv2.erode(h_opened, np.ones((4, 1), np.uint8))
        h_dilated = cv2.dilate(h_eroded, np.ones((8, 1), np.uint8))
        h_grid = cv2.subtract(h_opened, h_dilated)
        self.save_debug(h_grid, "03_h_grid_thin", "H-grid (thin only)")
        
        # Vertical opening
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
        v_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        self.save_debug(v_opened, "04_v_opened", "Vertical opening")
        
        # Filter to thin only
        v_eroded = cv2.erode(v_opened, np.ones((1, 4), np.uint8))
        v_dilated = cv2.dilate(v_eroded, np.ones((1, 8), np.uint8))
        v_grid = cv2.subtract(v_opened, v_dilated)
        self.save_debug(v_grid, "05_v_grid_thin", "V-grid (thin only)")
        
        # Combine and dilate
        grid_mask = cv2.bitwise_or(h_grid, v_grid)
        grid_mask = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=1)
        self.save_debug(grid_mask, "06_grid_mask", "Combined grid mask")
        
        # Remove from binary
        clean_binary = cv2.subtract(binary, grid_mask)
        self.save_debug(clean_binary, "07_binary_clean", f"Binary after grid removal")
        
        return clean_binary
    
    # ===== STEP 3: Scale-Based Separation =====
    
    def separate_by_width(self, binary: np.ndarray) -> tuple:
        """Separate thin wires from thick routes using morphological opening."""
        print("\n[STEP 3] Scale-based separation (thin vs thick)...")
        
        # THIN segments: opening with kernel width ~4-6px
        # Anything thinner survives, thicker doesn't
        thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
        thin_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, thin_kernel)
        self.save_debug(thin_opened, "08_thin_opened", "Opening with thin kernel (1×6)")
        
        # Also horizontal
        thin_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
        thin_opened_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, thin_kernel_h)
        self.save_debug(thin_opened_h, "09_thin_opened_h", "Opening with thin kernel (6×1)")
        
        # Combine thin
        thin_combined = cv2.bitwise_or(thin_opened, thin_opened_h)
        self.save_debug(thin_combined, "10_thin_combined", "Combined thin segments")
        
        # THICK routes: opening with kernel width ~25px
        # Only structures wider than ~25px survive
        thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        thick_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, thick_kernel)
        self.save_debug(thick_opened, "11_thick_opened", "Opening with thick kernel (1×25)")
        
        thick_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        thick_opened_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, thick_kernel_h)
        self.save_debug(thick_opened_h, "12_thick_opened_h", "Opening with thick kernel (25×1)")
        
        thick_combined = cv2.bitwise_or(thick_opened, thick_opened_h)
        self.save_debug(thick_combined, "13_thick_combined", "Combined thick routes")
        
        return thin_combined, thick_combined
    
    # ===== STEP 4: Remove Route from Segments =====
    
    def separate_hierarchy(self, thin: np.ndarray, thick: np.ndarray) -> np.ndarray:
        """Remove thick route from thin segments."""
        print("\n[STEP 4] Hierarchical separation...")
        
        # Expand thick to exclude boundary
        thick_expanded = cv2.dilate(thick, np.ones((20, 20), np.uint8), iterations=2)
        self.save_debug(thick_expanded, "14_thick_expanded", "Thick routes expanded (20px dilation)")
        
        # Subtract from thin
        thin_clean = cv2.subtract(thin, thick_expanded)
        self.save_debug(thin_clean, "15_thin_clean", "Thin segments after route removal")
        
        return thin_clean
    
    # ===== STEP 5: Morphological Cleanup =====
    
    def cleanup_morphology(self, thin: np.ndarray, thick: np.ndarray) -> tuple:
        """Close gaps and fill holes."""
        print("\n[STEP 5] Morphological cleanup...")
        
        # Thin: close small gaps
        thin_closed = cv2.morphologyEx(
            thin, cv2.MORPH_CLOSE, 
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 
            iterations=1
        )
        self.save_debug(thin_closed, "16_thin_closed", "Thin after morphological close")
        
        # Thick: close and fill
        thick_closed = cv2.morphologyEx(
            thick, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            iterations=3
        )
        self.save_debug(thick_closed, "17_thick_closed", "Thick after morphological close")
        
        return thin_closed, thick_closed
    
    # ===== STEP 6: CCL Filtering =====
    
    def ccl_filter_thin(self, mask: np.ndarray) -> np.ndarray:
        """Filter thin segments by aspect ratio and area."""
        print("\n[STEP 6a] CCL filtering (thin segments)...")
        
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        print(f"  Found {n-1} components")
        
        final_mask = np.zeros_like(mask)
        kept = 0
        
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            
            if area < 60:  # Too small
                continue
            
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect < 2.0:  # Not elongated enough
                continue
            
            fill = area / (w * h)
            if fill > 0.65:  # Too dense (blob, not wire)
                continue
            
            final_mask[labels == i] = 255
            kept += 1
        
        print(f"  Kept {kept} thin components")
        self.save_debug(final_mask, "18_thin_ccl_filtered", f"Thin after CCL ({kept} kept)")
        
        return final_mask
    
    def ccl_filter_thick(self, mask: np.ndarray) -> np.ndarray:
        """Filter thick routes (keep large components)."""
        print("\n[STEP 6b] CCL filtering (thick routes)...")
        
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        print(f"  Found {n-1} components")
        
        final_mask = np.zeros_like(mask)
        areas = [(stats[i][4], i) for i in range(1, n)]
        areas.sort(reverse=True)
        
        kept = 0
        for area, i in areas:
            if area < 2000:
                break
            final_mask[labels == i] = 255
            kept += 1
        
        print(f"  Kept {kept} thick components (min area 2000)")
        self.save_debug(final_mask, "19_thick_ccl_filtered", f"Thick after CCL ({kept} kept)")
        
        return final_mask
    
    # ===== STEP 7: Skeletonization =====
    
    def skeletonize_and_prune(self, thin: np.ndarray, thick: np.ndarray) -> tuple:
        """Convert to skeletons and prune spurs."""
        print("\n[STEP 7] Skeletonization...")
        
        # Thin skeleton
        skel_thin = skeletonize(thin > 0).astype(np.uint8) * 255
        self.save_debug(skel_thin, "20_thin_skeleton_raw", "Thin skeleton (raw)")
        
        skel_thin = prune_spurs(skel_thin, min_branch_length=12)
        skel_thin = (skel_thin * 255).astype(np.uint8)
        self.save_debug(skel_thin, "21_thin_skeleton_pruned", "Thin skeleton (spurs pruned)")
        
        # Thick skeleton
        skel_thick = skeletonize(thick > 0).astype(np.uint8) * 255
        self.save_debug(skel_thick, "22_thick_skeleton_raw", "Thick skeleton (raw)")
        
        skel_thick = prune_spurs(skel_thick, min_branch_length=25)
        skel_thick = (skel_thick * 255).astype(np.uint8)
        self.save_debug(skel_thick, "23_thick_skeleton_pruned", "Thick skeleton (spurs pruned)")
        
        return skel_thin, skel_thick
    
    # ===== FINAL VISUALIZATION =====
    
    def create_visualizations(self, thin_final: np.ndarray, thick_final: np.ndarray,
                             skel_thin: np.ndarray, skel_thick: np.ndarray):
        """Create composite visualizations."""
        print("\n[FINAL] Creating composite visualizations...")
        
        # Overlay skeletons on original
        result_color = self.img_color.copy()
        result_color[skel_thin > 0] = [0, 255, 0]  # Green for thin
        result_color[skel_thick > 0] = [0, 0, 255]  # Red for thick
        self.save_debug(result_color, "24_final_combined_overlay", 
                       "Final: thin (green) + thick (red)")
        
        # Side-by-side: masks
        thin_3ch = cv2.cvtColor(thin_final, cv2.COLOR_GRAY2BGR)
        thick_3ch = cv2.cvtColor(thick_final, cv2.COLOR_GRAY2BGR)
        masks_side = np.hstack([thin_3ch, thick_3ch])
        self.save_debug(masks_side, "25_masks_sbs", "Side-by-side: thin vs thick masks")
        
        # Side-by-side: skeletons
        skel_thin_3ch = cv2.cvtColor(skel_thin, cv2.COLOR_GRAY2BGR)
        skel_thick_3ch = cv2.cvtColor(skel_thick, cv2.COLOR_GRAY2BGR)
        skels_side = np.hstack([skel_thin_3ch, skel_thick_3ch])
        self.save_debug(skels_side, "26_skeletons_sbs", "Side-by-side: thin vs thick skeletons")
    
    # ===== MAIN RUN =====
    
    def run(self):
        """Execute full pipeline."""
        print("\n" + "="*70)
        print("BINARY-MORPHOLOGICAL PIPELINE FOR WIRE DETECTION")
        print("="*70)
        
        # Step 1: Binarize
        binary = self.binarize()
        
        # Step 2: Remove grid
        clean_binary = self.remove_grid_lines(binary, L=150)
        
        # Step 3: Separate by scale
        thin, thick = self.separate_by_width(clean_binary)
        
        # Step 4: Hierarchical
        thin_clean = self.separate_hierarchy(thin, thick)
        
        # Step 5: Cleanup
        thin_closed, thick_closed = self.cleanup_morphology(thin_clean, thick)
        
        # Step 6: CCL filter
        thin_final = self.ccl_filter_thin(thin_closed)
        thick_final = self.ccl_filter_thick(thick_closed)
        
        # Step 7: Skeletonize
        skel_thin, skel_thick = self.skeletonize_and_prune(thin_final, thick_final)
        
        # Final visualization
        self.create_visualizations(thin_final, thick_final, skel_thin, skel_thick)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Debug images: {self.debug_dir}")
        
        return {
            "skel_thin": skel_thin,
            "skel_thick": skel_thick,
            "mask_thin": thin_final,
            "mask_thick": thick_final,
        }


if __name__ == "__main__":
    img_path = "real_images/cemm-wire-harness-assembly.jpeg"
    
    if not Path(img_path).exists():
        print(f"Error: {img_path} not found")
        exit(1)
    
    pipeline = BinaryMorphPipeline(img_path)
    results = pipeline.run()
    
    print(f"\nThin skeleton: {results['skel_thin'].shape}, {np.sum(results['skel_thin'] > 0)} pixels")
    print(f"Thick skeleton: {results['skel_thick'].shape}, {np.sum(results['skel_thick'] > 0)} pixels")
