#!/usr/bin/env python3
"""
Dual-Scale Frangi Pipeline for Wire Detection

Separates grid lines, thin wire segments, and thick route bundles using:
1. Grid line morphological detection + removal
2. Dual Frangi vesselness (σ for thin vs thick)
3. Hierarchical masking
4. CCL filtering with aspect ratio constraints
5. Skeletonization with pruning

Generates intermediate debug images at each step.
"""

import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize
from pathlib import Path
import os

# Import existing utilities
from src.skeleton_graph import prune_spurs
from src.component_masker import _erase_rect


class FrangiDualScalePipeline:
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
        
        # Debug counter
        self.step_num = 0
    
    def save_debug(self, image: np.ndarray, name: str, description: str = ""):
        """Save intermediate debug image."""
        self.step_num += 1
        filename = f"{self.step_num:02d}_{name}.png"
        path = self.debug_dir / filename
        cv2.imwrite(str(path), image)
        print(f"  [{self.step_num:02d}] {name:<40} {description}")
        return path
    
    # ===== STEP 1: Pre-processing =====
    
    def preprocess(self) -> np.ndarray:
        """Enhanced grayscale with CLAHE."""
        print("\n[STEP 1] Pre-processing...")
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(self.gray)
        
        self.save_debug(self.gray, "00_original_gray", "Original grayscale")
        self.save_debug(enhanced, "01_enhanced_clahe", "After CLAHE contrast enhancement")
        
        return enhanced
    
    # ===== STEP 2: Grid Line Detection & Removal =====
    
    def remove_grid_lines(self, binary: np.ndarray, L: int = 150) -> np.ndarray:
        """Morphological grid line detection and removal."""
        print("\n[STEP 2] Grid line detection and removal...")
        
        self.save_debug(binary, "02_binary_otsu", "Otsu binary (with grid lines)")
        
        # Horizontal opening: long horizontal structures
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
        h_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        self.save_debug(h_opened, "03_h_opened_L150", f"Horizontal opening (L={L})")
        
        # Vertical opening
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
        v_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        self.save_debug(v_opened, "04_v_opened_L150", f"Vertical opening (L={L})")
        
        # Filter to keep only thin structures
        # H: erode in Y to remove anything > 4px tall
        h_erosion_kernel = np.ones((4, 1), np.uint8)
        h_eroded = cv2.erode(h_opened, h_erosion_kernel)
        h_dilated = cv2.dilate(h_eroded, np.ones((8, 1), np.uint8))
        h_grid = cv2.subtract(h_opened, h_dilated)
        self.save_debug(h_grid, "05_h_grid_thin_only", "Horizontal grid (thin lines)")
        
        # V: erode in X to remove anything > 4px wide
        v_erosion_kernel = np.ones((1, 4), np.uint8)
        v_eroded = cv2.erode(v_opened, v_erosion_kernel)
        v_dilated = cv2.dilate(v_eroded, np.ones((1, 8), np.uint8))
        v_grid = cv2.subtract(v_opened, v_dilated)
        self.save_debug(v_grid, "06_v_grid_thin_only", "Vertical grid (thin lines)")
        
        # Combine grid masks
        grid_mask = cv2.bitwise_or(h_grid, v_grid)
        self.save_debug(grid_mask, "07_combined_grid", "Combined H+V grid mask")
        
        # Dilate grid mask to ensure full coverage
        grid_mask = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=1)
        self.save_debug(grid_mask, "08_dilated_grid", "Dilated grid mask (coverage)")
        
        # Remove grid from binary
        clean_binary = cv2.subtract(binary, grid_mask)
        self.save_debug(clean_binary, "09_binary_grid_removed", "Binary after grid removal")
        
        return clean_binary
    
    # ===== STEP 3: Frangi Vesselness =====
    
    def frangi_segments(self, enhanced: np.ndarray) -> np.ndarray:
        """Frangi for thin wire segments (σ = 2.5 to 5.5)."""
        print("\n[STEP 3a] Frangi vesselness for segments (σ = 2.5–5.5)...")
        
        # Convert to float
        img_float = enhanced.astype(np.float64) / 255.0
        
        # Frangi with small sigma range
        sigmas = np.arange(2.5, 6.0, 0.5)
        print(f"  Sigmas: {sigmas}")
        
        response = frangi(
            img_float,
            sigmas=sigmas,
            alpha=0.5,
            beta=0.5,
            gamma=15,
            black_ridges=True
        )
        
        # Save raw Frangi response (normalized for visualization)
        response_viz = (response * 255).astype(np.uint8)
        self.save_debug(response_viz, "10_frangi_seg_raw", "Frangi response (segments, raw)")
        
        # Threshold
        THRESH = 0.05
        seg_mask = (response > THRESH).astype(np.uint8) * 255
        self.save_debug(seg_mask, "11_frangi_seg_threshed", f"Frangi segments (thresh={THRESH})")
        
        return seg_mask, response
    
    def frangi_routes(self, enhanced: np.ndarray) -> np.ndarray:
        """Frangi for thick route bundles (σ = 10 to 38)."""
        print("\n[STEP 3b] Frangi vesselness for routes (σ = 10–38)...")
        
        # Convert to float
        img_float = enhanced.astype(np.float64) / 255.0
        
        # Frangi with large sigma range
        sigmas = np.arange(10.0, 42.0, 4.0)
        print(f"  Sigmas: {sigmas}")
        
        response = frangi(
            img_float,
            sigmas=sigmas,
            alpha=0.5,
            beta=0.5,
            gamma=15,
            black_ridges=True
        )
        
        # Save raw response
        response_viz = (response * 255).astype(np.uint8)
        self.save_debug(response_viz, "12_frangi_route_raw", "Frangi response (routes, raw)")
        
        # Threshold
        THRESH = 0.08
        route_mask = (response > THRESH).astype(np.uint8) * 255
        self.save_debug(route_mask, "13_frangi_route_threshed", f"Frangi routes (thresh={THRESH})")
        
        return route_mask, response
    
    # ===== STEP 4: Gate with Clean Binary =====
    
    def gate_with_binary(self, seg_mask: np.ndarray, route_mask: np.ndarray, 
                         clean_binary: np.ndarray) -> tuple:
        """AND-gate Frangi masks with clean binary."""
        print("\n[STEP 4] Gating Frangi masks with clean binary...")
        
        seg_gated = cv2.bitwise_and(seg_mask, clean_binary)
        self.save_debug(seg_gated, "14_seg_gated", "Segment mask gated with binary")
        
        route_gated = cv2.bitwise_and(route_mask, clean_binary)
        self.save_debug(route_gated, "15_route_gated", "Route mask gated with binary")
        
        return seg_gated, route_gated
    
    # ===== STEP 5: Hierarchical Separation =====
    
    def separate_hierarchy(self, seg_gated: np.ndarray, route_gated: np.ndarray) -> np.ndarray:
        """Remove route region from segment mask."""
        print("\n[STEP 5] Hierarchical separation (route exclusion)...")
        
        # Expand route mask to cover boundary bleed
        route_exclusion = cv2.dilate(route_gated, np.ones((20, 20), np.uint8), iterations=2)
        self.save_debug(route_exclusion, "16_route_exclusion", "Route exclusion zone (20px dilation)")
        
        # Subtract route from segments
        seg_clean = cv2.subtract(seg_gated, route_exclusion)
        self.save_debug(seg_clean, "17_seg_hierarchical", "Segments after route removal")
        
        return seg_clean
    
    # ===== STEP 6: Morphological Cleanup =====
    
    def morphological_cleanup(self, seg_clean: np.ndarray, route_gated: np.ndarray) -> tuple:
        """Close gaps and fill holes."""
        print("\n[STEP 6] Morphological cleanup...")
        
        # Segments: close small gaps
        seg_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_closed = cv2.morphologyEx(seg_clean, cv2.MORPH_CLOSE, seg_close_kernel, iterations=1)
        self.save_debug(seg_closed, "18_seg_morphclose", "Segments after morphological close")
        
        # Route: fill holes
        route_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        route_filled = cv2.morphologyEx(route_gated, cv2.MORPH_CLOSE, route_close_kernel, iterations=3)
        self.save_debug(route_filled, "19_route_morphclose", "Route after morphological close")
        
        return seg_closed, route_filled
    
    # ===== STEP 7: CCL Filtering =====
    
    def ccl_filter_segments(self, seg_mask: np.ndarray) -> np.ndarray:
        """Filter segments by aspect ratio and area."""
        print("\n[STEP 7a] Connected Components filtering (segments)...")
        
        n, labels, stats, _ = cv2.connectedComponentsWithStats(seg_mask, connectivity=8)
        print(f"  Found {n-1} components")
        
        final_mask = np.zeros_like(seg_mask)
        
        kept = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            
            # Filter by area
            if area < 60:
                continue
            
            # Filter by aspect ratio
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect < 2.5:
                continue
            
            # Filter by fill ratio
            fill = area / (w * h)
            if fill > 0.65:
                continue
            
            final_mask[labels == i] = 255
            kept += 1
        
        print(f"  Kept {kept} segments after filtering")
        self.save_debug(final_mask, "20_seg_ccl_filtered", f"Segments after CCL filtering ({kept} kept)")
        
        return final_mask
    
    def ccl_filter_routes(self, route_mask: np.ndarray) -> np.ndarray:
        """Filter routes by area (keep largest components)."""
        print("\n[STEP 7b] Connected Components filtering (routes)...")
        
        n, labels, stats, _ = cv2.connectedComponentsWithStats(route_mask, connectivity=8)
        print(f"  Found {n-1} components")
        
        final_mask = np.zeros_like(route_mask)
        
        # Sort by area, keep large components
        areas_idx = sorted([(stats[i][4], i) for i in range(1, n)], reverse=True)
        
        kept = 0
        for area, i in areas_idx:
            if area < 2000:
                break
            final_mask[labels == i] = 255
            kept += 1
        
        print(f"  Kept {kept} route components (min area 2000)")
        self.save_debug(final_mask, "21_route_ccl_filtered", f"Routes after CCL filtering ({kept} kept)")
        
        return final_mask
    
    # ===== STEP 8: Skeletonization =====
    
    def skeletonize_masks(self, seg_mask: np.ndarray, route_mask: np.ndarray) -> tuple:
        """Convert masks to 1-pixel-wide skeletons."""
        print("\n[STEP 8] Skeletonization...")
        
        # Segments
        skel_seg = skeletonize(seg_mask > 0).astype(np.uint8) * 255
        self.save_debug(skel_seg, "22_seg_skeleton_raw", "Segments skeleton (raw)")
        
        skel_seg = prune_spurs(skel_seg, min_branch_length=12)
        self.save_debug(skel_seg, "23_seg_skeleton_pruned", "Segments skeleton (spurs pruned)")
        
        # Routes
        skel_route = skeletonize(route_mask > 0).astype(np.uint8) * 255
        self.save_debug(skel_route, "24_route_skeleton_raw", "Routes skeleton (raw)")
        
        skel_route = prune_spurs(skel_route, min_branch_length=25)
        self.save_debug(skel_route, "25_route_skeleton_pruned", "Routes skeleton (spurs pruned)")
        
        return skel_seg, skel_route
    
    # ===== MAIN PIPELINE =====
    
    def run(self):
        """Execute full pipeline."""
        print("\n" + "="*70)
        print("DUAL-SCALE FRANGI PIPELINE FOR WIRE DETECTION")
        print("="*70)
        
        # Step 1: Preprocess
        enhanced = self.preprocess()
        
        # Step 2: Grid removal
        binary = cv2.threshold(enhanced, 0, 255, 
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        clean_binary = self.remove_grid_lines(binary, L=150)
        
        # Step 3: Frangi
        seg_mask, seg_response = self.frangi_segments(enhanced)
        route_mask, route_response = self.frangi_routes(enhanced)
        
        # Step 4: Gate with binary
        seg_gated, route_gated = self.gate_with_binary(seg_mask, route_mask, clean_binary)
        
        # Step 5: Hierarchical separation
        seg_clean = self.separate_hierarchy(seg_gated, route_gated)
        
        # Step 6: Morphology
        seg_closed, route_filled = self.morphological_cleanup(seg_clean, route_gated)
        
        # Step 7: CCL filtering
        seg_final = self.ccl_filter_segments(seg_closed)
        route_final = self.ccl_filter_routes(route_filled)
        
        # Step 8: Skeletonization
        skel_seg, skel_route = self.skeletonize_masks(seg_final, route_final)
        
        # Final composite visualization
        print("\n[FINAL] Creating composite visualizations...")
        
        # Overlay skeleton on original
        skel_seg_vis = cv2.cvtColor(seg_final, cv2.COLOR_GRAY2BGR)
        skel_seg_vis[skel_seg > 0] = [0, 255, 0]  # Green
        self.save_debug(skel_seg_vis, "26_seg_skeleton_overlay", "Segments skeleton overlaid on mask")
        
        skel_route_vis = cv2.cvtColor(route_final, cv2.COLOR_GRAY2BGR)
        skel_route_vis[skel_route > 0] = [0, 0, 255]  # Red
        self.save_debug(skel_route_vis, "27_route_skeleton_overlay", "Routes skeleton overlaid on mask")
        
        # Combined on original image
        result = self.img_color.copy()
        result[skel_seg > 0] = [0, 255, 0]  # Green for segments
        result[skel_route > 0] = [0, 0, 255]  # Red for routes
        self.save_debug(result, "28_final_combined_overlay", "Final result: segments (green) + routes (red)")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Debug images saved to: {self.debug_dir}")
        
        return {
            "skel_segments": skel_seg,
            "skel_routes": skel_route,
            "final_segments": seg_final,
            "final_routes": route_final,
        }


if __name__ == "__main__":
    # Find the image
    img_path = "real_images/cemm-wire-harness-assembly.jpeg"
    
    if not Path(img_path).exists():
        print(f"Error: Image not found at {img_path}")
        exit(1)
    
    # Run pipeline
    pipeline = FrangiDualScalePipeline(img_path)
    results = pipeline.run()
    
    print(f"\nResults saved. Skeletons in memory.")
    print(f"Segment skeleton shape: {results['skel_segments'].shape}")
    print(f"Route skeleton shape: {results['skel_routes'].shape}")
