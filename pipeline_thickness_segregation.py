#!/usr/bin/env python3
"""
Revised Thickness-Based Segregation Pipeline

Uses clean binary from Step 2 and applies 2-step thickness separation:
1. Filter 1: Extract segments (thickness 4-12px) → eliminates grid lines and routes
2. Filter 2: Extract routes (thickness 20-80px) → eliminates segments and fine components
3. Component elimination: Remove blob-like structures (high fill ratio, low aspect)

Hierarchy: Grid < Segments < Routes ≤ Components
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from pathlib import Path

from src.skeleton_graph import prune_spurs


class ThicknessSegregationPipeline:
    """Thickness-based wire segregation with component elimination."""
    
    def __init__(self, input_image_path: str, output_dir: str = "real_images"):
        self.input_path = Path(input_image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.debug_dir = self.output_dir / "debug_v2"
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
        
        if image.dtype == bool:
            image = image.astype(np.uint8) * 255
        
        cv2.imwrite(str(path), image)
        print(f"  [{self.step_num:02d}] {name:<40} {description}")
        return path
    
    # ===== STEP 1: Create Clean Binary =====
    
    def create_clean_binary(self) -> np.ndarray:
        """Create clean binary with grid lines removed."""
        print("\n[STEP 1] Creating clean binary...")
        
        # Otsu binary
        _, binary = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        self.save_debug(self.gray, "00_original_gray", "Original grayscale")
        self.save_debug(binary, "01_binary_otsu", "Otsu binary (with grid)")
        
        # Grid removal via morphological opening
        print("  Removing grid lines...")
        
        # Horizontal grid
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
        h_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        h_eroded = cv2.erode(h_opened, np.ones((4, 1), np.uint8))
        h_dilated = cv2.dilate(h_eroded, np.ones((8, 1), np.uint8))
        h_grid = cv2.subtract(h_opened, h_dilated)
        
        # Vertical grid
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
        v_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        v_eroded = cv2.erode(v_opened, np.ones((1, 4), np.uint8))
        v_dilated = cv2.dilate(v_eroded, np.ones((1, 8), np.uint8))
        v_grid = cv2.subtract(v_opened, v_dilated)
        
        # Combine and remove
        grid_mask = cv2.bitwise_or(h_grid, v_grid)
        grid_mask = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=1)
        clean_binary = cv2.subtract(binary, grid_mask)
        
        self.save_debug(grid_mask, "02_grid_mask", "Detected grid lines")
        self.save_debug(clean_binary, "03_binary_clean", "Binary after grid removal")
        
        print(f"  Grid removed: {np.sum(binary > 0) - np.sum(clean_binary > 0)} pixels")
        
        return clean_binary
    
    # ===== STEP 2: Thickness-Based Segregation =====
    
    def extract_segments_by_thickness(self, binary: np.ndarray) -> np.ndarray:
        """
        Extract thin wires (segments) using width threshold.
        
        Grid lines are 1-2px — removed by 5x5 lower-bound filter.
        Routes are 20px+ — removed by 13x13 upper-bound filter.
        Segments (4-12px) are what remains between those two.
        """
        print("\n[STEP 2a] Extracting segments (4-12px thickness)...")
        
        # Lower bound: 5x5 removes structures < 4px (grid lines), then restore
        grid_filter = cv2.erode(binary, np.ones((5, 5), np.uint8), iterations=1)
        grid_filter_restored = cv2.dilate(grid_filter, np.ones((5, 5), np.uint8), iterations=1)
        binary_no_grid = cv2.subtract(binary, cv2.subtract(binary, grid_filter_restored))
        self.save_debug(binary_no_grid, "04a_segments_no_grid", "After lower-bound filter (5x5, removes <4px grid lines)")
        
        # Upper bound: 13x13 erode keeps only structures > 12px (routes/components)
        thick_only = cv2.erode(binary_no_grid, np.ones((15, 15), np.uint8), iterations=1)
        thick_restored = cv2.dilate(thick_only, np.ones((15, 15), np.uint8), iterations=1)
        
        # Segments = what's left after removing thick structures
        segments_raw = cv2.subtract(binary_no_grid, thick_restored)
        self.save_debug(segments_raw, "04b_segments_raw", "Segments only (4-12px, routes subtracted)")
        
        return segments_raw
    
    def extract_routes_by_thickness(self, binary: np.ndarray) -> np.ndarray:
        """
        Extract thick routes using width threshold.
        
        Routes are typically 20-80px thick.
        Use erosion kernel: only structures > 12px thick survive 13px erosion.
        """
        print("\n[STEP 2b] Extracting routes (thickness 20-80px)...")
        
        # Erode by 13px to remove thin structures (segments) and keep only thick routes
        eroded = cv2.erode(binary, np.ones((13, 13), np.uint8), iterations=1)
        self.save_debug(eroded, "06_eroded_13x13", "After 13×13 erosion (keeps thick only)")
        
        # Dilate back to original size
        routes_raw = cv2.dilate(eroded, np.ones((13, 13), np.uint8), iterations=1)
        
        self.save_debug(routes_raw, "07_routes_raw", "Raw routes (20-80px range)")
        
        return routes_raw
    
    # ===== STEP 3: Component Elimination =====
    
    def eliminate_components(self, mask: np.ndarray, component_type: str = "segments") -> np.ndarray:
        """
        Eliminate blob-like structures (components: connectors, clips).
        
        Wires have low solidity (convex hull contains empty space due to curves).
        Blobs have high solidity (fill their convex hull).
        """
        print(f"\n[STEP 3] Component elimination ({component_type})...")
        
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        print(f"  Found {n-1} components")
        
        final_mask = np.zeros_like(mask)
        kept = 0
        removed_components = []
        
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            
            # Skip if too small
            if area < 30:
                removed_components.append(("area<30", area))
                continue
            
            # Fill ratio: wires are sparse, components are dense
            fill = area / (w * h)
            
            # Solidity: ratio of area to convex hull area
            # Wires (straight/curved) have low solidity (~0.15-0.55)
            # Blobs (compact shapes) have high solidity (~0.85-1.0)
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 1.0
            
            # For SEGMENTS: wires should be sparse with low solidity (accommodate curves)
            if component_type == "segments":
                # Primary filter: fill ratio (wires ~30-50%, blobs ~70%+)
                # Secondary filter: solidity for very compact shapes (> 0.75 = likely blob)
                if fill > 0.62:
                    removed_components.append(("fill>0.62", fill))
                    continue
                if solidity > 0.75:
                    removed_components.append(("solidity>0.75", solidity))
                    continue
            
            # For ROUTES: wires should have low solidity, medium-high fill
            elif component_type == "routes":
                # Remove isolated components on far left (artifact connectors)
                if x < 150:
                    removed_components.append(("x<150", x))
                    continue
                # Rectangular components: compact (aspect < 3.0) AND dense (fill > 0.55)
                aspect = max(w, h) / max(min(w, h), 1)
    
                # Large route sections are always kept regardless of shape
                if area > 5000:
                    pass  # skip filtering for large components — they're legitimate route sections
                # Small-to-medium compact+hollow components = panels/electronic components
                elif aspect < 4.0 and fill < 0.70 and solidity < 0.82:
                    removed_components.append(("panel:compact+hollow", fill))
                    continue
                # Route: solidity < 0.80 (thicker wires can have higher solidity), fill < 0.72
                # if solidity > 0.80:
                #     removed_components.append(("solidity>0.80", solidity))
                #     continue
                # if fill > 0.72:
                #     removed_components.append(("fill>0.72", fill))
                #     continue
            
            final_mask[labels == i] = 255
            kept += 1
        
        print(f"  Kept {kept} components, removed {n-1-kept}")
        if removed_components:
            # Print sample removals
            removal_counts = {}
            for reason, _ in removed_components:
                removal_counts[reason] = removal_counts.get(reason, 0) + 1
            for reason, count in sorted(removal_counts.items()):
                print(f"    - {reason}: {count}")
        
        return final_mask
    
    # ===== STEP 4: Morphological Cleanup =====
    
    def cleanup_morphology(self, segments: np.ndarray, routes: np.ndarray) -> tuple:
        """Close small gaps."""
        print("\n[STEP 4] Morphological cleanup...")
        
        # Segments: light morphological close
        seg_closed = cv2.morphologyEx(
            segments, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1
        )
        self.save_debug(seg_closed, "11_seg_morphclose", "Segments after morphological close")
        
        # Routes: heavier morphological close
        route_closed = cv2.morphologyEx(
            routes, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            iterations=2
        )
        self.save_debug(route_closed, "12_route_morphclose", "Routes after morphological close")
        
        return seg_closed, route_closed
    
    # ===== STEP 5: Skeletonization =====
    
    def skeletonize_and_prune(self, segments: np.ndarray, routes: np.ndarray) -> tuple:
        """Convert to skeletons and prune spurs."""
        print("\n[STEP 5] Skeletonization and spur pruning...")
        
        # Segments
        skel_seg = skeletonize(segments > 0).astype(np.uint8) * 255
        self.save_debug(skel_seg, "13_seg_skeleton_raw", "Segment skeleton (raw)")
        
        skel_seg = prune_spurs(skel_seg, min_branch_length=12)
        skel_seg = (skel_seg * 255).astype(np.uint8)
        self.save_debug(skel_seg, "14_seg_skeleton_pruned", "Segment skeleton (pruned)")
        
        # Routes
        skel_route = skeletonize(routes > 0).astype(np.uint8) * 255
        self.save_debug(skel_route, "15_route_skeleton_raw", "Route skeleton (raw)")
        
        skel_route = prune_spurs(skel_route, min_branch_length=20)
        skel_route = (skel_route * 255).astype(np.uint8)
        self.save_debug(skel_route, "16_route_skeleton_pruned", "Route skeleton (pruned)")
        
        return skel_seg, skel_route
    
    # ===== VISUALIZATION =====
    
    def create_visualizations(self, seg_final: np.ndarray, route_final: np.ndarray,
                             skel_seg: np.ndarray, skel_route: np.ndarray):
        """Create composite visualizations."""
        print("\n[FINAL] Creating composite visualizations...")
        
        # Final overlay on original
        result = self.img_color.copy()
        result[skel_seg > 0] = [0, 255, 0]  # Green for segments
        result[skel_route > 0] = [0, 0, 255]  # Red for routes
        self.save_debug(result, "17_final_overlay_color", 
                       "Final: segments (green) + routes (red)")
        
        # Masks side-by-side
        seg_3ch = cv2.cvtColor(seg_final, cv2.COLOR_GRAY2BGR)
        route_3ch = cv2.cvtColor(route_final, cv2.COLOR_GRAY2BGR)
        masks_sbs = np.hstack([seg_3ch, route_3ch])
        self.save_debug(masks_sbs, "18_masks_sbs", "Masks: segments (L) vs routes (R)")
        
        # Skeletons side-by-side
        skel_seg_3ch = cv2.cvtColor(skel_seg, cv2.COLOR_GRAY2BGR)
        skel_route_3ch = cv2.cvtColor(skel_route, cv2.COLOR_GRAY2BGR)
        skels_sbs = np.hstack([skel_seg_3ch, skel_route_3ch])
        self.save_debug(skels_sbs, "19_skeletons_sbs", "Skeletons: segments (L) vs routes (R)")
        
        # Composite: clean binary + final segmentation
        clean_bin_3ch = cv2.cvtColor(self.clean_binary, cv2.COLOR_GRAY2BGR)
        self.save_debug(np.hstack([clean_bin_3ch, result]), "20_before_after",
                       "Before (clean binary) vs After (segmentation)")
    
    # ===== MAIN EXECUTION =====
    
    def run(self):
        """Execute full pipeline."""
        print("\n" + "="*70)
        print("THICKNESS-BASED SEGREGATION PIPELINE")
        print("Hierarchy: Grid < Segments < Routes ≤ Components")
        print("="*70)
        
        # Step 1: Clean binary
        self.clean_binary = self.create_clean_binary()
        
        # Step 2: Thickness segregation
        segments_raw = self.extract_segments_by_thickness(self.clean_binary)
        routes_raw = self.extract_routes_by_thickness(self.clean_binary)
        
        # Step 3: Component elimination
        segments_cleaned = self.eliminate_components(segments_raw, component_type="segments")
        self.save_debug(segments_cleaned, "08_seg_components_removed", 
                       "Segments after component elimination")
        
        routes_cleaned = self.eliminate_components(routes_raw, component_type="routes")
        self.save_debug(routes_cleaned, "09_route_components_removed",
                       "Routes after component elimination")
        
        # Step 4: Morphology
        seg_closed, route_closed = self.cleanup_morphology(segments_cleaned, routes_cleaned)
        
        # Step 5: Skeletonize
        skel_seg, skel_route = self.skeletonize_and_prune(seg_closed, route_closed)
        
        # Visualization
        self.create_visualizations(seg_closed, route_closed, skel_seg, skel_route)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Debug images: {self.debug_dir}")
        
        return {
            "skel_segments": skel_seg,
            "skel_routes": skel_route,
            "mask_segments": seg_closed,
            "mask_routes": route_closed,
        }


if __name__ == "__main__":
    img_path = "real_images/cemm-wire-harness-assembly.jpeg"
    
    if not Path(img_path).exists():
        print(f"Error: {img_path} not found")
        exit(1)
    
    pipeline = ThicknessSegregationPipeline(img_path)
    results = pipeline.run()
    
    seg_pixels = np.sum(results['skel_segments'] > 0)
    route_pixels = np.sum(results['skel_routes'] > 0)
    print(f"\nSegment skeleton: {seg_pixels} pixels")
    print(f"Route skeleton: {route_pixels} pixels")
