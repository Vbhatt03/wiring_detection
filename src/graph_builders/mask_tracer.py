"""Mask-based wire tracer — BFS label propagation.

Pipeline:
    1. Morphological CLOSE  — bridges dash-dot gaps in thick wires
    2. CCL area filter      — removes tiny noise fragments
    3. Seed component circles onto wire pixels
    4. Multi-source BFS     — flood each component's label outward along
                              connected white pixels
    5. Edge detection       — wherever two different labels share a pixel
                              boundary → wire connection between those components
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

import cv2
import networkx as nx
import numpy as np

# ── Tunable parameters ──────────────────────────────────────────────────────
# px seed-circle radius around each component center
COMPONENT_EXPAND = 25

# Wire-pixel blobs below this area are treated as noise and removed
MIN_WIRE_AREA = 80
MAX_SEED_REACH=200
# Node types that are property annotations, NOT physical wire endpoints.
# Tape nodes (VT-BK, AT-BK, COT-BK …) sit on the wire and label its type;
# they must not become connection endpoints.
ENDPOINT_EXCLUDED_TYPES: Set[str] = {"length", "dimension", "annotation"}
# ────────────────────────────────────────────────────────────────────────────

_DIRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
          (-1, -1), (-1, 1), (1, -1), (1, 1)]


# ── Step 1: mask cleaning ────────────────────────────────────────────────────

def clean_wire_mask(wire_mask: np.ndarray) -> np.ndarray:
    """3-pass closing to bridge dash-dot gaps in all directions.

    Pass 1+2: directional rectangles target horizontal and vertical dashes.
    Pass 3: isotropic ellipse cleans up diagonal/irregular gaps.
    No opening applied — opening erases thin wire sections.
    """
    # Horizontal dashes (the long wire bus runs left-right)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, k_h, iterations=1)

    # Vertical dashes
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_v, iterations=1)

    # Isotropic close for diagonal/irregular gaps
    k_iso = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_iso, iterations=2)

    return mask

def _remove_tiny_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Zero-out CCL blobs whose area is below min_area (text chars, arrowheads)."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )
    out = mask.copy()
    for bid in range(1, n):
        if int(stats[bid, cv2.CC_STAT_AREA]) < min_area:
            out[lbl == bid] = 0
    return out


# ── Step 2-3: seed + BFS ─────────────────────────────────────────────────────

def _seed_and_bfs(
    cleaned: np.ndarray,
    eligible_ids: List[str],          # ONLY connectors/clips/junctions — no tapes
    nodes_dict: Dict,
    expand: int,
) -> Tuple[np.ndarray, Set[Tuple[int, int]], Dict[int, str]]:
    h, w = cleaned.shape
    label_img: np.ndarray = np.where(cleaned > 0, 0, -1).astype(np.int32)
    queue: deque = deque()
    seed_idx_to_nodeid: Dict[int, str] = {}
    seed_idx = 0

    for nid in eligible_ids:
        info = nodes_dict[nid]
        seed_idx += 1
        seed_idx_to_nodeid[seed_idx] = nid

        cx, cy = int(info["x"]), int(info["y"])
        r = expand + (15 if info.get("type") == "junction" else 0)

        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        xs_roi = np.arange(x0, x1)
        ys_roi = np.arange(y0, y1)
        gx, gy = np.meshgrid(xs_roi, ys_roi)

        in_circle = (gx - cx) ** 2 + (gy - cy) ** 2 <= r ** 2
        seeable = in_circle & (label_img[y0:y1, x0:x1] == 0)

        ys_seed, xs_seed = gy[seeable], gx[seeable]
        label_img[ys_seed, xs_seed] = seed_idx
        for y, x in zip(ys_seed.ravel().tolist(), xs_seed.ravel().tolist()):
            queue.append((y, x, seed_idx))

        # Fallback: if center is inside an erased bbox (connector body was masked out),
        # scan a wider radius for the nearest wire pixel approaching the component.
        if len(ys_seed) == 0:
            ey0 = max(0, cy - MAX_SEED_REACH)
            ey1 = min(h, cy + MAX_SEED_REACH + 1)
            ex0 = max(0, cx - MAX_SEED_REACH)
            ex1 = min(w, cx + MAX_SEED_REACH + 1)
            roi_ys, roi_xs = np.where(label_img[ey0:ey1, ex0:ex1] == 0)
            if len(roi_ys) > 0:
                dists = (roi_ys + ey0 - cy) ** 2 + (roi_xs + ex0 - cx) ** 2
                nearest = np.argsort(dists)[:20]
                seeded = 0
                for ni in nearest:
                    wy, wx = int(roi_ys[ni] + ey0), int(roi_xs[ni] + ex0)
                    if label_img[wy, wx] == 0:
                        label_img[wy, wx] = seed_idx
                        queue.append((wy, wx, seed_idx))
                        seeded += 1
                print(f"    [MaskTrace] {nid}: erased center, "
                      f"fallback seeded {seeded} px "
                      f"(nearest wire ~{int(dists[nearest[0]]**0.5)}px away)")
            else:
                print(f"    [MaskTrace] {nid}: no wire within {MAX_SEED_REACH}px")

    connections: Set[Tuple[int, int]] = set()
    while queue:
        y, x, lbl = queue.popleft()
        for dy, dx in _DIRS8:
            ny, nx_ = y + dy, x + dx
            if ny < 0 or ny >= h or nx_ < 0 or nx_ >= w:
                continue
            v = label_img[ny, nx_]
            if v == 0:
                label_img[ny, nx_] = lbl
                queue.append((ny, nx_, lbl))
            elif v > 0 and v != lbl:
                a, b = (lbl, v) if lbl < v else (v, lbl)
                connections.add((a, b))

    return label_img, connections, seed_idx_to_nodeid


# ── Step 4: path points ───────────────────────────────────────────────────────

def _path_pts_between(
    label_img: np.ndarray,
    idx_a: int,
    idx_b: int,
    pos_a: Tuple[int, int],
    pos_b: Tuple[int, int],
    max_pts: int = 150,
) -> List[Tuple[int, int]]:
    """Ordered (x, y) points sampled along the wire between component A and B.

    Collects all pixels claimed by either label, projects them onto the A→B
    axis, and subsamples to ≤ max_pts points.  The result is used by
    assign_wire_properties to find tape/length labels within 80 px of the wire.
    """
    mask = (label_img == idx_a) | (label_img == idx_b)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [pos_a, pos_b]

    pts = np.column_stack((xs, ys)).astype(np.float32)

    ax, ay = float(pos_a[0]), float(pos_a[1])
    bx, by = float(pos_b[0]), float(pos_b[1])
    ab = np.array([bx - ax, by - ay], dtype=np.float32)
    ab_len = float(np.linalg.norm(ab))
    if ab_len < 1:
        return [pos_a, pos_b]

    ab_unit = ab / ab_len
    proj = (pts - np.array([ax, ay])) @ ab_unit
    sampled = pts[np.argsort(proj)][:: max(1, len(pts) // max_pts)]

    return [pos_a] + [(int(p[0]), int(p[1])) for p in sampled] + [pos_b]


# ── Public entry point ────────────────────────────────────────────────────────

def trace_mask_connectivity(
    wire_mask: np.ndarray,
    nodes_dict: Dict,
    component_expand: int = COMPONENT_EXPAND,
) -> nx.Graph:
    h, w = wire_mask.shape[:2]

    cleaned = clean_wire_mask(wire_mask)
    cleaned = _remove_tiny_blobs(cleaned, MIN_WIRE_AREA)
    cv2.imwrite("debug_cleaned_mask.png", cleaned)

    # Pre-filter: only connectors, clips, junctions are wire endpoints.
    # Tape/length/annotation nodes sit ON wires — never connection endpoints.
    eligible_ids = [
        nid for nid, info in nodes_dict.items()
        if info.get("type", "") not in ENDPOINT_EXCLUDED_TYPES
    ]
    print(f"    [MaskTrace] {len(eligible_ids)} eligible endpoint nodes "
          f"({len(nodes_dict) - len(eligible_ids)} annotation nodes excluded)")

    label_img, connections, seed_idx_to_nodeid = _seed_and_bfs(
        cleaned, eligible_ids, nodes_dict, component_expand
    )
    print(f"    [MaskTrace] Seeded {len(seed_idx_to_nodeid)} components, "
          f"BFS found {len(connections)} raw connections")

    g = nx.Graph()
    for nid, info in nodes_dict.items():
        g.add_node(nid, **info)

    for idx_a, idx_b in connections:
        na = seed_idx_to_nodeid[idx_a]
        nb = seed_idx_to_nodeid[idx_b]
        if na == nb:
            continue
        type_a = nodes_dict[na].get("type", "")
        type_b = nodes_dict[nb].get("type", "")
        if type_a == "clip" and type_b == "clip":
            continue

        pos_a = (nodes_dict[na]["x"], nodes_dict[na]["y"])
        pos_b = (nodes_dict[nb]["x"], nodes_dict[nb]["y"])
        dist_px = float(np.hypot(pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]))
        path_pts = _path_pts_between(label_img, idx_a, idx_b, pos_a, pos_b)

        if not g.has_edge(na, nb):
            g.add_edge(
                na, nb,
                path_pts=path_pts,
                path_length_px=dist_px,
                wire_type=None,
                length_mm=None,
            )
            print(f"    [MaskTrace] {na} ↔ {nb}  dist={dist_px:.0f}px")

    print(f"    [MaskTrace] Result: {g.number_of_edges()} edges "
          f"across {g.number_of_nodes()} components")
    palette = [
        (255, 60, 60), (60, 255, 60), (60, 60, 255),
        (255, 255, 60), (255, 60, 255), (60, 255, 255),
        (255, 160, 60), (60, 255, 160), (160, 60, 255),
        (200, 200, 60), (60, 200, 200), (200, 60, 200),
    ]
    dbg = np.zeros((h, w, 3), dtype=np.uint8)
    for sidx, nid in seed_idx_to_nodeid.items():
        col = palette[sidx % len(palette)]
        dbg[label_img == sidx] = col
        # Draw component center label
        cx, cy = int(nodes_dict[nid]["x"]), int(nodes_dict[nid]["y"])
        cv2.circle(dbg, (cx, cy), 5, (255, 255, 255), -1)
        cv2.putText(dbg, nid.split("-")[0] + str(sidx),
                    (cx + 6, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.imwrite("debug_label_regions.png", dbg)
    print(f"    [MaskTrace] Saved debug_label_regions.png")
    return g