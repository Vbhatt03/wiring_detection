"""Skeletonization and graph extraction utilities for segment diagrams."""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import networkx as nx
import numpy as np
import sknw
from skimage.morphology import skeletonize


OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _neighbors(y: int, x: int, shape: Sequence[int]) -> Iterable[Tuple[int, int]]:
    h, w = shape
    for dy, dx in OFFSETS_8:
        ny, nx_ = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx_ < w:
            yield ny, nx_


def _neighbor_count(skel: np.ndarray, y: int, x: int) -> int:
    return sum(1 for ny, nx_ in _neighbors(y, x, skel.shape) if skel[ny, nx_])


def _prune_isolated_pixels(skel: np.ndarray) -> np.ndarray:
    out = skel.copy()
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        if _neighbor_count(skel, y, x) == 0:
            out[y, x] = False
    return out


def _trace_branch(skel: np.ndarray, start: Tuple[int, int], min_branch_length: int) -> Optional[List[Tuple[int, int]]]:
    path = [start]
    prev = None
    cur = start
    while True:
        y, x = cur
        nbrs = [(ny, nx_) for ny, nx_ in _neighbors(y, x, skel.shape) if skel[ny, nx_] and (ny, nx_) != prev]
        degree = _neighbor_count(skel, y, x)
        if len(path) > 1 and degree >= 3:
            break
        if not nbrs:
            break
        nxt = nbrs[0]
        prev, cur = cur, nxt
        path.append(cur)
        if len(path) > min_branch_length:
            return None
    return path


def prune_spurs(skeleton: np.ndarray, min_branch_length: int = 10) -> np.ndarray:
    """Iteratively remove short endpoint-to-junction branches."""
    skel = skeleton.copy().astype(bool)
    changed = True
    while changed:
        changed = False
        endpoints = [(y, x) for y, x in zip(*np.where(skel)) if _neighbor_count(skel, y, x) == 1]
        for ep in endpoints:
            if not skel[ep[0], ep[1]]:
                continue
            branch = _trace_branch(skel, ep, min_branch_length=min_branch_length)
            if branch is not None and len(branch) < min_branch_length:
                for y, x in branch:
                    skel[y, x] = False
                changed = True
    return skel


def skeletonize_segment_mask(
    segment_mask: np.ndarray,
    j20_hint: Optional[Tuple[int, int]] = None,
    j20_region_radius: int = 30,
    j20_dilate_kernel: int = 5,
    min_branch_length: int = 10,
) -> np.ndarray:
    """Convert segment mask to a cleaned 1-pixel-wide skeleton."""
    mask_bool = segment_mask > 0

    # Local dilation around J20 can help force one clean hub before thinning.
    if j20_hint is not None:
        x, y = int(j20_hint[0]), int(j20_hint[1])
        h, w = mask_bool.shape
        x1 = max(0, x - j20_region_radius)
        x2 = min(w, x + j20_region_radius)
        y1 = max(0, y - j20_region_radius)
        y2 = min(h, y + j20_region_radius)
        roi = mask_bool[y1:y2, x1:x2].astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (j20_dilate_kernel, j20_dilate_kernel))
        roi = cv2.dilate(roi, kernel, iterations=1)
        mask_bool[y1:y2, x1:x2] = roi > 0

    skel = skeletonize(mask_bool)
    skel = _prune_isolated_pixels(skel)
    skel = prune_spurs(skel, min_branch_length=min_branch_length)
    return skel.astype(bool)


def _edge_payload(attrs: Dict) -> Tuple[float, List[Tuple[int, int]]]:
    pts = attrs.get("pts")
    if pts is None:
        pts_xy: List[Tuple[int, int]] = []
    else:
        pts_xy = [(int(p[1]), int(p[0])) for p in np.asarray(pts)]
    length_px = float(attrs.get("weight", 0.0))
    return length_px, pts_xy


def extract_skeleton_graph(skeleton: np.ndarray) -> nx.MultiGraph:
    """Build a graph from skeleton pixels using sknw."""
    raw = sknw.build_sknw(skeleton.astype(np.uint16), multi=True)
    g = nx.MultiGraph()

    for n, attrs in raw.nodes(data=True):
        oy, ox = attrs.get("o", (0, 0))
        g.add_node(
            int(n),
            x=float(ox),
            y=float(oy),
            pos=(float(ox), float(oy)),
        )

    for u, v, k, attrs in raw.edges(keys=True, data=True):
        length_px, pts_xy = _edge_payload(attrs)
        g.add_edge(
            int(u),
            int(v),
            key=int(k),
            path_length_px=length_px,
            path_pts=pts_xy,
        )

    return g


def _merge_parallel_edges(g: nx.MultiGraph, u: int, v: int) -> None:
    keys = list(g[u][v].keys())
    if len(keys) <= 1:
        return
    payloads = [g[u][v][k] for k in keys]
    best = max(payloads, key=lambda e: e.get("path_length_px", 0.0))
    for k in keys:
        g.remove_edge(u, v, key=k)
    g.add_edge(u, v, path_length_px=float(best.get("path_length_px", 0.0)), path_pts=list(best.get("path_pts", [])))


def consolidate_junctions(graph: nx.MultiGraph, radius: float = 15.0) -> nx.MultiGraph:
    g = graph.copy()
    changed = True
    while changed:
        changed = False
        junctions = [n for n in g.nodes if g.degree(n) >= 3]
        for a, b in combinations(junctions, 2):
            pa = np.array([g.nodes[a]["x"], g.nodes[a]["y"]], dtype=float)
            pb = np.array([g.nodes[b]["x"], g.nodes[b]["y"]], dtype=float)
            if np.linalg.norm(pa - pb) > radius:
                continue

            # Move node a to centroid and resegment b neighbors into a.
            centroid = (float((pa[0] + pb[0]) / 2.0), float((pa[1] + pb[1]) / 2.0))
            g.nodes[a]["x"], g.nodes[a]["y"], g.nodes[a]["pos"] = centroid[0], centroid[1], centroid

            for nbr, keydict in list(g[b].items()):
                if nbr == a:
                    continue
                for key, attrs in list(keydict.items()):
                    g.add_edge(a, nbr, path_length_px=float(attrs.get("path_length_px", 0.0)), path_pts=list(attrs.get("path_pts", [])))
            if g.has_node(b):
                g.remove_node(b)
            changed = True
            break
        if changed:
            # Clean up multi-edges generated by merge.
            for u, v in list(g.edges()):
                if g.has_edge(u, v):
                    _merge_parallel_edges(g, u, v)
    return g


def contract_degree2_nodes(graph: nx.MultiGraph) -> nx.MultiGraph:
    g = graph.copy()
    changed = True
    while changed:
        changed = False
        for n in list(g.nodes):
            if not g.has_node(n):
                continue
            if g.degree(n) != 2:
                continue
            nbrs = list(g.neighbors(n))
            if len(nbrs) != 2:
                continue
            u, v = nbrs
            attrs_un = next(iter(g.get_edge_data(u, n).values()))
            attrs_nv = next(iter(g.get_edge_data(n, v).values()))
            merged_len = float(attrs_un.get("path_length_px", 0.0)) + float(attrs_nv.get("path_length_px", 0.0))
            pts = list(attrs_un.get("path_pts", [])) + list(attrs_nv.get("path_pts", []))
            g.remove_node(n)
            g.add_edge(u, v, path_length_px=merged_len, path_pts=pts)
            _merge_parallel_edges(g, u, v)
            changed = True
            break
    return g


def prune_short_edges(graph: nx.MultiGraph, min_weight: float = 20.0) -> nx.MultiGraph:
    g = graph.copy()
    changed = True
    while changed:
        changed = False
        for u, v, k, attrs in list(g.edges(keys=True, data=True)):
            w = float(attrs.get("path_length_px", 0.0))
            if w >= min_weight:
                continue
            du, dv = g.degree(u), g.degree(v)
            if du >= 3 and dv >= 3:
                # Collapse nearby false split junctions.
                if g.has_node(u) and g.has_node(v):
                    pa = np.array([g.nodes[u]["x"], g.nodes[u]["y"]], dtype=float)
                    pb = np.array([g.nodes[v]["x"], g.nodes[v]["y"]], dtype=float)
                    g.nodes[u]["x"], g.nodes[u]["y"] = float((pa[0] + pb[0]) / 2), float((pa[1] + pb[1]) / 2)
                    for nbr, keydict in list(g[v].items()):
                        if nbr == u:
                            continue
                        for key2, attrs2 in list(keydict.items()):
                            g.add_edge(u, nbr, path_length_px=float(attrs2.get("path_length_px", 0.0)), path_pts=list(attrs2.get("path_pts", [])))
                    g.remove_node(v)
            elif du == 1 and g.has_node(u):
                g.remove_node(u)
            elif dv == 1 and g.has_node(v):
                g.remove_node(v)
            else:
                if g.has_edge(u, v, k):
                    g.remove_edge(u, v, key=k)
            changed = True
            break
        if changed:
            for u, v in list(g.edges()):
                if g.has_edge(u, v):
                    _merge_parallel_edges(g, u, v)
    return g


def bridge_nearby_components(graph: nx.MultiGraph, bridge_radius: float = 50.0) -> nx.MultiGraph:
    """Connect nearby endpoints that belong to different connected components.

    Schematic segment masks often have tiny gaps where dash-dot patterns or
    component maskings create breaks.  This step bridges those gaps at the
    graph level by adding synthetic edges between endpoint nodes (degree <= 1)
    in different CCs that are within *bridge_radius* pixels of each other.
    """
    g = graph.copy()
    changed = True
    while changed:
        changed = False
        ug = nx.Graph(g)
        ccs = list(nx.connected_components(ug))
        cc_of = {}
        for ci, cc in enumerate(ccs):
            for n in cc:
                cc_of[n] = ci

        # Collect endpoints (degree <= 1) and junction endpoints
        candidates = []
        for n in g.nodes:
            deg = g.degree(n)
            if deg <= 1:
                candidates.append(n)

        # Also consider high-degree (junction) nodes at CC boundaries
        for n in g.nodes:
            if g.degree(n) >= 3:
                candidates.append(n)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                if cc_of.get(a) == cc_of.get(b):
                    continue
                pa = np.array([g.nodes[a]['x'], g.nodes[a]['y']])
                pb = np.array([g.nodes[b]['x'], g.nodes[b]['y']])
                dist = float(np.linalg.norm(pa - pb))
                if dist <= bridge_radius:
                    g.add_edge(a, b, path_length_px=dist, path_pts=[
                        (int(pa[0]), int(pa[1])),
                        (int(pb[0]), int(pb[1])),
                    ])
                    changed = True
                    break
            if changed:
                break
    return g


def filter_skeleton_graph(
    graph: nx.MultiGraph,
    consolidate_radius: float = 15.0,
    min_edge_weight: float = 20.0,
    bridge_radius: float = 50.0,
) -> nx.MultiGraph:
    """Apply graph cleanup passes similar to NEFI post-filtering.

    Pipeline:
        1. Consolidate nearby junction nodes into one.
        2. Contract pass-through (degree-2) nodes.
        3. Bridge nearby endpoints in different CCs (gap repair).
        4. Re-contract any new degree-2 nodes created by bridging.
    """
    g = consolidate_junctions(graph, radius=consolidate_radius)
    g = contract_degree2_nodes(g)
    g = bridge_nearby_components(g, bridge_radius=bridge_radius)
    g = contract_degree2_nodes(g)  # clean up after bridging
    return g
