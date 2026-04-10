"""Phase 1: Find the main root (taproot) centerline path.

Strategy: graph-based shortest path for correct topology (won't drift
onto lateral roots), followed by iterative cross-sectional centering so
the path sits at the geometric center of the root.
"""

import copy
from dataclasses import dataclass

import hdbscan
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree

from .config import PipelineConfig
from .graph import (
    build_knn_graph,
    adjust_weights_by_density,
    compute_avg_distance,
    find_nearest_node,
    find_shortest_path,
)


@dataclass
class MainPathResult:
    path_points: np.ndarray       # (M, 3) centered centerline coordinates
    avg_distance: float           # mean nearest-neighbor distance in the cloud


def extract_main_path(
    pcd: o3d.geometry.PointCloud,
    start_point: np.ndarray,
    end_point: np.ndarray,
    config: PipelineConfig,
) -> MainPathResult:
    """Find the taproot centerline between two manually-picked endpoints.

    Three stages:
      1. **Topology**: density-weighted Dijkstra on a KNN graph gives a
         path that follows the correct root and never drifts onto laterals.
      2. **Pre-smoothing**: heavy Gaussian smoothing on the raw graph path
         to get stable tangent estimates despite the zigzag Dijkstra output.
      3. **Iterative centering**: repeatedly compute cross-sectional slices
         using the current tangents, snap each node to the slice centroid,
         then recompute tangents from the improved path. Each iteration
         produces better tangents which produce better centroids.

    Parameters
    ----------
    pcd : the full (downsampled) point cloud
    start_point : root crown position (3,)
    end_point : root tip position (3,)
    config : pipeline parameters

    Returns
    -------
    MainPathResult with centered path coordinates and avg_distance.
    """
    points = np.asarray(pcd.points)
    avg_distance = compute_avg_distance(points)

    # --- Stage 1: graph-based shortest path (correct topology) ---
    graph = build_knn_graph(points, k=config.knn_k)
    adjusted = adjust_weights_by_density(
        copy.deepcopy(graph), points, radius=avg_distance
    )

    source = find_nearest_node(points, start_point)
    target = find_nearest_node(points, end_point)
    path_indices = find_shortest_path(adjusted, source, target)
    raw_path = points[path_indices]

    print(
        f"Main path (graph): {len(path_indices)} nodes, "
        f"avg_distance={avg_distance:.6f}"
    )

    # --- Stage 2: pre-smooth the raw graph path ---
    # The Dijkstra path zigzags between surface points. Heavy smoothing
    # removes the zigzag while preserving the overall trajectory, giving
    # us stable tangent directions for the first centering pass.
    smoothed = _smooth_path(raw_path, sigma=config.centering_presmooth_sigma)

    # --- Stage 3: iterative cross-sectional centering ---
    tree = KDTree(points)
    search_radius = avg_distance * config.centering_search_k
    plane_tolerance = avg_distance * config.centering_tolerance_k

    current_path = smoothed
    for iteration in range(config.centering_iterations):
        tangents = _compute_tangents(current_path, sigma=config.tangent_sigma)
        current_path = _center_one_pass(
            current_path, tangents, points, tree,
            search_radius, plane_tolerance,
            min_cluster_size=config.min_cluster_size,
        )
        # Light smoothing between iterations to prevent jitter accumulation
        current_path = _smooth_path(
            current_path, sigma=config.centering_smooth_sigma
        )

    # Pin endpoints to the user-picked positions
    current_path[0] = points[source]
    current_path[-1] = points[target]

    print(
        f"Main path (centered, {config.centering_iterations} iterations): "
        f"{len(current_path)} points"
    )
    return MainPathResult(
        path_points=current_path,
        avg_distance=avg_distance,
    )


def _center_one_pass(
    path_points: np.ndarray,
    tangents: np.ndarray,
    all_points: np.ndarray,
    tree: KDTree,
    search_radius: float,
    plane_tolerance: float,
    min_cluster_size: int = 5,
) -> np.ndarray:
    """One pass of cross-sectional centering.

    For each path node:
      1. Slice the cloud with the tangent-normal plane.
      2. Cluster the slice with HDBSCAN to separate the primary root
         from any lateral roots that intersect the same plane.
      3. Keep only the cluster closest to the current path position.
      4. Replace the node with that cluster's centroid.
    """
    centered = path_points.copy()

    for i, (pt, tangent) in enumerate(zip(path_points, tangents)):
        indices = tree.query_ball_point(pt, search_radius)
        if len(indices) < min_cluster_size:
            continue

        candidates = all_points[indices]

        # Slice: keep points near the tangent-normal plane
        offsets = candidates - pt
        dist_to_plane = np.abs(offsets @ tangent)
        on_plane = candidates[dist_to_plane < plane_tolerance]

        if len(on_plane) < min_cluster_size:
            continue

        # Cluster the cross-section to separate primary from lateral
        primary_pts = _extract_primary_cluster(on_plane, pt, min_cluster_size)
        centered[i] = primary_pts.mean(axis=0)

    return centered


def _extract_primary_cluster(
    plane_points: np.ndarray,
    path_point: np.ndarray,
    min_cluster_size: int,
) -> np.ndarray:
    """From a cross-sectional slice, return only the cluster that belongs
    to the primary root (the one closest to the current path position).

    If clustering fails or finds only noise, falls back to all points.
    """
    if len(plane_points) < min_cluster_size * 2:
        # Too few points to meaningfully cluster — use all of them
        return plane_points

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(plane_points)

    unique_labels = set(labels)
    unique_labels.discard(-1)  # ignore noise

    if not unique_labels:
        # All noise — fall back to all points
        return plane_points

    # Pick the cluster whose points are closest to the path position
    best_label = None
    best_dist = np.inf
    for label in unique_labels:
        cluster = plane_points[labels == label]
        dist = np.linalg.norm(cluster.mean(axis=0) - path_point)
        if dist < best_dist:
            best_dist = dist
            best_label = label

    return plane_points[labels == best_label]


def _smooth_path(path: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth each coordinate axis of a path independently."""
    smoothed = path.copy()
    for axis in range(3):
        smoothed[:, axis] = gaussian_filter1d(smoothed[:, axis], sigma=sigma)
    return smoothed


def _compute_tangents(
    path_points: np.ndarray, sigma: float = 3.0
) -> np.ndarray:
    """Smoothed unit tangent vectors along a path."""
    tangents = np.gradient(path_points, axis=0)
    for axis in range(3):
        tangents[:, axis] = gaussian_filter1d(tangents[:, axis], sigma=sigma)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tangents / norms
