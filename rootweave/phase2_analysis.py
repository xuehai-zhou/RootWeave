"""Phase 2+3: Cylinder-shell branch detection.

A single pass that produces:
  - The main root volume (points belonging to the primary root)
  - Branch origin detections (where lateral roots pierce a shell around the taproot)

Strategy:
  1. Sweep cross-sections to measure the taproot radius profile R(t).
  2. Build a true hollow cylinder (shell) around the taproot:
     inner boundary = R(t) * inner_factor, outer = R(t) * outer_factor.
  3. Find all cloud points inside the shell.
  4. Cluster ALL shell points at once with DBSCAN — each cluster is one
     lateral root passing through the shell.
  5. For each cluster, compute direction and filter re-entries.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import hdbscan
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, argrelextrema
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN as SklearnDBSCAN

from .config import PipelineConfig
from .phase3_branch_detection import BranchOrigin


@dataclass
class AnalysisResult:
    main_root_points: np.ndarray
    branch_origins: Dict[int, BranchOrigin]
    radii: np.ndarray                  # smoothed radius profile R(t)
    shell_points: Optional[np.ndarray] = None
    shell_ring_labels: Optional[np.ndarray] = None
    shell_cluster_labels: Optional[np.ndarray] = None


def analyze_cross_sections(
    points: np.ndarray,
    path_points: np.ndarray,
    avg_distance: float,
    config: PipelineConfig,
) -> AnalysisResult:
    """Cylinder-shell branch detection + main root volume extraction."""

    tangents = _compute_tangent_vectors(path_points, sigma=config.tangent_sigma)
    tree = KDTree(points)

    # ------------------------------------------------------------------
    # Step 1: measure radius profile R(t) via cross-section clustering
    # ------------------------------------------------------------------
    search_radius = avg_distance * 100
    plane_tolerance = avg_distance * 1.5

    raw_radii = []
    for ref_point, tangent in zip(path_points, tangents):
        r = _measure_primary_radius(
            ref_point, tangent, points, tree,
            search_radius, plane_tolerance, config.min_cluster_size,
        )
        if r is not None:
            raw_radii.append(r)
        elif raw_radii:
            raw_radii.append(raw_radii[-1])
        else:
            raw_radii.append(avg_distance * 5)

    radii = np.array(raw_radii)
    radii = _remove_outliers(radii, threshold=config.outlier_std_threshold)
    radii = _smooth_radii(
        radii,
        window_length=config.radius_smoothing_window,
        polyorder=config.radius_smoothing_polyorder,
    )
    radii = _suppress_peaks(radii)

    # ------------------------------------------------------------------
    # Step 2: build main root volume
    # ------------------------------------------------------------------
    main_indices = set()
    for ref_point, r in zip(path_points, radii):
        idx = tree.query_ball_point(ref_point, r=r * config.main_branch_factor)
        main_indices.update(idx)
    main_root_points = points[list(main_indices)]

    # ------------------------------------------------------------------
    # Step 3: build the true hollow cylinder shell
    # ------------------------------------------------------------------
    path_tree = KDTree(path_points)
    _, nearest_ring = path_tree.query(points, k=1)

    # Perpendicular distance to the centerline axis
    offsets = points - path_points[nearest_ring]
    along_tangent = np.sum(offsets * tangents[nearest_ring], axis=1)
    perp_vectors = offsets - along_tangent[:, None] * tangents[nearest_ring]
    dist_to_axis = np.linalg.norm(perp_vectors, axis=1)

    # Local inner/outer radius per point
    inner_r = radii[nearest_ring] * config.shell_inner_factor
    outer_r = radii[nearest_ring] * config.shell_outer_factor

    shell_mask = (dist_to_axis > inner_r) & (dist_to_axis < outer_r)
    shell_global_indices = np.where(shell_mask)[0]
    shell_pts = points[shell_global_indices]
    shell_rings = nearest_ring[shell_mask]

    print(
        f"Radius profile: min={radii.min():.4f}, max={radii.max():.4f}, "
        f"mean={radii.mean():.4f}"
    )
    print(
        f"Shell: {len(shell_pts)} points "
        f"(inner={config.shell_inner_factor:.1f}R, "
        f"outer={config.shell_outer_factor:.1f}R)"
    )

    if len(shell_pts) == 0:
        return AnalysisResult(
            main_root_points=main_root_points,
            branch_origins={},
            radii=radii,
            shell_points=shell_pts,
        )

    # ------------------------------------------------------------------
    # Step 4: cluster ALL shell points at once
    # ------------------------------------------------------------------
    # Each connected cluster of shell points = one lateral root passing
    # through the shell. No ring sampling, no merge step.
    shell_eps = avg_distance * config.shell_cluster_eps_k
    clusterer = SklearnDBSCAN(
        eps=shell_eps, min_samples=config.shell_min_cluster
    )
    shell_labels = clusterer.fit_predict(shell_pts)

    unique_labels = set(shell_labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)
    n_noise = int(np.sum(shell_labels == -1))
    print(
        f"Shell clustering: {n_clusters} clusters, "
        f"{n_noise} noise points "
        f"(eps={shell_eps:.4f}, min_samples={config.shell_min_cluster})"
    )

    # ------------------------------------------------------------------
    # Step 5: build branch origins from shell clusters
    # ------------------------------------------------------------------
    branch_origins: Dict[int, BranchOrigin] = {}
    branch_id = 0
    filtered_size = 0
    filtered_reentry = 0

    for label in unique_labels:
        cmask = shell_labels == label
        cluster_pts = shell_pts[cmask]
        cluster_global = shell_global_indices[cmask]
        cluster_rings = shell_rings[cmask]

        # Min seed points filter
        if len(cluster_pts) < config.min_seed_points:
            filtered_size += 1
            continue

        centroid = cluster_pts.mean(axis=0)

        # Find the nearest path point to the centroid for direction calc
        _, nearest_path_idx = path_tree.query(centroid)
        path_pt = path_points[nearest_path_idx]
        tangent = tangents[nearest_path_idx]

        # Direction: radial vector from centerline to centroid,
        # projected onto the plane perpendicular to the tangent
        offset = centroid - path_pt
        radial = offset - np.dot(offset, tangent) * tangent
        radial_norm = np.linalg.norm(radial)
        if radial_norm < 1e-10:
            filtered_reentry += 1
            continue
        radial /= radial_norm

        # Re-entry filter: radial must point outward from axis
        # (the perpendicular offset from axis to centroid should align
        # with the radial direction — if not, it's pointing inward)
        perp_to_centroid = centroid - path_pt
        perp_to_centroid -= np.dot(perp_to_centroid, tangent) * tangent
        if np.dot(radial, perp_to_centroid) < 0:
            filtered_reentry += 1
            continue

        # Attachment point: path point at the earliest ring in this cluster
        earliest_ring = int(cluster_rings.min())
        attachment = path_points[earliest_ring]

        dist = float(np.min(np.linalg.norm(cluster_pts - attachment, axis=1)))

        branch_origins[branch_id] = BranchOrigin(
            label=branch_id,
            points=cluster_pts,
            direction=radial,
            distance_to_root=dist,
        )
        branch_id += 1

    if filtered_size or filtered_reentry:
        print(
            f"  Filtered: {filtered_size} clusters < {config.min_seed_points} pts, "
            f"{filtered_reentry} re-entry/degenerate"
        )
    print(f"Branch detection: {n_clusters} shell clusters -> {len(branch_origins)} branches")

    return AnalysisResult(
        main_root_points=main_root_points,
        branch_origins=branch_origins,
        radii=radii,
        shell_points=shell_pts,
        shell_ring_labels=shell_rings,
        shell_cluster_labels=shell_labels,
    )


# ---------------------------------------------------------------------------
# Radius measurement
# ---------------------------------------------------------------------------

def _measure_primary_radius(
    ref_point, tangent, points, tree,
    search_radius, plane_tolerance, min_cluster_size,
) -> Optional[float]:
    near_idx = tree.query_ball_point(ref_point, search_radius)
    if len(near_idx) < min_cluster_size:
        return None

    candidates = points[near_idx]
    offsets = candidates - ref_point
    dist_to_plane = np.abs(offsets @ tangent) / np.linalg.norm(tangent)
    plane_pts = candidates[dist_to_plane < plane_tolerance]

    if len(plane_pts) < min_cluster_size:
        return None

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(plane_pts)

    dists = np.linalg.norm(plane_pts - ref_point, axis=1)
    primary_label = labels[np.argmin(dists)]
    if primary_label == -1:
        return None

    primary_pts = plane_pts[labels == primary_label]
    return float(np.max(np.linalg.norm(primary_pts - ref_point, axis=1)))


# ---------------------------------------------------------------------------
# Radius smoothing helpers
# ---------------------------------------------------------------------------

def _compute_tangent_vectors(path_points, sigma=5.0):
    tangents = np.gradient(path_points, axis=0)
    for ax in range(3):
        tangents[:, ax] = gaussian_filter1d(tangents[:, ax], sigma=sigma)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tangents / norms


def _remove_outliers(data, threshold=3.0):
    mean, std = np.mean(data), np.std(data)
    lo, hi = mean - threshold * std, mean + threshold * std
    mask = (data >= lo) & (data <= hi)
    if mask.sum() == 0:
        return data
    return np.interp(np.arange(len(data)), np.where(mask)[0], data[mask])


def _smooth_radii(data, window_length=51, polyorder=3):
    if len(data) < window_length:
        window_length = len(data) // 2
        if window_length % 2 == 0:
            window_length = max(window_length - 1, polyorder + 1)
    if window_length <= polyorder:
        return data
    return savgol_filter(data, window_length, polyorder)


def _suppress_peaks(data):
    local_min = argrelextrema(data, np.less)[0]
    local_max = argrelextrema(data, np.greater)[0]
    min_t = np.mean(data[local_min]) if len(local_min) > 0 else np.inf
    max_t = np.mean(data[local_max]) if len(local_max) > 0 else -np.inf
    threshold = min(min_t, max_t)
    if not np.isfinite(threshold):
        threshold = np.median(data)
    return np.minimum(data, threshold)
