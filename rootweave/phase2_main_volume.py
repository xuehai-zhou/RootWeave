"""Phase 2: Main root volume extraction.

Measures the taproot radius profile R(t) along the centerline by cross-
sectional clustering, smooths it, then marks every point within
R(t) * main_branch_factor of the centerline as belonging to the main
root volume.

This module intentionally does NOT perform branch detection. The
shrinking-shell tip seeding is phase 3's job.
"""

from dataclasses import dataclass
from typing import Optional

import hdbscan
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema, savgol_filter
from scipy.spatial import KDTree

from .config import PipelineConfig


@dataclass
class MainVolumeResult:
    main_root_points: np.ndarray      # (M, 3) points inside the taproot volume
    main_root_mask: np.ndarray        # (N,) bool mask into the full cloud
    radii: np.ndarray                 # smoothed R(t) along the taproot path
    tangents: np.ndarray              # unit tangents along the taproot path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_main_volume(
    points: np.ndarray,
    path_points: np.ndarray,
    avg_distance: float,
    config: PipelineConfig,
) -> MainVolumeResult:
    """Measure the taproot radius profile and extract its volume from the cloud."""
    tangents = _compute_tangent_vectors(path_points, sigma=config.tangent_sigma)
    tree = KDTree(points)

    # --- Step 1: radius profile R(t) via cross-section clustering ---
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

    # --- Step 2: main root volume mask ---
    n_pts = len(points)
    main_mask = np.zeros(n_pts, dtype=bool)
    for ref_point, r in zip(path_points, radii):
        idx = tree.query_ball_point(ref_point, r=r * config.main_branch_factor)
        if idx:
            main_mask[idx] = True
    main_root_points = points[main_mask]

    print(
        f"Radius profile: min={radii.min():.4f}, max={radii.max():.4f}, "
        f"mean={radii.mean():.4f}"
    )
    print(f"Main root volume: {int(main_mask.sum())} / {n_pts} points")

    return MainVolumeResult(
        main_root_points=main_root_points,
        main_root_mask=main_mask,
        radii=radii,
        tangents=tangents,
    )


# ---------------------------------------------------------------------------
# Radius measurement
# ---------------------------------------------------------------------------

def _measure_primary_radius(
    ref_point: np.ndarray,
    tangent: np.ndarray,
    points: np.ndarray,
    tree: KDTree,
    search_radius: float,
    plane_tolerance: float,
    min_cluster_size: int,
) -> Optional[float]:
    """Measure the taproot radius at one path point.

    Slices the cloud by a plane normal to the local tangent, clusters
    the slice with HDBSCAN, keeps the cluster closest to the path point
    (the primary root, not a lateral passing through the same plane),
    and returns that cluster's max distance from the ref point.
    """
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
# Radius-profile smoothing helpers
# ---------------------------------------------------------------------------

def _compute_tangent_vectors(path_points: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Smoothed unit tangent at each path point."""
    tangents = np.gradient(path_points, axis=0)
    for ax in range(3):
        tangents[:, ax] = gaussian_filter1d(tangents[:, ax], sigma=sigma)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tangents / norms


def _remove_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Replace samples more than *threshold* std-devs from the mean with
    linear interpolation from their surviving neighbors.
    """
    mean, std = np.mean(data), np.std(data)
    lo, hi = mean - threshold * std, mean + threshold * std
    mask = (data >= lo) & (data <= hi)
    if mask.sum() == 0:
        return data
    return np.interp(np.arange(len(data)), np.where(mask)[0], data[mask])


def _smooth_radii(data: np.ndarray, window_length: int = 51, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing of the radius profile."""
    if len(data) < window_length:
        window_length = len(data) // 2
        if window_length % 2 == 0:
            window_length = max(window_length - 1, polyorder + 1)
    if window_length <= polyorder:
        return data
    return savgol_filter(data, window_length, polyorder)


def _suppress_peaks(data: np.ndarray) -> np.ndarray:
    """Clip the profile to the smaller of its mean local-min and mean local-max.

    This suppresses occasional spikes from lateral roots leaking into the
    taproot cross-section without affecting the overall shape.
    """
    local_min = argrelextrema(data, np.less)[0]
    local_max = argrelextrema(data, np.greater)[0]
    min_t = np.mean(data[local_min]) if len(local_min) > 0 else np.inf
    max_t = np.mean(data[local_max]) if len(local_max) > 0 else -np.inf
    threshold = min(min_t, max_t)
    if not np.isfinite(threshold):
        threshold = np.median(data)
    return np.minimum(data, threshold)
