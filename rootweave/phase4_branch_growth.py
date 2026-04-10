"""Phase 4: PCA tracking + graph-guided bend crossing.

Primary: PCA tracking with fixed step size, cross-sectional cluster
filtering, and directional biases.

When PCA gets stuck:
  1. Try local direction probing (24 small perturbations, same step size).
  2. If probing fails, use a local graph lookahead to find where the root
     continues, then take small PCA steps toward that target.
  No large jumps — the graph only suggests a direction, not a destination.

Bidirectional: inward (toward taproot) and outward (toward tip).
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from .config import PipelineConfig
from .graph import build_knn_graph, adjust_weights_by_density
from .phase3_branch_detection import BranchOrigin


@dataclass
class BranchPath:
    label: int
    path: List[np.ndarray]
    surrounding_indices: List[int]


def grow_branches(
    all_points: np.ndarray,
    main_root_points: np.ndarray,
    branch_origins: Dict[int, BranchOrigin],
    avg_distance: float,
    path_points: np.ndarray,
    radii: np.ndarray,
    config: PipelineConfig,
) -> List[BranchPath]:
    """Trace all lateral roots with adaptive PCA + local direction probing."""
    if not branch_origins:
        return []

    path_tree = KDTree(path_points)
    tangents = _compute_tangents(path_points)

    # --- Build two search spaces ---
    # Inward: all_points (needs to cross through the junction region)
    tree_all = KDTree(all_points)

    # Outward: exclude the taproot zone so tracking can't follow residual
    # primary root points. Exclusion is generous (R(t) * exclusion_factor).
    outward_pts, _ = _build_outward_points(
        all_points, path_points, tangents, radii, config.taproot_exclusion_factor
    )
    tree_out = KDTree(outward_pts)
    print(
        f"Outward search space: {len(outward_pts)}/{len(all_points)} points "
        f"(excluded {len(all_points) - len(outward_pts)} within "
        f"{config.taproot_exclusion_factor:.1f}×R of taproot)"
    )

    # Build KNN graphs for graph-guided lookahead (not for jumping)
    print("Building KNN graphs for bend lookahead...")
    graph_all = adjust_weights_by_density(
        copy.deepcopy(build_knn_graph(all_points, k=config.knn_k)),
        all_points, radius=avg_distance,
    )
    graph_out = adjust_weights_by_density(
        copy.deepcopy(build_knn_graph(outward_pts, k=config.knn_k)),
        outward_pts, radius=avg_distance,
    )

    base_nbr = avg_distance * config.neighborhood_k
    base_step = avg_distance * config.step_k
    base_snap = avg_distance * config.snap_k

    print(
        f"Growth params: neighborhood={base_nbr:.4f}, "
        f"step={base_step:.4f}, snap={base_snap:.4f}"
    )

    results = []
    for label, origin in branch_origins.items():
        bp = _grow_one_branch(
            label, origin,
            all_points, tree_all, graph_all,
            outward_pts, tree_out, graph_out,
            path_points, path_tree, tangents, radii,
            base_nbr, base_step, base_snap,
            avg_distance, config,
        )
        if bp is not None:
            results.append(bp)

    print(f"Grew {len(results)} branch paths from {len(branch_origins)} origins")
    return results


def _build_outward_points(all_points, path_points, tangents, radii, exclusion_factor):
    """Build the outward search space by excluding points within the
    taproot exclusion zone (perpendicular distance < R(t) * factor)."""
    path_tree = KDTree(path_points)
    _, nearest_ring = path_tree.query(all_points, k=1)

    # Perpendicular distance to taproot axis
    offsets = all_points - path_points[nearest_ring]
    along = np.sum(offsets * tangents[nearest_ring], axis=1)
    perp = offsets - along[:, None] * tangents[nearest_ring]
    perp_dist = np.linalg.norm(perp, axis=1)

    # Exclusion threshold: local radius * factor
    thresholds = radii[nearest_ring] * exclusion_factor
    keep_mask = perp_dist > thresholds

    outward_pts = all_points[keep_mask]
    return outward_pts, np.where(keep_mask)[0]


def _grow_one_branch(
    label, origin,
    all_points, tree_all, graph_all,
    outward_pts, tree_out, graph_out,
    path_points, path_tree, tangents, radii,
    base_nbr, base_step, base_snap,
    avg_distance, config,
) -> Optional[BranchPath]:
    cluster_pts = origin.points
    centroid = cluster_pts.mean(axis=0)

    # --- Snap centroid to the local root center ---
    centroid = _snap_to_root_center(centroid, all_points, tree_all, base_nbr)

    # --- Determine initial directions via PCA ---
    dir_a, dir_b = _estimate_initial_directions(
        centroid, origin.direction, all_points, tree_all,
        path_points, path_tree, base_nbr,
    )

    # --- Neutral short exploration to determine which is inward/outward ---
    # Grow both directions on all_points (same search space, no biases,
    # no arrival checks) for a few steps. Use the distance trend to
    # decide which direction approaches the taproot.
    _PROBE_STEPS = 10
    probe_config = copy.copy(config)
    probe_config.max_growth_steps = _PROBE_STEPS
    probe_config.max_bend_recoveries = 0  # no recovery during probing
    probe_config.inward_attraction_weight = 0.0
    probe_config.antigravity_weight = 0.0
    probe_config.outward_repulsion_weight = 0.0
    probe_config.gravity_weight = 0.0

    probe_a, _ = _adaptive_track(
        centroid, dir_a, all_points, tree_all, graph_all,
        base_nbr, base_step, base_snap, probe_config,
        inward_mode=False,
        path_points=path_points, path_tree=path_tree, tangents=None, radii=None,
    )
    probe_b, _ = _adaptive_track(
        centroid, dir_b, all_points, tree_all, graph_all,
        base_nbr, base_step, base_snap, probe_config,
        inward_mode=False,
        path_points=path_points, path_tree=path_tree, tangents=None, radii=None,
    )

    # Compute distance trend for each probe
    trend_a = _distance_trend(probe_a, path_points, tangents, path_tree)
    trend_b = _distance_trend(probe_b, path_points, tangents, path_tree)

    # Negative trend = getting closer to taproot = inward
    if trend_a < trend_b:
        inward_dir, outward_dir = dir_a, dir_b
    else:
        inward_dir, outward_dir = dir_b, dir_a

    # --- Now grow for real with correct assignments ---
    out_path, out_surr = _adaptive_track(
        centroid, outward_dir, outward_pts, tree_out, graph_out,
        base_nbr, base_step, base_snap, config,
        inward_mode=False,
        path_points=path_points, path_tree=path_tree, tangents=None, radii=None,
    )

    in_path, in_surr = _adaptive_track(
        centroid, inward_dir, all_points, tree_all, graph_all,
        base_nbr, base_step, base_snap, config,
        inward_mode=True,
        path_points=path_points, path_tree=path_tree,
        tangents=tangents, radii=radii,
    )

    # Concatenate: inward(reversed) + outward
    full = in_path[::-1]
    if len(out_path) > 1:
        full = full + out_path[1:]

    if len(full) < 2:
        return None

    return BranchPath(
        label=label, path=full,
        surrounding_indices=list(set(in_surr + out_surr)),
    )


# ---------------------------------------------------------------------------
# Starting point snapping
# ---------------------------------------------------------------------------

def _snap_to_root_center(centroid, all_points, tree, base_nbr):
    """Snap the shell centroid to the center of the local root cross-section.

    Finds nearby points, estimates the local root axis via PCA, projects
    onto the perpendicular plane, clusters, and returns the centroid of
    the nearest cluster. This moves an edge-biased starting point to
    the true center of the root.
    """
    idx = tree.query_ball_point(centroid, base_nbr)
    if len(idx) < 3:
        idx = tree.query_ball_point(centroid, base_nbr * 2)
        if len(idx) < 3:
            return centroid

    pts = all_points[idx]

    # Local PCA to get the root axis
    pca = PCA(n_components=min(3, len(pts)))
    pca.fit(pts - centroid)
    axis = pca.components_[0]

    # Filter to the cross-sectional plane
    offsets = pts - centroid
    along = np.abs(offsets @ axis)
    plane_mask = along < base_nbr * 0.3  # thin slice
    plane_pts = pts[plane_mask]

    if len(plane_pts) < 3:
        return centroid

    # Cluster the cross-section and take nearest cluster centroid
    filtered = _filter_to_nearest_cluster(plane_pts, centroid, axis, base_nbr * 0.5, 2)
    if filtered is not None and len(filtered) >= 2:
        return filtered.mean(axis=0)
    return plane_pts.mean(axis=0)


# ---------------------------------------------------------------------------
# Initial direction estimation
# ---------------------------------------------------------------------------

def _estimate_initial_directions(
    centroid, radial_direction, all_points, tree,
    path_points, path_tree, base_nbr,
):
    """Estimate initial inward/outward directions using PCA on a wide
    neighborhood around the centroid, rather than just the shell cluster.

    Tries progressively wider radii (1x, 2x, 3x base_nbr) until enough
    points are found for a reliable PCA. Falls back to the radial
    direction from the shell detection if PCA fails entirely.
    """
    best_axis = None
    for mult in [1.0, 2.0, 3.0]:
        idx = tree.query_ball_point(centroid, base_nbr * mult)
        if len(idx) < 5:
            continue
        pts = all_points[idx]
        pca = PCA(n_components=min(3, len(pts)))
        pca.fit(pts - centroid)
        # Only trust PC1 if it explains a clear majority (elongated = root)
        if pca.explained_variance_ratio_[0] > 0.4:
            best_axis = pca.components_[0]
            break

    if best_axis is None:
        # Fallback to radial direction from shell detection
        outward_dir = radial_direction / np.linalg.norm(radial_direction)
        return -outward_dir, outward_dir

    # Orient: inward end points toward the taproot centerline
    _, nearest_idx = path_tree.query(centroid)
    to_tap = path_points[nearest_idx] - centroid
    if np.dot(best_axis, to_tap) > 0:
        return best_axis, -best_axis  # inward, outward
    else:
        return -best_axis, best_axis


# ---------------------------------------------------------------------------
# Adaptive PCA tracking
# ---------------------------------------------------------------------------

def _adaptive_track(
    start, direction, all_points, tree, graph,
    base_nbr, base_step, base_snap, config,
    inward_mode,
    path_points, path_tree, tangents, radii,
) -> Tuple[List[np.ndarray], List[int]]:
    """Adaptive PCA tracking with graph-assisted bend recovery.

    Inward mode adds:
      - Taproot arrival detection
      - Attraction bias toward the taproot path
      - Small upward anti-gravity bias

    Outward mode adds:
      - Self-intersection detection (retracing = past the tip)
      - Post-loop tail trimming (remove turn-back nodes)
    """
    direction = direction / np.linalg.norm(direction)
    current = start.copy()
    path = [current.copy()]
    surrounding = []

    smooth = config.smoothing_factor
    recoveries_left = config.max_bend_recoveries

    # Inward arrival state
    min_perp_dist = np.inf
    moving_away_count = 0

    # Outward self-intersection detection
    _RETRACE_SKIP = 8
    _retrace_count = 0

    # For outward tail trimming: track max distance from start
    max_dist_from_start = 0.0
    max_dist_index = 0

    # Disable graph recovery for the first few outward steps
    _OUTWARD_GRACE_STEPS = 5
    step_count = 0

    for _ in range(config.max_growth_steps):
        step_count += 1
        _allow_recovery = recoveries_left > 0 and \
            (inward_mode or step_count > _OUTWARD_GRACE_STEPS)

        # --- Inward arrival check ---
        if inward_mode:
            should_stop, snap_pt = _check_taproot_arrival(
                current, direction, all_points, tree,
                path_points, path_tree, tangents, radii,
                config, min_perp_dist, moving_away_count,
            )
            if should_stop:
                if snap_pt is not None:
                    path.append(snap_pt)
                break

            perp_d = _perp_dist_to_axis(current, path_points, tangents, path_tree)
            if perp_d < min_perp_dist:
                min_perp_dist = perp_d
                moving_away_count = 0
            else:
                moving_away_count += 1

        # --- Outward: self-intersection detection ---
        if not inward_mode:
            d_from_start = np.linalg.norm(current - start)
            if d_from_start > max_dist_from_start:
                max_dist_from_start = d_from_start
                max_dist_index = len(path) - 1

            if len(path) > _RETRACE_SKIP + 3:
                old_pts = np.array(path[:-_RETRACE_SKIP])
                dists_to_old = np.linalg.norm(old_pts - current, axis=1)
                min_dist_to_old = float(dists_to_old.min())
                if min_dist_to_old < base_step * 3:
                    _retrace_count += 1
                    if _retrace_count >= 3:
                        break
                else:
                    _retrace_count = 0

        # --- Compute bias ---
        bias = None
        if inward_mode and path_tree is not None:
            bias = _compute_inward_bias(current, path_tree, path_points, config)
        elif not inward_mode and path_tree is not None:
            bias = _compute_outward_bias(current, path_tree, path_points, config)

        # --- Try a PCA step (fixed step size) ---
        result = _pca_step(
            current, direction, all_points, tree,
            base_nbr, base_step, base_snap, smooth, config.pca_min_points,
            bias=bias,
        )

        if result is not None:
            next_pt, new_dir, cosine = result

            if cosine < config.min_step_cosine:
                # PCA direction deviates too much — try local probing
                probe = _probe_directions(
                    current, direction, all_points, tree,
                    base_nbr, base_step, base_snap, smooth,
                    config.pca_min_points, config.min_step_cosine, bias,
                ) if _allow_recovery else None
                if probe is not None:
                    path.append(probe[0])
                    direction = probe[1]
                    current = probe[0]
                    recoveries_left -= 1
                    continue

                # Probing failed — graph lookahead: check if root
                # continues ahead, get a target direction, then step
                # toward it with small PCA steps
                guided = _graph_guided_step(
                    current, direction, path, all_points, tree, graph,
                    base_nbr, base_step, base_snap, smooth,
                    config.pca_min_points, bias,
                ) if _allow_recovery else None
                if guided is not None:
                    path.append(guided[0])
                    direction = guided[1]
                    current = guided[0]
                    recoveries_left -= 1
                    continue
                else:
                    break

            path.append(next_pt)
            direction = new_dir
            current = next_pt
            surr = tree.query_ball_point(current, base_snap)
            surrounding.extend(surr)

        else:
            # PCA failed — try probing, then graph lookahead
            probe = _probe_directions(
                current, direction, all_points, tree,
                base_nbr, base_step, base_snap, smooth,
                config.pca_min_points, config.min_step_cosine, bias,
            ) if _allow_recovery else None
            if probe is not None:
                path.append(probe[0])
                direction = probe[1]
                current = probe[0]
                recoveries_left -= 1
                continue

            guided = _graph_guided_step(
                current, direction, path, all_points, tree, graph,
                base_nbr, base_step, base_snap, smooth,
                config.pca_min_points, bias,
            ) if _allow_recovery else None
            if guided is not None:
                path.append(guided[0])
                direction = guided[1]
                current = guided[0]
                recoveries_left -= 1
            else:
                break

    # --- Outward tail trimming ---
    if not inward_mode and _retrace_count >= 3 and max_dist_index < len(path) - 1:
        path = path[:max_dist_index + 1]

    return path, surrounding


def _compute_inward_bias(current, path_tree, path_points, config):
    """Compute a bias vector for inward tracking:
       attraction toward taproot path + small upward anti-gravity.

    Returns a unit-length bias vector, or None.
    """
    # Attraction toward nearest taproot path point
    _, nearest_idx = path_tree.query(current)
    nearest_pt = path_points[nearest_idx]
    to_taproot = nearest_pt - current
    dist = np.linalg.norm(to_taproot)
    if dist < 1e-10:
        attraction = np.zeros(3)
    else:
        attraction = to_taproot / dist

    # Anti-gravity: small upward component (Z-up in our coordinate system)
    antigrav = np.array([0.0, 0.0, 1.0])

    # Combine with configurable weights
    bias = config.inward_attraction_weight * attraction + \
           config.antigravity_weight * antigrav

    norm = np.linalg.norm(bias)
    if norm < 1e-10:
        return None
    return bias / norm


def _compute_outward_bias(current, path_tree, path_points, config):
    """Compute a bias vector for outward tracking:
       repulsion away from taproot path + small downward gravity.

    Returns a unit-length bias vector, or None.
    """
    # Repulsion: away from nearest taproot path point
    _, nearest_idx = path_tree.query(current)
    nearest_pt = path_points[nearest_idx]
    away_from_taproot = current - nearest_pt
    dist = np.linalg.norm(away_from_taproot)
    if dist < 1e-10:
        repulsion = np.zeros(3)
    else:
        repulsion = away_from_taproot / dist

    # Gravity: small downward component (negative Z)
    gravity = np.array([0.0, 0.0, -1.0])

    bias = config.outward_repulsion_weight * repulsion + \
           config.gravity_weight * gravity

    norm = np.linalg.norm(bias)
    if norm < 1e-10:
        return None
    return bias / norm


# ---------------------------------------------------------------------------
# Taproot arrival detection
# ---------------------------------------------------------------------------

def _check_taproot_arrival(
    current, direction, all_points, tree,
    path_points, path_tree, tangents, radii,
    config, min_perp_dist, moving_away_count,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Check if inward tracking has reached the taproot.

    Uses three criteria:
      1. Perpendicular distance to taproot axis < R(t) * arrival_factor
         (we're geometrically inside the inflated taproot cylinder)
      2. Local PCA direction aligns with the taproot tangent
         (the local structure is oriented along the taproot, not a
         lateral root that happens to pass near the axis)
      3. Closest-approach monitor: if we got close then moved away for
         several steps, we've passed the junction

    Returns (should_stop, snap_point).
    """
    # Perpendicular distance to the taproot axis
    _, nearest_ring = path_tree.query(current)
    nearest_ring = int(nearest_ring)
    tangent = tangents[nearest_ring]
    local_radius = radii[nearest_ring]
    arrival_threshold = local_radius * config.arrival_factor
    hard_threshold = local_radius * config.hard_arrival_factor

    offset = current - path_points[nearest_ring]
    along = np.dot(offset, tangent)
    perp_vec = offset - along * tangent
    perp_dist = np.linalg.norm(perp_vec)

    snap_pt = path_points[nearest_ring].copy()

    # Criterion 0 (hard floor): inside the tight taproot cylinder — stop
    # unconditionally. No alignment check needed. This prevents the tracker
    # from following residual primary root points sideways.
    if perp_dist < hard_threshold:
        return True, snap_pt

    # Criterion 1 + 2: inside the inflated cylinder AND aligned with taproot
    if perp_dist < arrival_threshold:
        local_idx = tree.query_ball_point(current, arrival_threshold)
        if len(local_idx) >= 3:
            local_pts = all_points[local_idx]
            pca = PCA(n_components=min(3, len(local_pts)))
            pca.fit(local_pts - current)
            local_pc1 = pca.components_[0]
            alignment = abs(float(np.dot(local_pc1, tangent)))
            if alignment > config.arrival_alignment_threshold:
                return True, snap_pt

    # Criterion 3: closest-approach monitor
    if moving_away_count >= 4 and min_perp_dist < arrival_threshold * 2.5:
        return True, snap_pt

    return False, None


def _perp_dist_to_axis(current, path_points, tangents, path_tree):
    """Perpendicular distance from current to the taproot axis."""
    _, nearest_ring = path_tree.query(current)
    nearest_ring = int(nearest_ring)
    offset = current - path_points[nearest_ring]
    along = np.dot(offset, tangents[nearest_ring])
    perp = offset - along * tangents[nearest_ring]
    return float(np.linalg.norm(perp))


def _distance_trend(probe_path, path_points, tangents, path_tree):
    """Compute the distance trend of a probe path relative to the taproot.

    Returns a scalar: negative = approaching taproot, positive = moving away.
    Uses the difference between mean distance in the second half vs first half.
    """
    if len(probe_path) < 4:
        # Too short to compute a trend — use endpoint distance
        if len(probe_path) >= 2:
            d0 = _perp_dist_to_axis(probe_path[0], path_points, tangents, path_tree)
            d1 = _perp_dist_to_axis(probe_path[-1], path_points, tangents, path_tree)
            return d1 - d0
        return 0.0

    dists = [_perp_dist_to_axis(pt, path_points, tangents, path_tree)
             for pt in probe_path]
    mid = len(dists) // 2
    first_half_mean = float(np.mean(dists[:mid]))
    second_half_mean = float(np.mean(dists[mid:]))
    return second_half_mean - first_half_mean


def _compute_tangents(path_points, sigma=3.0):
    tangents = np.gradient(path_points, axis=0)
    for ax in range(3):
        tangents[:, ax] = gaussian_filter1d(tangents[:, ax], sigma=sigma)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tangents / norms


# ---------------------------------------------------------------------------
# Single PCA step
# ---------------------------------------------------------------------------

def _pca_step(current, direction, all_points, tree,
              nbr_r, step_sz, snap_r, smooth, min_pts,
              bias=None):
    """One PCA tracking step with cross-sectional cluster filtering.

    Direction = PCA (dominant) + optional bias (attraction + anti-gravity).

    Returns (next_pt, new_direction, step_cosine) or None.
    """
    idx = tree.query_ball_point(current, nbr_r)
    if len(idx) < min_pts:
        idx = tree.query_ball_point(current, nbr_r * 2)
        if len(idx) < min_pts:
            return None

    pts = all_points[idx]

    # Forward hemisphere filter
    vecs = pts - current
    norms = np.linalg.norm(vecs, axis=1)
    norms[norms == 0] = 1.0
    cos = (vecs / norms[:, None]) @ direction
    fwd_mask = cos > 0
    fwd = pts[fwd_mask]

    if len(fwd) < min_pts:
        fwd = pts
        fwd_mask = np.ones(len(pts), dtype=bool)
    if len(fwd) < min_pts:
        return None

    # Cross-sectional cluster filtering
    fwd = _filter_to_nearest_cluster(fwd, current, direction, snap_r, min_pts)
    if fwd is None or len(fwd) < min_pts:
        return None

    # PCA on the filtered points
    pca = PCA(n_components=min(3, len(fwd)))
    pca.fit(fwd - current)
    pc1 = pca.components_[0]
    if np.dot(pc1, direction) < 0:
        pc1 = -pc1

    # Blend: PCA (dominant) + bias (secondary)
    # Direction composition: smooth * PCA + (1-smooth) * previous + bias_weight * bias
    blended = smooth * pc1 + (1.0 - smooth) * direction
    if bias is not None:
        blended += bias  # bias is already weighted and normalized in _compute_inward_bias
    n = np.linalg.norm(blended)
    if n < 1e-10:
        return None
    blended /= n

    # Step + cross-sectional centering snap
    candidate = current + step_sz * blended
    next_pt = _cross_section_snap(
        candidate, blended, all_points, tree, snap_r, min_pts
    )
    if next_pt is None:
        return None

    # Step direction
    sv = next_pt - current
    sn = np.linalg.norm(sv)
    if sn < 1e-10:
        return None
    sd = sv / sn
    cosine = float(np.dot(sd, direction))

    # New direction (smoothed)
    nd = smooth * sd + (1.0 - smooth) * blended
    nn = np.linalg.norm(nd)
    nd = nd / nn if nn > 1e-10 else sd

    return next_pt, nd, cosine


def _cross_section_snap(candidate, direction, all_points, tree, snap_r, min_pts):
    """Snap a candidate position to the center of the root cross-section
    using iterative centering.

    Problem: if the candidate is near the root edge, a single snap pass
    only sees half the cross-section and the centroid is still off-center.

    Solution: two passes. The first pass gets approximately onto the root.
    The second pass searches from the corrected position with a wider
    radius, sees the full cross-section, and centers properly.

    Returns the snapped 3D point, or None.
    """
    pos = candidate.copy()

    # Two centering passes: first rough, then refined from better position
    for iteration in range(2):
        # Wider search on second pass (first pass position is closer to center)
        r = snap_r if iteration == 0 else snap_r * 2
        centered = _single_centering_pass(pos, direction, all_points, tree, r, snap_r, min_pts)
        if centered is None:
            if iteration == 0:
                return None
            break  # second pass failed, keep first result
        pos = centered

    return pos


def _single_centering_pass(center, direction, all_points, tree, search_r, cluster_eps, min_pts):
    """One pass of cross-sectional centering: search, slice, cluster, centroid."""
    from sklearn.cluster import DBSCAN as _DBSCAN

    si = tree.query_ball_point(center, search_r)
    if not si:
        return None

    pts = all_points[si]

    # Plane basis
    d = direction / np.linalg.norm(direction)
    ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    # Slice perpendicular to direction
    offsets = pts - center
    along = offsets @ d
    plane_mask = np.abs(along) < cluster_eps * 0.5
    plane_pts = pts[plane_mask]

    if len(plane_pts) < max(2, min_pts):
        return pts.mean(axis=0)

    # Project to 2D and cluster
    offsets_2d = plane_pts - center
    proj_2d = np.column_stack([offsets_2d @ u, offsets_2d @ v])

    labels = _DBSCAN(eps=cluster_eps * 0.5, min_samples=max(1, min_pts)).fit_predict(proj_2d)

    unique = set(labels)
    unique.discard(-1)

    if not unique:
        center_2d = proj_2d.mean(axis=0)
    else:
        best_label = max(unique, key=lambda lab: int(np.sum(labels == lab)))
        center_2d = proj_2d[labels == best_label].mean(axis=0)

    # Reconstruct 3D: cross-section center + along-axis from original center
    return center + center_2d[0] * u + center_2d[1] * v


def _filter_to_nearest_cluster(pts, center, direction, eps, min_pts):
    """Project points onto the cross-sectional plane perpendicular to
    *direction*, cluster with DBSCAN, and return only the 3D points
    belonging to the cluster whose 2D centroid is closest to *center*.

    Returns the filtered 3D points, or None if clustering fails.
    """
    from sklearn.cluster import DBSCAN as _DBSCAN

    if len(pts) < 3:
        return pts

    # Build orthonormal basis for the cross-sectional plane
    d = direction / np.linalg.norm(direction)
    ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    # Project to 2D
    offsets = pts - center
    proj_2d = np.column_stack([offsets @ u, offsets @ v])

    # Cluster in 2D
    labels = _DBSCAN(eps=eps, min_samples=max(1, min_pts)).fit_predict(proj_2d)

    unique = set(labels)
    unique.discard(-1)
    if not unique:
        return pts  # all noise — return everything

    # Pick the largest cluster — the root we're on fills most of the
    # cross-section; a neighbor at the edge has fewer points.
    best_label = max(unique, key=lambda lab: int(np.sum(labels == lab)))

    return pts[labels == best_label]


# ---------------------------------------------------------------------------
# Local direction probing (replaces graph-based bend recovery)
# ---------------------------------------------------------------------------

def _probe_directions(current, direction, all_points, tree,
                      base_nbr, base_step, base_snap, smooth,
                      min_pts, min_cosine, bias):
    """When PCA tracking fails or deviates too much, try several
    small perturbations of the current direction at the same step size.

    Generates candidate directions by rotating the current direction
    by small angles in the plane perpendicular to it. For each, attempts
    a PCA step. Returns the best result that passes min_cosine, or None.
    """
    # Build two orthonormal vectors perpendicular to direction
    d = direction / np.linalg.norm(direction)
    ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    # Probe angles: 20°, 40°, 60° in 8 azimuthal directions = 24 probes
    best_result = None
    best_cosine = -np.inf

    for tilt_deg in [20, 40, 60]:
        tilt = np.radians(tilt_deg)
        for azimuth_idx in range(8):
            azimuth = azimuth_idx * np.pi / 4  # 0, 45, 90, ... 315 degrees
            # Rotate direction by tilt angle in the (u, v) plane
            perp = np.cos(azimuth) * u + np.sin(azimuth) * v
            probe_dir = np.cos(tilt) * d + np.sin(tilt) * perp
            probe_dir /= np.linalg.norm(probe_dir)

            result = _pca_step(
                current, probe_dir, all_points, tree,
                base_nbr, base_step, base_snap, smooth, min_pts,
                bias=bias,
            )
            if result is None:
                continue

            _, _, cosine = result
            # The cosine here is vs the *probe* direction, but we care
            # about whether the result is reasonable relative to original
            actual_dir = result[0] - current
            actual_norm = np.linalg.norm(actual_dir)
            if actual_norm < 1e-10:
                continue
            actual_cos = float(np.dot(actual_dir / actual_norm, direction))

            # Accept if it's the best valid probe and doesn't reverse
            if actual_cos > max(min_cosine * 0.5, -0.3) and cosine > best_cosine:
                best_cosine = cosine
                best_result = result

    return best_result


# ---------------------------------------------------------------------------
# Graph-guided lookahead (confirms direction, does NOT jump)
# ---------------------------------------------------------------------------

def _graph_guided_step(
    current, _direction, path, all_points, tree, graph,
    base_nbr, base_step, base_snap, smooth, min_pts, bias,
):
    """Use a local graph search to find where the root continues past a
    sharp bend, then take ONE small PCA step toward that target.

    Unlike the old graph recovery, this does NOT jump to the target.
    The graph only tells us which direction to look. The actual movement
    is a normal small PCA step.

    Returns (next_pt, new_direction) or None (true tip).
    """
    # Find current node in the graph
    _, cur_node = tree.query(current)
    cur_node = int(cur_node)

    # Local Dijkstra with tight cutoff
    search_r = base_nbr * 2
    try:
        local_dists = nx.single_source_dijkstra_path_length(
            graph, cur_node, cutoff=search_r, weight="weight"
        )
    except nx.NetworkXError:
        return None

    if len(local_dists) < 3:
        return None

    # Backward rejection: recent path points
    n_recent = min(5, len(path))
    recent = np.array(path[-n_recent:])

    # Find the best "lookahead target" — a point that is:
    #   - ahead (not behind recent path)
    #   - at moderate distance (1-3x step size away)
    #   - has an elongated local neighborhood (still on a root)
    best_target = None
    best_score = -np.inf

    for node, gdist in local_dists.items():
        if node == cur_node:
            continue

        pt = all_points[node]
        spatial_dist = np.linalg.norm(pt - current)

        # Must be within search radius
        if spatial_dist > search_r:
            continue

        # Prefer moderate distance (not too close, not too far)
        if spatial_dist < base_step * 0.5:
            continue

        # Backward rejection
        dist_to_recent = np.min(np.linalg.norm(recent - pt, axis=1))
        if dist_to_recent < spatial_dist * 0.7:
            continue

        # Elongation check at target
        local_idx = tree.query_ball_point(pt, base_nbr * 0.7)
        if len(local_idx) < min_pts:
            continue

        pca = PCA(n_components=min(3, len(local_idx)))
        pca.fit(all_points[local_idx] - pt)
        elong = pca.explained_variance_ratio_[0]
        if elong < 0.4:
            continue

        score = elong + min(spatial_dist, base_step * 3) / (base_step * 3) * 0.3

        if score > best_score:
            best_score = score
            best_target = pt

    if best_target is None:
        return None

    # We found a continuation target. Take ONE small PCA step toward it.
    target_dir = best_target - current
    tn = np.linalg.norm(target_dir)
    if tn < 1e-10:
        return None
    target_dir /= tn

    # PCA step using the graph-suggested direction (not a jump)
    result = _pca_step(
        current, target_dir, all_points, tree,
        base_nbr, base_step, base_snap, smooth, min_pts,
        bias=bias,
    )
    if result is not None:
        return result[0], result[1]  # next_pt, new_direction

    # PCA couldn't step in that direction — step exactly base_step
    # toward the target and snap with tight radius (no large jumps)
    step_pt = current + base_step * target_dir
    snap_idx = tree.query_ball_point(step_pt, base_snap)
    if not snap_idx:
        return None  # no fallback to wider snap — avoid large jumps

    next_pt = all_points[snap_idx].mean(axis=0)

    # Hard cap: reject if actual displacement exceeds 1.5× step size
    sv = next_pt - current
    sn = np.linalg.norm(sv)
    if sn < 1e-10 or sn > base_step * 1.5:
        return None

    return next_pt, sv / sn
