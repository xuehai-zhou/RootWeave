"""Phase 4: Inward-only tracking from tip seeds.

Each seed (from phase3_tip_seeds) is grown from its tip toward a parent structure
— either the taproot or a previously traced lateral. Seeds are processed
LONGEST FIRST (largest shell radius first) so that a short inner root
never steals a long outer root's points.

Stop conditions for a single tracker
------------------------------------
1. Taproot arrival — three criteria combined:
   a. hard floor: perpendicular distance to taproot axis < R(t) * hard_arrival_factor
   b. geometric + alignment: perp < R(t) * arrival_factor AND local PCA
      aligned with the taproot tangent
   c. closest-approach monitor: got close, then moved away for several steps
2. Arrival at a previously finished lateral (KDTree query within
   avg_distance * other_path_arrival_k).
3. Direction collapses (cosine below min_step_cosine, even after probing
   and graph-guided recovery).
4. max_growth_steps reached.

Step mechanics
--------------
At each step, the tracker calls ``_pca_step`` which:
  - gathers points within a forward hemisphere of the neighborhood
  - filters them to the nearest cross-sectional cluster
  - runs PCA on the filtered points
  - blends the principal direction with the previous direction and a
    small inward bias (radial inward + anti-gravity + optional taproot
    attraction)
  - snaps the candidate to the root cross-section center

If the PCA step fails or the cosine with the previous direction is too
small, recovery is attempted first by local direction probing (24 small
perturbations), then by a local graph lookahead that suggests a
continuation target without jumping to it.

All helpers in this module are private (leading underscore). The only
public entry points are ``grow_inward_from_seeds`` and the
``BranchPath`` dataclass.
"""

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from .config import PipelineConfig
from .graph import adjust_weights_by_density, build_knn_graph
from .phase3_tip_seeds import ShellSeed


@dataclass
class BranchPath:
    label: int
    path: List[np.ndarray]
    surrounding_indices: List[int]
    parent: str                       # "taproot" | "lateral" | "unknown"
    parent_label: Optional[int] = None  # if parent is a lateral, its label


# ===========================================================================
# Public API
# ===========================================================================

def grow_inward_from_seeds(
    all_points: np.ndarray,
    seeds: List[ShellSeed],
    avg_distance: float,
    path_points: np.ndarray,
    radii: np.ndarray,
    tangents: np.ndarray,
    config: PipelineConfig,
) -> List[BranchPath]:
    """Track each seed inward, in longest-first order."""
    if not seeds:
        return []

    # Longest roots first: process seeds with the largest shell_radius first
    ordered = sorted(seeds, key=lambda s: s.shell_radius, reverse=True)

    tree_all = KDTree(all_points)
    path_tree = KDTree(path_points)

    print("Building KNN graph for bend recovery...")
    graph_all = adjust_weights_by_density(
        copy.deepcopy(build_knn_graph(all_points, k=config.knn_k)),
        all_points, radius=avg_distance,
    )

    base_nbr = avg_distance * config.neighborhood_k
    base_step = avg_distance * config.step_k
    base_snap = avg_distance * config.snap_k
    other_path_r = avg_distance * config.other_path_arrival_k

    print(
        f"Growth params: neighborhood={base_nbr:.4f}, "
        f"step={base_step:.4f}, snap={base_snap:.4f}, "
        f"other_path_r={other_path_r:.4f}"
    )

    finished: List[BranchPath] = []
    finished_nodes: List[np.ndarray] = []
    finished_labels: List[int] = []
    finished_tree: Optional[KDTree] = None

    for seed in ordered:
        path, surr, stop_reason, parent_label = _track_one_seed(
            seed,
            all_points, tree_all, graph_all,
            path_points, path_tree, tangents, radii,
            finished_tree, finished_labels,
            base_nbr, base_step, base_snap, other_path_r,
            config,
        )
        if len(path) < 2:
            continue

        parent = {
            "taproot": "taproot",
            "lateral": "lateral",
        }.get(stop_reason, "unknown")

        bp = BranchPath(
            label=seed.label,
            path=path,
            surrounding_indices=list(set(surr)),
            parent=parent,
            parent_label=parent_label,
        )
        finished.append(bp)

        # Add this path's nodes to the finished-nodes KDTree so later
        # seeds can arrive on it.
        for node in path:
            finished_nodes.append(np.asarray(node))
            finished_labels.append(seed.label)
        finished_tree = KDTree(np.asarray(finished_nodes))

    n_taproot = sum(1 for b in finished if b.parent == "taproot")
    n_lateral = sum(1 for b in finished if b.parent == "lateral")
    print(
        f"Tracked {len(finished)}/{len(seeds)} seeds "
        f"({n_taproot} arrived at taproot, {n_lateral} arrived at another lateral)"
    )
    return finished


# ===========================================================================
# Per-seed inward tracker
# ===========================================================================

def _track_one_seed(
    seed: ShellSeed,
    all_points: np.ndarray,
    tree_all: KDTree,
    graph_all: nx.Graph,
    path_points: np.ndarray,
    path_tree: KDTree,
    tangents: np.ndarray,
    radii: np.ndarray,
    finished_tree: Optional[KDTree],
    finished_labels: List[int],
    base_nbr: float,
    base_step: float,
    base_snap: float,
    other_path_r: float,
    config: PipelineConfig,
) -> Tuple[List[np.ndarray], List[int], str, Optional[int]]:
    """Grow a single seed inward.

    Returns (path, surrounding_indices, stop_reason, parent_label).
    stop_reason is one of {"taproot", "lateral", "maxsteps"}.
    parent_label is the finished-path label if stop_reason == "lateral",
    otherwise None.
    """
    # Snap seed to the local root cross-section center
    start = _snap_to_root_center(seed.seed_point, all_points, tree_all, base_nbr)
    direction = seed.initial_direction / (np.linalg.norm(seed.initial_direction) + 1e-12)

    current = start.copy()
    path: List[np.ndarray] = [current.copy()]
    surrounding: List[int] = []

    smooth = config.smoothing_factor
    recoveries_left = config.max_bend_recoveries

    # Taproot arrival bookkeeping (closest-approach monitor)
    min_perp_dist = np.inf
    moving_away_count = 0

    for step_i in range(config.max_growth_steps):
        allow_recovery = recoveries_left > 0

        # --- Arrival at taproot ---
        should_stop, snap_pt = _check_taproot_arrival(
            current, all_points, tree_all,
            path_points, path_tree, tangents, radii,
            config, min_perp_dist, moving_away_count,
        )
        if should_stop:
            if snap_pt is not None:
                path.append(snap_pt)
            return path, surrounding, "taproot", None

        # Update closest-approach monitor
        perp_d = _perp_dist_to_axis(current, path_points, tangents, path_tree)
        if perp_d < min_perp_dist:
            min_perp_dist = perp_d
            moving_away_count = 0
        else:
            moving_away_count += 1

        # --- Arrival at a previously finished lateral ---
        if (
            finished_tree is not None
            and step_i >= config.other_path_min_steps
        ):
            d, nn_idx = finished_tree.query(current)
            if d < other_path_r:
                parent_label = finished_labels[int(nn_idx)]
                # Snap to that node for a clean attachment
                attach_pt = finished_tree.data[int(nn_idx)].copy()
                path.append(attach_pt)
                return path, surrounding, "lateral", parent_label

        # --- Compute per-step bias ---
        bias = _compute_inward_bias(current, path_tree, path_points, config)

        # --- One PCA step, with recovery if it stalls or bends too hard ---
        result = _pca_step(
            current, direction, all_points, tree_all,
            base_nbr, base_step, base_snap, smooth,
            config.pca_min_points, bias=bias,
        )

        if result is not None:
            next_pt, new_dir, cosine = result
            if cosine < config.min_step_cosine:
                recovered = _recover_step(
                    current, direction, path, all_points, tree_all, graph_all,
                    base_nbr, base_step, base_snap, smooth, config, bias,
                    allow_recovery,
                )
                if recovered is None:
                    break
                next_pt, new_dir = recovered
                path.append(next_pt)
                direction = new_dir
                current = next_pt
                recoveries_left -= 1
                continue

            # Accept the step
            path.append(next_pt)
            direction = new_dir
            current = next_pt
            surr = tree_all.query_ball_point(current, base_snap)
            surrounding.extend(surr)

        else:
            # PCA failed outright — try recovery
            recovered = _recover_step(
                current, direction, path, all_points, tree_all, graph_all,
                base_nbr, base_step, base_snap, smooth, config, bias,
                allow_recovery,
            )
            if recovered is None:
                break
            next_pt, new_dir = recovered
            path.append(next_pt)
            direction = new_dir
            current = next_pt
            recoveries_left -= 1

    return path, surrounding, "maxsteps", None


def _recover_step(
    current: np.ndarray,
    direction: np.ndarray,
    path: List[np.ndarray],
    all_points: np.ndarray,
    tree: KDTree,
    graph: nx.Graph,
    base_nbr: float,
    base_step: float,
    base_snap: float,
    smooth: float,
    config: PipelineConfig,
    bias: Optional[np.ndarray],
    allow_recovery: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Two-stage recovery when a PCA step fails or bends too sharply:
    first try local direction probing, then graph-guided lookahead.
    Returns (next_pt, new_direction) or None if both fail.
    """
    if not allow_recovery:
        return None

    probe = _probe_directions(
        current, direction, all_points, tree,
        base_nbr, base_step, base_snap, smooth,
        config.pca_min_points, config.min_step_cosine, bias,
    )
    if probe is not None:
        next_pt, new_dir, _ = probe
        return next_pt, new_dir

    guided = _graph_guided_step(
        current, path, all_points, tree, graph,
        base_nbr, base_step, base_snap, smooth,
        config.pca_min_points, bias,
    )
    if guided is not None:
        return guided

    return None


# ===========================================================================
# Inward bias
# ===========================================================================

def _compute_inward_bias(
    current: np.ndarray,
    path_tree: KDTree,
    path_points: np.ndarray,
    config: PipelineConfig,
) -> Optional[np.ndarray]:
    """Combine three weak inward cues:
       - radial inward: XY unit vector from current toward the vertical axis
       - taproot attraction: 3D unit vector toward the nearest taproot point
       - anti-gravity: small upward nudge (Z-up)

    Each weight can be zero to disable the cue. Returns a unit-length
    bias vector, or None when all cues are negligible.
    """
    # Radial inward (XY only) — the vertical axis is the mean XY of the taproot
    axis_xy = path_points[:, :2].mean(axis=0)
    radial = np.zeros(3, dtype=float)
    rx = axis_xy - current[:2]
    rn = float(np.linalg.norm(rx))
    if rn > 1e-10:
        radial[0] = rx[0] / rn
        radial[1] = rx[1] / rn

    # Taproot attraction (3D) — toward nearest taproot path point
    attraction = np.zeros(3, dtype=float)
    _, nearest_idx = path_tree.query(current)
    to_tap = path_points[int(nearest_idx)] - current
    tn = float(np.linalg.norm(to_tap))
    if tn > 1e-10:
        attraction = to_tap / tn

    # Anti-gravity (small upward nudge)
    antigrav = np.array([0.0, 0.0, 1.0])

    bias = (
        config.inward_radial_weight * radial
        + config.inward_attraction_weight * attraction
        + config.antigravity_weight * antigrav
    )
    n = float(np.linalg.norm(bias))
    if n < 1e-10:
        return None
    return bias / n


# ===========================================================================
# Taproot arrival check
# ===========================================================================

def _check_taproot_arrival(
    current: np.ndarray,
    all_points: np.ndarray,
    tree: KDTree,
    path_points: np.ndarray,
    path_tree: KDTree,
    tangents: np.ndarray,
    radii: np.ndarray,
    config: PipelineConfig,
    min_perp_dist: float,
    moving_away_count: int,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Three-criteria arrival check for the taproot:

    0. Hard floor (unconditional): perpendicular distance to taproot
       axis < R(t) * hard_arrival_factor.
    1. Inflated cylinder + alignment: perp < R(t) * arrival_factor AND
       the local PC1 direction aligns with the taproot tangent.
    2. Closest-approach monitor: got close and moved away for 4+ steps.

    Returns (should_stop, snap_point).
    """
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

    # Criterion 0 — hard floor
    if perp_dist < hard_threshold:
        return True, snap_pt

    # Criterion 1 — inflated cylinder + alignment
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

    # Criterion 2 — closest-approach monitor
    if moving_away_count >= 4 and min_perp_dist < arrival_threshold * 1.2:
        return True, snap_pt

    return False, None


def _perp_dist_to_axis(
    current: np.ndarray,
    path_points: np.ndarray,
    tangents: np.ndarray,
    path_tree: KDTree,
) -> float:
    """Perpendicular distance from *current* to the local taproot axis."""
    _, nearest_ring = path_tree.query(current)
    nearest_ring = int(nearest_ring)
    offset = current - path_points[nearest_ring]
    along = np.dot(offset, tangents[nearest_ring])
    perp = offset - along * tangents[nearest_ring]
    return float(np.linalg.norm(perp))


def _compute_tangents(path_points: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Smoothed unit tangent at each path point (Gaussian-smoothed gradient)."""
    tangents = np.gradient(path_points, axis=0)
    for ax in range(3):
        tangents[:, ax] = gaussian_filter1d(tangents[:, ax], sigma=sigma)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tangents / norms


# ===========================================================================
# Starting-point snap
# ===========================================================================

def _snap_to_root_center(
    centroid: np.ndarray,
    all_points: np.ndarray,
    tree: KDTree,
    base_nbr: float,
) -> np.ndarray:
    """Snap a seed point (usually near a root edge) to the local root center.

    Finds nearby points, estimates the local root axis via PCA, slices
    the cloud perpendicular to that axis, clusters the slice, and returns
    the centroid of the nearest cluster.
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

    # Slice the cross-section with a thin plane perpendicular to the axis
    offsets = pts - centroid
    along = np.abs(offsets @ axis)
    plane_mask = along < base_nbr * 0.3
    plane_pts = pts[plane_mask]

    if len(plane_pts) < 3:
        return centroid

    filtered = _filter_to_nearest_cluster(plane_pts, centroid, axis, base_nbr * 0.5, 2)
    if filtered is not None and len(filtered) >= 2:
        return filtered.mean(axis=0)
    return plane_pts.mean(axis=0)


# ===========================================================================
# Single PCA step
# ===========================================================================

def _pca_step(
    current: np.ndarray,
    direction: np.ndarray,
    all_points: np.ndarray,
    tree: KDTree,
    nbr_r: float,
    step_sz: float,
    snap_r: float,
    smooth: float,
    min_pts: int,
    bias: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """One PCA tracking step with cross-sectional cluster filtering.

    Direction = PCA (dominant) + optional bias (inward radial + attraction
    + anti-gravity, already combined into one unit vector).

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

    # Blend PCA (dominant) with previous direction and optional bias
    blended = smooth * pc1 + (1.0 - smooth) * direction
    if bias is not None:
        blended += bias
    n = np.linalg.norm(blended)
    if n < 1e-10:
        return None
    blended /= n

    # Step + cross-sectional centering snap
    candidate = current + step_sz * blended
    next_pt = _cross_section_snap(
        candidate, blended, all_points, tree, snap_r, min_pts,
    )
    if next_pt is None:
        return None

    # Derive the actual step direction and cosine with the previous one
    sv = next_pt - current
    sn = np.linalg.norm(sv)
    if sn < 1e-10:
        return None
    sd = sv / sn
    cosine = float(np.dot(sd, direction))

    # Smoothed new direction for the next iteration
    nd = smooth * sd + (1.0 - smooth) * blended
    nn = np.linalg.norm(nd)
    nd = nd / nn if nn > 1e-10 else sd

    return next_pt, nd, cosine


# ===========================================================================
# Cross-sectional snapping
# ===========================================================================

def _cross_section_snap(
    candidate: np.ndarray,
    direction: np.ndarray,
    all_points: np.ndarray,
    tree: KDTree,
    snap_r: float,
    min_pts: int,
) -> Optional[np.ndarray]:
    """Snap a candidate position to the root cross-section center using
    two centering passes.

    The first pass gets approximately onto the root; the second pass
    widens the search from the corrected position so it sees the full
    cross-section and centers properly.
    """
    pos = candidate.copy()
    for iteration in range(2):
        r = snap_r if iteration == 0 else snap_r * 2
        centered = _single_centering_pass(pos, direction, all_points, tree, r, snap_r, min_pts)
        if centered is None:
            if iteration == 0:
                return None
            break  # second pass failed, keep first result
        pos = centered
    return pos


def _single_centering_pass(
    center: np.ndarray,
    direction: np.ndarray,
    all_points: np.ndarray,
    tree: KDTree,
    search_r: float,
    cluster_eps: float,
    min_pts: int,
) -> Optional[np.ndarray]:
    """One pass of cross-sectional centering: search, slice, cluster, centroid."""
    si = tree.query_ball_point(center, search_r)
    if not si:
        return None

    pts = all_points[si]

    # Plane basis perpendicular to *direction*
    d = direction / np.linalg.norm(direction)
    ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    # Thin slice perpendicular to direction
    offsets = pts - center
    along = offsets @ d
    plane_mask = np.abs(along) < cluster_eps * 0.5
    plane_pts = pts[plane_mask]

    if len(plane_pts) < max(2, min_pts):
        return pts.mean(axis=0)

    # Project to 2D and cluster
    offsets_2d = plane_pts - center
    proj_2d = np.column_stack([offsets_2d @ u, offsets_2d @ v])

    labels = DBSCAN(eps=cluster_eps * 0.5, min_samples=max(1, min_pts)).fit_predict(proj_2d)

    unique = set(labels)
    unique.discard(-1)

    if not unique:
        center_2d = proj_2d.mean(axis=0)
    else:
        best_label = max(unique, key=lambda lab: int(np.sum(labels == lab)))
        center_2d = proj_2d[labels == best_label].mean(axis=0)

    return center + center_2d[0] * u + center_2d[1] * v


def _filter_to_nearest_cluster(
    pts: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
    eps: float,
    min_pts: int,
) -> Optional[np.ndarray]:
    """Project points onto the plane perpendicular to *direction*, cluster
    them in 2D with DBSCAN, and return only the 3D points belonging to
    the largest cluster. This filters out neighboring roots that happen
    to fall inside the search radius.
    """
    if len(pts) < 3:
        return pts

    d = direction / np.linalg.norm(direction)
    ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    offsets = pts - center
    proj_2d = np.column_stack([offsets @ u, offsets @ v])

    labels = DBSCAN(eps=eps, min_samples=max(1, min_pts)).fit_predict(proj_2d)

    unique = set(labels)
    unique.discard(-1)
    if not unique:
        return pts  # all noise — return everything

    # The root we're on fills most of the cross-section; neighboring
    # roots at the edge have fewer points.
    best_label = max(unique, key=lambda lab: int(np.sum(labels == lab)))
    return pts[labels == best_label]


# ===========================================================================
# Recovery: local direction probing
# ===========================================================================

def _probe_directions(
    current: np.ndarray,
    direction: np.ndarray,
    all_points: np.ndarray,
    tree: KDTree,
    base_nbr: float,
    base_step: float,
    base_snap: float,
    smooth: float,
    min_pts: int,
    min_cosine: float,
    bias: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """When a PCA step fails or bends too hard, try 24 small perturbations
    of the current direction (3 tilts × 8 azimuths) at the same step
    size. Returns the best valid probe, or None.
    """
    d = direction / np.linalg.norm(direction)
    ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    best_result = None
    best_cosine = -np.inf

    for tilt_deg in (20, 40, 60):
        tilt = np.radians(tilt_deg)
        for azimuth_idx in range(8):
            azimuth = azimuth_idx * np.pi / 4  # 0, 45, ... 315 degrees
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

            # Score the probe by its cosine with the *original* direction
            # — we want the best continuation, not the best probe rotation.
            next_pt, _, cosine = result
            actual_dir = next_pt - current
            actual_norm = np.linalg.norm(actual_dir)
            if actual_norm < 1e-10:
                continue
            actual_cos = float(np.dot(actual_dir / actual_norm, direction))

            if actual_cos > max(min_cosine * 0.5, -0.3) and cosine > best_cosine:
                best_cosine = cosine
                best_result = result

    return best_result


# ===========================================================================
# Recovery: graph-guided lookahead (suggests a direction, does NOT jump)
# ===========================================================================

def _graph_guided_step(
    current: np.ndarray,
    path: List[np.ndarray],
    all_points: np.ndarray,
    tree: KDTree,
    graph: nx.Graph,
    base_nbr: float,
    base_step: float,
    base_snap: float,
    smooth: float,
    min_pts: int,
    bias: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Use a local graph search to find where the root continues past a
    sharp bend, then take ONE small PCA step toward that target.

    The graph only suggests a direction — the actual movement is a
    normal-sized PCA step, not a jump to the target.
    """
    _, cur_node = tree.query(current)
    cur_node = int(cur_node)

    # Local Dijkstra with a tight cutoff
    search_r = base_nbr * 2
    try:
        local_dists = nx.single_source_dijkstra_path_length(
            graph, cur_node, cutoff=search_r, weight="weight",
        )
    except nx.NetworkXError:
        return None

    if len(local_dists) < 3:
        return None

    # Backward rejection: recent path points
    n_recent = min(5, len(path))
    recent = np.array(path[-n_recent:])

    best_target = None
    best_score = -np.inf

    for node, _gdist in local_dists.items():
        if node == cur_node:
            continue

        pt = all_points[node]
        spatial_dist = np.linalg.norm(pt - current)

        if spatial_dist > search_r:
            continue
        if spatial_dist < base_step * 0.5:
            continue

        # Backward rejection — don't look behind
        dist_to_recent = np.min(np.linalg.norm(recent - pt, axis=1))
        if dist_to_recent < spatial_dist * 0.7:
            continue

        # The target must still be on a root (elongated local neighborhood)
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

    # Take one small PCA step toward the continuation target
    target_dir = best_target - current
    tn = np.linalg.norm(target_dir)
    if tn < 1e-10:
        return None
    target_dir /= tn

    result = _pca_step(
        current, target_dir, all_points, tree,
        base_nbr, base_step, base_snap, smooth, min_pts,
        bias=bias,
    )
    if result is not None:
        return result[0], result[1]

    # PCA refused — fall back to a single fixed step toward the target,
    # snapped tightly so we can't make large jumps.
    step_pt = current + base_step * target_dir
    snap_idx = tree.query_ball_point(step_pt, base_snap)
    if not snap_idx:
        return None

    next_pt = all_points[snap_idx].mean(axis=0)

    # Hard cap: reject steps that exceed 1.5× the base step size
    sv = next_pt - current
    sn = np.linalg.norm(sv)
    if sn < 1e-10 or sn > base_step * 1.5:
        return None

    return next_pt, sv / sn
