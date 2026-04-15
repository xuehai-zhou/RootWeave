"""Phase 3: Shrinking-tube tip seed detection with KNN-graph verification.

Conceptual flow
---------------
The "shell" is a tube around the taproot POLYLINE (not around a single
vertical axis).  For each cloud point we compute:
    perp_tap       = perpendicular distance to the taproot polyline
                     (using the local tangent at the nearest path point)
    perp_effective = perp_tap - taproot_clearance_k * R(t_nearest)

The sweep shrinks *perp_effective* from its maximum down toward zero.
At perp_effective = 0, the tube's inner wall is exactly
(clearance_k × R(t)) away from the taproot surface — this cushion
absorbs any under-estimate of R(t) and is the intentional stop line.
Very short laterals whose tips sit inside this cushion are skipped
(they're too short to track reliably).

At each shell step we DBSCAN the unclaimed shell points and classify
each cluster using a KNN graph built once on all non-main-root points.

Classification (for each cluster at each shell step):

  1. *Connected to claimed?* — bounded BFS through the graph, searching
     for any already-claimed node within a few hops.  If yes, the
     cluster belongs to a root that was already seeded somewhere.

  2. If connected:
       *Terminus?* — BFS through UNCLAIMED graph nodes only; if the
       BFS exhausts quickly (reaches few nodes beyond the cluster), the
       cluster sits at the end of some unclaimed region → SEED it (this
       recovers actual tips of curving roots whose outer bend was
       seeded earlier).  Otherwise → continuation, claim only.

  3. If NOT connected:
       *Has unclaimed outward extension?* — BFS through UNCLAIMED graph
       nodes; if any reached node has perp_effective > cluster_max + margin,
       the root continues further out → skip this shell step (the
       cluster will be re-examined at a larger r).  Otherwise the
       cluster is the outermost end of an unseen root → SEED.

Graph BFS respects the root's actual connectivity, which matters in
two concrete cases:
  - Mid-shaft clusters on a straight root are correctly identified as
    continuations (graph reaches the unclaimed inner shaft) rather than
    falsely flagged as tips just because their outer neighbors were
    claimed by a previous shell.
  - Curving roots whose actual tip lies inward of their outermost bend
    can still get a second seed at the tip (it's graph-terminal in the
    unclaimed subgraph).

The graph is built on non-main-root points only, so taproot proximity
cannot leak into the BFS.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from .config import PipelineConfig


@dataclass
class ShellSeed:
    label: int
    seed_point: np.ndarray         # (3,) tip position (cluster centroid)
    initial_direction: np.ndarray  # (3,) unit vector, points inward
    shell_radius: float            # the shell radius at which this seed was found
    cluster_points: np.ndarray     # (K, 3) the cluster that produced it


@dataclass
class ShellSweepResult:
    seeds: List[ShellSeed]
    axis_xy: np.ndarray            # (2,) vertical axis position in XY
    r_start: float
    r_stop: float
    dr: float
    thickness: float
    shell_history: list = field(default_factory=list)  # optional debug trace


# ===========================================================================
# Public API
# ===========================================================================

def detect_tip_seeds(
    points: np.ndarray,
    main_root_mask: np.ndarray,
    path_points: np.ndarray,
    radii: np.ndarray,
    avg_distance: float,
    config: PipelineConfig,
) -> ShellSweepResult:
    """Sweep a shrinking cylindrical shell and collect root-tip seeds.

    Parameters
    ----------
    points : (N, 3) full cloud (in Z-up normalized space).
    main_root_mask : (N,) bool, True for points inside the taproot volume.
    path_points : (M, 3) taproot centerline.
    radii : (M,) smoothed taproot radius profile.
    avg_distance : average nearest-neighbor distance in the cloud.
    config : pipeline configuration.
    """
    # ------------------------------------------------------------------
    # Reference axis: the taproot POLYLINE, not a single vertical line.
    # For each cloud point we find its nearest path point and take the
    # perpendicular component of (point - path_point) relative to the
    # local tangent.  This gives a distance that respects the taproot's
    # actual curvature.
    # ------------------------------------------------------------------
    tangents = _compute_path_tangents(path_points, sigma=config.tangent_sigma)
    path_tree = KDTree(path_points)
    _, nearest_ring = path_tree.query(points, k=1)          # (N,)
    nearest_ring = nearest_ring.astype(np.int64)

    offsets = points - path_points[nearest_ring]            # (N, 3)
    tan_at_pt = tangents[nearest_ring]                      # (N, 3)
    along = np.einsum('ij,ij->i', offsets, tan_at_pt)       # (N,)
    perp_vec = offsets - along[:, None] * tan_at_pt         # (N, 3)
    perp_tap = np.linalg.norm(perp_vec, axis=1)             # (N,)

    # Effective distance: subtract a clearance cushion = clearance_k * R(t)
    # around the extracted taproot surface.  Tips inside this cushion are
    # excluded from the search.
    local_radius = radii[nearest_ring]                      # (N,)
    perp_effective = perp_tap - config.taproot_clearance_k * local_radius

    # Kept for backwards-compat with ShellSweepResult consumers; also used
    # as a coarse visualization reference.
    axis_xy = path_points[:, :2].mean(axis=0)

    # ------------------------------------------------------------------
    # Search volume: full Z extent of the cloud (padded)
    # ------------------------------------------------------------------
    z_pad = avg_distance * config.shell_z_pad_k
    z_min = float(points[:, 2].min()) - z_pad
    z_max = float(points[:, 2].max()) + z_pad
    z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)

    # ------------------------------------------------------------------
    # Shell parameters (all in "effective distance" units — offsets above
    # the clearance-inflated taproot surface).
    # ------------------------------------------------------------------
    # Only points OUTSIDE the clearance cushion are searchable.
    searchable = z_mask & ~main_root_mask & (perp_effective > 0)
    if not np.any(searchable):
        print("Shell sweep: no searchable points outside the taproot clearance.")
        return ShellSweepResult(
            seeds=[], axis_xy=axis_xy,
            r_start=0.0, r_stop=0.0, dr=0.0, thickness=0.0,
        )

    max_outer = float(perp_effective[searchable].max())
    r_start = max_outer * config.shell_start_margin
    r_stop = avg_distance * config.shell_stop_effective_k
    if r_stop >= r_start:
        r_stop = 0.0  # fall back: sweep all the way to the clearance surface

    dr = avg_distance * config.shell_step_k
    thickness = avg_distance * config.shell_thickness_k
    cluster_eps = avg_distance * config.shell_cluster_eps_k
    tip_margin = avg_distance * config.tip_margin_k
    seed_dedupe_r = avg_distance * config.seed_dedupe_k
    seed_init_r = avg_distance * config.seed_init_neighborhood_k

    n_shells = max(1, int(np.ceil((r_start - r_stop) / dr)))
    print(
        f"Tube sweep: r_start={r_start:.4f} -> r_stop={r_stop:.4f} "
        f"(effective; clearance={config.taproot_clearance_k}×R(t)), "
        f"dr={dr:.4f}, thickness={thickness:.4f}, n_shells={n_shells}"
    )
    print(
        f"  cluster_eps={cluster_eps:.4f}, tip_margin={tip_margin:.4f}, "
        f"conn_depth={config.graph_connection_depth}, "
        f"out_depth={config.graph_outward_depth}, "
        f"term_depth={config.graph_terminus_depth}, "
        f"term_max={config.graph_terminus_max_reached}"
    )

    # ------------------------------------------------------------------
    # KNN graph on non-main-root points.  Main-root points are excluded
    # entirely so taproot proximity cannot connect unrelated clusters via
    # BFS.  All bookkeeping below lives in "lateral index" space.
    # ------------------------------------------------------------------
    lateral_full_indices = np.where(~main_root_mask)[0]
    n_lat = len(lateral_full_indices)
    lateral_pts = points[lateral_full_indices]

    # Map full-cloud index → lateral index (or -1 if the point is main root)
    full_to_lat = np.full(len(points), -1, dtype=np.int64)
    full_to_lat[lateral_full_indices] = np.arange(n_lat)

    print(f"Building KNN graph on {n_lat} lateral points (k={config.knn_k})...")
    adj = _build_adjacency(lateral_pts, k=config.knn_k)

    # perp_effective in lateral-index space (precomputed)
    lat_perp_eff = perp_effective[lateral_full_indices]

    # ------------------------------------------------------------------
    # Claim state
    # ------------------------------------------------------------------
    # Full-cloud mask used by shell selection to exclude main root +
    # already-claimed points.
    claimed_full = main_root_mask.copy()
    # Lateral-only mask used by graph BFS checks.  Main-root points are
    # NOT represented here (they're not in the graph at all).
    claimed_lat = np.zeros(n_lat, dtype=bool)

    # KD-tree over the full cloud for seed initial-direction PCA
    tree_full = KDTree(points)

    seeds: List[ShellSeed] = []
    seed_points_list: List[np.ndarray] = []
    seed_tree: Optional[KDTree] = None
    shell_history = []
    n_continuations_total = 0

    current_r = r_start
    step = 0
    for step in range(n_shells):
        lower = current_r - thickness / 2.0
        upper = current_r + thickness / 2.0
        shell_mask = (
            (perp_effective >= lower) & (perp_effective <= upper)
            & z_mask & ~claimed_full
        )
        shell_idx = np.where(shell_mask)[0]  # full-cloud indices

        n_in = int(len(shell_idx))
        n_clusters = 0
        n_new_seeds = 0
        n_continuations = 0

        if n_in >= config.shell_min_cluster:
            shell_pts = points[shell_idx]
            labels = DBSCAN(
                eps=cluster_eps, min_samples=config.shell_min_cluster,
            ).fit_predict(shell_pts)

            unique_labels = [int(l) for l in set(labels) if l != -1]
            n_clusters = len(unique_labels)

            # Collect cluster info using the CURRENT claim snapshot, so all
            # clusters in this step see the same state.
            cluster_infos = []
            for lbl in unique_labels:
                cmask = labels == lbl
                c_full = shell_idx[cmask]
                c_pts = points[c_full]
                if len(c_pts) < config.shell_min_cluster:
                    continue
                c_lat = full_to_lat[c_full]
                # All shell points are non-main-root by construction,
                # so c_lat never contains -1.
                cluster_max_perp = float(lat_perp_eff[c_lat].max())
                cluster_infos.append((c_full, c_lat, c_pts, cluster_max_perp))

            to_claim_full: List[np.ndarray] = []
            to_claim_lat: List[np.ndarray] = []

            for c_full, c_lat, c_pts, cluster_max_perp in cluster_infos:
                connected = _graph_connected_to_claimed(
                    c_lat, adj, claimed_lat, config.graph_connection_depth,
                )

                if connected:
                    # Already part of a seeded root.  Seed only if it's a
                    # graph terminus (e.g., actual tip of a curving root).
                    is_terminus = _is_graph_terminus(
                        c_lat, adj, claimed_lat,
                        config.graph_terminus_depth,
                        config.graph_terminus_max_reached,
                    )
                    if is_terminus:
                        emitted = _try_emit_seed(
                            c_pts, current_r,
                            points, tree_full,
                            path_points, path_tree, seed_init_r,
                            seed_tree, seed_dedupe_r, seeds, seed_points_list,
                            config,
                        )
                        if emitted:
                            n_new_seeds += 1
                            seed_tree = KDTree(np.asarray(seed_points_list))
                        to_claim_full.append(c_full)
                        to_claim_lat.append(c_lat)
                    else:
                        # Mid-shaft continuation: claim only.
                        to_claim_full.append(c_full)
                        to_claim_lat.append(c_lat)
                        n_continuations += 1

                else:
                    # Unseen root.  Seed iff this cluster is the outermost
                    # extent (no unclaimed graph node reaches further out).
                    has_outward = _has_unclaimed_graph_outward(
                        c_lat, adj, claimed_lat, lat_perp_eff,
                        cluster_max_perp + tip_margin,
                        config.graph_outward_depth,
                    )
                    if not has_outward:
                        emitted = _try_emit_seed(
                            c_pts, current_r,
                            points, tree_full,
                            path_points, path_tree, seed_init_r,
                            seed_tree, seed_dedupe_r, seeds, seed_points_list,
                            config,
                        )
                        if emitted:
                            n_new_seeds += 1
                            seed_tree = KDTree(np.asarray(seed_points_list))
                        to_claim_full.append(c_full)
                        to_claim_lat.append(c_lat)
                    # else: SKIP — outer portion still pending.  The sweep
                    # is inward-only, so the outer portion was either
                    # already visited (and we're re-seeing leftover noise)
                    # or it's genuinely unclaimed; either way, deferring
                    # is safe because this cluster will stay unclaimed and
                    # re-appear at the next shell.

            # Apply all claims after classification
            for idx_arr in to_claim_full:
                claimed_full[idx_arr] = True
            for idx_arr in to_claim_lat:
                claimed_lat[idx_arr] = True

        n_continuations_total += n_continuations
        shell_history.append({
            "r": current_r,
            "shell_points": n_in,
            "clusters": n_clusters,
            "new_seeds": n_new_seeds,
            "continuations": n_continuations,
        })

        current_r -= dr
        if current_r <= r_stop:
            break

    print(
        f"Shell sweep found {len(seeds)} tip seeds across {step + 1} shells "
        f"({n_continuations_total} continuation clusters claimed)"
    )
    return ShellSweepResult(
        seeds=seeds,
        axis_xy=axis_xy,
        r_start=r_start,
        r_stop=r_stop,
        dr=dr,
        thickness=thickness,
        shell_history=shell_history,
    )


# ===========================================================================
# KNN adjacency
# ===========================================================================

def _build_adjacency(pts: np.ndarray, k: int) -> List[np.ndarray]:
    """KNN adjacency list: adj[i] = np.ndarray of k neighbor indices (excluding self).

    Faster than networkx for BFS: flat numpy arrays, no edge-attribute
    lookups.  Symmetry is not enforced (if j is a neighbor of i but not
    vice-versa, BFS only traverses i → j, not j → i).  For our purposes
    this is fine because we only care about local connectivity.
    """
    tree = KDTree(pts)
    # k + 1 because the closest neighbor is always the point itself.
    _, idx = tree.query(pts, k=k + 1)
    return [row[1:].astype(np.int64) for row in idx]


# ===========================================================================
# Graph BFS classification helpers
# ===========================================================================

def _graph_connected_to_claimed(
    start_nodes: Sequence[int],
    adj: List[np.ndarray],
    claimed_lat: np.ndarray,
    max_depth: int,
) -> bool:
    """True iff BFS from *start_nodes* reaches any claimed node within
    max_depth hops.  Traversal does NOT pass through claimed nodes — the
    goal is to find a claimed neighbor, not to route through them.
    """
    if max_depth <= 0:
        return False
    visited = set(int(i) for i in start_nodes)
    frontier = list(visited)
    for _ in range(max_depth):
        next_frontier = []
        for node in frontier:
            for nbr in adj[node]:
                nbr_i = int(nbr)
                if nbr_i in visited:
                    continue
                if claimed_lat[nbr_i]:
                    return True
                visited.add(nbr_i)
                next_frontier.append(nbr_i)
        frontier = next_frontier
        if not frontier:
            break
    return False


def _has_unclaimed_graph_outward(
    start_nodes: Sequence[int],
    adj: List[np.ndarray],
    claimed_lat: np.ndarray,
    lat_perp: np.ndarray,
    threshold: float,
    max_depth: int,
) -> bool:
    """True iff BFS through UNCLAIMED graph nodes reaches any node with
    ``lat_perp[node] > threshold``.  Used for the unseen-root case: if
    the root continues further out (in perp-to-taproot distance) via
    unclaimed points, this cluster is not the outermost extent.
    """
    if max_depth <= 0:
        return False
    visited = set(int(i) for i in start_nodes)
    frontier = list(visited)
    for _ in range(max_depth):
        next_frontier = []
        for node in frontier:
            for nbr in adj[node]:
                nbr_i = int(nbr)
                if nbr_i in visited or claimed_lat[nbr_i]:
                    continue
                if lat_perp[nbr_i] > threshold:
                    return True
                visited.add(nbr_i)
                next_frontier.append(nbr_i)
        frontier = next_frontier
        if not frontier:
            break
    return False


def _is_graph_terminus(
    start_nodes: Sequence[int],
    adj: List[np.ndarray],
    claimed_lat: np.ndarray,
    max_depth: int,
    max_reached: int,
) -> bool:
    """True iff BFS through UNCLAIMED graph nodes reaches ≤ *max_reached*
    new nodes beyond *start_nodes*.  Used for the connected-cluster case:
    a terminus is a cluster whose unclaimed expansion saturates quickly
    — e.g., the actual tip of a curving root whose outer portion was
    already claimed via the bend.
    """
    if max_depth <= 0:
        return True
    start_set = set(int(i) for i in start_nodes)
    visited = set(start_set)
    frontier = list(start_set)
    new_reached = 0
    for _ in range(max_depth):
        next_frontier = []
        for node in frontier:
            for nbr in adj[node]:
                nbr_i = int(nbr)
                if nbr_i in visited or claimed_lat[nbr_i]:
                    continue
                visited.add(nbr_i)
                next_frontier.append(nbr_i)
                new_reached += 1
                if new_reached > max_reached:
                    return False
        frontier = next_frontier
        if not frontier:
            break
    return new_reached <= max_reached


# ===========================================================================
# Seed emission (shared between the connected-terminus and unseen-root paths)
# ===========================================================================

def _try_emit_seed(
    cluster_pts: np.ndarray,
    current_r: float,
    points: np.ndarray,
    tree_full: KDTree,
    path_points: np.ndarray,
    path_tree: KDTree,
    seed_init_r: float,
    seed_tree: Optional[KDTree],
    seed_dedupe_r: float,
    seeds: List[ShellSeed],
    seed_points_list: List[np.ndarray],
    config: PipelineConfig,
) -> bool:
    """Try to emit a seed for a cluster.  Returns True on success, False
    if duplicate-deduped or if the initial direction can't be estimated.
    The caller handles claim bookkeeping.
    """
    centroid = cluster_pts.mean(axis=0)

    # Spatial dedupe against existing seeds
    if seed_tree is not None:
        d, _ = seed_tree.query(centroid)
        if d < seed_dedupe_r:
            return False

    init_dir = _estimate_inward_direction(
        centroid, points, tree_full, path_points, path_tree,
        neighborhood_r=seed_init_r,
        w_pca=config.init_dir_pca_weight,
        w_taproot=config.init_dir_taproot_weight,
        w_antigravity=config.init_dir_antigravity_weight,
    )
    if init_dir is None:
        return False

    seeds.append(ShellSeed(
        label=len(seeds),
        seed_point=centroid,
        initial_direction=init_dir,
        shell_radius=current_r,
        cluster_points=cluster_pts,
    ))
    seed_points_list.append(centroid)
    return True


# ===========================================================================
# Initial inward direction for a seed
# ===========================================================================

def _estimate_inward_direction(
    centroid: np.ndarray,
    points: np.ndarray,
    tree: KDTree,
    path_points: np.ndarray,
    path_tree: KDTree,
    neighborhood_r: float,
    w_pca: float = 0.60,
    w_taproot: float = 0.25,
    w_antigravity: float = 0.15,
) -> Optional[np.ndarray]:
    """Ensemble initial direction from three independent cues:

      1. **PCA** of the local neighborhood — captures the root's local
         axis orientation.  Has an inherent sign ambiguity (it's a line,
         not a vector) that is resolved against the preference direction.
      2. **Toward nearest taproot path point** — the most natural "inward"
         signal, but ambiguous when the root is parallel to the taproot
         (toward-taproot has small magnitude perpendicular to the root
         axis, so it barely nudges).
      3. **Anti-gravity (+Z)** — roots tips grow down; tracking inward
         from a tip generally means going UP.  This is independent of
         the taproot geometry and is the tie-breaker for bent roots
         whose tip is below the seed: anti-gravity points away from the
         tip even when toward-taproot is near-zero.

    The three cues combine in two steps:
      a. Form a *preference vector* = ``w_taproot * toward_tap
         + w_antigravity * antigrav`` (normalized).  This is what the
         PCA axis gets flipped to agree with.
      b. Return ``normalize(w_pca * PC1_oriented + w_taproot * toward_tap
         + w_antigravity * antigrav)``.

    If PCA can't find an elongated neighborhood, return the preference
    vector (taproot + anti-gravity without PCA).
    """
    # --- Cue 2: toward the nearest taproot path point ---
    _, nearest_ring = path_tree.query(centroid)
    nearest_ring = int(nearest_ring)
    to_tap = path_points[nearest_ring] - centroid
    tn = float(np.linalg.norm(to_tap))
    if tn > 1e-10:
        toward_tap = to_tap / tn
    else:
        toward_tap = np.zeros(3)

    # --- Cue 3: anti-gravity (+Z) ---
    antigrav = np.array([0.0, 0.0, 1.0])

    # --- Preference vector (used to orient PCA) ---
    preference = w_taproot * toward_tap + w_antigravity * antigrav
    pref_norm = float(np.linalg.norm(preference))
    if pref_norm < 1e-10:
        # All non-PCA weights are zero AND we have no toward-taproot —
        # nothing to anchor on.
        if np.linalg.norm(toward_tap) < 1e-10:
            return None
        preference = toward_tap
        pref_norm = 1.0
    preference /= pref_norm

    # --- Cue 1: PCA on a local neighborhood ---
    best_axis: Optional[np.ndarray] = None
    for mult in (1.0, 1.5, 2.5):
        idx = tree.query_ball_point(centroid, neighborhood_r * mult)
        if len(idx) < 5:
            continue
        pts = points[idx]
        pca = PCA(n_components=min(3, len(pts)))
        pca.fit(pts - centroid)
        if pca.explained_variance_ratio_[0] > 0.4:
            best_axis = pca.components_[0]
            break

    # --- PCA failed: return pure preference ---
    if best_axis is None:
        return preference

    # --- Orient PCA so it agrees with the preference direction ---
    if np.dot(best_axis, preference) < 0:
        best_axis = -best_axis

    # --- Blend all three cues ---
    mixed = (
        w_pca * best_axis
        + w_taproot * toward_tap
        + w_antigravity * antigrav
    )
    n = float(np.linalg.norm(mixed))
    if n < 1e-10:
        return preference
    return mixed / n


# ===========================================================================
# Taproot tangent helper (copied from phase2_main_volume for phase3's
# self-contained use)
# ===========================================================================

def _compute_path_tangents(path_points: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Smoothed unit tangent at each taproot path point.

    Gaussian-smoothed gradient — same formulation as
    ``phase2_main_volume._compute_tangent_vectors`` and
    ``phase4_inward_tracking._compute_tangents``.  Reimplemented locally
    so phase3_tip_seeds can compute perp-to-polyline distances without a
    cross-module dependency.
    """
    from scipy.ndimage import gaussian_filter1d
    tangents = np.gradient(path_points, axis=0)
    for ax in range(3):
        tangents[:, ax] = gaussian_filter1d(tangents[:, ax], sigma=sigma)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tangents / norms
