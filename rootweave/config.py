"""Pipeline configuration for rootweave.

Single source of truth for every tunable parameter the five phases use.
Edit the defaults here, or override per-call via the
:class:`PipelineConfig` constructor or the CLI (``run_pipeline.py``).
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # --- Loading & preprocessing ---
    voxel_size: float = 0.003
    nifti_threshold: float = 0.0

    # --- Phase 1: taproot centerline ---
    knn_k: int = 10
    centering_presmooth_sigma: float = 5.0
    centering_search_k: float = 20.0
    centering_tolerance_k: float = 3.0
    centering_smooth_sigma: float = 2.0
    centering_iterations: int = 3
    tangent_sigma: float = 5.0

    # --- Phase 2: main root volume ---
    main_branch_factor: float = 0.85
    min_cluster_size: int = 5
    radius_smoothing_window: int = 51
    radius_smoothing_polyorder: int = 3
    outlier_std_threshold: float = 3.0

    # --- Phase 3: shrinking-shell tip seed detection ---
    # The "shell" is a tube around the taproot polyline (not around a single
    # vertical axis).  For each cloud point we compute:
    #     perp_tap       = perpendicular distance to the taproot polyline
    #     perp_effective = perp_tap - taproot_clearance_k * R(t_nearest)
    # The sweep shrinks perp_effective from max(perp_effective) down to
    # shell_stop_effective_k * avg_distance.  At perp_effective = 0 the tube
    # wall is exactly (clearance_k * R(t)) away from the taproot surface —
    # this is the cushion that absorbs error in the taproot-radius estimate.
    # Laterals whose tips sit inside this cushion are intentionally skipped
    # (they're too short to care about).
    taproot_clearance_k: float = 2.0     # perp_eff = perp_tap - this * R(t)
    shell_start_margin: float = 1.05     # r_start = max(perp_eff) * this
    shell_stop_effective_k: float = 0.0  # r_stop = avg_distance * this (in effective units)
    shell_step_k: float = 2.0            # dr = avg_distance * this
    shell_thickness_k: float = 12.0      # shell thickness = avg_distance * this
    shell_z_pad_k: float = 2.0           # Z extent padding = avg_distance * this

    shell_cluster_eps_k: float = 4.0     # DBSCAN eps = avg_distance * this
    shell_min_cluster: int = 3           # min points per shell cluster
    seed_dedupe_k: float = 10.0          # seeds merged if within avg_distance * this

    # Graph-based cluster classification.
    # A KNN graph is built on non-main-root points in phase 3.  For each
    # shell cluster, bounded BFS decides whether it is:
    #   - connected to a previously-claimed region (continuation OR terminus)
    #   - has unclaimed outward extension (skip; defer to an outer shell)
    #   - a graph terminus (end of a root; seed it)
    graph_connection_depth: int = 3       # BFS hops for connection-to-claim check
    graph_outward_depth: int = 5          # BFS hops for outward-extension check
    graph_terminus_depth: int = 5         # BFS hops for terminus check
    graph_terminus_max_reached: int = 5   # terminus iff unclaimed BFS reaches ≤ this many new nodes
    tip_margin_k: float = 1.5             # outward-extension threshold = avg_distance * this

    # Initial direction at each seed is a weighted blend of three cues:
    #   - PCA of the local neighborhood (captures the root's local axis)
    #   - toward the nearest taproot path point (breaks PCA sign ambiguity)
    #   - anti-gravity (+Z; disambiguates bent roots where the tip points
    #     downward and toward-taproot is weak, e.g. roots parallel to the
    #     taproot).  When PCA succeeds, its sign is flipped so PC1 agrees
    #     with the taproot + anti-gravity preference, then blended.  When
    #     PCA fails, only taproot + anti-gravity are used.
    seed_init_neighborhood_k: float = 15.0
    init_dir_pca_weight: float = 0.60
    init_dir_taproot_weight: float = 0.20
    init_dir_antigravity_weight: float = 0.20

    # --- Phase 4: inward-only tracker ---
    neighborhood_k: float = 15.0
    step_k: float = 5.0
    snap_k: float = 5.0
    arrival_factor: float = 2.0
    hard_arrival_factor: float = 1.2
    arrival_alignment_threshold: float = 0.5
    smoothing_factor: float = 0.7
    pca_min_points: int = 3
    min_step_cosine: float = 0.1
    max_growth_steps: int = 300
    max_bend_recoveries: int = 5

    # Inward directional biases. The primary attractor is the XY radial
    # toward the vertical axis (not gravity), because some laterals grow
    # nearly horizontally.
    inward_radial_weight: float = 0.00   # pull toward axis in XY
    inward_attraction_weight: float = 0.00  # pull toward nearest taproot point
    antigravity_weight: float = 0.00     # small upward nudge

    # Arrival at another already-tracked lateral
    other_path_arrival_k: float = 2.5    # arrived if within avg_distance * this
    other_path_min_steps: int = 5        # don't check until we have this many steps

    # --- Post-processing ---
    min_branch_path_length: int = 6
    overlap_proximity_k: float = 8.0
    overlap_threshold: float = 0.5

    # --- Phase 5: classify + merge.  For each branch that stopped at
    # another lateral we analyze the attachment point to decide whether
    # it's:
    #   * REDUNDANT — tiny over-seeding artifact; drop.
    #   * SPLIT-EXTENSION — child extends past parent's tip; merge the
    #     two paths into a single longer branch and absorb the parent.
    #   * SIBLING-SECONDARY — attached near parent's taproot end; the
    #     child is really a sibling secondary, extend to the taproot.
    #   * TERTIARY — mid-shaft branching; keep as a tertiary of parent.
    split_min_child_length: int = 5     # paths with fewer nodes are dropped as redundant
    split_tip_zone_ratio: float = 0.10  # attach_idx/len(parent) < this → tip-extension split
    split_base_zone_ratio: float = 0.90 # attach_idx/len(parent) > this → sibling secondary
