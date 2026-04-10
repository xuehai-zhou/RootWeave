"""Pipeline configuration as a single dataclass."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    # --- Loading & preprocessing ---
    voxel_size: float = 0.003
    nifti_threshold: float = 0.0  # voxels above this value become points

    # --- Phase 1: taproot path finding ---
    knn_k: int = 10
    centering_presmooth_sigma: float = 5.0  # heavy smoothing on raw Dijkstra path
    centering_search_k: float = 20.0        # cross-section search = avg_distance * this
    centering_tolerance_k: float = 3.0      # plane thickness = avg_distance * this
    centering_smooth_sigma: float = 2.0     # light smoothing between iterations
    centering_iterations: int = 3           # number of center-recompute-tangent cycles

    # --- Phase 2: main root volume ---
    main_branch_factor: float = 0.85
    tangent_sigma: float = 5.0
    min_cluster_size: int = 5
    radius_smoothing_window: int = 51
    radius_smoothing_polyorder: int = 3
    outlier_std_threshold: float = 3.0

    # --- Phase 3: branch detection (cylinder-shell) ---
    shell_inner_factor: float = 6.0    # shell inner boundary = R(t) * this
    shell_outer_factor: float = 8.0    # shell outer boundary = R(t) * this
    shell_cluster_eps_k: float = 5.0   # DBSCAN eps = avg_distance * this
    shell_min_cluster: int = 2         # min points for a shell cluster (2 = catch thin roots)
    min_seed_points: int = 2           # minimum total seed points per cluster
    min_branch_path_length: int = 8    # post-growth filter: discard paths shorter than this

    # --- Phase 4: adaptive PCA + graph-assisted growth ---
    neighborhood_k: float = 15.0     # base neighborhood = avg_distance * this
    step_k: float = 5.0             # base step = avg_distance * this
    snap_k: float = 5.0             # base snap = avg_distance * this
    arrival_factor: float = 2.0      # inward stops within R(t) * this (with alignment check)
    hard_arrival_factor: float = 1.2  # inward stops unconditionally within R(t) * this
    taproot_exclusion_factor: float = 2.0  # outward tracking ignores points within R(t) * this
    arrival_alignment_threshold: float = 0.5  # |cos| between local PC1 and taproot tangent
    smoothing_factor: float = 0.7    # base PCA smoothing weight
    pca_min_points: int = 3          # minimum neighbors for PCA
    min_step_cosine: float = 0.3     # hard floor on direction change
    max_growth_steps: int = 200      # max iterations per direction
    # Inward tracking biases
    inward_attraction_weight: float = 0.00  # strength of attraction toward taproot path
    antigravity_weight: float = 0.03        # small upward bias (Z-up)
    # Outward tracking biases
    outward_repulsion_weight: float = 0.0  # strength of repulsion from taproot path
    gravity_weight: float = 0.0            # small downward bias (negative Z)
    # Direction probing (when PCA gets stuck)
    max_bend_recoveries: int = 5     # max probe attempts per growth direction

    # --- Post-growth: deduplication ---
    overlap_proximity_k: float = 8.0   # two nodes match if within avg_distance * this
    overlap_threshold: float = 0.5     # fraction of shorter path's nodes that must match
