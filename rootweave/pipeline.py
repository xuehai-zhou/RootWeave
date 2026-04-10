"""Main pipeline orchestration: ties all phases together."""

import time
from typing import Optional

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from .config import PipelineConfig
from . import io
from . import viz
from .phase1_main_path import extract_main_path, MainPathResult
from .phase2_analysis import analyze_cross_sections
from .phase4_branch_growth import grow_branches, BranchPath


def run(
    input_path: str,
    config: Optional[PipelineConfig] = None,
    output_path: Optional[str] = None,
    endpoints_path: Optional[str] = None,
    visualize: bool = True,
) -> dict:
    """Execute the full root extraction pipeline.

    Parameters
    ----------
    input_path : path to .ply or .nii.gz file
    config : pipeline parameters (uses defaults if None)
    output_path : where to save results (pickle); skips save if None
    endpoints_path : JSON file with saved endpoint indices; if None, prompts interactive picking
    visualize : whether to show Open3D windows at each stage

    Returns
    -------
    dict with keys: all_points, main_path, main_root_points, branch_origins,
    branch_paths, config, timings.
    """
    config = config or PipelineConfig()
    timings = {}
    t0 = time.time()

    # --- Load ---
    pcd, points, norm_params = io.load_point_cloud(
        input_path,
        voxel_size=config.voxel_size,
        nifti_threshold=config.nifti_threshold,
    )
    if visualize:
        viz.show_point_cloud(pcd, title=input_path)

    # --- Pick endpoints (auto-save to ./endpoints/) ---
    from pathlib import Path
    sample_name = Path(input_path).name
    # Strip all suffixes (handles .nii.gz)
    stem = sample_name
    while Path(stem).suffix:
        stem = Path(stem).stem
    ep_dir = Path("endpoints")
    ep_dir.mkdir(exist_ok=True)
    default_ep_path = str(ep_dir / f"{stem}_endpoints.json")

    if endpoints_path and Path(endpoints_path).exists():
        start_pt, end_pt = io.load_endpoints(endpoints_path, points)
        print(f"Loaded endpoints from {endpoints_path}")
    elif Path(default_ep_path).exists():
        start_pt, end_pt = io.load_endpoints(default_ep_path, points)
        print(f"Loaded endpoints from {default_ep_path}")
    else:
        start_pt, end_pt = io.pick_endpoints(pcd)
        ep_save = endpoints_path or default_ep_path
        picked_tree = KDTree(points)
        _, si = picked_tree.query(start_pt)
        _, ei = picked_tree.query(end_pt)
        io.save_endpoints(ep_save, (int(si), int(ei)))
        print(f"Saved endpoints to {ep_save}")

    # --- Phase 1: Main root path ---
    t1 = time.time()
    main_path = extract_main_path(pcd, start_pt, end_pt, config)
    timings["phase1_main_path"] = time.time() - t1

    if visualize:
        viz.show_path_on_cloud(pcd, main_path.path_points)

    # --- Phase 2+3: Cross-section analysis (main volume + branch detection) ---
    t2 = time.time()
    analysis = analyze_cross_sections(
        points, main_path.path_points, main_path.avg_distance, config
    )
    main_root_points = analysis.main_root_points
    branch_origins = analysis.branch_origins
    timings["phase2_3_analysis"] = time.time() - t2

    if visualize:
        viz.show_main_root(pcd, main_root_points)

    if visualize:
        viz.show_shell_analysis(
            pcd,
            main_path.path_points,
            analysis.radii,
            shell_points=analysis.shell_points,
            shell_cluster_labels=analysis.shell_cluster_labels,
            branch_origins=branch_origins if branch_origins else None,
            shell_inner_factor=config.shell_inner_factor,
            shell_outer_factor=config.shell_outer_factor,
        )

    if visualize and branch_origins:
        viz.show_branches(
            pcd,
            main_root_points,
            {k: b.points for k, b in branch_origins.items()},
            {k: b.direction for k, b in branch_origins.items()},
        )

    # --- Phase 4: Branch growth ---
    t4 = time.time()
    branch_paths = grow_branches(
        points, main_root_points, branch_origins,
        main_path.avg_distance,
        main_path.path_points,
        analysis.radii,
        config,
    )

    # Post-growth filter: discard short paths
    before = len(branch_paths)
    branch_paths = [
        bp for bp in branch_paths
        if len(bp.path) >= config.min_branch_path_length
    ]
    if before > len(branch_paths):
        print(
            f"Post-growth filter: {before} -> {len(branch_paths)} "
            f"(removed {before - len(branch_paths)} short paths)"
        )

    # Deduplicate overlapping paths (re-entering roots produce near-identical paths)
    before_dedup = len(branch_paths)
    branch_paths = _deduplicate_paths(branch_paths, main_path.avg_distance, config)
    if before_dedup > len(branch_paths):
        print(
            f"Deduplication: {before_dedup} -> {len(branch_paths)} "
            f"(removed {before_dedup - len(branch_paths)} overlapping paths)"
        )

    timings["phase4_branch_growth"] = time.time() - t4

    if visualize and branch_paths:
        viz.show_branch_paths(
            points,
            [b.path for b in branch_paths],
            [b.surrounding_indices for b in branch_paths],
            taproot_path=main_path.path_points,
        )

    timings["total"] = time.time() - t0
    _print_timings(timings)

    # --- Save (in original physical coordinates) ---
    result = {
        "all_points": points,
        "main_path": main_path,
        "main_root_points": main_root_points,
        "branch_origins": branch_origins,
        "branch_paths": branch_paths,
        "config": config,
        "norm_params": norm_params,
        "timings": timings,
    }
    if output_path:
        io.save_results(
            output_path, points, main_root_points, branch_paths,
            norm_params, config,
            taproot_path=main_path.path_points,
        )

    return result


def run_phase1_only(
    input_path: str,
    config: Optional[PipelineConfig] = None,
    endpoints_path: Optional[str] = None,
    visualize: bool = True,
) -> MainPathResult:
    """Run only loading + Phase 1 (useful for debugging the main path)."""
    config = config or PipelineConfig()

    pcd, points, norm_params = io.load_point_cloud(
        input_path,
        voxel_size=config.voxel_size,
        nifti_threshold=config.nifti_threshold,
    )
    if visualize:
        viz.show_point_cloud(pcd, title=input_path)

    if endpoints_path:
        start_pt, end_pt = io.load_endpoints(endpoints_path, points)
    else:
        start_pt, end_pt = io.pick_endpoints(pcd)

    main_path = extract_main_path(pcd, start_pt, end_pt, config)

    if visualize:
        viz.show_path_on_cloud(pcd, main_path.path_points)

    return main_path


def _deduplicate_paths(branch_paths, avg_distance, config):
    """Remove duplicate paths caused by re-entering roots.

    Two paths overlap if a large fraction of one path's nodes are close
    to any node of the other path. When overlap is detected, keep the
    longer path (more complete coverage).
    """
    if len(branch_paths) <= 1:
        return branch_paths

    proximity = avg_distance * config.overlap_proximity_k
    overlap_threshold = config.overlap_threshold
    n = len(branch_paths)

    # Sort by path length descending — longer paths get priority
    indexed = sorted(enumerate(branch_paths), key=lambda x: len(x[1].path), reverse=True)

    keep = [True] * n  # indexed by original position

    for i in range(len(indexed)):
        orig_i = indexed[i][0]
        if not keep[orig_i]:
            continue
        path_i = np.array(indexed[i][1].path)
        tree_i = KDTree(path_i)

        for j in range(i + 1, len(indexed)):
            orig_j = indexed[j][0]
            if not keep[orig_j]:
                continue
            path_j = np.array(indexed[j][1].path)

            # Check: what fraction of path_j's nodes are close to path_i?
            dists, _ = tree_i.query(path_j)
            fraction_close = np.mean(dists < proximity)

            if fraction_close >= overlap_threshold:
                # path_j is mostly a subset of path_i — discard path_j
                keep[orig_j] = False

    result = [bp for bp, k in zip(branch_paths, keep) if k]
    return result


def _print_timings(timings: dict) -> None:
    print("\n--- Timings ---")
    for key, val in timings.items():
        print(f"  {key}: {val:.2f}s")
