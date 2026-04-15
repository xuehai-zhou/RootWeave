"""Pipeline orchestration for rootweave.

Phases
------
1. Load point cloud (io.load_point_cloud).
2. Taproot centerline (phase1_main_path.extract_main_path).
3. Main root volume + R(t) profile (phase2_main_volume.extract_main_volume).
4. Shrinking-shell tip seed detection (phase3_tip_seeds.detect_tip_seeds).
5. Inward-only tracking from each seed
   (phase4_inward_tracking.grow_inward_from_seeds).
6. Post-process (length filter, deduplication).
7. Classify + merge: each "lateral"-stopped branch is classified at
   its attachment point as REDUNDANT / SPLIT-EXTENSION /
   SIBLING-SECONDARY / TERTIARY, and split artifacts are merged into
   their parents (phase5_classify_merge.classify_and_merge).
8. Save in original physical coordinates (io.save_results).
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from . import io, viz
from .config import PipelineConfig
from .phase1_main_path import extract_main_path, MainPathResult
from .phase2_main_volume import extract_main_volume
from .phase3_tip_seeds import detect_tip_seeds
from .phase4_inward_tracking import grow_inward_from_seeds, BranchPath
from .phase5_classify_merge import classify_and_merge, ClassifiedBranch


def run(
    input_path: str,
    config: Optional[PipelineConfig] = None,
    output_path: Optional[str] = None,
    endpoints_path: Optional[str] = None,
    visualize: bool = True,
) -> dict:
    """Execute the full pipeline."""
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

    # --- Pick endpoints (cached at ./endpoints/<stem>_endpoints.json) ---
    sample_name = Path(input_path).name
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
        tree_tmp = KDTree(points)
        _, si = tree_tmp.query(start_pt)
        _, ei = tree_tmp.query(end_pt)
        io.save_endpoints(ep_save, (int(si), int(ei)))
        print(f"Saved endpoints to {ep_save}")

    # --- Phase 1: Taproot path ---
    t1 = time.time()
    main_path: MainPathResult = extract_main_path(pcd, start_pt, end_pt, config)
    timings["phase1_main_path"] = time.time() - t1
    if visualize:
        viz.show_path_on_cloud(pcd, main_path.path_points)

    # --- Phase 2: Main root volume + R(t) profile ---
    t2 = time.time()
    volume = extract_main_volume(
        points, main_path.path_points, main_path.avg_distance, config,
    )
    timings["phase2_main_volume"] = time.time() - t2
    if visualize:
        viz.show_main_root(pcd, volume.main_root_points)

    # --- Phase 3: Shrinking-shell tip seed detection ---
    t3 = time.time()
    sweep = detect_tip_seeds(
        points=points,
        main_root_mask=volume.main_root_mask,
        path_points=main_path.path_points,
        radii=volume.radii,
        avg_distance=main_path.avg_distance,
        config=config,
    )
    timings["phase3_tip_seeds"] = time.time() - t3

    if visualize and sweep.seeds:
        _visualize_seeds(points, main_path.path_points, volume.main_root_points, sweep)

    # --- Phase 4: Inward-only tracking ---
    t4 = time.time()
    branch_paths = grow_inward_from_seeds(
        all_points=points,
        seeds=sweep.seeds,
        avg_distance=main_path.avg_distance,
        path_points=main_path.path_points,
        radii=volume.radii,
        tangents=volume.tangents,
        config=config,
    )

    # Post-growth filter: discard too-short paths
    before = len(branch_paths)
    branch_paths = [
        bp for bp in branch_paths if len(bp.path) >= config.min_branch_path_length
    ]
    if before > len(branch_paths):
        print(
            f"Length filter: {before} -> {len(branch_paths)} "
            f"(removed {before - len(branch_paths)} short paths)"
        )

    # Deduplicate overlapping paths (re-tracked regions)
    before_dd = len(branch_paths)
    branch_paths = _deduplicate_paths(branch_paths, main_path.avg_distance, config)
    if before_dd > len(branch_paths):
        print(
            f"Dedup: {before_dd} -> {len(branch_paths)} "
            f"(removed {before_dd - len(branch_paths)} overlapping paths)"
        )

    timings["phase4_inward_tracking"] = time.time() - t4

    # --- Phase 5: Classify each lateral-stopped branch at its attachment
    # point and merge split artifacts into their parents.  Produces a
    # flat list of ClassifiedBranch with explicit order + parent_label
    # (the root-order tree: secondaries, tertiaries, …).
    t5 = time.time()
    classified_branches = classify_and_merge(branch_paths, config)
    timings["phase5_classify_merge"] = time.time() - t5

    # In-pipeline preview: render the MERGED paths (one per root, already
    # classified into orders internally but drawn here in distinct colors
    # per branch WITHOUT separating by order).  Surrounding-point indices
    # aren't tracked through classify_and_merge, so pass empty lists.
    if visualize and classified_branches:
        drawable = [cb for cb in classified_branches if len(cb.path) >= 2]
        viz.show_branch_paths(
            points,
            [cb.path for cb in drawable],
            [[] for _ in drawable],
            taproot_path=main_path.path_points,
        )

    timings["total"] = time.time() - t0
    _print_timings(timings)

    # --- Save (in original physical coordinates) ---
    result = {
        "all_points": points,
        "main_path": main_path,
        "main_root_points": volume.main_root_points,
        "radii": volume.radii,
        "shell_sweep": sweep,
        "branch_paths": branch_paths,
        "classified_branches": classified_branches,
        "config": config,
        "norm_params": norm_params,
        "timings": timings,
        "version": "rootweave/1.0.0",
    }
    if output_path:
        io.save_results(
            output_path, points, volume.main_root_points, branch_paths,
            norm_params, config,
            taproot_path=main_path.path_points,
            classified_branches=classified_branches,
        )

    return result


def run_phase1_only(
    input_path: str,
    config: Optional[PipelineConfig] = None,
    endpoints_path: Optional[str] = None,
    visualize: bool = True,
) -> MainPathResult:
    """Phase 1 only — useful for debugging the taproot extraction."""
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


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _deduplicate_paths(branch_paths, avg_distance, config):
    """Remove duplicate paths (near-identical overlapping tracks).

    Two paths overlap if a large fraction of one path's nodes are close
    to any node of the other path. When overlap is detected, keep the
    longer path.
    """
    if len(branch_paths) <= 1:
        return branch_paths

    proximity = avg_distance * config.overlap_proximity_k
    overlap_threshold = config.overlap_threshold
    n = len(branch_paths)

    indexed = sorted(
        enumerate(branch_paths), key=lambda x: len(x[1].path), reverse=True
    )
    keep = [True] * n

    for i in range(len(indexed)):
        oi = indexed[i][0]
        if not keep[oi]:
            continue
        pi = np.array(indexed[i][1].path)
        tree_i = KDTree(pi)

        for j in range(i + 1, len(indexed)):
            oj = indexed[j][0]
            if not keep[oj]:
                continue
            pj = np.array(indexed[j][1].path)
            dists, _ = tree_i.query(pj)
            frac = float(np.mean(dists < proximity))
            if frac >= overlap_threshold:
                keep[oj] = False

    return [bp for bp, k in zip(branch_paths, keep) if k]


def _print_timings(timings: dict) -> None:
    print("\n--- Timings ---")
    for k, v in timings.items():
        print(f"  {k}: {v:.2f}s")


def _visualize_seeds(points, path_points, main_root_points, sweep):
    """Quick visualization: show each seed as a blue sphere on top of the cloud."""
    try:
        import open3d as o3d
    except ImportError:
        return

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color([0.7, 0.7, 0.7])

    main = o3d.geometry.PointCloud()
    main.points = o3d.utility.Vector3dVector(main_root_points)
    main.paint_uniform_color([1.0, 0.2, 0.2])

    # Taproot path as a line set
    n = len(path_points)
    path_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(path_points),
        lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(n - 1)]),
    )
    path_ls.colors = o3d.utility.Vector3dVector([[0.1, 0.1, 0.1]] * (n - 1))

    geoms = [cloud, main, path_ls]
    for seed in sweep.seeds:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        sphere.translate(seed.seed_point - np.asarray(sphere.get_center()))
        sphere.paint_uniform_color([0.2, 0.6, 1.0])
        geoms.append(sphere)

        # Arrow for initial inward direction
        arrow_end = seed.seed_point + seed.initial_direction * 0.05
        arrow_ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([seed.seed_point, arrow_end]),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        arrow_ls.colors = o3d.utility.Vector3dVector([[0.1, 0.4, 1.0]])
        geoms.append(arrow_ls)

    o3d.visualization.draw_geometries(
        geoms, window_name=f"Tip seeds: {len(sweep.seeds)}"
    )
