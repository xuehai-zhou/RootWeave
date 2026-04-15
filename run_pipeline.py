#!/usr/bin/env python3
"""Command-line entry point for the rootweave pipeline.

Path conventions:
  - Inputs read from ``./samples/``
  - Endpoints saved/loaded from ``./endpoints/``
  - Results saved to ``./skl_res/``

Pipeline overview (see ``rootweave.pipeline.run`` for details):

  1. Load point cloud (``.nii.gz`` / ``.ply`` / ``.xyz``)
  2. Taproot centerline (Dijkstra + iterative cross-sectional centering)
  3. R(t) radius profile + main-root volume mask
  4. Shrinking-tube tip seed detection with KNN-graph classification
  5. Inward-only tracking from each seed
  6. Classify + merge at each attachment point → ClassifiedBranch list
     (flat root-order tree: secondaries, tertiaries, …)

Usage examples:

    # Run a sample (auto-resolves paths)
    python run_pipeline.py B2T3G16S3

    # Explicit full path
    python run_pipeline.py samples/B2T3G16S3.nii.gz

    # No visualization (headless)
    python run_pipeline.py B2T3G16S3 --no-viz

    # Phase 1 only — taproot centerline for debugging
    python run_pipeline.py B2T3G16S3 --phase1-only
"""

import argparse
from pathlib import Path

from rootweave import PipelineConfig, run, run_phase1_only

SAMPLES_DIR = Path("samples")
RESULTS_DIR = Path("skl_res")


def _resolve_input(name: str) -> str:
    p = Path(name)
    if p.exists():
        return str(p)
    if (SAMPLES_DIR / name).exists():
        return str(SAMPLES_DIR / name)
    for ext in [".nii.gz", ".ply", ".xyz"]:
        candidate = SAMPLES_DIR / (name + ext)
        if candidate.exists():
            return str(candidate)
    return name


def _resolve_output(input_path: str) -> str:
    stem = Path(input_path).name
    while Path(stem).suffix:
        stem = Path(stem).stem
    RESULTS_DIR.mkdir(exist_ok=True)
    return str(RESULTS_DIR / f"{stem}.pkl")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "rootweave: inward-only root skeletonization with shrinking-shell "
            "tip seeding and KNN-graph classification."
        )
    )
    parser.add_argument("input", help="Sample name or path to .ply/.nii.gz file")
    parser.add_argument(
        "-o", "--output",
        help="Output pickle path (default: ./skl_res/<name>.pkl)",
    )
    parser.add_argument("--endpoints", help="JSON file with saved endpoint indices")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--phase1-only", action="store_true", help="Run only Phase 1")

    # Config overrides
    _d = PipelineConfig()
    parser.add_argument("--voxel-size", type=float, default=_d.voxel_size)
    parser.add_argument("--nifti-threshold", type=float, default=_d.nifti_threshold)
    parser.add_argument("--knn-k", type=int, default=_d.knn_k)
    parser.add_argument("--main-branch-factor", type=float, default=_d.main_branch_factor)
    parser.add_argument("--neighborhood-k", type=float, default=_d.neighborhood_k)
    parser.add_argument("--step-k", type=float, default=_d.step_k)

    # Shell sweep (phase 3)
    parser.add_argument("--shell-start-margin", type=float, default=_d.shell_start_margin)
    parser.add_argument("--shell-stop-effective-k", type=float, default=_d.shell_stop_effective_k,
                        help="r_stop = avg_distance * this (effective-distance units)")
    parser.add_argument("--taproot-clearance-k", type=float, default=_d.taproot_clearance_k,
                        help="perp_effective = perp_tap - this * R(t); safety cushion around the taproot")
    parser.add_argument("--shell-step-k", type=float, default=_d.shell_step_k)
    parser.add_argument("--shell-thickness-k", type=float, default=_d.shell_thickness_k)
    parser.add_argument("--shell-cluster-eps-k", type=float, default=_d.shell_cluster_eps_k)
    parser.add_argument("--shell-min-cluster", type=int, default=_d.shell_min_cluster)
    parser.add_argument("--tip-margin-k", type=float, default=_d.tip_margin_k)
    parser.add_argument("--seed-dedupe-k", type=float, default=_d.seed_dedupe_k)

    # Graph classification (phase 3)
    parser.add_argument(
        "--graph-connection-depth", type=int, default=_d.graph_connection_depth,
        help="BFS hops for the connection-to-claim check",
    )
    parser.add_argument(
        "--graph-outward-depth", type=int, default=_d.graph_outward_depth,
        help="BFS hops for the outward-extension check",
    )
    parser.add_argument(
        "--graph-terminus-depth", type=int, default=_d.graph_terminus_depth,
        help="BFS hops for the terminus check",
    )
    parser.add_argument(
        "--graph-terminus-max-reached", type=int, default=_d.graph_terminus_max_reached,
        help="Cluster is a terminus if unclaimed BFS reaches ≤ this many new nodes",
    )

    args = parser.parse_args()

    input_path = _resolve_input(args.input)
    output_path = args.output or _resolve_output(input_path)

    config = PipelineConfig(
        voxel_size=args.voxel_size,
        nifti_threshold=args.nifti_threshold,
        knn_k=args.knn_k,
        main_branch_factor=args.main_branch_factor,
        neighborhood_k=args.neighborhood_k,
        step_k=args.step_k,
        shell_start_margin=args.shell_start_margin,
        shell_stop_effective_k=args.shell_stop_effective_k,
        taproot_clearance_k=args.taproot_clearance_k,
        shell_step_k=args.shell_step_k,
        shell_thickness_k=args.shell_thickness_k,
        shell_cluster_eps_k=args.shell_cluster_eps_k,
        shell_min_cluster=args.shell_min_cluster,
        tip_margin_k=args.tip_margin_k,
        seed_dedupe_k=args.seed_dedupe_k,
        graph_connection_depth=args.graph_connection_depth,
        graph_outward_depth=args.graph_outward_depth,
        graph_terminus_depth=args.graph_terminus_depth,
        graph_terminus_max_reached=args.graph_terminus_max_reached,
    )

    if args.phase1_only:
        run_phase1_only(
            input_path,
            config=config,
            endpoints_path=args.endpoints,
            visualize=not args.no_viz,
        )
    else:
        run(
            input_path,
            config=config,
            output_path=output_path,
            endpoints_path=args.endpoints,
            visualize=not args.no_viz,
        )


if __name__ == "__main__":
    main()
