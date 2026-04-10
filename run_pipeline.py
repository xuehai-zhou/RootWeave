#!/usr/bin/env python3
"""Command-line entry point for the RootWeave root extraction pipeline.

Input files are read from ./samples/.
Endpoints are saved/loaded from ./endpoints/.
Results are saved to ./skl_res/.

Usage examples:

    # Run a sample (auto-resolves paths)
    python run_pipeline.py B2T3G16S3

    # Explicit full path
    python run_pipeline.py samples/B2T3G16S3.nii.gz

    # No visualization (headless)
    python run_pipeline.py B2T3G16S3 --no-viz

    # Phase 1 only
    python run_pipeline.py B2T3G16S3 --phase1-only
"""

import argparse
from pathlib import Path

from rootweave import PipelineConfig, run, run_phase1_only

SAMPLES_DIR = Path("samples")
RESULTS_DIR = Path("skl_res")


def _resolve_input(name: str) -> str:
    """Resolve a sample name to a full input path.

    Accepts:
      - Full path: samples/B2T3G16S3.nii.gz
      - Just the name: B2T3G16S3 (auto-finds .nii.gz or .ply in ./samples/)
      - Name with extension: B2T3G16S3.nii.gz (prepends ./samples/)
    """
    p = Path(name)
    if p.exists():
        return str(p)

    # Try in samples dir
    if (SAMPLES_DIR / name).exists():
        return str(SAMPLES_DIR / name)

    # Try adding common extensions
    for ext in [".nii.gz", ".ply", ".xyz"]:
        candidate = SAMPLES_DIR / (name + ext)
        if candidate.exists():
            return str(candidate)

    # Nothing found — return as-is (will fail with a clear error in io.load)
    return name


def _resolve_output(input_path: str) -> str:
    """Generate output path in ./skl_res/ from the input path."""
    stem = Path(input_path).name
    while Path(stem).suffix:
        stem = Path(stem).stem
    RESULTS_DIR.mkdir(exist_ok=True)
    return str(RESULTS_DIR / f"{stem}.pkl")


def main():
    parser = argparse.ArgumentParser(
        description="RootWeave: 3D plant root structure extraction"
    )
    parser.add_argument("input", help="Sample name or path to .ply/.nii.gz file")
    parser.add_argument("-o", "--output", help="Output pickle path (default: ./skl_res/<name>.pkl)")
    parser.add_argument("--endpoints", help="JSON file with saved endpoint indices")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--phase1-only", action="store_true", help="Run only Phase 1")

    # Config overrides (defaults come from PipelineConfig, not hardcoded here)
    _d = PipelineConfig()
    parser.add_argument("--voxel-size", type=float, default=_d.voxel_size)
    parser.add_argument("--nifti-threshold", type=float, default=_d.nifti_threshold)
    parser.add_argument("--knn-k", type=int, default=_d.knn_k)
    parser.add_argument("--main-branch-factor", type=float, default=_d.main_branch_factor)
    parser.add_argument("--shell-inner", type=float, default=_d.shell_inner_factor)
    parser.add_argument("--shell-outer", type=float, default=_d.shell_outer_factor)
    parser.add_argument("--neighborhood-k", type=float, default=_d.neighborhood_k)
    parser.add_argument("--step-k", type=float, default=_d.step_k)

    args = parser.parse_args()

    input_path = _resolve_input(args.input)
    output_path = args.output or _resolve_output(input_path)

    config = PipelineConfig(
        voxel_size=args.voxel_size,
        nifti_threshold=args.nifti_threshold,
        knn_k=args.knn_k,
        main_branch_factor=args.main_branch_factor,
        shell_inner_factor=args.shell_inner,
        shell_outer_factor=args.shell_outer,
        neighborhood_k=args.neighborhood_k,
        step_k=args.step_k,
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
