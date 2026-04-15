# rootweave

3D soybean root system skeletonization from point clouds — taproot centerline
extraction, lateral-root tracking, and automatic
secondary / tertiary classification in a single pass.

<!-- Add a rendered sample image here once one is ready, e.g.
     ![rootweave pipeline](docs/example.png) -->

## What it does

Given a 3D point cloud of a plant root system (`.nii.gz`, `.ply`, or
`.xyz`) and two user-picked endpoints on the taproot (root crown and
root tip), **rootweave** produces:

1. A smoothed **taproot centerline** + per-position radius profile `R(t)`.
2. A **binary mask** of the taproot volume inside the cloud.
3. A set of **lateral-root seed points** located at the outermost tips of
   each lateral, found by sweeping a shrinking tube around the taproot
   polyline with KNN-graph-assisted classification.
4. **Inward-tracked paths** from each seed to its parent structure.
5. A **classified root tree** — each root tagged with its order
   (secondary, tertiary, …) and an explicit `parent_label` link so
   tertiaries know which secondary they branch from.

## Install

```bash
git clone https://github.com/<user>/rootweave.git
cd rootweave
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ is required. The dependencies (`numpy`, `scipy`,
`scikit-learn`, `networkx`, `hdbscan`, `nibabel`, `open3d`, `matplotlib`)
are all on PyPI.

## Quick start

```bash
# 1. Put a .nii.gz / .ply point cloud in ./samples/
cp /path/to/my_root.nii.gz samples/

# 2. Run the full pipeline
python run_pipeline.py my_root
# On the first run, an Open3D window opens for you to pick the taproot
# crown and tip (Shift + left-click).  Endpoints are saved to
# ./endpoints/my_root_endpoints.json and reused next time.

# 3. Visualize the result
python visualize_result.py skl_res/my_root.pkl
```

The pipeline writes a pickle at `skl_res/<name>.pkl` whose
`classified_branches` field is the complete root-order tree.

## Visualization modes

`visualize_result.py` supports four modes, selectable with `--mode`:

| Mode | What's shown |
|---|---|
| `classified` (default) | Taproot + all secondaries (blue) + all tertiaries (green) with black links from tertiaries to their parents |
| `secondary` | Focus on secondaries only, recolored in distinct categorical hues |
| `tertiary`  | Focus on tertiaries only, recolored in distinct categorical hues |
| `raw`       | Pre-classification view (every branch colored by label) |

Try:
```bash
python visualize_result.py skl_res/my_root.pkl --mode secondary
python visualize_result.py skl_res/my_root.pkl --mode tertiary
python visualize_result.py skl_res/my_root.pkl --no-cloud          # hide the full point cloud
python visualize_result.py skl_res/my_root.pkl --point-size 2.0    # enlarge the cloud dots
```

## Pipeline overview (five phases)

```
Phase 1   phase1_main_path       — Dijkstra + iterative cross-sectional
                                   centering → taproot centerline
Phase 2   phase2_main_volume     — R(t) radius profile + main-root volume mask
Phase 3   phase3_tip_seeds       — shrinking-tube sweep with KNN-graph
                                   classification → one seed per lateral tip
Phase 4   phase4_inward_tracking — PCA step, probe, graph-lookahead recovery;
                                   tracks each seed inward to its parent
Phase 5   phase5_classify_merge  — per-attachment analysis:
                                      REDUNDANT, SPLIT-EXTENSION,
                                      SIBLING-SECONDARY, or TERTIARY
```

Every tunable parameter lives in [`rootweave/config.py`](rootweave/config.py)
as a single `PipelineConfig` dataclass — the source comments document
each field.

## Programmatic API

```python
from rootweave import PipelineConfig, run

config = PipelineConfig()
# Override any parameter:
# config.taproot_clearance_k = 2.5
# config.graph_terminus_max_reached = 7

result = run(
    "samples/my_root.nii.gz",
    config=config,
    output_path="skl_res/my_root.pkl",
    visualize=False,
)

# Inspect the root tree
for cb in result["classified_branches"]:
    parent = "taproot" if cb.parent_label is None else f"#{cb.parent_label}"
    print(f"#{cb.label}  order={cb.order}  parent={parent}  "
          f"classification={cb.classification}  "
          f"n_nodes={len(cb.path)}")
```

## Output format

Each entry in `result["classified_branches"]` (and in the saved pickle) is:

```python
{
    "label": int,                            # unique id
    "order": int,                            # 0=unknown, 2=secondary, 3=tertiary, ...
    "classification": str,                   # taproot-direct | split-extension
                                             # | sibling-secondary | tertiary | unknown
    "parent_label": Optional[int],           # None = attached to taproot
    "path": List[ndarray],                   # 3D points, tip → taproot (secondaries)
                                             # or tip → attachment (tertiaries)
    "attachment_point": Optional[ndarray],
    "attachment_index_on_parent": Optional[int],
    "absorbed_labels": List[int],            # labels folded in via SPLIT-EXTENSION
}
```

Finding a tertiary's parent is a dict lookup:

```python
by_label = {cb["label"]: cb for cb in data["classified_branches"]}
for cb in data["classified_branches"]:
    if cb["order"] == 3:
        parent = by_label[cb["parent_label"]]
        # ...
```

## Trait extraction

After a run, compute per-root and system-wide traits from the pickle:

```bash
python compute_traits.py B2T3G16S3          # or skl_res/B2T3G16S3.pkl
```

Traits are printed to the console and also written to
`traits/<sample>.json` and `traits/<sample>.txt`. Per-root output is
grouped by order (secondaries first, then tertiaries, …) and includes
`length_mm`, `angle_to_taproot_deg`, `angle_to_parent_deg` (for
tertiaries), `absolute_angle_deg`, frustum-model `volume_mm3` +
`surface_area_mm2`, and `mean_radius_mm`.

## Interactive per-root viewer

```bash
python interactive_viewer.py B2T3G16S3
```

Step through each root (`N` / `B`) and see its tube mesh + traits one
at a time. Secondaries render in a blue family, tertiaries in a green
family; tertiary attach points show as small black spheres on their
parent's path.

## Citation

If you find RootWeave useful in your research, please consider citing the following paper:

> Zhou, X., Yang, T., Xu, R., Bucksch, A., Dutilleul, P., Torkamaneh, D., & Sun, S. (2025). 3D skeletonization and phenotyping for soybean root system architecture using a bio-inspired algorithm. *Computers and Electronics in Agriculture*, 239, 110890. https://doi.org/10.1016/j.compag.2025.110890

```bibtex
@article{ZHOU2025110890,
  title     = {3D skeletonization and phenotyping for soybean root system architecture using a bio-inspired algorithm},
  journal   = {Computers and Electronics in Agriculture},
  volume    = {239},
  pages     = {110890},
  year      = {2025},
  issn      = {0168-1699},
  doi       = {https://doi.org/10.1016/j.compag.2025.110890},
  author    = {Xuehai Zhou and Tianzi Yang and Rui Xu and Alexander Bucksch and Pierre Dutilleul and Davoud Torkamaneh and Shangpeng Sun},
  keywords  = {Plant phenotyping, 3D point cloud analysis, Adaptive path tracking, Root system segmentation, Individual root extraction},
}
```

## License

MIT — see [LICENSE](LICENSE).
