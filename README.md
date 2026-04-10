# RootWeave

**3D root system architecture segmentation, skeletonization, and trait computation tool.**

RootWeave takes a 3D point cloud or volumetric image (`.ply`, `.nii.gz`) of a plant root system and automatically extracts the skeleton: the primary root (taproot) centerline and all detectable lateral root paths. From these skeletons, it computes a comprehensive set of root system architecture (RSA) traits including root length, branching angles, volume, and surface area.

## Method overview

RootWeave processes a root system in four phases:

1. **Primary root path finding** -- The user selects two endpoints (root crown and tip). A density-weighted shortest path on a KNN graph traces the taproot centerline, followed by iterative cross-sectional centering to place the path at the geometric center of the root.

2. **Main root volume extraction + branch detection** -- Cross-sectional planes sweep along the taproot to measure the local radius profile. A hollow cylinder shell is constructed around the taproot at a configurable distance. Point cloud clusters within this shell identify where lateral roots emerge.

3. **Bidirectional lateral root tracing** -- From each detected branch origin, PCA-based tracking with adaptive step size grows the root path in both directions: inward (toward the taproot junction) and outward (toward the root tip). Cross-sectional cluster filtering prevents jumping between adjacent roots. Direction probing and graph-guided lookahead handle sharp bends. A neutral short exploration determines inward vs outward directions before committing to direction-specific search spaces.

4. **Trait computation** -- Each root path is modeled as a sequence of frustum (truncated cone) segments for volume and surface area estimation. PCA of each path gives root orientation for angle measurements relative to the taproot and the vertical axis.

## Project structure

```
rootweave/                    Core library
  config.py                 All pipeline parameters (single source of truth)
  io.py                     Data loading (.ply, .nii.gz), saving, endpoints
  graph.py                  KNN graph construction, Dijkstra path finding
  phase1_main_path.py       Taproot centerline extraction
  phase2_analysis.py        Main root volume + cylinder-shell branch detection
  phase3_branch_detection.py  BranchOrigin data structure
  phase4_branch_growth.py   Bidirectional PCA tracking with graph assist
  pipeline.py               Full pipeline orchestration
  viz.py                    Open3D visualization helpers

run_pipeline.py             CLI entry point
compute_traits.py           Trait computation from skeleton results
interactive_viewer.py       3D interactive trait inspection tool
visualize_result.py         Static result visualization

samples/                    Input data (.nii.gz or .ply files)
endpoints/                  Saved taproot endpoint selections (auto-generated)
skl_res/                    Skeleton extraction results (.pkl)
traits/                     Computed traits (.json and .txt)
```

## Setup

### Requirements

- Python 3.9+
- The following packages:

```
numpy
scipy
open3d
networkx
scikit-learn
hdbscan
nibabel
matplotlib
tqdm
```

### Installation

```bash
git clone https://github.com/your-username/RootWeave.git
cd RootWeave

# Create a conda environment (recommended)
conda create -n rootweave python=3.10
conda activate rootweave

# Install dependencies
pip install numpy scipy open3d networkx scikit-learn hdbscan nibabel matplotlib tqdm
```

No additional environment variables are needed. All configuration is managed through `rootweave/config.py` and CLI arguments.

### Directory setup

```bash
mkdir -p samples endpoints skl_res traits
```

Place your input files (`.nii.gz` or `.ply`) in `./samples/`.

## Usage

### Run the full pipeline

```bash
# By sample name (auto-finds in ./samples/)
python run_pipeline.py B2T3G16S3

# With explicit path
python run_pipeline.py samples/B2T3G16S3.nii.gz

# Without visualization (headless mode)
python run_pipeline.py B2T3G16S3 --no-viz
```

On the first run for a sample, an Open3D window opens for you to select the taproot endpoints (root crown and root tip) using `Shift + Left Click`. These are saved to `./endpoints/` and reused automatically on subsequent runs.

Results are saved to `./skl_res/<sample_name>.pkl`.

### Compute traits

```bash
python compute_traits.py B2T3G16S3
```

Outputs are saved to `./traits/` in both JSON and plain text table format.

### Visualize results

```bash
# Static skeleton view
python visualize_result.py B2T3G16S3

# Interactive viewer with tube mesh and per-root trait display
python interactive_viewer.py B2T3G16S3
```

Interactive viewer controls:
- **N / B** : Next / Previous root
- **T** : Toggle tube mesh
- **C** : Toggle point cloud
- **D** : Deselect
- **Q** : Quit

### Configuration

All parameters are centralized in `rootweave/config.py` as a `PipelineConfig` dataclass. Key parameters can also be overridden via CLI:

```bash
python run_pipeline.py B2T3G16S3 --shell-inner 5.0 --shell-outer 7.0 --step-k 5.0
```

### NIfTI spacing note

For `.nii.gz` files, RootWeave overrides the affine with the correct voxel spacing `[0.39, 0.39, 0.2]` mm (hardcoded in `rootweave/io.py`). If your data has different spacing, update the `CORRECT_SPACING` constant in that file.

## Computed traits

| Trait | Description | Unit |
|---|---|---|
| Individual root length | Arc length along the skeleton path | mm |
| Relative root angle | Angle between lateral root PCA and taproot PCA | degrees |
| Absolute root angle | Angle between root PCA and the vertical (Z) axis | degrees |
| Individual root volume | Frustum model from path + local cross-section radii | mm^3 |
| Individual root surface area | Frustum lateral area + end caps | mm^2 |
| Total root length | Sum of taproot + all lateral root lengths | mm |
| Total root volume | Sum of all individual volumes | mm^3 |
| Total root surface area | Sum of all individual surface areas | mm^2 |
| Number of lateral roots | Count of detected and traced lateral roots | - |
| Root system depth | Vertical (Z) extent of the point cloud | mm |
| Root system width | Maximum horizontal (X or Y) extent | mm |

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

MIT License
