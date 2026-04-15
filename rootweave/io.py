"""Data loading (PLY, NIfTI), point picking, and result saving.

Normalization is applied on load (center + scale to [-1, 1]) so all
pipeline computation happens in a uniform coordinate space. The inverse
transform is stored and applied on save so results are output in the
original physical scale.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import open3d as o3d


@dataclass
class NormParams:
    """Stores the normalization transform so it can be inverted on output."""
    center: np.ndarray       # (3,) mean subtracted during normalization
    scale: float             # scalar multiplied during normalization

    def to_original(self, points: np.ndarray) -> np.ndarray:
        """Convert normalized points back to original coordinates."""
        return points / self.scale + self.center

    def to_normalized(self, points: np.ndarray) -> np.ndarray:
        """Convert original points to normalized coordinates."""
        return (points - self.center) * self.scale


def load_point_cloud(
    file_path: str,
    voxel_size: float = 0.003,
    nifti_threshold: float = 0.0,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, NormParams]:
    """Load a point cloud from .ply, .xyz, or .nii.gz and return
    a normalized, downsampled Open3D point cloud, its points array,
    and the normalization parameters for inverse transform.
    """
    path = Path(file_path)
    suffix = "".join(path.suffixes).lower()

    if suffix in (".nii", ".nii.gz"):
        pcd = _load_nifti(file_path, nifti_threshold)
    elif suffix == ".ply":
        pcd = o3d.io.read_point_cloud(file_path)
    elif suffix == ".xyz":
        pcd = o3d.io.read_point_cloud(file_path, format="xyz")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {file_path}")

    # Rotate so Y-up becomes Z-up (original script convention)
    rotation = pcd.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    pcd.rotate(rotation, center=(0, 0, 0))

    # Normalize to [-1, 1] along the longest axis, centered at the mean
    pcd, norm_params = _normalize(pcd)

    # Voxel downsample
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    points = np.asarray(pcd.points)
    print(f"Loaded {file_path}: {points.shape[0]} points after downsampling")
    return pcd, points, norm_params


CORRECT_SPACING = np.array([0.39, 0.39, 0.2], dtype=np.float32)


def _load_nifti(file_path: str, threshold: float) -> o3d.geometry.PointCloud:
    """Load a NIfTI volume and convert above-threshold voxels to a point cloud.

    The NIfTI metadata often has incorrect spacing (e.g., [0.39, 0.39, 0.5]).
    We override with the known correct spacing and ignore the affine.
    """
    img = nib.load(file_path)
    data = np.asarray(img.dataobj, dtype=np.float32)

    voxel_coords = np.argwhere(data > threshold)

    if voxel_coords.shape[0] == 0:
        raise ValueError(
            f"No voxels above threshold {threshold} in {file_path}"
        )

    # Use correct spacing instead of the affine (which has wrong Z spacing)
    world_coords = voxel_coords.astype(np.float32) * CORRECT_SPACING

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_coords)
    return pcd


def _normalize(
    pcd: o3d.geometry.PointCloud,
) -> Tuple[o3d.geometry.PointCloud, NormParams]:
    """Center at mean and scale so the longest axis spans [-1, 1].
    Returns the modified point cloud and the normalization parameters."""
    points = np.asarray(pcd.points)
    center = points.mean(axis=0).copy()
    ranges = points.max(axis=0) - points.min(axis=0)
    scale = float(2.0 / ranges.max())

    points = (points - center) * scale
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd, NormParams(center=center, scale=scale)


def pick_endpoints(
    pcd: o3d.geometry.PointCloud,
) -> Tuple[np.ndarray, np.ndarray]:
    """Open an interactive viewer for the user to pick exactly 2 points
    (root crown and root tip). Returns (start_point, end_point)."""
    print(
        "Pick 2 points: root crown first, then root tip.\n"
        "  [Shift + Left Click] to pick a point.\n"
        "  Close the window when done."
    )
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick 2 endpoints")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    if len(picked) < 2:
        raise RuntimeError(
            f"Need 2 points, but only {len(picked)} were picked."
        )

    points = np.asarray(pcd.points)
    start = points[picked[0]]
    end = points[picked[1]]
    print(f"Start index: {picked[0]}, End index: {picked[1]}")
    return start, end


def save_endpoints(
    path: str, indices: Tuple[int, int]
) -> None:
    """Save picked endpoint indices for reproducibility."""
    with open(path, "w") as f:
        json.dump({"start_index": indices[0], "end_index": indices[1]}, f)


def load_endpoints(
    path: str, points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Load previously saved endpoint indices."""
    with open(path) as f:
        data = json.load(f)
    return points[data["start_index"]], points[data["end_index"]]


def save_results(
    path: str,
    all_points: np.ndarray,
    main_root_points: np.ndarray,
    branch_paths: list,
    norm_params: NormParams,
    config: Optional[object] = None,
    taproot_path: Optional[np.ndarray] = None,
    classified_branches: Optional[list] = None,
) -> None:
    """Save pipeline results in original (physical) coordinates.

    All point arrays are inverse-transformed before saving.

    Saved pickle keys:
      - ``all_points``            — (N, 3) full cloud
      - ``main_root_points``      — (M, 3) taproot volume mask
      - ``taproot_path``          — (P, 3) taproot centerline (order-1 root)
      - ``branch_paths``          — raw phase-4 paths (pre-classification)
      - ``classified_branches``   — post-classification flat list (NEW);
        only present when classified_branches is provided.  Each entry is
        a dict with keys: label, order, classification, parent_label,
        path, attachment_point, attachment_index_on_parent,
        absorbed_labels.  Use ``order`` and ``parent_label`` to reconstruct
        the secondary/tertiary tree.
      - ``config``                — the PipelineConfig used
      - ``norm_params``           — normalization transform
      - ``version``               — ``"rootweave/1.0.0"`` when classified_branches is present
    """
    data = {
        "all_points": norm_params.to_original(all_points),
        "main_root_points": norm_params.to_original(main_root_points),
        "taproot_path": norm_params.to_original(taproot_path) if taproot_path is not None else None,
        "branch_paths": [
            _denormalize_branch_path(bp, norm_params)
            for bp in branch_paths
        ],
        "config": config,
        "norm_params": norm_params,
    }
    if classified_branches is not None:
        data["classified_branches"] = [
            _denormalize_classified_branch(cb, norm_params)
            for cb in classified_branches
        ]
        data["version"] = "rootweave/1.0.0"
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_cls = len(classified_branches) if classified_branches else 0
    print(
        f"Results saved to {path} (original scale) — "
        f"{len(branch_paths)} raw branches, {n_cls} classified branches"
    )


def _denormalize_classified_branch(cb, norm_params: NormParams) -> dict:
    """Convert a ClassifiedBranch (or equivalent dict) to a plain dict
    with coordinates in original physical units.
    """
    def get(attr, default=None):
        if hasattr(cb, attr):
            return getattr(cb, attr)
        return cb.get(attr, default)

    path = [norm_params.to_original(np.asarray(p)) for p in get("path", [])]
    attach = get("attachment_point")
    if attach is not None:
        attach = norm_params.to_original(np.asarray(attach))

    return {
        "label": get("label"),
        "order": get("order"),
        "classification": get("classification", "unknown"),
        "parent_label": get("parent_label"),
        "path": path,
        "attachment_point": attach,
        "attachment_index_on_parent": get("attachment_index_on_parent"),
        "absorbed_labels": list(get("absorbed_labels", []) or []),
    }


def _denormalize_branch_path(bp, norm_params: NormParams):
    """Convert a BranchPath's coordinates back to original scale.
    Works with both dataclass instances and dicts."""
    if hasattr(bp, "path"):
        path = bp.path
        surr = bp.surrounding_indices
        label = bp.label
    else:
        path = bp.get("path", [])
        surr = bp.get("surrounding_indices", [])
        label = bp.get("label", 0)

    original_path = [norm_params.to_original(np.array(pt)) for pt in path]

    return {
        "label": label,
        "path": original_path,
        "surrounding_indices": surr,
    }


def load_results(path: str) -> dict:
    """Load previously saved pipeline results."""
    with open(path, "rb") as f:
        return pickle.load(f)
