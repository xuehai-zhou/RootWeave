#!/usr/bin/env python3
"""
Compute root system traits from a RootWeave result.pkl file.

Traits computed:
  1.  Individual root length (mm)
  2.  Relative root angle (degrees) — angle between each lateral root PCA and taproot PCA
  3.  Absolute root angle (degrees) — angle between each root PCA and the vertical (Z) axis
  4.  Individual root volume (mm³) — frustum model from path + local cross-section radius
  5.  Individual root surface area (mm²) — frustum model
  6.  Total root length (mm)
  7.  Total root volume (mm³)
  8.  Total root surface area (mm²)
  9.  Number of lateral roots
  10. Overall depth and width (mm)

Height axis: Z (+Z = up).

Usage:
    python compute_traits.py B2T3G16S3
    python compute_traits.py skl_res/B2T3G16S3.pkl
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

# Height axis: Z (index 2), up = +Z
HEIGHT_AXIS = np.array([0.0, 0.0, 1.0])

TRAITS_DIR = Path("traits")


def load_result(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _get_path_array(bp):
    """Extract path as numpy array from a BranchPath (dataclass or dict)."""
    if hasattr(bp, "path"):
        return np.array(bp.path)
    return np.array(bp.get("path", []))


def path_length(path_arr):
    """Total arc length of a path (mm)."""
    if len(path_arr) < 2:
        return 0.0
    diffs = np.diff(path_arr, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def path_pca_direction(path_arr):
    """PCA first component of a path (unit vector along the root)."""
    if len(path_arr) < 2:
        return np.array([0.0, 0.0, 1.0])
    pca = PCA(n_components=1)
    pca.fit(path_arr)
    return pca.components_[0]


def angle_between(v1, v2):
    """Angle in degrees between two vectors (0-90, folded)."""
    cos = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
    return float(np.degrees(np.arccos(abs(cos))))


def angle_between_signed(v1, v2):
    """Angle in degrees between two vectors (0-180, not folded)."""
    cos = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
    return float(np.degrees(np.arccos(cos)))


# ---------------------------------------------------------------------------
# Volume and surface area via frustum (truncated cone) model
# ---------------------------------------------------------------------------

def estimate_tube_geometry(path_arr, all_points, tree):
    """Estimate volume and surface area using a frustum (truncated cone) model.

    Each path segment is modeled as a frustum with radii r1, r2 at the
    two endpoints:
      volume = pi * L * (r1^2 + r1*r2 + r2^2) / 3
      lateral_area = pi * (r1 + r2) * slant_height
      slant_height = sqrt(L^2 + (r1 - r2)^2)

    This is more accurate than a piecewise cylinder model because it
    correctly accounts for radius changes along the root (tapering).

    Returns (volume_mm3, surface_area_mm2, radii_per_node).
    """
    if len(path_arr) < 2:
        return 0.0, 0.0, []

    # Tangent vectors for cross-section slicing
    tangents = np.gradient(path_arr, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents = tangents / norms

    radii = []
    for pt, tangent in zip(path_arr, tangents):
        r = _local_radius(pt, tangent, all_points, tree)
        radii.append(r)

    radii = np.array(radii)
    volume = 0.0
    surface = 0.0

    for i in range(len(path_arr) - 1):
        seg_len = np.linalg.norm(path_arr[i + 1] - path_arr[i])
        r1 = radii[i]
        r2 = radii[i + 1]

        # Frustum volume: exact for linearly tapering segment
        volume += np.pi * seg_len * (r1**2 + r1 * r2 + r2**2) / 3.0

        # Frustum lateral surface area
        slant = np.sqrt(seg_len**2 + (r1 - r2)**2)
        surface += np.pi * (r1 + r2) * slant

    # Add end caps (two circles)
    surface += np.pi * radii[0] ** 2 + np.pi * radii[-1] ** 2

    return float(volume), float(surface), radii.tolist()


def _local_radius(point, tangent, all_points, tree, search_k=20):
    """Estimate the root radius at a path node from nearby points."""
    dists, indices = tree.query(point, k=min(search_k, len(all_points)))
    nearby = all_points[indices]

    offsets = nearby - point
    along = offsets @ tangent
    perp = offsets - along[:, None] * tangent
    perp_dists = np.linalg.norm(perp, axis=1)

    plane_mask = np.abs(along) < np.percentile(np.abs(along), 50)
    if plane_mask.sum() < 3:
        return float(np.median(perp_dists))

    return float(np.percentile(perp_dists[plane_mask], 75))


# ---------------------------------------------------------------------------
# Main trait computation
# ---------------------------------------------------------------------------

def compute_traits(data):
    """Compute all traits from a loaded result dict."""
    all_points = data["all_points"]
    taproot_path = data.get("taproot_path")
    branch_paths = data.get("branch_paths", [])

    tree = KDTree(all_points)

    # Taproot
    tap_arr = np.array(taproot_path) if taproot_path is not None else np.empty((0, 3))
    tap_length = path_length(tap_arr)
    tap_pca = path_pca_direction(tap_arr) if len(tap_arr) >= 2 else HEIGHT_AXIS
    tap_abs_angle = angle_between(tap_pca, HEIGHT_AXIS)
    tap_vol, tap_sa, tap_radii = estimate_tube_geometry(tap_arr, all_points, tree)

    # Per-branch traits
    branch_traits = []
    total_branch_length = 0.0
    total_branch_volume = 0.0
    total_branch_sa = 0.0

    for idx, bp in enumerate(branch_paths):
        path_arr = _get_path_array(bp)
        if len(path_arr) < 2:
            continue

        length = path_length(path_arr)
        pca_dir = path_pca_direction(path_arr)
        rel_angle = angle_between(pca_dir, tap_pca)
        abs_angle = angle_between(pca_dir, HEIGHT_AXIS)
        vol, sa, radii = estimate_tube_geometry(path_arr, all_points, tree)

        branch_traits.append({
            "branch_id": idx,
            "length_mm": round(length, 3),
            "relative_angle_deg": round(rel_angle, 2),
            "absolute_angle_deg": round(abs_angle, 2),
            "volume_mm3": round(vol, 4),
            "surface_area_mm2": round(sa, 4),
            "n_nodes": len(path_arr),
            "mean_radius_mm": round(float(np.mean(radii)), 4) if radii else 0.0,
        })

        total_branch_length += length
        total_branch_volume += vol
        total_branch_sa += sa

    # Overall dimensions
    z_vals = all_points[:, 2]
    depth = float(np.max(z_vals) - np.min(z_vals))

    xy_extent = all_points[:, :2]
    width_x = float(np.max(xy_extent[:, 0]) - np.min(xy_extent[:, 0]))
    width_y = float(np.max(xy_extent[:, 1]) - np.min(xy_extent[:, 1]))
    width = max(width_x, width_y)

    summary = {
        "taproot_length_mm": round(tap_length, 3),
        "taproot_absolute_angle_deg": round(tap_abs_angle, 2),
        "taproot_volume_mm3": round(tap_vol, 4),
        "taproot_surface_area_mm2": round(tap_sa, 4),
        "taproot_mean_radius_mm": round(float(np.mean(tap_radii)), 4) if tap_radii else 0.0,
        "total_root_length_mm": round(tap_length + total_branch_length, 3),
        "total_root_volume_mm3": round(tap_vol + total_branch_volume, 4),
        "total_root_surface_area_mm2": round(tap_sa + total_branch_sa, 4),
        "n_lateral_roots": len(branch_traits),
        "depth_mm": round(depth, 3),
        "width_mm": round(width, 3),
    }

    return summary, branch_traits


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_traits(summary, branch_traits):
    """Print traits to console."""
    print("\n" + "=" * 60)
    print("ROOT SYSTEM TRAITS (frustum model)")
    print("=" * 60)

    print(f"\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")

    print(f"\n--- Taproot ---")
    print(f"  {'Length (mm)':35s}: {summary['taproot_length_mm']}")
    print(f"  {'Absolute angle (deg)':35s}: {summary['taproot_absolute_angle_deg']}")
    print(f"  {'Volume (mm³)':35s}: {summary['taproot_volume_mm3']}")
    print(f"  {'Surface area (mm²)':35s}: {summary['taproot_surface_area_mm2']}")
    print(f"  {'Mean radius (mm)':35s}: {summary['taproot_mean_radius_mm']}")

    if branch_traits:
        print(f"\n--- Lateral Roots ({len(branch_traits)}) ---")
        print(f"  {'ID':>4s} {'Length':>10s} {'RelAngle':>10s} {'AbsAngle':>10s} "
              f"{'Volume':>12s} {'SurfArea':>12s} {'Nodes':>6s} {'MeanR':>8s}")
        for bt in branch_traits:
            print(f"  {bt['branch_id']:4d} "
                  f"{bt['length_mm']:10.2f} "
                  f"{bt['relative_angle_deg']:10.2f} "
                  f"{bt['absolute_angle_deg']:10.2f} "
                  f"{bt['volume_mm3']:12.4f} "
                  f"{bt['surface_area_mm2']:12.4f} "
                  f"{bt['n_nodes']:6d} "
                  f"{bt['mean_radius_mm']:8.4f}")


def save_traits(summary, branch_traits, sample_name):
    """Save traits to ./traits/ in both plain text table and JSON formats."""
    TRAITS_DIR.mkdir(exist_ok=True)

    # --- JSON ---
    json_path = TRAITS_DIR / f"{sample_name}.json"
    json_data = {
        "sample": sample_name,
        "summary": summary,
        "lateral_roots": branch_traits,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nJSON saved to {json_path}")

    # --- Plain text table ---
    txt_path = TRAITS_DIR / f"{sample_name}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Sample: {sample_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        for k, v in summary.items():
            f.write(f"  {k:40s}: {v}\n")
        f.write("\n")

        if branch_traits:
            f.write(f"LATERAL ROOTS ({len(branch_traits)})\n")
            f.write("-" * 80 + "\n")

            header = (f"  {'ID':>4s} {'Length':>10s} {'RelAngle':>10s} {'AbsAngle':>10s} "
                      f"{'Volume':>12s} {'SurfArea':>12s} {'Nodes':>6s} {'MeanR':>8s}\n")
            f.write(header)
            f.write("  " + "-" * 76 + "\n")

            for bt in branch_traits:
                f.write(f"  {bt['branch_id']:4d} "
                        f"{bt['length_mm']:10.2f} "
                        f"{bt['relative_angle_deg']:10.2f} "
                        f"{bt['absolute_angle_deg']:10.2f} "
                        f"{bt['volume_mm3']:12.4f} "
                        f"{bt['surface_area_mm2']:12.4f} "
                        f"{bt['n_nodes']:6d} "
                        f"{bt['mean_radius_mm']:8.4f}\n")

    print(f"Text table saved to {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute root system traits")
    parser.add_argument("input", help="Path to result.pkl or sample name")
    args = parser.parse_args()

    # Resolve input
    input_path = args.input
    if not Path(input_path).exists():
        candidate = Path("skl_res") / (input_path + ".pkl")
        if candidate.exists():
            input_path = str(candidate)

    # Derive sample name
    stem = Path(input_path).name
    while Path(stem).suffix:
        stem = Path(stem).stem

    print(f"Loading {input_path}...")
    data = load_result(input_path)

    summary, branch_traits = compute_traits(data)
    print_traits(summary, branch_traits)
    save_traits(summary, branch_traits, stem)


if __name__ == "__main__":
    main()
