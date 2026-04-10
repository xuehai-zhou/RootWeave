#!/usr/bin/env python3
"""
Visualize a RootWeave result.pkl file with node-edge representation.

    Red         : primary root path (nodes + edges)
    Colored     : lateral root paths (distinct color per path)
    Light gray  : full root system point cloud

Usage:
    python visualize_result.py skl_res/B2T3G16S3.pkl
    python visualize_result.py skl_res/B2T3G16S3.pkl --no-cloud
    python visualize_result.py skl_res/B2T3G16S3.pkl --no-branches
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

COLOR_CLOUD = [0.75, 0.75, 0.75]
COLOR_TAPROOT = [1.0, 0.0, 0.0]


def load_result(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _auto_marker_size(all_points):
    """Compute a reasonable sphere radius from the point cloud extent."""
    if all_points is None or len(all_points) < 2:
        return 0.5
    extent = np.max(all_points, axis=0) - np.min(all_points, axis=0)
    return float(np.max(extent)) * 0.003  # 0.3% of the longest axis


def create_sphere(center, radius=0.5, color=None):
    color = color or [0.3, 0.3, 0.3]
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    s.translate(center - s.get_center())
    s.paint_uniform_color(color)
    return s


def create_lineset(pts, color):
    if len(pts) < 2:
        return None
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(pts) - 1)]),
    )
    ls.colors = o3d.utility.Vector3dVector([color] * (len(pts) - 1))
    return ls


def visualize(data, show_cloud=True, show_branches=True):
    all_points = data.get("all_points")
    taproot_path = data.get("taproot_path")
    branch_paths = data.get("branch_paths", [])

    marker = _auto_marker_size(all_points)
    geometries = []

    # Point cloud background
    if show_cloud and all_points is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.paint_uniform_color(COLOR_CLOUD)
        geometries.append(pcd)

    # Primary root path in red
    if taproot_path is not None and len(taproot_path) >= 2:
        tap_arr = np.array(taproot_path)
        ls = create_lineset(tap_arr, COLOR_TAPROOT)
        if ls:
            geometries.append(ls)
        for pt in tap_arr:
            geometries.append(create_sphere(pt, marker, COLOR_TAPROOT))

    # Lateral root paths in distinct colors
    if show_branches and branch_paths:
        cmap = plt.get_cmap("tab20")
        for idx, bp in enumerate(branch_paths):
            if hasattr(bp, "path"):
                path = bp.path
            else:
                path = bp.get("path", [])

            if len(path) < 2:
                continue

            path_arr = np.array(path)
            color = list(cmap(idx % 20)[:3])

            ls = create_lineset(path_arr, color)
            if ls:
                geometries.append(ls)
            for pt in path_arr:
                geometries.append(create_sphere(pt, marker, color))
            # Larger sphere at start
            geometries.append(create_sphere(path_arr[0], marker * 1.5, color))

    n_branches = len(branch_paths) if branch_paths else 0
    n_tap = len(taproot_path) if taproot_path is not None else 0
    print(f"Visualizing:")
    print(f"  Taproot    : {n_tap} nodes (red)")
    print(f"  Branches   : {n_branches} paths (colored)")
    if all_points is not None:
        print(f"  Cloud      : {len(all_points)} points")

    o3d.visualization.draw_geometries(geometries, window_name="RootWeave Result")


def main():
    parser = argparse.ArgumentParser(description="Visualize RootWeave result.pkl")
    parser.add_argument("input", help="Path to result.pkl (or sample name)")
    parser.add_argument("--no-cloud", action="store_true", help="Hide point cloud")
    parser.add_argument("--no-branches", action="store_true", help="Hide branch paths")
    args = parser.parse_args()

    # Resolve input: accept "B2T3G16S3" or "skl_res/B2T3G16S3.pkl"
    input_path = args.input
    if not Path(input_path).exists():
        candidate = Path("skl_res") / (input_path + ".pkl")
        if candidate.exists():
            input_path = str(candidate)

    data = load_result(input_path)
    visualize(data, show_cloud=not args.no_cloud, show_branches=not args.no_branches)


if __name__ == "__main__":
    main()
