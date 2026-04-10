#!/usr/bin/env python3
"""
Interactive Root Trait Viewer

Displays the root system with all paths. Navigate between paths to
inspect individual traits. A smooth tubular mesh is constructed for the
selected path to visualize and compute volume/surface area.

Controls:
    N / B   : Next / Previous path (cycles through all roots including taproot)
    T       : Toggle tube mesh for selected path
    C       : Toggle point cloud visibility
    R       : Reset view
    H       : Print help
    Q       : Quit

The selected path is shown with larger, brighter spheres.
Trait information is printed to the terminal on each selection change.

Usage:
    python interactive_viewer.py B2T3G16S3
    python interactive_viewer.py skl_res/B2T3G16S3.pkl
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

# Import trait computation
from compute_traits import (
    path_length, path_pca_direction, angle_between,
    estimate_tube_geometry, HEIGHT_AXIS, _get_path_array,
)


# ---------------------------------------------------------------------------
# Tube mesh construction
# ---------------------------------------------------------------------------

def build_tube_mesh(path_arr, radii, n_sides=16, color=[0.2, 0.7, 1.0]):
    """Build a smooth tubular triangle mesh around a path.

    Uses cubic spline resampling for a smooth centerline, Savitzky-Golay
    smoothed radii, and the double-reflection rotation-minimizing frame
    (RMF) algorithm for twist-free cross-sections.

    Parameters
    ----------
    path_arr : (N, 3) path nodes
    radii : (N,) radius at each node
    n_sides : number of sides for the tube cross-section
    color : RGB color for the mesh

    Returns
    -------
    o3d.geometry.TriangleMesh
    """
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter

    if len(path_arr) < 2:
        return None

    path_arr = np.array(path_arr, dtype=np.float64)
    radii = np.array(radii, dtype=np.float64)

    # --- Resample the path with a cubic spline for smoothness ---
    diffs = np.diff(path_arr, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    t_orig = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = t_orig[-1]
    if total_len < 1e-8:
        return None

    # Resample at ~2x the original density for a smoother tube
    n_resample = max(len(path_arr) * 2, 20)
    t_dense = np.linspace(0, total_len, n_resample)

    cs = CubicSpline(t_orig, path_arr, bc_type="natural")
    smooth_path = cs(t_dense)

    # Interpolate radii to the resampled points + smooth
    radii_interp = np.interp(t_dense, t_orig, radii)
    win = min(len(radii_interp) // 2, 15)
    if win >= 3 and win % 2 == 0:
        win -= 1
    if win >= 3:
        radii_interp = savgol_filter(radii_interp, win, 2)
    radii_interp = np.maximum(radii_interp, 0.01)  # floor

    # --- Compute tangents ---
    tangents = np.gradient(smooth_path, axis=0)
    tnorms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tnorms[tnorms == 0] = 1.0
    tangents = tangents / tnorms

    # --- Double-reflection rotation-minimizing frames (RMF) ---
    # Reference: Wang et al., "Computation of Rotation Minimizing Frames"
    t0 = tangents[0]
    ref = np.array([1.0, 0, 0]) if abs(t0[0]) < 0.9 else np.array([0, 1.0, 0])
    r0 = np.cross(t0, ref)
    r0 /= np.linalg.norm(r0)

    normals = np.zeros_like(tangents)
    normals[0] = r0

    for i in range(len(tangents) - 1):
        # First reflection: reflect r_i across the bisector of t_i and t_{i+1}
        v1 = smooth_path[i + 1] - smooth_path[i]
        c1 = np.dot(v1, v1)
        if c1 < 1e-20:
            normals[i + 1] = normals[i]
            continue
        rL = normals[i] - (2.0 / c1) * np.dot(v1, normals[i]) * v1
        tL = tangents[i] - (2.0 / c1) * np.dot(v1, tangents[i]) * v1

        # Second reflection: reflect across the bisector of tL and t_{i+1}
        v2 = tangents[i + 1] - tL
        c2 = np.dot(v2, v2)
        if c2 < 1e-20:
            normals[i + 1] = rL
            continue
        normals[i + 1] = rL - (2.0 / c2) * np.dot(v2, rL) * v2

        # Re-normalize
        nn = np.linalg.norm(normals[i + 1])
        if nn > 1e-10:
            normals[i + 1] /= nn
        else:
            normals[i + 1] = normals[i]

    binormals = np.cross(tangents, normals)
    bn = np.linalg.norm(binormals, axis=1, keepdims=True)
    bn[bn == 0] = 1.0
    binormals /= bn

    path_arr = smooth_path
    radii = radii_interp

    # Generate vertices: rings of n_sides points at each path node
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    vertices = []
    for i, (center, r, n, b) in enumerate(zip(path_arr, radii, normals, binormals)):
        for a in angles:
            pt = center + r * (np.cos(a) * n + np.sin(a) * b)
            vertices.append(pt)
    vertices = np.array(vertices)

    # Generate triangles connecting adjacent rings
    triangles = []
    for i in range(len(path_arr) - 1):
        for j in range(n_sides):
            j_next = (j + 1) % n_sides
            # Current ring start index
            c0 = i * n_sides
            c1 = (i + 1) * n_sides
            # Two triangles per quad
            triangles.append([c0 + j, c1 + j, c1 + j_next])
            triangles.append([c0 + j, c1 + j_next, c0 + j_next])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Interactive Viewer
# ---------------------------------------------------------------------------

class InteractiveViewer:
    def __init__(self, data):
        self.all_points = data["all_points"]
        self.taproot_path = np.array(data["taproot_path"]) if data.get("taproot_path") is not None else None
        self.branch_paths = data.get("branch_paths", [])

        self.tree = KDTree(self.all_points)

        # Build list of all paths: taproot first, then branches
        self.paths = []  # list of (name, path_arr, color)
        self.path_traits = []  # computed traits per path

        cmap = plt.get_cmap("tab20")

        # Taproot
        if self.taproot_path is not None and len(self.taproot_path) >= 2:
            self.paths.append(("Taproot", self.taproot_path, [1.0, 0.0, 0.0]))
            tap_pca = path_pca_direction(self.taproot_path)
        else:
            tap_pca = HEIGHT_AXIS

        # Branches
        for idx, bp in enumerate(self.branch_paths):
            path_arr = _get_path_array(bp)
            if len(path_arr) < 2:
                continue
            color = list(cmap(idx % 20)[:3])
            self.paths.append((f"Lateral {idx}", path_arr, color))

        # Compute traits for each path
        for name, path_arr, _ in self.paths:
            length = path_length(path_arr)
            pca_dir = path_pca_direction(path_arr)
            rel_angle = angle_between(pca_dir, tap_pca)
            abs_angle = angle_between(pca_dir, HEIGHT_AXIS)
            vol, sa, radii = estimate_tube_geometry(path_arr, self.all_points, self.tree)
            self.path_traits.append({
                "name": name,
                "length_mm": length,
                "relative_angle_deg": rel_angle,
                "absolute_angle_deg": abs_angle,
                "volume_mm3": vol,
                "surface_area_mm2": sa,
                "n_nodes": len(path_arr),
                "mean_radius_mm": float(np.mean(radii)) if radii else 0.0,
                "radii": radii,
            })

        # Auto marker size from point cloud extent
        extent = np.max(self.all_points, axis=0) - np.min(self.all_points, axis=0)
        self.marker = float(np.max(extent)) * 0.002
        self.marker_selected = self.marker * 2.5

        # State
        self.selected_idx = -1  # -1 = no selection
        self.show_tube = True
        self.show_cloud = True
        self.vis = None

    def _build_geometries(self):
        """Build all geometries for the current state."""
        geoms = []

        # Point cloud
        if self.show_cloud:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.all_points)
            pcd.paint_uniform_color([0.75, 0.75, 0.75])
            geoms.append(pcd)

        # Draw each path
        for i, (name, path_arr, color) in enumerate(self.paths):
            is_selected = (i == self.selected_idx)

            if is_selected:
                # Selected path: show ONLY the tube mesh, no nodes/edges
                traits = self.path_traits[i]
                radii = traits["radii"]
                if radii and len(path_arr) >= 2 and len(radii) == len(path_arr):
                    tube_color = [min(1, c * 0.6 + 0.4) for c in color]
                    mesh = build_tube_mesh(path_arr, radii, n_sides=16, color=tube_color)
                    if mesh is not None:
                        mesh.compute_vertex_normals()
                        geoms.append(mesh)
            else:
                # Non-selected paths: node-edge representation
                if len(path_arr) >= 2:
                    ls = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(path_arr),
                        lines=o3d.utility.Vector2iVector(
                            [[j, j + 1] for j in range(len(path_arr) - 1)]
                        ),
                    )
                    ls.colors = o3d.utility.Vector3dVector([color] * (len(path_arr) - 1))
                    geoms.append(ls)

                for pt in path_arr:
                    s = o3d.geometry.TriangleMesh.create_sphere(
                        radius=self.marker, resolution=8
                    )
                    s.translate(pt - s.get_center())
                    s.paint_uniform_color(color)
                    geoms.append(s)

        return geoms

    def _reload(self, vis, reset_bbox=False):
        vis.clear_geometries()
        for g in self._build_geometries():
            vis.add_geometry(g, reset_bounding_box=reset_bbox)
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.mesh_show_back_face = True
        vis.poll_events()
        vis.update_renderer()

    def _print_traits(self):
        if self.selected_idx < 0:
            print("\n  No path selected")
            return
        t = self.path_traits[self.selected_idx]
        name = t["name"]
        print(f"\n{'=' * 50}")
        print(f"  SELECTED: {name} ({self.selected_idx + 1}/{len(self.paths)})")
        print(f"{'=' * 50}")
        print(f"  Length           : {t['length_mm']:.2f} mm")
        print(f"  Relative angle   : {t['relative_angle_deg']:.2f}°")
        print(f"  Absolute angle   : {t['absolute_angle_deg']:.2f}°")
        print(f"  Volume           : {t['volume_mm3']:.4f} mm³")
        print(f"  Surface area     : {t['surface_area_mm2']:.4f} mm²")
        print(f"  Nodes            : {t['n_nodes']}")
        print(f"  Mean radius      : {t['mean_radius_mm']:.4f} mm")
        if self.show_tube:
            print(f"  Tube mesh        : ON")
        print(f"{'=' * 50}")

    def _next(self, vis):
        if not self.paths:
            return False
        self.selected_idx = (self.selected_idx + 1) % len(self.paths)
        self._print_traits()
        self._reload(vis)
        return False

    def _prev(self, vis):
        if not self.paths:
            return False
        self.selected_idx = (self.selected_idx - 1) % len(self.paths)
        self._print_traits()
        self._reload(vis)
        return False

    def _toggle_tube(self, vis):
        self.show_tube = not self.show_tube
        print(f"  Tube mesh: {'ON' if self.show_tube else 'OFF'}")
        self._reload(vis)
        return False

    def _toggle_cloud(self, vis):
        self.show_cloud = not self.show_cloud
        print(f"  Point cloud: {'ON' if self.show_cloud else 'OFF'}")
        self._reload(vis)
        return False

    def _deselect(self, vis):
        self.selected_idx = -1
        print("\n  Deselected")
        self._reload(vis)
        return False

    def _reset_view(self, vis):
        vis.reset_view_point(True)
        return False

    def _print_help(self, vis):
        print("\n  Controls:")
        print("    N     : Next path")
        print("    B     : Previous path")
        print("    T     : Toggle tube mesh")
        print("    C     : Toggle point cloud")
        print("    D     : Deselect")
        print("    R     : Reset view")
        print("    Q     : Quit")
        return False

    def _quit(self, vis):
        vis.close()
        return False

    def run(self):
        print(f"\nLoaded {len(self.paths)} paths "
              f"({len(self.all_points)} cloud points)")
        print("Press H for help, N/B to navigate paths")

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="RootWeave Interactive Viewer",
            width=1400, height=1000,
        )

        self.vis.register_key_callback(ord("N"), self._next)
        self.vis.register_key_callback(ord("B"), self._prev)
        self.vis.register_key_callback(ord("T"), self._toggle_tube)
        self.vis.register_key_callback(ord("C"), self._toggle_cloud)
        self.vis.register_key_callback(ord("D"), self._deselect)
        self.vis.register_key_callback(ord("R"), self._reset_view)
        self.vis.register_key_callback(ord("H"), self._print_help)
        self.vis.register_key_callback(ord("Q"), self._quit)

        for g in self._build_geometries():
            self.vis.add_geometry(g)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.mesh_show_back_face = True

        self.vis.run()
        self.vis.destroy_window()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive Root Trait Viewer")
    parser.add_argument("input", help="Path to result.pkl or sample name")
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        candidate = Path("skl_res") / (input_path + ".pkl")
        if candidate.exists():
            input_path = str(candidate)

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded: {input_path}")
    viewer = InteractiveViewer(data)
    viewer.run()


if __name__ == "__main__":
    main()
