#!/usr/bin/env python3
"""Interactive Root Trait Viewer.

Reads a rootweave result pickle (``skl_res/<name>.pkl``) and lets you
navigate each root's path one at a time, printing traits to the console
and showing a smooth tube mesh for the selected root.

Reads ``data["classified_branches"]`` (the post-phase-5 flat root-order
tree) when present; falls back to raw ``data["branch_paths"]`` for
pickles saved before phase 5 was run.

Controls
--------
    N / B   : Next / Previous root (cycles through taproot → all laterals)
    T       : Toggle tube mesh for the selected root
    C       : Toggle point cloud visibility
    D       : Deselect (hide all per-path highlighting)
    R       : Reset view
    H       : Print help
    Q       : Quit

Each non-selected root is drawn as a node-edge polyline in a distinct
colour.  Tertiaries get a black attach-sphere at their junction with
their parent.  The selected root is rendered as a smooth tube mesh with
per-node radius.  Trait info is printed to the terminal on every
selection change.

Usage:
    python interactive_viewer.py B2T3G16S3
    python interactive_viewer.py skl_res/B2T3G16S3.pkl
"""

import argparse
import colorsys
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# Reuse the trait math from the shipped compute_traits.py
from compute_traits import (
    _collect_branches, _get, _get_path_array,
    _parent_local_direction,
    path_length, path_pca_direction, angle_between,
    estimate_tube_geometry, HEIGHT_AXIS,
)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

COLOR_TAPROOT = [0.90, 0.10, 0.10]
COLOR_ATTACH = [0.0, 0.0, 0.0]


def _order_color(order: int, idx_in_order: int):
    """Distinct per-root colour, by order.  Secondaries in blues,
    tertiaries in greens, higher-order in purples.
    """
    if order == 2:
        cmap = plt.get_cmap("Blues")
    elif order == 3:
        cmap = plt.get_cmap("Greens")
    elif order >= 4:
        cmap = plt.get_cmap("Purples")
    else:
        return [0.55, 0.55, 0.55]
    return list(cmap(0.45 + 0.50 * (idx_in_order % 7) / 7.0)[:3])


# ---------------------------------------------------------------------------
# Tube mesh (smooth tubular triangle mesh around a path)
# ---------------------------------------------------------------------------

def build_tube_mesh(path_arr, radii, n_sides=16, color=(0.2, 0.7, 1.0)):
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter

    if len(path_arr) < 2:
        return None

    path_arr = np.asarray(path_arr, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    diffs = np.diff(path_arr, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    t_orig = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = t_orig[-1]
    if total_len < 1e-8:
        return None

    n_resample = max(len(path_arr) * 2, 20)
    t_dense = np.linspace(0, total_len, n_resample)
    smooth_path = CubicSpline(t_orig, path_arr, bc_type="natural")(t_dense)

    radii_interp = np.interp(t_dense, t_orig, radii)
    win = min(len(radii_interp) // 2, 15)
    if win >= 3 and win % 2 == 0:
        win -= 1
    if win >= 3:
        radii_interp = savgol_filter(radii_interp, win, 2)
    radii_interp = np.maximum(radii_interp, 0.01)

    tangents = np.gradient(smooth_path, axis=0)
    tnorms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tnorms[tnorms == 0] = 1.0
    tangents = tangents / tnorms

    # Double-reflection rotation-minimizing frames
    t0 = tangents[0]
    ref = np.array([1.0, 0, 0]) if abs(t0[0]) < 0.9 else np.array([0, 1.0, 0])
    r0 = np.cross(t0, ref)
    r0 /= np.linalg.norm(r0)

    normals = np.zeros_like(tangents)
    normals[0] = r0
    for i in range(len(tangents) - 1):
        v1 = smooth_path[i + 1] - smooth_path[i]
        c1 = np.dot(v1, v1)
        if c1 < 1e-20:
            normals[i + 1] = normals[i]
            continue
        rL = normals[i] - (2.0 / c1) * np.dot(v1, normals[i]) * v1
        tL = tangents[i] - (2.0 / c1) * np.dot(v1, tangents[i]) * v1
        v2 = tangents[i + 1] - tL
        c2 = np.dot(v2, v2)
        if c2 < 1e-20:
            normals[i + 1] = rL
            continue
        normals[i + 1] = rL - (2.0 / c2) * np.dot(v2, rL) * v2
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

    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    vertices = []
    for center, r, n, b in zip(path_arr, radii, normals, binormals):
        for a in angles:
            vertices.append(center + r * (np.cos(a) * n + np.sin(a) * b))
    vertices = np.array(vertices)

    triangles = []
    for i in range(len(path_arr) - 1):
        for j in range(n_sides):
            j_next = (j + 1) % n_sides
            c0 = i * n_sides
            c1 = (i + 1) * n_sides
            triangles.append([c0 + j, c1 + j, c1 + j_next])
            triangles.append([c0 + j, c1 + j_next, c0 + j_next])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(list(color))
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Interactive Viewer
# ---------------------------------------------------------------------------

class InteractiveViewer:
    """Paths list order = [taproot, laterals sorted by (order, label)].

    Each entry carries: name, path_arr, color, order, parent_label,
    classification, attachment_point, tube_radii.
    """

    def __init__(self, data):
        self.all_points = data["all_points"]
        self.taproot_path = (
            np.array(data["taproot_path"]) if data.get("taproot_path") is not None
            else None
        )
        branches, source = _collect_branches(data)
        self.source = source  # "classified" or "legacy"

        self.tree = KDTree(self.all_points)

        # --- Build the per-path entries ---
        self.paths = []       # list of dicts
        self.path_traits = [] # parallel list

        # Taproot first
        if self.taproot_path is not None and len(self.taproot_path) >= 2:
            self.paths.append({
                "name": "Taproot",
                "path_arr": self.taproot_path,
                "color": COLOR_TAPROOT,
                "order": 1,
                "parent_label": None,
                "classification": "taproot",
                "attachment_point": None,
            })
            tap_pca = path_pca_direction(self.taproot_path)
        else:
            tap_pca = HEIGHT_AXIS

        # Laterals: skip unknown/order-0, sort by (order, label)
        good_branches = []
        for i, b in enumerate(branches):
            order = int(_get(b, "order", 0) or 0)
            if order == 0 or len(_get_path_array(b)) < 2:
                continue
            good_branches.append((order, int(_get(b, "label", i)), b))
        good_branches.sort(key=lambda t: (t[0], t[1]))

        # Colour index per order
        per_order_i = {}
        by_label = {int(_get(b, "label", i)): b for i, b in enumerate(branches)}

        for order, label, b in good_branches:
            idx_in_order = per_order_i.get(order, 0)
            per_order_i[order] = idx_in_order + 1
            color = _order_color(order, idx_in_order)
            path_arr = _get_path_array(b)
            parent_label = _get(b, "parent_label", None)
            classification = _get(b, "classification", "legacy")

            attach_pt = _get(b, "attachment_point", None)
            if attach_pt is not None:
                attach_pt = np.asarray(attach_pt, dtype=float)

            name_kind = {
                2: "Secondary",
                3: "Tertiary",
            }.get(order, f"Order-{order}")
            self.paths.append({
                "name": f"{name_kind} #{label}",
                "path_arr": path_arr,
                "color": color,
                "order": order,
                "parent_label": parent_label,
                "classification": classification,
                "attachment_point": attach_pt,
                "_branch": b,
            })

        # --- Trait computation (once, cached) ---
        for entry in self.paths:
            path_arr = entry["path_arr"]
            length = path_length(path_arr)
            pca_dir = path_pca_direction(path_arr)
            abs_angle = angle_between(pca_dir, HEIGHT_AXIS)
            angle_tap = angle_between(pca_dir, tap_pca)
            vol, sa, radii = estimate_tube_geometry(
                path_arr, self.all_points, self.tree
            )

            # Parent angle: only meaningful for entries with a non-None
            # parent_label that points to another entry in the list.
            angle_par: Optional[float] = None
            parent_label = entry["parent_label"]
            if parent_label is not None and parent_label in by_label:
                parent = by_label[parent_label]
                parent_arr = _get_path_array(parent)
                if len(parent_arr) >= 2:
                    attach_idx = _get(parent, "attachment_index_on_parent", None)
                    # For this viewer we re-locate the attachment on the
                    # parent's CURRENT path by nearest-neighbour to the
                    # child's last point.  Saved attachment_index_on_parent
                    # was the index on the parent at phase-5 time; the
                    # path may have been absorbed/re-labelled since.
                    if entry["attachment_point"] is not None:
                        dists = np.linalg.norm(parent_arr - entry["attachment_point"], axis=1)
                        attach_idx = int(np.argmin(dists))
                    par_dir = _parent_local_direction(parent_arr, attach_idx)
                    angle_par = angle_between(pca_dir, par_dir)

            self.path_traits.append({
                "name": entry["name"],
                "order": entry["order"],
                "classification": entry["classification"],
                "parent_label": parent_label,
                "length_mm": length,
                "angle_to_taproot_deg": angle_tap,
                "angle_to_parent_deg": angle_par,
                "absolute_angle_deg": abs_angle,
                "volume_mm3": vol,
                "surface_area_mm2": sa,
                "n_nodes": int(len(path_arr)),
                "mean_radius_mm": float(np.mean(radii)) if radii else 0.0,
                "radii": radii,
            })

        # Visual scale
        extent = np.max(self.all_points, axis=0) - np.min(self.all_points, axis=0)
        self.marker = float(np.max(extent)) * 0.002

        # State
        self.selected_idx = -1
        self.show_tube = True
        self.show_cloud = True
        self.vis = None

    # -- Geometry ------------------------------------------------------

    def _build_geometries(self):
        geoms = []

        if self.show_cloud:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.all_points)
            pcd.paint_uniform_color([0.75, 0.75, 0.75])
            geoms.append(pcd)

        for i, entry in enumerate(self.paths):
            path_arr = entry["path_arr"]
            color = entry["color"]
            is_selected = (i == self.selected_idx)

            if is_selected and self.show_tube:
                radii = self.path_traits[i]["radii"]
                if radii and len(path_arr) >= 2 and len(radii) == len(path_arr):
                    tube_color = [min(1, c * 0.6 + 0.4) for c in color]
                    mesh = build_tube_mesh(
                        path_arr, radii, n_sides=16, color=tube_color,
                    )
                    if mesh is not None:
                        geoms.append(mesh)
                        continue  # skip node-edge rendering for selected

            # Non-selected (or tube disabled): node-edge polyline
            if len(path_arr) >= 2:
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(path_arr),
                    lines=o3d.utility.Vector2iVector(
                        [[j, j + 1] for j in range(len(path_arr) - 1)]
                    ),
                )
                ls.colors = o3d.utility.Vector3dVector(
                    [color] * (len(path_arr) - 1)
                )
                geoms.append(ls)

                node_r = self.marker if not is_selected else self.marker * 1.6
                for pt in path_arr:
                    s = o3d.geometry.TriangleMesh.create_sphere(
                        radius=node_r, resolution=8,
                    )
                    s.translate(pt - s.get_center())
                    s.paint_uniform_color(color)
                    geoms.append(s)

            # Tertiary attachment marker
            if entry["attachment_point"] is not None and entry["order"] >= 3:
                s = o3d.geometry.TriangleMesh.create_sphere(
                    radius=self.marker * 2.0, resolution=10,
                )
                s.translate(entry["attachment_point"] - s.get_center())
                s.paint_uniform_color(COLOR_ATTACH)
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

    # -- Trait printing ------------------------------------------------

    def _print_traits(self):
        if self.selected_idx < 0:
            print("\n  (nothing selected)")
            return
        t = self.path_traits[self.selected_idx]
        e = self.paths[self.selected_idx]
        print("\n" + "=" * 54)
        print(f"  {t['name']}  ({self.selected_idx + 1}/{len(self.paths)})")
        print(f"    order         : {t['order']}")
        print(f"    classification: {t['classification']}")
        if t["parent_label"] is not None:
            print(f"    parent        : #{t['parent_label']}")
        else:
            print(f"    parent        : taproot")
        print("-" * 54)
        print(f"  Length              : {t['length_mm']:.2f} mm")
        print(f"  Angle → taproot     : {t['angle_to_taproot_deg']:.2f}°")
        if t["angle_to_parent_deg"] is not None:
            print(f"  Angle → parent      : {t['angle_to_parent_deg']:.2f}°")
        print(f"  Absolute angle (+Z) : {t['absolute_angle_deg']:.2f}°")
        print(f"  Volume              : {t['volume_mm3']:.4f} mm³")
        print(f"  Surface area        : {t['surface_area_mm2']:.4f} mm²")
        print(f"  Nodes               : {t['n_nodes']}")
        print(f"  Mean radius         : {t['mean_radius_mm']:.4f} mm")
        print(f"  Tube mesh           : {'ON' if self.show_tube else 'OFF'}")
        print("=" * 54)

    # -- Key callbacks -------------------------------------------------

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
        print("    N     : Next root")
        print("    B     : Previous root")
        print("    T     : Toggle tube mesh")
        print("    C     : Toggle point cloud")
        print("    D     : Deselect")
        print("    R     : Reset view")
        print("    H     : Print this help")
        print("    Q     : Quit")
        return False

    def _quit(self, vis):
        vis.close()
        return False

    # -- Main loop -----------------------------------------------------

    def run(self):
        n_sec = sum(1 for e in self.paths if e.get("order") == 2)
        n_ter = sum(1 for e in self.paths if e.get("order") == 3)
        print(
            f"\nLoaded {len(self.paths)} paths "
            f"(taproot + {n_sec} secondaries + {n_ter} tertiaries)  "
            f"source={self.source}"
        )
        print("Press H for help, N/B to navigate")

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="rootweave Interactive Viewer",
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
