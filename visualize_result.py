#!/usr/bin/env python3
"""Visualize a RootWeave result.pkl with explicit root-order hierarchy.

The saved pickle contains (among other things):

    all_points           — full point cloud
    main_root_points     — taproot volume mask (the "taproot point cloud")
    taproot_path         — order-1 root (taproot centerline)
    branch_paths         — RAW phase-4 paths (pre-classification)
    classified_branches  — POST-classification flat list, one entry per root:
        {
          "label": int,
          "order": int,                        # 0=unknown, 2=secondary, 3=tertiary, ...
          "classification": str,               # taproot-direct | split-extension
                                               # | sibling-secondary | tertiary | unknown
          "parent_label": int | None,          # None means attached to the taproot
          "path": [(x,y,z), ...],              # tip→taproot for secondaries,
                                               # tip→attachment for tertiaries
          "attachment_point": (x,y,z) | None,
          "attachment_index_on_parent": int | None,
          "absorbed_labels": [int, ...],
        }

Display modes (``--mode``)
--------------------------

The full point cloud is ALWAYS displayed (gray, small point size so it
doesn't block the paths).  Use ``--no-cloud`` to force it off.

    classified (default)
        Full cloud + taproot centerline (red) + ALL secondaries (BLUE
        palette) + ALL tertiaries (GREEN palette).  Attachment links
        draw black lines from tertiaries to their parent secondaries.

    secondary
        Full cloud + taproot centerline (red) + taproot volume cloud
        (salmon) + secondaries ONLY, each drawn in a DIFFERENT hue
        chosen from a purple/cool-biased qualitative palette so
        adjacent secondaries are clearly distinguishable.  Tertiaries
        hidden.

    tertiary
        Full cloud + taproot centerline (red) + taproot volume cloud
        (salmon) + tertiaries ONLY, each drawn in a DIFFERENT hue
        chosen from a warm/orange-biased qualitative palette.  Sibling
        tertiaries are clearly distinguishable from each other.
        Secondaries hidden.  The black attach-sphere on each tertiary
        still marks where it joined its (now invisible) parent.

    raw
        Legacy pre-classification view: all branches colored by label
        with no order distinction.  Falls back to this automatically if
        the pickle doesn't contain ``classified_branches``.

Component color families:
    taproot centerline       → red
    taproot point cloud      → salmon (shown in focused modes)
    full point cloud         → gray (small point size)
    secondaries (default)    → blue palette
    secondaries (focused)    → purple palette
    tertiaries (default)     → green palette
    tertiaries (focused)     → orange palette
    higher-order roots       → purple
    unknown / dropped        → gray
    attach markers + links   → black

Usage:
    python visualize_result.py skl_res/B2T3G16S3.pkl
    python visualize_result.py skl_res/B2T3G16S3.pkl --mode secondary
    python visualize_result.py skl_res/B2T3G16S3.pkl --mode tertiary
    python visualize_result.py skl_res/B2T3G16S3.pkl --raw
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Component colors
COLOR_CLOUD = [0.75, 0.75, 0.75]          # full point cloud — neutral gray
COLOR_TAPROOT = [0.90, 0.10, 0.10]        # taproot centerline — red
COLOR_TAPROOT_CLOUD = [1.00, 0.55, 0.55]  # main_root_points — salmon
COLOR_UNKNOWN = [0.55, 0.55, 0.55]        # order-0 / unknown — gray
COLOR_ATTACH = [0.0, 0.0, 0.0]            # attach markers + parent links — black


# Qualitative palettes for focused modes.
#
# Base set = matplotlib's ``tab10`` (default 10-color categorical scheme,
# maximally distinct hues), minus:
#   - index 3 (red)  — reserved for the taproot
#   - index 7 (gray) — too close to the full-cloud color
#
# That leaves 8 distinct, high-contrast colors.  The secondary-focused
# and tertiary-focused palettes use the same base set but start at
# different hues so the two modes feel visually different at a glance.

# Hand-ordered so consecutive slots are in different hue families (warm /
# cool / neutral alternation), which minimizes the chance that two roots
# visible side-by-side get similar colors.

PALETTE_SECONDARY_FOCUSED = [
    [0.580, 0.404, 0.741],  # purple          (cool)
    [1.000, 0.498, 0.055],  # orange          (warm)
    [0.090, 0.745, 0.812],  # cyan            (cool)
    [0.890, 0.467, 0.761],  # pink            (warm)
    [0.173, 0.627, 0.173],  # green           (cool/neutral)
    [0.549, 0.337, 0.294],  # brown           (warm/neutral)
    [0.122, 0.467, 0.706],  # blue            (cool)
    [0.737, 0.741, 0.133],  # olive           (warm/neutral)
]

# Tertiary mode: SAME 8 distinct hues but starting with orange and
# alternating differently, so the mode has an obviously different
# "first look" from the secondary-focused mode.
PALETTE_TERTIARY_FOCUSED = [
    [1.000, 0.498, 0.055],  # orange
    [0.122, 0.467, 0.706],  # blue
    [0.737, 0.741, 0.133],  # olive
    [0.580, 0.404, 0.741],  # purple
    [0.890, 0.467, 0.761],  # pink
    [0.090, 0.745, 0.812],  # cyan
    [0.549, 0.337, 0.294],  # brown
    [0.173, 0.627, 0.173],  # green
]


def load_result(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _auto_marker_size(all_points):
    if all_points is None or len(all_points) < 2:
        return 0.5
    extent = np.max(all_points, axis=0) - np.min(all_points, axis=0)
    return float(np.max(extent)) * 0.003  # 0.3% of the longest axis


def create_sphere(center, radius=0.5, color=None):
    color = color or [0.3, 0.3, 0.3]
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    s.translate(np.asarray(center) - np.asarray(s.get_center()))
    s.paint_uniform_color(color)
    return s


def create_lineset(pts, color):
    if len(pts) < 2:
        return None
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(pts) - 1)]
        ),
    )
    ls.colors = o3d.utility.Vector3dVector([color] * (len(pts) - 1))
    return ls


# ---------------------------------------------------------------------------
# Order → palette
# ---------------------------------------------------------------------------

def _color_for_order(order: int, idx_in_order: int, override: dict = None):
    """Pick a color based on root order, cycling within-order.

    ``override`` maps order -> palette.  A palette can be either:
      * a ``str`` — a matplotlib colormap name; colors are sampled along
        it (gradient, used for default mode's within-order shading);
      * a ``list`` of RGB triples — used as a qualitative palette,
        cycled by ``idx_in_order``.  This is used by focused modes so
        adjacent paths get genuinely different hues, not similar shades.
    """
    if override and order in override:
        palette = override[order]
        if isinstance(palette, str):
            cmap = plt.get_cmap(palette)
            return list(cmap(0.45 + 0.50 * (idx_in_order % 7) / 7.0)[:3])
        # list/tuple of colors — cycle through it
        return list(palette[idx_in_order % len(palette)])

    if order == 2:
        cmap = plt.get_cmap("Blues")
        return list(cmap(0.45 + 0.50 * (idx_in_order % 7) / 7.0)[:3])
    if order == 3:
        cmap = plt.get_cmap("Greens")
        return list(cmap(0.45 + 0.50 * (idx_in_order % 7) / 7.0)[:3])
    if order >= 4:
        cmap = plt.get_cmap("Purples")
        return list(cmap(0.50 + 0.45 * (idx_in_order % 7) / 7.0)[:3])
    return COLOR_UNKNOWN


# ---------------------------------------------------------------------------
# Component geometry helpers
# ---------------------------------------------------------------------------

def _show(geoms, window_name: str, point_size: float = 1.5):
    """Render geoms in an Open3D window with a configurable point size.

    Uses Visualizer instead of draw_geometries so the full point cloud
    can be shown at a small size without visually blocking the paths.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=1000)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()


def _add_full_cloud(geoms, all_points):
    if all_points is None:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.paint_uniform_color(COLOR_CLOUD)
    geoms.append(pcd)


def _add_taproot_cloud(geoms, main_root_points):
    if main_root_points is None or len(main_root_points) == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(main_root_points)
    pcd.paint_uniform_color(COLOR_TAPROOT_CLOUD)
    geoms.append(pcd)


def _add_taproot_centerline(geoms, taproot_path, marker):
    if taproot_path is None or len(taproot_path) < 2:
        return
    tap_arr = np.asarray(taproot_path)
    ls = create_lineset(tap_arr, COLOR_TAPROOT)
    if ls is not None:
        geoms.append(ls)
    for pt in tap_arr:
        geoms.append(create_sphere(pt, marker, COLOR_TAPROOT))


# ---------------------------------------------------------------------------
# Classified view
# ---------------------------------------------------------------------------

def _visualize_classified(
    data,
    show_full_cloud: bool,
    show_taproot_cloud: bool,
    show_attach_links: bool,
    order_filter,
    color_override: dict = None,
    point_size: float = 1.5,
):
    all_points = data.get("all_points")
    main_root_points = data.get("main_root_points")
    taproot_path = data.get("taproot_path")
    classified = data.get("classified_branches", [])

    marker = _auto_marker_size(all_points)
    geoms = []

    if show_full_cloud:
        _add_full_cloud(geoms, all_points)
    if show_taproot_cloud:
        _add_taproot_cloud(geoms, main_root_points)
    _add_taproot_centerline(geoms, taproot_path, marker)

    # Color each root deterministically by (order, sorted-label)
    label_to_entry = {cb["label"]: cb for cb in classified if "label" in cb}
    label_to_color = {}
    for order in sorted({cb["order"] for cb in classified}):
        of_this_order = sorted(
            [cb for cb in classified if cb["order"] == order],
            key=lambda c: c["label"],
        )
        for i, cb in enumerate(of_this_order):
            label_to_color[cb["label"]] = _color_for_order(
                order, i, override=color_override,
            )

    per_order_count = {}
    n_hidden_links = 0

    for cb in classified:
        order = cb["order"]
        if order_filter is not None and order not in order_filter:
            continue
        path = cb.get("path", [])
        if len(path) < 2:
            continue

        color = label_to_color.get(cb["label"], COLOR_UNKNOWN)
        path_arr = np.asarray(path)

        ls = create_lineset(path_arr, color)
        if ls is not None:
            geoms.append(ls)
        for pt in path_arr:
            geoms.append(create_sphere(pt, marker, color))
        # Bigger tip marker
        geoms.append(create_sphere(path_arr[0], marker * 1.8, color))

        # Tertiary (and higher) attachment markers
        if order >= 3 and cb.get("attachment_point") is not None:
            attach = np.asarray(cb["attachment_point"])
            geoms.append(create_sphere(attach, marker * 2.0, COLOR_ATTACH))

            # Only draw the link to the parent if the parent is VISIBLE
            if show_attach_links and cb.get("parent_label") is not None:
                parent = label_to_entry.get(cb["parent_label"])
                parent_visible = (
                    parent is not None
                    and (order_filter is None or parent["order"] in order_filter)
                )
                if parent_visible:
                    idx = cb.get("attachment_index_on_parent")
                    if idx is not None and 0 <= idx < len(parent.get("path", [])):
                        anchor = np.asarray(parent["path"][idx])
                    else:
                        anchor = attach
                    link = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector([attach, anchor]),
                        lines=o3d.utility.Vector2iVector([[0, 1]]),
                    )
                    link.colors = o3d.utility.Vector3dVector([COLOR_ATTACH])
                    geoms.append(link)
                else:
                    n_hidden_links += 1

        per_order_count[order] = per_order_count.get(order, 0) + 1

    n_tap = len(taproot_path) if taproot_path is not None else 0
    n_tap_cloud = len(main_root_points) if main_root_points is not None else 0
    print(f"Taproot (order 1): {n_tap} centerline nodes (red)")
    if show_taproot_cloud:
        print(f"  taproot point cloud: {n_tap_cloud} points (salmon)")
    default_palette = {2: "blue", 3: "green"}
    for order in sorted(per_order_count.keys()):
        palette = (color_override or {}).get(order, None)
        if palette is None:
            palette_label = default_palette.get(order, f"order-{order}")
        elif isinstance(palette, str):
            palette_label = palette.lower()
        else:
            palette_label = f"{len(palette)} distinct hues"
        base_name = {2: "secondaries", 3: "tertiaries"}.get(
            order, f"order-{order} roots"
        )
        print(f"  {base_name} ({palette_label}): {per_order_count[order]}")
    if n_hidden_links:
        print(
            f"  ({n_hidden_links} tertiary attachment-links hidden "
            f"because parent not in current view)"
        )
    if show_full_cloud and all_points is not None:
        print(f"  full cloud: {len(all_points)} points")

    if order_filter is None:
        _print_tree_summary(classified)

    _show(geoms, "rootweave — Classified", point_size=point_size)


def _print_tree_summary(classified):
    if not classified:
        return

    by_label = {cb["label"]: cb for cb in classified}
    children_of = {None: []}
    for cb in classified:
        p = cb.get("parent_label")
        children_of.setdefault(p, []).append(cb["label"])

    print("\n  Root tree (label — classification, arc length)")
    print("  taproot")

    def arc_len(path):
        if len(path) < 2:
            return 0.0
        arr = np.asarray(path)
        return float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))

    def print_subtree(parent_label, indent):
        for lbl in sorted(children_of.get(parent_label, [])):
            cb = by_label[lbl]
            print(
                f"  {'  ' * indent}└── #{lbl}  order={cb['order']}  "
                f"{cb['classification']}  len≈{arc_len(cb['path']):.2f}"
            )
            print_subtree(lbl, indent + 1)

    print_subtree(None, 1)


# ---------------------------------------------------------------------------
# Legacy / raw view (pre-classification)
# ---------------------------------------------------------------------------

def _visualize_raw(data, show_full_cloud: bool, point_size: float = 1.5):
    all_points = data.get("all_points")
    taproot_path = data.get("taproot_path")
    branch_paths = data.get("branch_paths", [])

    marker = _auto_marker_size(all_points)
    geoms = []

    if show_full_cloud:
        _add_full_cloud(geoms, all_points)
    _add_taproot_centerline(geoms, taproot_path, marker)

    if branch_paths:
        cmap = plt.get_cmap("tab20")
        for idx, bp in enumerate(branch_paths):
            path = bp.path if hasattr(bp, "path") else bp.get("path", [])
            if len(path) < 2:
                continue
            path_arr = np.asarray(path)
            color = list(cmap(idx % 20)[:3])
            ls = create_lineset(path_arr, color)
            if ls is not None:
                geoms.append(ls)
            for pt in path_arr:
                geoms.append(create_sphere(pt, marker, color))
            geoms.append(create_sphere(path_arr[0], marker * 1.5, color))

    n_branches = len(branch_paths) if branch_paths else 0
    n_tap = len(taproot_path) if taproot_path is not None else 0
    print(f"RAW view — Taproot: {n_tap} nodes, Branches: {n_branches} paths")
    if show_full_cloud and all_points is not None:
        print(f"  full cloud: {len(all_points)} points")

    _show(geoms, "rootweave — RAW (pre-classification)",
          point_size=point_size)


# ---------------------------------------------------------------------------
# Mode presets
# ---------------------------------------------------------------------------

# Each preset returns a set of display options; individual --no-* flags can
# still override afterward.

def _apply_mode(mode: str):
    """Map a mode name to display options.

    Full cloud is always enabled; user can disable with ``--no-cloud``.
    Focused modes also carry a ``color_override`` that swaps the default
    per-order palette so the mode is visually distinct from the default.
    """
    if mode == "secondary":
        return dict(
            full_cloud=True, taproot_cloud=True, order_filter={2},
            color_override={2: PALETTE_SECONDARY_FOCUSED},
        )
    if mode == "tertiary":
        return dict(
            full_cloud=True, taproot_cloud=True, order_filter={3},
            color_override={3: PALETTE_TERTIARY_FOCUSED},
        )
    # "classified" / default — default palettes untouched
    return dict(
        full_cloud=True, taproot_cloud=False, order_filter=None,
        color_override=None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a rootweave result.pkl.  "
            "Default mode shows the classified root tree (taproot / "
            "secondaries / tertiaries).  Use --mode secondary or "
            "--mode tertiary to focus on one lateral layer."
        )
    )
    parser.add_argument("input", help="Path to result.pkl (or sample name)")
    parser.add_argument(
        "--mode",
        choices=["classified", "secondary", "tertiary", "raw"],
        default="classified",
        help=(
            "classified (default): full cloud + taproot + all laterals.  "
            "secondary: taproot + taproot cloud + secondaries only.  "
            "tertiary: taproot + taproot cloud + tertiaries only.  "
            "raw: pre-classification view (all branches same palette)."
        ),
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Alias for --mode raw.  Kept for backward compatibility.",
    )
    parser.add_argument(
        "--no-cloud", action="store_true",
        help="Hide the full point cloud (effective in classified / raw modes).",
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Force-enable the full point cloud in focused modes.",
    )
    parser.add_argument(
        "--no-taproot-cloud", action="store_true",
        help="Hide the taproot point cloud (main_root_points).",
    )
    parser.add_argument(
        "--taproot-cloud", action="store_true",
        help="Force-enable the taproot point cloud in classified / raw modes.",
    )
    parser.add_argument(
        "--no-attach-links", action="store_true",
        help="Don't draw the black lines linking tertiaries to their parents.",
    )
    parser.add_argument(
        "--order", type=int, nargs="+", default=None,
        help="Override mode's order filter (e.g. --order 2 3).",
    )
    parser.add_argument(
        "--point-size", type=float, default=1.5,
        help=(
            "Open3D point-size for the rendering.  Smaller values keep "
            "the full cloud from visually blocking the paths.  Default 1.5."
        ),
    )
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        candidate = Path("skl_res") / (input_path + ".pkl")
        if candidate.exists():
            input_path = str(candidate)

    data = load_result(input_path)
    has_classified = bool(data.get("classified_branches"))

    # Resolve mode
    mode = args.mode
    if args.raw:
        mode = "raw"
    if mode in ("classified", "secondary", "tertiary") and not has_classified:
        print(
            "  (classified_branches missing in this pickle → falling back to raw)"
        )
        mode = "raw"

    print(f"Loaded: {input_path}")
    print(
        f"  version: {data.get('version', 'legacy')}  |  "
        f"mode: {mode}"
    )

    if mode == "raw":
        show_full_cloud = not args.no_cloud
        _visualize_raw(
            data,
            show_full_cloud=show_full_cloud,
            point_size=args.point_size,
        )
        return

    preset = _apply_mode(mode)

    # Resolve individual toggles (explicit flags override the preset)
    show_full_cloud = preset["full_cloud"]
    if args.no_cloud:
        show_full_cloud = False
    if args.cloud:
        show_full_cloud = True

    show_taproot_cloud = preset["taproot_cloud"]
    if args.no_taproot_cloud:
        show_taproot_cloud = False
    if args.taproot_cloud:
        show_taproot_cloud = True

    order_filter = preset["order_filter"]
    if args.order is not None:
        order_filter = set(args.order)

    _visualize_classified(
        data,
        show_full_cloud=show_full_cloud,
        show_taproot_cloud=show_taproot_cloud,
        show_attach_links=not args.no_attach_links,
        order_filter=order_filter,
        color_override=preset.get("color_override"),
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
