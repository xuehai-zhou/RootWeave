#!/usr/bin/env python3
"""Compute root-system traits from a rootweave result pickle.

Reads ``data["classified_branches"]`` (the post-phase-5 flat root-order
tree) and computes per-root + system-wide traits.

Traits computed per lateral root
--------------------------------
  * ``length_mm``                arc length along the path
  * ``angle_to_taproot_deg``     angle between this root's PCA and the taproot PCA (0–90°, folded)
  * ``angle_to_parent_deg``      angle between this root's PCA and the PCA
                                 of the PORTION of its parent at the attachment.
                                 ``None`` if ``parent_label is None`` (attached to taproot).
  * ``absolute_angle_deg``       angle between this root's PCA and +Z (0–90°, folded)
  * ``volume_mm3``               frustum-model volume from path + local radii
  * ``surface_area_mm2``         frustum-model lateral area + two end caps
  * ``mean_radius_mm``           average local radius along the path
  * ``n_nodes``                  number of path nodes
  * ``order``                    2 = secondary, 3 = tertiary, 4+ = higher
  * ``classification``           taproot-direct | split-extension | sibling-secondary | tertiary
  * ``parent_label``             None if attached to taproot, else the parent root's label

Summary traits
--------------
  * ``taproot_length_mm``, ``taproot_volume_mm3``, ``taproot_surface_area_mm2``, etc.
  * ``total_root_length_mm``     = taproot_length + Σ lateral lengths
  * ``total_root_volume_mm3``    = taproot_volume + Σ lateral volumes
  * ``total_root_surface_area_mm2``
  * ``n_secondary_roots`` / ``n_tertiary_roots`` / ``n_higher_order_roots``
  * ``depth_mm``, ``width_mm``   bbox extents of the full cloud

Entries with ``order == 0`` (incomplete ``unknown``-stopped tracks) are
excluded from the summary totals but still reported individually.

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

HEIGHT_AXIS = np.array([0.0, 0.0, 1.0])
TRAITS_DIR = Path("traits")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_result(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _get_path_array(obj):
    """Extract the path from either a classified-branch dict or a BranchPath."""
    if hasattr(obj, "path"):
        path = obj.path
    elif isinstance(obj, dict):
        path = obj.get("path", [])
    else:
        path = obj
    return np.array(path, dtype=float)


def _get(obj, key, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def path_length(path_arr):
    if len(path_arr) < 2:
        return 0.0
    diffs = np.diff(path_arr, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def path_pca_direction(path_arr):
    """First PCA component of a path as a unit vector."""
    if len(path_arr) < 2:
        return np.array([0.0, 0.0, 1.0])
    pca = PCA(n_components=1)
    pca.fit(path_arr)
    v = pca.components_[0]
    return v / (np.linalg.norm(v) + 1e-12)


def angle_between(v1, v2):
    """Angle in degrees between two vectors, folded to [0, 90]."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return float(np.degrees(np.arccos(abs(cos))))


# ---------------------------------------------------------------------------
# Volume / surface area via frustum (truncated-cone) segments
# ---------------------------------------------------------------------------

def estimate_tube_geometry(path_arr, all_points, tree):
    if len(path_arr) < 2:
        return 0.0, 0.0, []

    tangents = np.gradient(path_arr, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents = tangents / norms

    radii = np.array([
        _local_radius(pt, tangent, all_points, tree)
        for pt, tangent in zip(path_arr, tangents)
    ])

    volume = 0.0
    surface = 0.0
    for i in range(len(path_arr) - 1):
        seg_len = np.linalg.norm(path_arr[i + 1] - path_arr[i])
        r1, r2 = radii[i], radii[i + 1]
        volume  += np.pi * seg_len * (r1 ** 2 + r1 * r2 + r2 ** 2) / 3.0
        surface += np.pi * (r1 + r2) * np.sqrt(seg_len ** 2 + (r1 - r2) ** 2)
    surface += np.pi * radii[0] ** 2 + np.pi * radii[-1] ** 2
    return float(volume), float(surface), radii.tolist()


def _local_radius(point, tangent, all_points, tree, search_k=20):
    _, indices = tree.query(point, k=min(search_k, len(all_points)))
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
# Parent-angle helper
# ---------------------------------------------------------------------------

def _parent_local_direction(parent_path_arr, attach_idx, window=6):
    """PCA direction of the parent's path in a short window around the
    attachment index, used to measure a meaningful "branch angle" for
    tertiaries rather than the global parent PCA.
    """
    if len(parent_path_arr) < 2:
        return HEIGHT_AXIS
    if attach_idx is None:
        return path_pca_direction(parent_path_arr)
    lo = max(0, attach_idx - window)
    hi = min(len(parent_path_arr), attach_idx + window + 1)
    chunk = parent_path_arr[lo:hi]
    if len(chunk) < 2:
        return path_pca_direction(parent_path_arr)
    return path_pca_direction(chunk)


# ---------------------------------------------------------------------------
# Input normalization: accept new (classified) or legacy (branch_paths) pickles
# ---------------------------------------------------------------------------

def _collect_branches(data):
    """Return (branches, source) where branches is a list of dicts/dataclasses
    each exposing: path, order, classification, parent_label,
    attachment_index_on_parent.  ``source`` is 'classified' or 'legacy'.
    """
    cb = data.get("classified_branches")
    if cb:
        return cb, "classified"
    # Legacy fallback: synthesize minimal "classified" entries from branch_paths
    legacy = data.get("branch_paths", [])
    synth = []
    for i, bp in enumerate(legacy):
        synth.append({
            "label": _get(bp, "label", i),
            "order": 2,  # treat all legacy branches as secondaries
            "classification": "legacy",
            "parent_label": None,
            "attachment_point": None,
            "attachment_index_on_parent": None,
            "path": list(_get_path_array(bp)),
        })
    return synth, "legacy"


# ---------------------------------------------------------------------------
# Main trait computation
# ---------------------------------------------------------------------------

def compute_traits(data):
    all_points = data["all_points"]
    taproot_path = data.get("taproot_path")
    branches, source = _collect_branches(data)

    tree = KDTree(all_points)

    # --- Taproot ---
    tap_arr = np.array(taproot_path) if taproot_path is not None else np.empty((0, 3))
    tap_length = path_length(tap_arr)
    tap_pca = path_pca_direction(tap_arr) if len(tap_arr) >= 2 else HEIGHT_AXIS
    tap_abs_angle = angle_between(tap_pca, HEIGHT_AXIS)
    tap_vol, tap_sa, tap_radii = estimate_tube_geometry(tap_arr, all_points, tree)

    # Index branches by label so tertiaries can look up their parent
    by_label = {_get(b, "label", i): b for i, b in enumerate(branches)}

    branch_traits = []
    totals = {"length": 0.0, "volume": 0.0, "surface": 0.0}
    count_by_order = {}

    for i, b in enumerate(branches):
        path_arr = _get_path_array(b)
        order = int(_get(b, "order", 0) or 0)
        if len(path_arr) < 2 or order == 0:
            # Skip incomplete (unknown) branches from the totals
            continue

        length = path_length(path_arr)
        pca_dir = path_pca_direction(path_arr)
        abs_angle = angle_between(pca_dir, HEIGHT_AXIS)
        angle_tap = angle_between(pca_dir, tap_pca)
        vol, sa, radii = estimate_tube_geometry(path_arr, all_points, tree)

        # Angle to parent (None for roots attached to the taproot)
        parent_label = _get(b, "parent_label", None)
        angle_parent = None
        if parent_label is not None and parent_label in by_label:
            parent = by_label[parent_label]
            parent_arr = _get_path_array(parent)
            if len(parent_arr) >= 2:
                attach_idx = _get(b, "attachment_index_on_parent", None)
                par_dir = _parent_local_direction(parent_arr, attach_idx)
                angle_parent = angle_between(pca_dir, par_dir)

        branch_traits.append({
            "label": int(_get(b, "label", i)),
            "order": order,
            "classification": _get(b, "classification", "legacy"),
            "parent_label": parent_label,
            "length_mm": round(length, 3),
            "angle_to_taproot_deg": round(angle_tap, 2),
            "angle_to_parent_deg": (
                round(angle_parent, 2) if angle_parent is not None else None
            ),
            "absolute_angle_deg": round(abs_angle, 2),
            "volume_mm3": round(vol, 4),
            "surface_area_mm2": round(sa, 4),
            "n_nodes": int(len(path_arr)),
            "mean_radius_mm": round(float(np.mean(radii)), 4) if radii else 0.0,
        })

        totals["length"]  += length
        totals["volume"]  += vol
        totals["surface"] += sa
        count_by_order[order] = count_by_order.get(order, 0) + 1

    # System-wide bbox extents
    z_vals = all_points[:, 2]
    depth = float(np.max(z_vals) - np.min(z_vals))
    xy_extent = all_points[:, :2]
    width_x = float(np.max(xy_extent[:, 0]) - np.min(xy_extent[:, 0]))
    width_y = float(np.max(xy_extent[:, 1]) - np.min(xy_extent[:, 1]))
    width = max(width_x, width_y)

    summary = {
        "source": source,
        "taproot_length_mm": round(tap_length, 3),
        "taproot_absolute_angle_deg": round(tap_abs_angle, 2),
        "taproot_volume_mm3": round(tap_vol, 4),
        "taproot_surface_area_mm2": round(tap_sa, 4),
        "taproot_mean_radius_mm": (
            round(float(np.mean(tap_radii)), 4) if tap_radii else 0.0
        ),
        "total_root_length_mm": round(tap_length + totals["length"], 3),
        "total_root_volume_mm3": round(tap_vol + totals["volume"], 4),
        "total_root_surface_area_mm2": round(tap_sa + totals["surface"], 4),
        "n_secondary_roots": count_by_order.get(2, 0),
        "n_tertiary_roots": count_by_order.get(3, 0),
        "n_higher_order_roots": sum(
            n for o, n in count_by_order.items() if o >= 4
        ),
        "depth_mm": round(depth, 3),
        "width_mm": round(width, 3),
    }

    return summary, branch_traits


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_traits(summary, branch_traits):
    print("\n" + "=" * 72)
    print("ROOT SYSTEM TRAITS (frustum model)")
    print("=" * 72)

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")

    print("\n--- Taproot ---")
    print(f"  {'Length (mm)':35s}: {summary['taproot_length_mm']}")
    print(f"  {'Absolute angle (deg)':35s}: {summary['taproot_absolute_angle_deg']}")
    print(f"  {'Volume (mm³)':35s}: {summary['taproot_volume_mm3']}")
    print(f"  {'Surface area (mm²)':35s}: {summary['taproot_surface_area_mm2']}")
    print(f"  {'Mean radius (mm)':35s}: {summary['taproot_mean_radius_mm']}")

    if not branch_traits:
        return

    # Group by order for readability
    by_order = {}
    for bt in branch_traits:
        by_order.setdefault(bt["order"], []).append(bt)

    order_name = {2: "SECONDARIES", 3: "TERTIARIES"}
    for order in sorted(by_order):
        label = order_name.get(order, f"ORDER {order}")
        rows = sorted(by_order[order], key=lambda x: x["label"])
        print(f"\n--- {label} ({len(rows)}) ---")
        print(f"  {'ID':>4s} {'parent':>7s} {'class':>18s} "
              f"{'Length':>9s} {'A→tap':>7s} {'A→par':>7s} {'Abs':>7s} "
              f"{'Vol':>10s} {'SurfA':>10s} {'N':>4s} {'MeanR':>7s}")
        for bt in rows:
            p = "-" if bt["parent_label"] is None else str(bt["parent_label"])
            ap = "-" if bt["angle_to_parent_deg"] is None else f"{bt['angle_to_parent_deg']:.1f}"
            print(f"  {bt['label']:4d} {p:>7s} "
                  f"{bt['classification']:>18s} "
                  f"{bt['length_mm']:9.2f} "
                  f"{bt['angle_to_taproot_deg']:7.1f} "
                  f"{ap:>7s} "
                  f"{bt['absolute_angle_deg']:7.1f} "
                  f"{bt['volume_mm3']:10.4f} "
                  f"{bt['surface_area_mm2']:10.4f} "
                  f"{bt['n_nodes']:4d} "
                  f"{bt['mean_radius_mm']:7.4f}")


def save_traits(summary, branch_traits, sample_name):
    TRAITS_DIR.mkdir(exist_ok=True)

    json_path = TRAITS_DIR / f"{sample_name}.json"
    with open(json_path, "w") as f:
        json.dump(
            {"sample": sample_name,
             "summary": summary,
             "lateral_roots": branch_traits},
            f, indent=2,
        )
    print(f"\nJSON saved to {json_path}")

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
            by_order = {}
            for bt in branch_traits:
                by_order.setdefault(bt["order"], []).append(bt)
            for order in sorted(by_order):
                label = {2: "SECONDARIES", 3: "TERTIARIES"}.get(
                    order, f"ORDER {order}"
                )
                rows = sorted(by_order[order], key=lambda x: x["label"])
                f.write(f"{label} ({len(rows)})\n")
                f.write("-" * 80 + "\n")
                f.write(f"  {'ID':>4s} {'parent':>7s} {'class':>18s} "
                        f"{'Length':>9s} {'A→tap':>7s} {'A→par':>7s} {'Abs':>7s} "
                        f"{'Vol':>10s} {'SurfA':>10s} {'N':>4s} {'MeanR':>7s}\n")
                for bt in rows:
                    p = "-" if bt["parent_label"] is None else str(bt["parent_label"])
                    ap = ("-" if bt["angle_to_parent_deg"] is None
                          else f"{bt['angle_to_parent_deg']:.1f}")
                    f.write(f"  {bt['label']:4d} {p:>7s} "
                            f"{bt['classification']:>18s} "
                            f"{bt['length_mm']:9.2f} "
                            f"{bt['angle_to_taproot_deg']:7.1f} "
                            f"{ap:>7s} "
                            f"{bt['absolute_angle_deg']:7.1f} "
                            f"{bt['volume_mm3']:10.4f} "
                            f"{bt['surface_area_mm2']:10.4f} "
                            f"{bt['n_nodes']:4d} "
                            f"{bt['mean_radius_mm']:7.4f}\n")
                f.write("\n")

    print(f"Text table saved to {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute root-system traits")
    parser.add_argument("input", help="Path to result.pkl or sample name")
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        candidate = Path("skl_res") / (input_path + ".pkl")
        if candidate.exists():
            input_path = str(candidate)

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
