"""Visualization helpers using Open3D.

Color scheme: the root system point cloud is always shown in neutral gray.
Only overlays (paths, markers, arrows) use accent colors to highlight
specific structures without visual clutter.
"""

from typing import Dict, List, Optional

import numpy as np
import open3d as o3d


MARKER_SIZE = 0.005

# Neutral palette
COLOR_CLOUD = [0.7, 0.7, 0.7]       # light gray for the full point cloud
COLOR_MAIN_ROOT = [1.0, 0.1, 0.1]   # medium gray for the main root volume
COLOR_BRANCH = [0.6, 0.6, 0.6]      # gray for branch clusters

# Accent colors (used sparingly for overlays only)
COLOR_PATH = [0.2, 0.2, 0.2]        # dark gray for path lines
COLOR_PATH_NODE = [0.3, 0.3, 0.3]   # dark gray for path node spheres
COLOR_START = [0.1, 0.1, 0.1]       # near-black for start point
COLOR_ARROW = [0.4, 0.4, 0.4]       # gray for direction arrows
COLOR_BRANCH_ORIGIN = [0.5, 0.8, 1.0]  # light blue for branch starting points
COLOR_SHELL = [0.85, 0.85, 0.75]      # warm light gray for the hollow cylinder shell
COLOR_SHELL_CLUSTER = [0.9, 0.6, 0.2] # orange for shell clusters


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def create_sphere(
    center: np.ndarray, radius: float = 0.005, color: list = None, resolution: int = 20
) -> o3d.geometry.TriangleMesh:
    """A solid colored sphere."""
    color = color or COLOR_PATH_NODE
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sphere.translate(center - np.array(sphere.get_center()))
    sphere.paint_uniform_color(color[:3])
    return sphere


def create_wireframe_sphere(
    center: np.ndarray, radius: float, color: list = None, resolution: int = 10
) -> o3d.geometry.LineSet:
    """A wireframe sphere."""
    color = color or [0, 0, 0]
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    mesh.translate(center)
    edges = set()
    for tri in mesh.triangles:
        for i in range(3):
            a, b = sorted([tri[i], tri[(i + 1) % 3]])
            edges.add((a, b))
    lines = list(edges)
    ls = o3d.geometry.LineSet(
        points=mesh.vertices,
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return ls


def create_path_lineset(
    path_points: np.ndarray, color: list = None
) -> o3d.geometry.LineSet:
    """A line set connecting sequential points."""
    color = color or COLOR_PATH
    n = len(path_points)
    if n < 2:
        return o3d.geometry.LineSet()
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(path_points),
        lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(n - 1)]),
    )
    ls.colors = o3d.utility.Vector3dVector([color for _ in range(n - 1)])
    return ls


def create_arrow(
    origin: np.ndarray, direction: np.ndarray, length: float = 0.05, color: list = None
) -> o3d.geometry.TriangleMesh:
    """An arrow pointing in *direction* from *origin*."""
    color = color or COLOR_ARROW
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=length * 0.1,
        cone_radius=length * 0.2,
        cylinder_height=length * 0.5,
        cone_height=length * 0.5,
    )
    arrow.translate(origin - np.array(arrow.get_center()))
    _rotate_arrow(arrow, direction)
    arrow.paint_uniform_color(color[:3])
    return arrow


def _rotate_arrow(arrow, direction):
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    z = np.array([0, 0, 1.0])
    cross = np.cross(z, d)
    if np.linalg.norm(cross) < 1e-10:
        if np.dot(z, d) < 0:
            rot = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            arrow.rotate(rot, center=arrow.get_center())
        return
    angle = np.arccos(np.clip(np.dot(z, d), -1, 1))
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(cross * angle)
    arrow.rotate(rot, center=arrow.get_center())


# ---------------------------------------------------------------------------
# High-level viewers
# ---------------------------------------------------------------------------

def show_point_cloud(
    pcd: o3d.geometry.PointCloud, title: str = "Point Cloud"
) -> None:
    display = o3d.geometry.PointCloud(pcd)
    display.paint_uniform_color(COLOR_CLOUD)
    o3d.visualization.draw_geometries([display], window_name=title)


def show_path_on_cloud(
    pcd: o3d.geometry.PointCloud,
    path_points: np.ndarray,
    title: str = "Main Root Path",
) -> None:
    """Show the point cloud (gray) with the main path overlaid (dark gray)."""
    bg = o3d.geometry.PointCloud(pcd)
    bg.paint_uniform_color(COLOR_CLOUD)

    line_set = create_path_lineset(path_points, color=COLOR_PATH)
    spheres = [
        create_sphere(
            pt, radius=MARKER_SIZE,
            color=COLOR_START if i == 0 else COLOR_PATH_NODE,
        )
        for i, pt in enumerate(path_points)
    ]
    o3d.visualization.draw_geometries(
        [bg, line_set, *spheres], window_name=title
    )


def show_main_root(
    pcd: o3d.geometry.PointCloud,
    main_root_points: np.ndarray,
    title: str = "Main Root Volume",
) -> None:
    """Show main root (medium gray) against the rest (light gray)."""
    root_pcd = o3d.geometry.PointCloud()
    root_pcd.points = o3d.utility.Vector3dVector(main_root_points)
    root_pcd.paint_uniform_color(COLOR_MAIN_ROOT)

    bg = o3d.geometry.PointCloud(pcd)
    bg.paint_uniform_color(COLOR_CLOUD)

    o3d.visualization.draw_geometries([bg, root_pcd], window_name=title)


def show_branches(
    pcd: o3d.geometry.PointCloud,
    main_root_points: np.ndarray,
    branch_clusters: Dict[int, np.ndarray],
    branch_directions: Optional[Dict[int, np.ndarray]] = None,
    title: str = "Branches",
) -> None:
    """Show main root and branch clusters in shades of gray,
    with optional direction arrows."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)

    root_pcd = o3d.geometry.PointCloud()
    root_pcd.points = o3d.utility.Vector3dVector(main_root_points)
    root_pcd.paint_uniform_color(COLOR_MAIN_ROOT)
    vis.add_geometry(root_pcd)

    # Alternate between two gray tones so adjacent clusters are distinguishable
    gray_tones = [[0.55, 0.55, 0.55], [0.65, 0.65, 0.65]]

    for idx, (label, pts) in enumerate(branch_clusters.items()):
        color = gray_tones[idx % len(gray_tones)]
        bpcd = o3d.geometry.PointCloud()
        bpcd.points = o3d.utility.Vector3dVector(pts)
        bpcd.paint_uniform_color(color)
        vis.add_geometry(bpcd)

        centroid = np.mean(pts, axis=0)

        # Light blue sphere at the branch starting point
        origin_sphere = create_sphere(
            centroid, radius=MARKER_SIZE * 0.5, color=COLOR_BRANCH_ORIGIN
        )
        vis.add_geometry(origin_sphere)

        if branch_directions and label in branch_directions:
            arr = create_arrow(centroid, branch_directions[label], color=COLOR_ARROW)
            vis.add_geometry(arr)

    vis.run()
    vis.destroy_window()


def show_branch_paths(
    points: np.ndarray,
    paths: List[List[np.ndarray]],
    surrounding_indices: Optional[List[List[int]]] = None,
    taproot_path: Optional[np.ndarray] = None,
    title: str = "Branch Paths",
) -> None:
    """Show all branch paths on the point cloud with the taproot in red."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(COLOR_CLOUD)

    geometries = []

    # Taproot path in red
    if taproot_path is not None and len(taproot_path) >= 2:
        geometries.append(create_path_lineset(taproot_path, color=[1, 0, 0]))
        geometries.extend([
            create_sphere(pt, radius=MARKER_SIZE, color=[1, 0, 0])
            for pt in taproot_path
        ])

    # Shade surrounding points slightly darker per branch
    if surrounding_indices:
        colors = np.asarray(pcd.colors).copy()
        for idx, s_idx in enumerate(surrounding_indices):
            if s_idx:
                tone = 0.45 + 0.05 * (idx % 4)  # subtle variation: 0.45-0.60
                valid = [i for i in s_idx if i < len(colors)]
                colors[valid] = [tone, tone, tone]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Distinct color per path so trajectories are easy to follow
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")

    for idx, path in enumerate(paths):
        if len(path) < 2:
            continue
        path_arr = np.array(path)
        color = list(cmap(idx % 20)[:3])

        geometries.append(create_path_lineset(path_arr, color=color))
        geometries.extend([
            create_sphere(pt, radius=MARKER_SIZE, color=color)
            for pt in path_arr
        ])
        # Slightly larger sphere at start of each branch
        geometries.append(
            create_sphere(path_arr[0], radius=MARKER_SIZE * 1.5, color=color)
        )

    geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries, window_name=title)


def show_shell_analysis(
    pcd: o3d.geometry.PointCloud,
    path_points: np.ndarray,
    radii: np.ndarray,
    shell_points: Optional[np.ndarray] = None,
    shell_cluster_labels: Optional[np.ndarray] = None,
    branch_origins: Optional[Dict] = None,
    shell_inner_factor: float = 1.1,
    shell_outer_factor: float = 4.0,
    title: str = "Shell Analysis",
) -> None:
    """Visualize the hollow cylinder shell, shell clusters, and detected
    branch starting points.

    Layers:
      - Gray: full root system point cloud
      - Warm gray: all shell points (the hollow cylinder)
      - Orange: shell clusters (per-ring HDBSCAN clusters)
      - Light blue spheres: detected branch starting points
      - Dark gray arrows: initial growth directions
      - Wireframe: inner/outer shell boundaries (sampled rings)
    """
    import matplotlib.pyplot as plt

    geometries = []

    # Background point cloud
    bg = o3d.geometry.PointCloud(pcd)
    bg.paint_uniform_color(COLOR_CLOUD)
    geometries.append(bg)

    # Centerline path
    geometries.append(create_path_lineset(path_points, color=COLOR_PATH))

    # Shell wireframe rings at sampled path points
    n_rings_to_show = min(30, len(path_points))
    ring_step = max(1, len(path_points) // n_rings_to_show)
    tangents = np.gradient(path_points, axis=0)
    tnorms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tnorms[tnorms == 0] = 1.0
    tangents = tangents / tnorms

    for i in range(0, len(path_points), ring_step):
        center = path_points[i]
        r_inner = radii[i] * shell_inner_factor
        r_outer = radii[i] * shell_outer_factor
        tangent = tangents[i]

        for r, alpha in [(r_inner, 0.3), (r_outer, 0.15)]:
            ring = _make_ring(center, tangent, r, n_segments=24)
            gray_val = 0.5 + alpha
            ring_ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(ring),
                lines=o3d.utility.Vector2iVector(
                    [[j, (j + 1) % len(ring)] for j in range(len(ring))]
                ),
            )
            ring_ls.colors = o3d.utility.Vector3dVector(
                [[gray_val, gray_val, gray_val]] * len(ring)
            )
            geometries.append(ring_ls)

    # Shell points
    if shell_points is not None and len(shell_points) > 0:
        shell_pcd = o3d.geometry.PointCloud()
        shell_pcd.points = o3d.utility.Vector3dVector(shell_points)

        if shell_cluster_labels is not None:
            # Color: unclustered shell in warm gray, clusters in distinct colors
            cmap = plt.get_cmap("tab20")
            colors = np.tile(COLOR_SHELL, (len(shell_points), 1))
            unique_labels = set(shell_cluster_labels)
            unique_labels.discard(-1)
            for label in unique_labels:
                mask = shell_cluster_labels == label
                c = cmap(label % 20)[:3]
                colors[mask] = c
            shell_pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            shell_pcd.paint_uniform_color(COLOR_SHELL)

        geometries.append(shell_pcd)

    # Branch starting points as light blue spheres + direction arrows
    if branch_origins:
        for label, origin in branch_origins.items():
            centroid = origin.points.mean(axis=0)

            # Light blue sphere
            sphere = create_sphere(
                centroid, radius=MARKER_SIZE * 0.67, color=COLOR_BRANCH_ORIGIN
            )
            geometries.append(sphere)

            # Direction arrow
            arr = create_arrow(
                centroid, origin.direction,
                length=0.03, color=COLOR_ARROW,
            )
            geometries.append(arr)

    o3d.visualization.draw_geometries(geometries, window_name=title)


def _make_ring(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    n_segments: int = 24,
) -> np.ndarray:
    """Generate ring points in 3D centered at *center* perpendicular to *normal*."""
    # Build two orthogonal vectors in the plane
    normal = normal / np.linalg.norm(normal)
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    ring = np.array([
        center + radius * (np.cos(a) * u + np.sin(a) * v)
        for a in angles
    ])
    return ring
