"""KNN graph construction, density-based weight adjustment, and path finding."""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from typing import List, Tuple


def compute_avg_distance(points: np.ndarray) -> float:
    """Average nearest-neighbor distance across all points."""
    tree = KDTree(points)
    distances, _ = tree.query(points, k=2)
    return float(np.mean(distances[:, 1]))


def build_knn_graph(points: np.ndarray, k: int = 10) -> nx.Graph:
    """Build an undirected KNN graph where edge weights are Euclidean distances."""
    tree = KDTree(points)
    graph = nx.Graph()

    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=k + 1)
        for j in range(1, k + 1):
            graph.add_edge(i, indices[j], weight=distances[j])

    return graph


def adjust_weights_by_density(
    graph: nx.Graph,
    points: np.ndarray,
    radius: float,
    reverse: bool = False,
) -> nx.Graph:
    """Adjust edge weights by local point density.

    Default (reverse=False): multiply weights so shortest paths favor
    high-density (interior) regions. reverse=True does the opposite.
    """
    tree = KDTree(points)
    densities = np.array(
        [len(tree.query_ball_point(p, r=radius)) for p in points]
    )
    max_density = densities.max()
    # Normalized inverse density: 0 = densest, 1 = sparsest
    inv_density = 1.0 - densities / max_density

    for u, v, d in graph.edges(data=True):
        factor = 1.0 + (inv_density[u] + inv_density[v]) / 2.0
        if reverse:
            d["weight"] /= factor
        else:
            d["weight"] *= factor

    return graph


def find_nearest_node(points: np.ndarray, query: np.ndarray) -> int:
    """Index of the point closest to *query*."""
    tree = KDTree(points)
    _, idx = tree.query(query)
    return int(idx)


def find_shortest_path(
    graph: nx.Graph, source: int, target: int
) -> List[int]:
    """Shortest weighted path between two nodes."""
    return nx.shortest_path(graph, source=source, target=target, weight="weight")


def find_shortest_path_with_weights(
    graph: nx.Graph, source: int, target: int
) -> Tuple[List[int], List[float], float]:
    """Shortest path plus per-edge weights and total weight."""
    path = find_shortest_path(graph, source, target)
    weights = [graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)]
    return path, weights, sum(weights)
