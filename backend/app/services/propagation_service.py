from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from app.core.logger import get_logger


logger = get_logger(__name__)


def _community_map(graph: nx.Graph) -> Dict[int, int]:
    if graph.number_of_nodes() == 0:
        return {}

    try:
        communities = list(greedy_modularity_communities(graph))
    except Exception:
        communities = [set(component) for component in nx.connected_components(graph)]

    mapping: Dict[int, int] = {}
    for index, community in enumerate(communities):
        for node in community:
            mapping[node] = index
    return mapping


def _echo_chamber_density(graph: nx.Graph) -> float:
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return 0.0

    community_map = _community_map(graph)
    intra_cluster_edges = 0
    for left, right in graph.edges:
        if community_map.get(left) == community_map.get(right):
            intra_cluster_edges += 1
    return round(intra_cluster_edges / total_edges, 4)


def calculate_propagation_metrics(
    graph: nx.Graph,
    exposure_steps: Dict[int, int],
    step_history: List[int],
) -> Dict[str, Any]:
    if not exposure_steps:
        return {
            "reach": 0,
            "depth": 0,
            "velocity": 0.0,
            "velocity_by_step": step_history,
            "echo_chamber_density": _echo_chamber_density(graph),
        }

    depth = max(exposure_steps.values())
    reach = len(exposure_steps)
    velocity = round(sum(step_history[1:]) / max(1, len(step_history) - 1), 4)
    echo_density = _echo_chamber_density(graph)

    logger.info("Propagation metrics calculated: reach=%s depth=%s", reach, depth)

    return {
        "reach": reach,
        "depth": depth,
        "velocity": velocity,
        "velocity_by_step": step_history,
        "echo_chamber_density": echo_density,
    }