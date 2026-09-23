from __future__ import annotations

import networkx as nx

from app.services.propagation_service import calculate_propagation_metrics


def test_empty_exposure_returns_zeroed_metrics() -> None:
    graph = nx.path_graph(5)
    metrics = calculate_propagation_metrics(graph, exposure_steps={}, step_history=[0])

    assert metrics["reach"] == 0
    assert metrics["depth"] == 0
    assert metrics["velocity"] == 0.0
    assert metrics["velocity_by_step"] == [0]


def test_reach_and_depth_reflect_exposure_steps() -> None:
    graph = nx.path_graph(5)
    exposure_steps = {0: 0, 1: 1, 2: 1, 3: 2}
    step_history = [1, 2, 1]

    metrics = calculate_propagation_metrics(graph, exposure_steps, step_history)

    assert metrics["reach"] == 4
    assert metrics["depth"] == 2
    assert metrics["velocity"] == round(sum(step_history[1:]) / (len(step_history) - 1), 4)


def test_echo_chamber_density_is_bounded() -> None:
    graph = nx.karate_club_graph()
    exposure_steps = {node: 0 for node in graph.nodes}
    metrics = calculate_propagation_metrics(graph, exposure_steps, step_history=[len(graph)])

    assert 0.0 <= metrics["echo_chamber_density"] <= 1.0


def test_echo_chamber_density_zero_for_edgeless_graph() -> None:
    graph = nx.empty_graph(5)
    metrics = calculate_propagation_metrics(graph, exposure_steps={0: 0}, step_history=[1])

    assert metrics["echo_chamber_density"] == 0.0
