from __future__ import annotations

from app.services.simulation_service import simulate_network


SAMPLE_TEXT = "Breaking: unverified claims are spreading rapidly across social media platforms today"


def test_simulate_network_returns_expected_shape() -> None:
    result = simulate_network(text=SAMPLE_TEXT, steps=8)

    assert set(result.keys()) == {"graph", "graph_stats", "propagation_metrics"}

    graph_stats = result["graph_stats"]
    for key in ("nodes", "edges", "density", "average_degree", "agent_counts", "bot_ratio"):
        assert key in graph_stats

    agent_counts = graph_stats["agent_counts"]
    assert sum(agent_counts.values()) == graph_stats["nodes"]

    propagation_metrics = result["propagation_metrics"]
    assert propagation_metrics["depth"] <= 8
    assert propagation_metrics["reach"] <= graph_stats["nodes"]


def test_simulate_network_is_deterministic_for_same_text() -> None:
    first = simulate_network(text=SAMPLE_TEXT, steps=6)
    second = simulate_network(text=SAMPLE_TEXT, steps=6)

    assert first["graph_stats"] == second["graph_stats"]
    assert first["propagation_metrics"] == second["propagation_metrics"]


def test_bot_ratio_override_shifts_agent_composition() -> None:
    low = simulate_network(text=SAMPLE_TEXT, steps=5, bot_ratio_override=0.0)
    high = simulate_network(text=SAMPLE_TEXT, steps=5, bot_ratio_override=0.9)

    assert high["graph_stats"]["agent_counts"]["bot"] >= low["graph_stats"]["agent_counts"]["bot"]


def test_seed_nodes_controls_initial_frontier_size() -> None:
    result = simulate_network(text=SAMPLE_TEXT, steps=1, seed_nodes=7)
    assert result["propagation_metrics"]["velocity_by_step"][0] == 7
