from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from app.core.config import ALPHA, RANDOM_SEED
from app.core.logger import get_logger
from app.services.propagation_service import calculate_propagation_metrics


logger = get_logger(__name__)


@dataclass
class AgentProperties:
    agent_type: str
    susceptibility: float
    sharing_probability: float
    credibility: float


class SimulationService:
    def __init__(self) -> None:
        self.alpha = ALPHA

    def _seed_from_text(self, text: str) -> int:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return (int(digest[:8], 16) + RANDOM_SEED) % (2**32)

    def _build_graph(
        self,
        text: str,
        bot_ratio_override: Optional[float] = None,
        influencer_strength: Optional[float] = None,
    ) -> nx.Graph:
        seed = self._seed_from_text(text)
        rng = np.random.default_rng(seed)
        node_count = int(np.clip(len(text.split()) * 3 + 20, 30, 120))
        attachment = max(1, min(4, node_count // 12))
        graph = nx.barabasi_albert_graph(node_count, attachment, seed=seed)

        for node in graph.nodes:
            degree = graph.degree[node]
            centrality = degree / max(1, node_count - 1)
            agent_type = self._assign_agent_type(rng.random(), centrality, bot_ratio_override=bot_ratio_override)
            properties = self._assign_properties(agent_type, rng, influencer_strength=influencer_strength)
            graph.nodes[node].update(
                {
                    "agent_type": properties.agent_type,
                    "susceptibility": properties.susceptibility,
                    "sharing_probability": properties.sharing_probability,
                    "credibility": properties.credibility,
                }
            )

        return graph

    def _assign_agent_type(self, roll: float, centrality: float, bot_ratio_override: Optional[float] = None) -> str:
        bot_threshold = float(np.clip(bot_ratio_override, 0.0, 1.0)) if bot_ratio_override is not None else 0.18
        if centrality > 0.18 or roll > 0.82:
            return "influencer"
        if roll < bot_threshold:
            return "bot"
        if roll < 0.36:
            return "skeptic"
        return "normal"

    def _assign_properties(
        self,
        agent_type: str,
        rng: np.random.Generator,
        influencer_strength: Optional[float] = None,
    ) -> AgentProperties:
        if agent_type == "bot":
            return AgentProperties(agent_type, susceptibility=0.85, sharing_probability=0.9, credibility=0.2)
        if agent_type == "influencer":
            strength = float(np.clip(influencer_strength, 0.0, 1.0)) if influencer_strength is not None else 0.7
            boosted_sharing = float(np.clip(0.55 + 0.4 * strength, 0.4, 0.99))
            boosted_credibility = float(np.clip(0.55 + 0.4 * strength, 0.4, 0.99))
            return AgentProperties(
                agent_type,
                susceptibility=0.7,
                sharing_probability=boosted_sharing,
                credibility=boosted_credibility,
            )
        if agent_type == "skeptic":
            return AgentProperties(agent_type, susceptibility=0.2, sharing_probability=0.15, credibility=0.95)
        return AgentProperties(
            agent_type,
            susceptibility=float(rng.uniform(0.35, 0.65)),
            sharing_probability=float(rng.uniform(0.25, 0.55)),
            credibility=float(rng.uniform(0.45, 0.75)),
        )

    def _initial_sources(self, graph: nx.Graph, seed_nodes: Optional[int] = None) -> List[int]:
        ranked_nodes = sorted(graph.degree, key=lambda item: item[1], reverse=True)
        source_count = int(np.clip(seed_nodes, 1, 50)) if seed_nodes is not None else 3
        sources = [node for node, _ in ranked_nodes[:source_count]]
        if not sources:
            sources = [0]
        return sources

    def _simulate_spread(self, graph: nx.Graph, steps: int, seed_nodes: Optional[int] = None) -> Tuple[Dict[int, int], List[int]]:
        exposure_steps: Dict[int, int] = {}
        frontier = self._initial_sources(graph, seed_nodes=seed_nodes)
        active_frontier = list(frontier)

        for source in frontier:
            exposure_steps[source] = 0

        step_history: List[int] = [len(frontier)]

        for step in range(1, steps + 1):
            next_frontier: List[int] = []
            for node in active_frontier:
                node_data = graph.nodes[node]
                for neighbor in graph.neighbors(node):
                    if neighbor in exposure_steps:
                        continue
                    neighbor_data = graph.nodes[neighbor]
                    share_chance = min(1.0, self.alpha * neighbor_data["susceptibility"] * neighbor_data["credibility"])
                    if np.random.default_rng(self._seed_from_text(f"{node}-{neighbor}-{step}")).random() < share_chance:
                        exposure_steps[neighbor] = step
                        next_frontier.append(neighbor)
            step_history.append(len(next_frontier))
            if not next_frontier:
                break
            active_frontier = next_frontier

        return exposure_steps, step_history

    def simulate(
        self,
        text: str,
        steps: int,
        bot_ratio_override: Optional[float] = None,
        seed_nodes: Optional[int] = None,
        influencer_strength: Optional[float] = None,
    ) -> Dict[str, Any]:
        logger.info("Running simulation for steps=%s", steps)
        graph = self._build_graph(
            text,
            bot_ratio_override=bot_ratio_override,
            influencer_strength=influencer_strength,
        )
        exposure_steps, step_history = self._simulate_spread(graph, steps, seed_nodes=seed_nodes)
        propagation_metrics = calculate_propagation_metrics(graph, exposure_steps, step_history)

        agent_counts = {agent_type: 0 for agent_type in ["normal", "influencer", "bot", "skeptic"]}
        for _, data in graph.nodes(data=True):
            agent_counts[data["agent_type"]] += 1

        total_nodes = graph.number_of_nodes()
        graph_stats = {
            "nodes": total_nodes,
            "edges": graph.number_of_edges(),
            "density": round(nx.density(graph), 4),
            "average_degree": round(sum(dict(graph.degree()).values()) / max(1, total_nodes), 4),
            "agent_counts": agent_counts,
            "bot_ratio": round(agent_counts["bot"] / max(1, total_nodes), 4),
            "influencer_ratio": round(agent_counts["influencer"] / max(1, total_nodes), 4),
            "skeptic_ratio": round(agent_counts["skeptic"] / max(1, total_nodes), 4),
        }

        return {
            "graph": graph,
            "graph_stats": graph_stats,
            "propagation_metrics": propagation_metrics,
        }


_simulation_service = SimulationService()


def simulate_network(
    text: str,
    steps: int,
    bot_ratio_override: Optional[float] = None,
    seed_nodes: Optional[int] = None,
    influencer_strength: Optional[float] = None,
) -> Dict[str, Any]:
    return _simulation_service.simulate(
        text=text,
        steps=steps,
        bot_ratio_override=bot_ratio_override,
        seed_nodes=seed_nodes,
        influencer_strength=influencer_strength,
    )