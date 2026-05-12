from __future__ import annotations

import re
import time
from typing import Dict, List, Optional
from uuid import UUID

import networkx as nx

from dnt.core.models import NeuronEdge, NeuronNode


class NeuronTree:
    """Neuron tree built on top of a NetworkX DiGraph."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        # id -> NeuronNode for fast lookups
        self._nodes: Dict[UUID, NeuronNode] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: NeuronNode) -> None:
        self._nodes[node.id] = node
        self._graph.add_node(str(node.id), **node.model_dump())

    def add_edge(self, edge: NeuronEdge) -> None:
        self._graph.add_edge(
            str(edge.source),
            str(edge.target),
            relation=edge.relation,
            weight=edge.weight,
            last_activated=edge.last_activated,
        )

    def update_edge_weight(
        self, source_id: UUID, target_id: UUID, weight: float
    ) -> None:
        u, v = str(source_id), str(target_id)
        if self._graph.has_edge(u, v):
            self._graph.edges[u, v]["weight"] = weight
            self._graph.edges[u, v]["last_activated"] = time.time()

    def decay_all_edges(self, decay_factor: float) -> None:
        for _, _, data in self._graph.edges(data=True):
            data["weight"] = max(0.0, data.get("weight", 0.1) * decay_factor)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_node(self, node_id: UUID) -> Optional[NeuronNode]:
        return self._nodes.get(node_id)

    def find_node_by_label(self, label: str) -> Optional[NeuronNode]:
        label_lower = label.lower()
        for node in self._nodes.values():
            if node.label.lower() == label_lower:
                return node
        return None

    def has_edge(self, source_id: UUID, target_id: UUID) -> bool:
        return self._graph.has_edge(str(source_id), str(target_id))

    def get_edge_weight(self, source_id: UUID, target_id: UUID) -> Optional[float]:
        u, v = str(source_id), str(target_id)
        if self._graph.has_edge(u, v):
            return float(self._graph.edges[u, v].get("weight", 0.1))
        return None

    def get_parents(self, node_id: UUID) -> List[NeuronNode]:
        parents: List[NeuronNode] = []
        for pred in self._graph.predecessors(str(node_id)):
            node = self._nodes.get(UUID(pred))
            if node:
                parents.append(node)
        return parents

    def all_nodes(self) -> List[NeuronNode]:
        return list(self._nodes.values())

    def get_active_neighbors(
        self, node_id: UUID, threshold: float
    ) -> List[tuple[NeuronNode, str]]:
        """Return (neighbor_node, relation) pairs for edges above threshold (both directions)."""
        results: List[tuple[NeuronNode, str]] = []
        nid = str(node_id)
        for neighbor in self._graph.successors(nid):
            data = self._graph.edges[nid, neighbor]
            if data.get("weight", 0) >= threshold:
                node = self._nodes.get(UUID(neighbor))
                if node:
                    results.append((node, data.get("relation", "related_to")))
        for neighbor in self._graph.predecessors(nid):
            data = self._graph.edges[neighbor, nid]
            if data.get("weight", 0) >= threshold:
                node = self._nodes.get(UUID(neighbor))
                if node:
                    results.append((node, f"←{data.get('relation', 'related_to')}"))
        return results

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def hop_traversal(
        self,
        query: str,
        hop_limit: int = 3,
        activation_threshold: float = 0.3,
    ) -> List[NeuronNode]:
        """
        Phase 1: traversal using simple string matching instead of embeddings.
        Finds seed nodes whose labels overlap with query tokens, then expands
        to neighbors up to hop_limit steps.
        """
        if not self._nodes:
            return []

        tokens = set(re.findall(r'\b\w+\b', query.lower()))
        seed_ids: List[str] = []

        for node in self._nodes.values():
            label_tokens = set(node.label.lower().split())
            if tokens & label_tokens:  # at least one token in common
                seed_ids.append(str(node.id))

        if not seed_ids:
            return []

        visited: set[str] = set()
        frontier = seed_ids[:3]  # cap seeds at 3

        for _ in range(hop_limit):
            next_frontier: List[str] = []
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                # follow edges in both directions so queries on target nodes
                # still surface their source neighbors
                for neighbor in self._graph.successors(nid):
                    edge_data = self._graph.edges[nid, neighbor]
                    if edge_data.get("weight", 0) >= activation_threshold:
                        if neighbor not in visited:
                            next_frontier.append(neighbor)
                for neighbor in self._graph.predecessors(nid):
                    edge_data = self._graph.edges[neighbor, nid]
                    if edge_data.get("weight", 0) >= activation_threshold:
                        if neighbor not in visited:
                            next_frontier.append(neighbor)
            frontier = next_frontier

        result: List[NeuronNode] = []
        for nid in visited:
            node = self._nodes.get(UUID(nid))
            if node:
                result.append(node)
        return result

    # ------------------------------------------------------------------
    # Stats / serialization
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def to_dict(self) -> dict:
        return {
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            "edges": [
                {"source": str(u), "target": str(v), **d}
                for u, v, d in self._graph.edges(data=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NeuronTree":
        tree = cls()
        for node_data in data.get("nodes", []):
            tree.add_node(NeuronNode(**node_data))
        for edge_data in data.get("edges", []):
            edge_data = dict(edge_data)
            source = UUID(edge_data.pop("source"))
            target = UUID(edge_data.pop("target"))
            tree.add_edge(NeuronEdge(source=source, target=target, **edge_data))
        return tree
