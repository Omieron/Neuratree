from __future__ import annotations

from dnt.config import DNTConfig
from dnt.core.models import NeuronEdge, NeuronNode
from dnt.core.tree import NeuronTree


class GHSOMGrower:
    """
    Simplified GHSOM (Growing Hierarchical Self-Organizing Map).
    Monitors each node's quantization_error; when it exceeds
    parent_QE * effective_tau1, the node expands into two child nodes.
    Phase 3: tau1 is adaptive — it scales with tree size to prevent
    runaway growth as the tree matures.
    """

    def __init__(self, config: DNTConfig) -> None:
        self._config = config

    def grow_if_needed(self, tree: NeuronTree) -> int:
        """
        Scan all nodes and expand overloaded ones.
        Returns the number of new nodes created.
        """
        candidates = tree.all_nodes()
        tau1 = self._effective_tau1(tree)
        new_nodes = 0
        for node in candidates:
            if self._should_grow(tree, node, tau1):
                new_nodes += self._expand(tree, node)
        return new_nodes

    # ------------------------------------------------------------------
    # Adaptive tau1 (Phase 3)
    # ------------------------------------------------------------------

    def _effective_tau1(self, tree: NeuronTree) -> float:
        n = tree.node_count
        return self._config.tau1 * (1.0 + n * 0.01)

    # ------------------------------------------------------------------
    # Growth decision
    # ------------------------------------------------------------------

    def _should_grow(self, tree: NeuronTree, node: NeuronNode, tau1: float) -> bool:
        if node.level >= self._config.max_depth:
            return False
        if node.quantization_error < self._config.tau2:
            return False
        parent_qe = self._parent_qe(tree, node)
        return node.quantization_error > parent_qe * tau1

    def _parent_qe(self, tree: NeuronTree, node: NeuronNode) -> float:
        """Root nodes use 1.0 as their reference QE."""
        if node.level == 0:
            return 1.0
        parents = tree.get_parents(node.id)
        if not parents:
            return 1.0
        return max(p.quantization_error for p in parents)

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand(self, tree: NeuronTree, node: NeuronNode) -> int:
        """
        Create two child nodes that inherit half the parent's QE.
        The parent's QE is reset so it won't trigger growth again immediately.
        """
        child_a = NeuronNode(
            label=f"{node.label}:a",
            level=node.level + 1,
            quantization_error=node.quantization_error * 0.5,
            source_ids=list(node.source_ids),
        )
        child_b = NeuronNode(
            label=f"{node.label}:b",
            level=node.level + 1,
            quantization_error=node.quantization_error * 0.5,
            source_ids=list(node.source_ids),
        )
        tree.add_node(child_a)
        tree.add_node(child_b)
        tree.add_edge(
            NeuronEdge(source=node.id, target=child_a.id, relation="expands_to", weight=0.5)
        )
        tree.add_edge(
            NeuronEdge(source=node.id, target=child_b.id, relation="expands_to", weight=0.5)
        )
        node.quantization_error *= 0.1
        return 2
