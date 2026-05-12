from __future__ import annotations

import time
from typing import List

from dnt.config import DNTConfig
from dnt.core.models import NeuronEdge, NeuronNode, Triplet
from dnt.core.tree import NeuronTree

# Differential plasticity: preference updates faster than facts
_LR_MULTIPLIERS = {"preference": 2.0, "rule": 1.5, "fact": 1.0}


class HebbianLearner:
    """
    Hebbian plasticity engine.
    LTP: strengthen edges between co-activated nodes.
    LTD: decay all edges every consolidation cycle.
    """

    def __init__(self, config: DNTConfig) -> None:
        self._config = config

    def update(self, tree: NeuronTree, triplets: List[Triplet]) -> None:
        """Apply one full LTD pass then LTP for each triplet."""
        self._apply_ltd(tree)
        for triplet in triplets:
            self._apply_ltp(tree, triplet)

    # ------------------------------------------------------------------
    # LTP
    # ------------------------------------------------------------------

    def _apply_ltp(self, tree: NeuronTree, triplet: Triplet) -> None:
        subject_node = self._get_or_create_node(tree, triplet.subject)
        object_node = self._get_or_create_node(tree, triplet.object)

        lr_mult = _LR_MULTIPLIERS.get(triplet.logic_type, 1.0)
        delta = self._config.hebbian_lr * lr_mult * triplet.confidence

        if tree.has_edge(subject_node.id, object_node.id):
            current = tree.get_edge_weight(subject_node.id, object_node.id) or 0.1
            tree.update_edge_weight(
                subject_node.id, object_node.id, min(1.0, current + delta)
            )
        else:
            tree.add_edge(
                NeuronEdge(
                    source=subject_node.id,
                    target=object_node.id,
                    relation=triplet.predicate,
                    weight=min(1.0, 0.1 + delta),
                    last_activated=time.time(),
                )
            )

        # activation increases quantization error, driving GHSOM growth
        subject_node.quantization_error += delta * 0.5
        object_node.quantization_error += delta * 0.5

    # ------------------------------------------------------------------
    # LTD
    # ------------------------------------------------------------------

    def _apply_ltd(self, tree: NeuronTree) -> None:
        tree.decay_all_edges(self._config.decay_factor)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _get_or_create_node(self, tree: NeuronTree, label: str) -> NeuronNode:
        existing = tree.find_node_by_label(label)
        if existing:
            return existing
        node = NeuronNode(label=label)
        tree.add_node(node)
        return node
