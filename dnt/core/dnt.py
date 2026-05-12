from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

from dnt.config import DNTConfig
from dnt.core.buffer import WorkingMemoryBuffer
from dnt.core.models import AtomicObservation, DNTStats
from dnt.core.tree import NeuronTree
from dnt.learning.ghsom import GHSOMGrower
from dnt.learning.hebbian import HebbianLearner
from dnt.learning.triplet import TripletExtractor


class DNT:
    """Developmental Neuron Tree — public API."""

    def __init__(self, config: Optional[DNTConfig] = None) -> None:
        self._config = config or DNTConfig()
        self._buffer = WorkingMemoryBuffer(max_size=self._config.buffer_size)
        self._tree = NeuronTree()
        self._observe_count: int = 0
        self._adapter = None

        self._triplet_extractor = TripletExtractor(self._config)
        self._hebbian = HebbianLearner(self._config)
        self._ghsom = GHSOMGrower(self._config)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def observe(self, data: Union[str, Dict[str, Any]]) -> None:
        """Add a new observation."""
        if isinstance(data, dict):
            raw_text = data.get("text") or json.dumps(data, ensure_ascii=False)
            source = data.get("source", "user")
            logic_type = data.get("logic_type", "fact")
        else:
            raw_text = str(data)
            source = "user"
            logic_type = "fact"

        # delegate to adapter if one is set
        if self._adapter is not None:
            obs = self._adapter.to_atomic(data)
        else:
            obs = AtomicObservation(
                raw_text=raw_text,
                source=source,
                logic_type=logic_type,  # type: ignore[arg-type]
            )

        self._buffer.push(obs)
        self._observe_count += 1

        # trigger consolidation when threshold is reached
        if self._observe_count % self._config.consolidate_every == 0:
            await self.consolidate()

    async def query(self, question: str) -> str:
        """Search L1 buffer and NeuronTree, return a context string."""
        buffer_hits = self._buffer.search(question)
        tree_hits = self._tree.hop_traversal(
            question,
            hop_limit=self._config.hop_limit,
            activation_threshold=self._config.activation_threshold,
        )

        parts: list[str] = []

        if buffer_hits:
            parts.append("=== Hot Memory (L1 Buffer) ===")
            for obs in buffer_hits[:5]:
                parts.append(f"[{obs.logic_type}] {obs.raw_text}")

        if tree_hits:
            parts.append("=== Neuron Tree ===")
            for node in tree_hits[:5]:
                label = node.summary or node.label
                parts.append(f"[node] {label}")

        if not parts:
            return "No relevant context found."

        return "\n".join(parts)

    async def consolidate(self) -> None:
        """
        Drain the buffer, extract triplets, apply Hebbian updates,
        and grow the tree via GHSOM if needed.
        """
        observations = self._buffer.flush()
        if not observations:
            return

        all_triplets = []
        for obs in observations:
            triplets = await self._triplet_extractor.extract(obs)
            all_triplets.extend(triplets)

        if all_triplets:
            self._hebbian.update(self._tree, all_triplets)
            self._ghsom.grow_if_needed(self._tree)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def export(self) -> dict:
        return {
            "config": self._config.model_dump(),
            "tree": self._tree.to_dict(),
            "buffer": [obs.model_dump(mode="json") for obs in self._buffer.peek()],
            "observe_count": self._observe_count,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "DNT":
        config = DNTConfig(**snapshot.get("config", {}))
        instance = cls(config=config)
        instance._tree = NeuronTree.from_dict(snapshot.get("tree", {}))
        for obs_data in snapshot.get("buffer", []):
            instance._buffer.push(AtomicObservation(**obs_data))
        instance._observe_count = snapshot.get("observe_count", 0)
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_adapter(self, adapter: Any) -> None:
        self._adapter = adapter

    def stats(self) -> DNTStats:
        return DNTStats(
            node_count=self._tree.node_count,
            edge_count=self._tree.edge_count,
            buffer_size=len(self._buffer),
            observe_count=self._observe_count,
        )

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"DNT(nodes={s.node_count}, edges={s.edge_count}, "
            f"buffer={s.buffer_size}, observed={s.observe_count})"
        )
