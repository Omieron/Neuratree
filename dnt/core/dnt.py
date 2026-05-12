from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from dnt.config import DNTConfig
from dnt.core.buffer import WorkingMemoryBuffer
from dnt.core.models import AtomicObservation, DNTStats
from dnt.core.tree import NeuronTree
from dnt.learning.ghsom import GHSOMGrower
from dnt.learning.hebbian import HebbianLearner
from dnt.learning.triplet import TripletExtractor
from dnt.memory.consolidate import ConsolidationEngine
from dnt.memory.snapshot import SnapshotManager


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
        self._engine = ConsolidationEngine(
            config=self._config,
            triplet_extractor=self._triplet_extractor,
            hebbian=self._hebbian,
            ghsom=self._ghsom,
            tree=self._tree,
        )

    # ------------------------------------------------------------------
    # Lifecycle (background mode)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background consolidation worker."""
        await self._engine.start()

    async def stop(self) -> None:
        """Drain the consolidation queue and stop the background worker."""
        await self._engine.stop()

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

        if self._observe_count % self._config.consolidate_every == 0:
            await self.consolidate()

    async def query(self, question: str) -> str:
        """
        ATP query: search L1 buffer and NeuronTree, return structured context.
        Tree results include active edge relations and weights so the LLM
        receives compact structural data instead of raw text.
        """
        buffer_hits = self._buffer.search(question)
        tree_hits = self._tree.hop_traversal(
            question,
            hop_limit=self._config.hop_limit,
            activation_threshold=self._config.activation_threshold,
        )

        parts: List[str] = []

        if buffer_hits:
            parts.append("=== Hot Memory (L1 Buffer) ===")
            for obs in buffer_hits[:5]:
                parts.append(f"[{obs.logic_type}] {obs.raw_text}")

        if tree_hits:
            parts.append("=== Neuron Tree (ATP active paths) ===")
            for node in tree_hits[:5]:
                label = node.summary or node.label
                neighbors = self._tree.get_active_neighbors(
                    node.id, self._config.activation_threshold
                )
                if neighbors:
                    rel_str = "; ".join(
                        f"{rel} → {n.label}" for n, rel in neighbors[:3]
                    )
                    parts.append(f"[L{node.level}] {label} | {rel_str}")
                else:
                    parts.append(f"[L{node.level}] {label}")

        if not parts:
            return "No relevant context found."

        return "\n".join(parts)

    async def consolidate(self) -> None:
        """
        Flush the buffer and run one consolidation cycle.
        In background mode, observations are queued and processed
        by the worker without blocking the caller.
        """
        observations = self._buffer.flush()
        if not observations:
            return

        if self._engine.is_running:
            await self._engine.enqueue(observations)
        else:
            await self._engine.run_once(observations)

    # ------------------------------------------------------------------
    # Snapshot — delegated to SnapshotManager
    # ------------------------------------------------------------------

    def export(self) -> dict:
        return SnapshotManager.export(
            self._config, self._tree, self._buffer, self._observe_count
        )

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "DNT":
        config, tree, buffer_items, observe_count = SnapshotManager.restore(snapshot)
        instance = cls(config=config)
        instance._tree = tree
        # re-wire engine to the restored tree
        instance._engine._tree = tree
        for obs in buffer_items:
            instance._buffer.push(obs)
        instance._observe_count = observe_count
        return instance

    def save(self, path: str) -> None:
        """Persist state to a JSON file."""
        SnapshotManager.save(self.export(), path)

    @classmethod
    def load(cls, path: str) -> "DNT":
        """Restore state from a JSON file."""
        return cls.from_snapshot(SnapshotManager.load(path))

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
