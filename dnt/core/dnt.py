from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

from dnt.config import DNTConfig
from dnt.core.buffer import WorkingMemoryBuffer
from dnt.core.models import AtomicObservation, DNTStats
from dnt.core.tree import NeuronTree


class DNT:
    """Developmental Neuron Tree — public API."""

    def __init__(self, config: Optional[DNTConfig] = None) -> None:
        self._config = config or DNTConfig()
        self._buffer = WorkingMemoryBuffer(max_size=self._config.buffer_size)
        self._tree = NeuronTree()
        self._observe_count: int = 0
        self._adapter = None

    # ------------------------------------------------------------------
    # Temel İşlemler
    # ------------------------------------------------------------------

    async def observe(self, data: Union[str, Dict[str, Any]]) -> None:
        """Yeni bir gözlem ekle."""
        if isinstance(data, dict):
            raw_text = data.get("text") or json.dumps(data, ensure_ascii=False)
            source = data.get("source", "user")
            logic_type = data.get("logic_type", "fact")
        else:
            raw_text = str(data)
            source = "user"
            logic_type = "fact"

        # Adaptör varsa dönüştür
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

        # Konsolidasyon eşiği doldu mu?
        if self._observe_count % self._config.consolidate_every == 0:
            await self.consolidate()

    async def query(self, soru: str) -> str:
        """
        Faz 1: L1 buffer + NeuronTree'yi basit string matching ile sorgula.
        Bağlam stringi döndür.
        """
        buffer_hits = self._buffer.search(soru)
        tree_hits = self._tree.hop_traversal(
            soru,
            hop_limit=self._config.hop_limit,
            activation_threshold=self._config.activation_threshold,
        )

        parts: list[str] = []

        if buffer_hits:
            parts.append("=== Sıcak Hafıza (L1 Buffer) ===")
            for obs in buffer_hits[:5]:
                parts.append(f"[{obs.logic_type}] {obs.raw_text}")

        if tree_hits:
            parts.append("=== Nöron Ağacı ===")
            for node in tree_hits[:5]:
                label = node.summary or node.label
                parts.append(f"[düğüm] {label}")

        if not parts:
            return "İlgili bağlam bulunamadı."

        return "\n".join(parts)

    async def consolidate(self) -> None:
        """
        Faz 1 stub: buffer'ı temizle.
        Faz 2'de gerçek triplet çıkarımı + Hebbian güncelleme gelecek.
        """
        self._buffer.flush()

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
    # Yardımcılar
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
