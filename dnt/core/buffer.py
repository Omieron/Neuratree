from __future__ import annotations

from collections import deque
from typing import List

from dnt.core.models import AtomicObservation


class WorkingMemoryBuffer:
    """Deque tabanlı L1 sıcak hafıza tamponu."""

    def __init__(self, max_size: int = 50) -> None:
        self._max_size = max_size
        self._buffer: deque[AtomicObservation] = deque(maxlen=max_size)

    def push(self, observation: AtomicObservation) -> None:
        self._buffer.append(observation)

    def search(self, query: str) -> List[AtomicObservation]:
        """
        Faz 1: Basit keyword eşleşmesi.
        Sorgu kelimelerinden en az biri raw_text'te geçiyorsa döndür.
        """
        tokens = set(query.lower().split())
        results: List[AtomicObservation] = []

        for obs in self._buffer:
            text_tokens = set(obs.raw_text.lower().split())
            if tokens & text_tokens:
                results.append(obs)

        # En yeni gözlemler önce
        return list(reversed(results))

    def flush(self) -> List[AtomicObservation]:
        """Tamponu boşalt ve içeriği döndür."""
        items = list(self._buffer)
        self._buffer.clear()
        return items

    def peek(self) -> List[AtomicObservation]:
        """Tamponu temizlemeden içeriği döndür."""
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"WorkingMemoryBuffer(size={len(self)}/{self._max_size})"
