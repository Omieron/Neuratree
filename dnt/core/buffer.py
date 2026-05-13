from __future__ import annotations

from collections import deque
from typing import List

from dnt.core.models import AtomicObservation


class WorkingMemoryBuffer:
    """Deque-based L1 hot memory buffer."""

    def __init__(self, max_size: int = 50) -> None:
        self._max_size = max_size
        self._buffer: deque[AtomicObservation] = deque(maxlen=max_size)

    def push(self, observation: AtomicObservation) -> None:
        self._buffer.append(observation)

    def search(self, query: str) -> List[AtomicObservation]:
        """
        Phase 1: simple keyword matching.
        Returns observations where at least one query token appears in raw_text.
        """
        import re
        tokens = set(re.findall(r'\b\w+\b', query.lower()))
        results: List[AtomicObservation] = []

        for obs in self._buffer:
            text_tokens = set(re.findall(r'\b\w+\b', obs.raw_text.lower()))
            if tokens & text_tokens:
                results.append(obs)

        # most recent first
        return list(reversed(results))

    def flush(self) -> List[AtomicObservation]:
        """Drain the buffer and return its contents."""
        items = list(self._buffer)
        self._buffer.clear()
        return items

    def peek(self) -> List[AtomicObservation]:
        """Return buffer contents without clearing."""
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"WorkingMemoryBuffer(size={len(self)}/{self._max_size})"
