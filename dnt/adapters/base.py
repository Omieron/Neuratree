from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dnt.core.models import AtomicObservation


class ProjectAdapter(ABC):
    """
    Convert project-specific data formats into AtomicObservation.
    Implement this to feed any data shape into DNT without modifying core logic.
    """

    @abstractmethod
    def to_atomic(self, data: Any) -> AtomicObservation:
        """Convert a single data item to an AtomicObservation."""
        ...


# ------------------------------------------------------------------
# Built-in adapters
# ------------------------------------------------------------------


class StringAdapter(ProjectAdapter):
    """Wraps plain strings. Optionally sets source and logic_type."""

    def __init__(
        self,
        source: str = "user",
        logic_type: str = "fact",
    ) -> None:
        self._source = source
        self._logic_type = logic_type

    def to_atomic(self, data: Any) -> AtomicObservation:
        return AtomicObservation(
            raw_text=str(data),
            source=self._source,
            logic_type=self._logic_type,  # type: ignore[arg-type]
        )


class ChatMessageAdapter(ProjectAdapter):
    """
    Converts chat-message dicts of the form {role, content} into observations.
    The role becomes the source; content becomes the raw_text.
    """

    def to_atomic(self, data: Any) -> AtomicObservation:
        if isinstance(data, dict):
            return AtomicObservation(
                raw_text=data.get("content", str(data)),
                source=data.get("role", "user"),
            )
        return AtomicObservation(raw_text=str(data))
