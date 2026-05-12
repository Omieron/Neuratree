from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from dnt.core.models import Triplet


class LLMProvider(ABC):
    """
    Abstract LLM backend. Swap providers without touching core logic.
    Implement this to add any LLM backend (OpenAI, Anthropic, local, etc.).
    """

    @abstractmethod
    async def extract_triplets(
        self,
        text: str,
        entities: List[str],
        logic_type: str,
    ) -> List[Triplet]:
        """Extract subject-predicate-object triplets from text.

        Args:
            text: Raw observation text.
            entities: Candidate entities detected by spaCy NER.
            logic_type: One of "fact", "rule", "preference".
        """
        ...

    @abstractmethod
    async def summarize(self, label: str, context: str) -> str:
        """Generate a one-line summary for a node.

        Args:
            label: Node label (entity name or concept).
            context: Surrounding triplet context for the node.
        """
        ...
