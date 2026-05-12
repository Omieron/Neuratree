from __future__ import annotations

import re
from typing import List, Optional

from dnt.config import DNTConfig
from dnt.core.models import AtomicObservation, Triplet
from dnt.llm.base import LLMProvider


class TripletExtractor:
    """
    Hybrid triplet extractor: spaCy NER for entity detection,
    LLMProvider for relation extraction.
    Falls back to heuristics when no provider is available.
    """

    def __init__(
        self,
        config: DNTConfig,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self._config = config
        self._llm = llm_provider
        self._nlp = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def extract(self, observation: AtomicObservation) -> List[Triplet]:
        entities = self._extract_entities(observation.raw_text)

        if self._llm is not None:
            try:
                return await self._llm.extract_triplets(
                    observation.raw_text, entities, observation.logic_type
                )
            except Exception:
                pass  # fall through to heuristic on any provider error

        return self._heuristic_extract(
            observation.raw_text, entities, observation.logic_type
        )

    # ------------------------------------------------------------------
    # Entity extraction (spaCy)
    # ------------------------------------------------------------------

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self._nlp = spacy.blank("en")
            except ImportError:
                self._nlp = None
        return self._nlp

    def _extract_entities(self, text: str) -> List[str]:
        nlp = self._get_nlp()
        entities: List[str] = []

        if nlp is not None:
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents]
            if not entities:
                try:
                    entities = [chunk.root.text for chunk in doc.noun_chunks]
                except ValueError:
                    entities = []

        # Fall back to regex when spaCy found nothing (blank model or not installed)
        if not entities:
            matches = re.findall(r'"([^"]+)"|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
            entities = [m[0] or m[1] for m in matches]

        # Last resort: pull out any content words (length > 3, not stopwords)
        if not entities:
            _STOP = {
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "has", "have", "had", "does", "did", "will", "would", "could",
                "should", "may", "might", "that", "this", "with", "from",
                "about", "into", "they", "them", "their", "also", "just",
            }
            entities = [
                w for w in re.findall(r'\b[a-zA-Z]{4,}\b', text)
                if w.lower() not in _STOP
            ][:8]

        # deduplicate, preserve order
        seen: set[str] = set()
        result: List[str] = []
        for e in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                result.append(e)
        return result

    # ------------------------------------------------------------------
    # Heuristic fallback (no LLM provider)
    # ------------------------------------------------------------------

    def _heuristic_extract(
        self, text: str, entities: List[str], logic_type: str
    ) -> List[Triplet]:
        if not entities:
            return []

        # Need at least 2 entities; if only 1, pair it with the full observation
        if len(entities) == 1:
            # Use the first non-trivial word that differs from the entity as object
            words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', text)
                     if w.lower() != entities[0].lower()]
            if not words:
                return []
            entities = [entities[0], words[0]]

        nlp = self._get_nlp()
        predicate = "related_to"
        if nlp is not None:
            doc = nlp(text)
            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    predicate = token.lemma_.lower().replace(" ", "_")
                    break

        return [
            Triplet(
                subject=entities[0],
                predicate=predicate,
                object=entities[i],
                logic_type=logic_type,  # type: ignore[arg-type]
            )
            for i in range(1, min(len(entities), 4))
        ]
