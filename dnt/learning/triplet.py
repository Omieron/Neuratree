from __future__ import annotations

import json
import re
from typing import List, Optional

from dnt.config import DNTConfig
from dnt.core.models import AtomicObservation, Triplet

# OpenAI tool schema for function calling
_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_triplets",
        "description": "Extract subject-predicate-object triplets from text.",
        "parameters": {
            "type": "object",
            "properties": {
                "triplets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {
                                "type": "string",
                                "description": "Relationship in snake_case (e.g. works_at, depends_on)",
                            },
                            "object": {"type": "string"},
                            "logic_type": {
                                "type": "string",
                                "enum": ["fact", "rule", "preference"],
                            },
                        },
                        "required": ["subject", "predicate", "object", "logic_type"],
                    },
                }
            },
            "required": ["triplets"],
        },
    },
}


class TripletExtractor:
    """Hybrid triplet extractor: spaCy NER for entities, OpenAI for relations."""

    def __init__(self, config: DNTConfig) -> None:
        self._config = config
        self._nlp = None
        self._client = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def extract(self, observation: AtomicObservation) -> List[Triplet]:
        entities = self._extract_entities(observation.raw_text)

        if self._config.openai_api_key and entities:
            try:
                return await self._openai_extract(
                    observation.raw_text, entities, observation.logic_type
                )
            except Exception:
                pass  # fall through to heuristic on any API error

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
                    # model not downloaded — use blank pipeline (no NER)
                    self._nlp = spacy.blank("en")
            except ImportError:
                self._nlp = None
        return self._nlp

    def _extract_entities(self, text: str) -> List[str]:
        nlp = self._get_nlp()
        if nlp is not None:
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents]
            if not entities:
                try:
                    entities = [chunk.root.text for chunk in doc.noun_chunks]
                except ValueError:
                    # blank model has no dependency parse — skip noun chunks
                    entities = []
        else:
            # spaCy not installed — regex: grab capitalised words / quoted phrases
            entities = re.findall(r'"([^"]+)"|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
            entities = [m[0] or m[1] for m in entities]

        # deduplicate while preserving order
        seen: set[str] = set()
        result: List[str] = []
        for e in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                result.append(e)
        return result

    # ------------------------------------------------------------------
    # Relation extraction (OpenAI)
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._config.openai_api_key)
        return self._client

    async def _openai_extract(
        self, text: str, entities: List[str], logic_type: str
    ) -> List[Triplet]:
        client = self._get_client()
        system_msg = (
            "Extract subject-predicate-object triplets from the text. "
            f"Detected entities: {', '.join(entities)}. "
            "Use snake_case for predicates."
        )
        response = await client.chat.completions.create(
            model=self._config.llm_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ],
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_triplets"}},
        )

        tool_call = response.choices[0].message.tool_calls[0]
        data = json.loads(tool_call.function.arguments)

        return [
            Triplet(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                logic_type=t.get("logic_type", logic_type),  # type: ignore[arg-type]
            )
            for t in data.get("triplets", [])
        ]

    # ------------------------------------------------------------------
    # Heuristic fallback (no OpenAI key)
    # ------------------------------------------------------------------

    def _heuristic_extract(
        self, text: str, entities: List[str], logic_type: str
    ) -> List[Triplet]:
        if len(entities) < 2:
            return []

        nlp = self._get_nlp()
        predicate = "related_to"

        if nlp is not None:
            doc = nlp(text)
            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    predicate = token.lemma_.lower().replace(" ", "_")
                    break

        # pair the first entity against up to 3 others
        return [
            Triplet(
                subject=entities[0],
                predicate=predicate,
                object=entities[i],
                logic_type=logic_type,  # type: ignore[arg-type]
            )
            for i in range(1, min(len(entities), 4))
        ]
