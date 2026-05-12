from __future__ import annotations

import json
from typing import List

from dnt.core.models import Triplet
from dnt.llm.base import LLMProvider

_EXTRACT_TOOL = {
    "name": "extract_triplets",
    "description": "Extract subject-predicate-object triplets from text.",
    "input_schema": {
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
                            "description": "Relationship in snake_case (e.g. works_at)",
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
}


class AnthropicProvider(LLMProvider):
    """Anthropic backend using tool_use for structured triplet extraction."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def extract_triplets(
        self, text: str, entities: List[str], logic_type: str
    ) -> List[Triplet]:
        client = self._get_client()
        system_msg = (
            "Extract subject-predicate-object triplets from the text. "
            f"Detected entities: {', '.join(entities)}. "
            "Use snake_case for predicates."
        )
        response = await client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system_msg,
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "extract_triplets"},
            messages=[{"role": "user", "content": text}],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        data = tool_block.input
        return [
            Triplet(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                logic_type=t.get("logic_type", logic_type),  # type: ignore[arg-type]
            )
            for t in data.get("triplets", [])
        ]

    async def summarize(self, label: str, context: str) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=60,
            system="Write a single concise sentence summarising the entity.",
            messages=[
                {
                    "role": "user",
                    "content": f"Entity: {label}\nContext:\n{context}",
                }
            ],
        )
        return response.content[0].text.strip()
