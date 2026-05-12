from __future__ import annotations

import json
from typing import List, Optional

from dnt.core.models import Triplet
from dnt.llm.base import LLMProvider

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
    },
}


class OpenAIProvider(LLMProvider):
    """OpenAI backend using function calling for structured triplet extraction."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
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
        response = await client.chat.completions.create(
            model=self._model,
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

    async def summarize(self, label: str, context: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "Write a single concise sentence summarising the entity.",
                },
                {
                    "role": "user",
                    "content": f"Entity: {label}\nContext:\n{context}",
                },
            ],
            max_tokens=60,
        )
        return response.choices[0].message.content.strip()
