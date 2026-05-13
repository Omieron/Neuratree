"""
DNT-powered Claude (Anthropic) wrapper.

Usage:
    from dnt.integrations.claude_chat import ClaudeChat
    import asyncio, os

    chat = ClaudeChat(api_key=os.environ["ANTHROPIC_API_KEY"])
    answer, stats = await chat.send("Who is the CEO of Acme Corp?")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from dnt import DNT, DNTConfig
from dnt.integrations.openai_chat import TurnStats   # reuse same dataclass


class ClaudeChat:
    """
    Drop-in Anthropic client with DNT long-term memory.
    Mirrors the OpenAIChat API exactly so you can swap providers.
    """

    DEFAULT_SYSTEM = (
        "You are a helpful assistant. "
        "When a memory context section is provided, use it to stay consistent "
        "with prior conversation. Do not mention the memory system to the user."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        system: str = DEFAULT_SYSTEM,
        dnt_config: Optional[DNTConfig] = None,
        max_tokens: int = 4096,
    ) -> None:
        import anthropic
        self._client     = anthropic.AsyncAnthropic(api_key=api_key)
        self._model      = model
        self._system     = system
        self._max_tokens = max_tokens
        self._dnt     = DNT(config=dnt_config or DNTConfig(
            consolidate_every=4,
            buffer_size=40,
            activation_threshold=0.05,
            hebbian_lr=0.3,
        ))
        self._raw_history: List[str] = []

    async def send(self, user_message: str) -> Tuple[str, TurnStats]:
        """Send a message, get a reply and usage stats."""

        # 1. Query DNT for compressed context
        context = await self._dnt.query(user_message)
        has_context = context and context != "No relevant context found."
        context_tokens = len(context) // 4 if has_context else 0

        # 2. Build system prompt
        system = self._system
        if has_context:
            system += f"\n\n[Memory Context]\n{context}\n[/Memory Context]"

        # 3. Call the model
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = response.content[0].text if response.content else ""
        usage  = response.usage

        # 4. Feed both sides back into DNT memory
        await self._dnt.observe(f"User: {user_message}")
        await self._dnt.observe(f"Assistant: {answer[:300]}")

        # 5. Track raw history size for comparison
        self._raw_history.append(f"User: {user_message}\nAssistant: {answer}")
        raw_tokens = sum(len(t) // 4 for t in self._raw_history)

        stats = TurnStats(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            context_tokens=context_tokens,
            raw_history_tokens=raw_tokens,
            context=context if has_context else "",
        )

        return answer, stats

    @property
    def dnt(self) -> DNT:
        return self._dnt
