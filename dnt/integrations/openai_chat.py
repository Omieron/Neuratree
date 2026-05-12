"""
DNT-powered ChatGPT wrapper.

Usage:
    from dnt.integrations.openai_chat import OpenAIChat
    import asyncio, os

    chat = OpenAIChat(api_key=os.environ["OPENAI_API_KEY"])
    answer, stats = await chat.send("Who is the CEO of Acme Corp?")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from dnt import DNT, DNTConfig


@dataclass
class TurnStats:
    prompt_tokens:      int
    completion_tokens:  int
    total_tokens:       int
    context_tokens:     int     # tokens DNT sent as context
    raw_history_tokens: int     # tokens full history would have cost
    context:            str     # the actual DNT context string

    @property
    def saved_tokens(self) -> int:
        return max(0, self.raw_history_tokens - self.context_tokens)

    @property
    def savings_pct(self) -> float:
        if self.raw_history_tokens == 0:
            return 0.0
        return round((1 - self.context_tokens / self.raw_history_tokens) * 100, 1)


class OpenAIChat:
    """
    Drop-in ChatGPT client with DNT long-term memory.

    Every message pair (user + assistant) is observed into the neuron tree.
    At query time, only the compressed relevant context is sent to the model
    instead of the full conversation history.
    """

    DEFAULT_SYSTEM = (
        "You are a helpful assistant. "
        "When a memory context section is provided, use it to stay consistent "
        "with prior conversation. Do not mention the memory system to the user."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        system: str = DEFAULT_SYSTEM,
        dnt_config: Optional[DNTConfig] = None,
    ) -> None:
        from openai import AsyncOpenAI
        self._client  = AsyncOpenAI(api_key=api_key)
        self._model   = model
        self._system  = system
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
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
        )
        answer = response.choices[0].message.content or ""
        usage  = response.usage

        # 4. Feed both sides back into DNT memory
        await self._dnt.observe(f"User: {user_message}")
        await self._dnt.observe(f"Assistant: {answer[:300]}")

        # 5. Track raw history size for comparison
        self._raw_history.append(f"User: {user_message}\nAssistant: {answer}")
        raw_tokens = sum(len(t) // 4 for t in self._raw_history)

        stats = TurnStats(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            context_tokens=context_tokens,
            raw_history_tokens=raw_tokens,
            context=context if has_context else "",
        )

        return answer, stats

    @property
    def dnt(self) -> DNT:
        return self._dnt
