"""Phase 4 LLM provider and adapter tests."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dnt.adapters.base import ChatMessageAdapter, ProjectAdapter, StringAdapter
from dnt.config import DNTConfig
from dnt.core.dnt import DNT
from dnt.core.models import AtomicObservation, Triplet
from dnt.learning.triplet import TripletExtractor
from dnt.llm.base import LLMProvider
from dnt.llm.openai_provider import OpenAIProvider
from dnt.llm.anthropic_provider import AnthropicProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _triplet(s: str, o: str) -> Triplet:
    return Triplet(subject=s, predicate="related_to", object=o)


class _StubProvider(LLMProvider):
    """Minimal concrete implementation for testing the interface."""

    def __init__(self, triplets: list[Triplet], summary: str = "stub summary") -> None:
        self._triplets = triplets
        self._summary = summary

    async def extract_triplets(self, text, entities, logic_type) -> list[Triplet]:
        return self._triplets

    async def summarize(self, label, context) -> str:
        return self._summary


# ---------------------------------------------------------------------------
# LLMProvider interface tests
# ---------------------------------------------------------------------------


class TestLLMProviderInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        p = _StubProvider([])
        assert isinstance(p, LLMProvider)

    @pytest.mark.asyncio
    async def test_stub_extract_triplets(self):
        expected = [_triplet("A", "B")]
        p = _StubProvider(expected)
        result = await p.extract_triplets("A relates to B", ["A", "B"], "fact")
        assert result == expected

    @pytest.mark.asyncio
    async def test_stub_summarize(self):
        p = _StubProvider([], summary="Alice is a CEO")
        result = await p.summarize("Alice", "context here")
        assert result == "Alice is a CEO"


# ---------------------------------------------------------------------------
# OpenAIProvider tests
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def test_lazy_client_init(self):
        provider = OpenAIProvider(api_key="sk-test")
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_extract_triplets_calls_api(self):
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")

        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = (
            '{"triplets": [{"subject": "Alice", "predicate": "works_at",'
            ' "object": "Google", "logic_type": "fact"}]}'
        )
        mock_choice = MagicMock()
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.extract_triplets("Alice works at Google", ["Alice", "Google"], "fact")
        assert len(result) == 1
        assert result[0].subject == "Alice"
        assert result[0].predicate == "works_at"
        assert result[0].object == "Google"

    @pytest.mark.asyncio
    async def test_summarize_calls_api(self):
        provider = OpenAIProvider(api_key="sk-test")

        mock_choice = MagicMock()
        mock_choice.message.content = "  Alice is the CEO of Acme Corp.  "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.summarize("Alice", "Alice is_ceo_of Acme Corp")
        assert result == "Alice is the CEO of Acme Corp."


# ---------------------------------------------------------------------------
# AnthropicProvider tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def test_lazy_client_init(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_extract_triplets_calls_api(self):
        provider = AnthropicProvider(api_key="sk-ant-test")

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.input = {
            "triplets": [
                {"subject": "Bob", "predicate": "is_cto_of", "object": "Acme", "logic_type": "fact"}
            ]
        }
        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]

        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.extract_triplets("Bob is CTO of Acme", ["Bob", "Acme"], "fact")
        assert len(result) == 1
        assert result[0].subject == "Bob"
        assert result[0].predicate == "is_cto_of"

    @pytest.mark.asyncio
    async def test_summarize_calls_api(self):
        provider = AnthropicProvider(api_key="sk-ant-test")

        mock_text_block = MagicMock()
        mock_text_block.text = "Bob is the chief technology officer."
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.summarize("Bob", "Bob is_cto_of Acme")
        assert result == "Bob is the chief technology officer."


# ---------------------------------------------------------------------------
# TripletExtractor with injected provider
# ---------------------------------------------------------------------------


class TestTripletExtractorWithProvider:
    @pytest.mark.asyncio
    async def test_uses_injected_provider(self):
        expected = [_triplet("Alice", "Google")]
        provider = _StubProvider(expected)
        extractor = TripletExtractor(DNTConfig(), llm_provider=provider)

        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "Alice"
        mock_doc.ents = [mock_ent]
        mock_doc.noun_chunks = []
        extractor._nlp = MagicMock(return_value=mock_doc)

        obs = AtomicObservation(raw_text="Alice works at Google")
        result = await extractor.extract(obs)
        assert result == expected

    @pytest.mark.asyncio
    async def test_falls_back_to_heuristic_when_provider_raises(self):
        class _FailProvider(LLMProvider):
            async def extract_triplets(self, text, entities, logic_type):
                raise RuntimeError("provider failure")
            async def summarize(self, label, context):
                return ""

        extractor = TripletExtractor(DNTConfig(), llm_provider=_FailProvider())

        mock_doc = MagicMock()
        mock_ent_a = MagicMock()
        mock_ent_a.text = "Alice"
        mock_ent_b = MagicMock()
        mock_ent_b.text = "Google"
        mock_doc.ents = [mock_ent_a, mock_ent_b]
        mock_doc.noun_chunks = []
        extractor._nlp = MagicMock(return_value=mock_doc)

        obs = AtomicObservation(raw_text="Alice works at Google")
        result = await extractor.extract(obs)
        assert len(result) >= 1  # heuristic produced something

    @pytest.mark.asyncio
    async def test_no_provider_uses_heuristic(self):
        extractor = TripletExtractor(DNTConfig(), llm_provider=None)

        mock_doc = MagicMock()
        mock_ent_a = MagicMock()
        mock_ent_a.text = "Alice"
        mock_ent_b = MagicMock()
        mock_ent_b.text = "Google"
        mock_doc.ents = [mock_ent_a, mock_ent_b]
        mock_doc.noun_chunks = []
        extractor._nlp = MagicMock(return_value=mock_doc)

        obs = AtomicObservation(raw_text="Alice at Google")
        result = await extractor.extract(obs)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestProjectAdapters:
    def test_project_adapter_is_abstract(self):
        with pytest.raises(TypeError):
            ProjectAdapter()  # type: ignore[abstract]

    def test_string_adapter_wraps_string(self):
        adapter = StringAdapter(source="system", logic_type="rule")
        obs = adapter.to_atomic("always log errors")
        assert obs.raw_text == "always log errors"
        assert obs.source == "system"
        assert obs.logic_type == "rule"

    def test_string_adapter_converts_non_string(self):
        adapter = StringAdapter()
        obs = adapter.to_atomic(42)
        assert obs.raw_text == "42"

    def test_chat_message_adapter_dict(self):
        adapter = ChatMessageAdapter()
        obs = adapter.to_atomic({"role": "assistant", "content": "Hello world"})
        assert obs.raw_text == "Hello world"
        assert obs.source == "assistant"

    def test_chat_message_adapter_missing_keys(self):
        adapter = ChatMessageAdapter()
        obs = adapter.to_atomic({"content": "just content"})
        assert obs.raw_text == "just content"
        assert obs.source == "user"

    def test_chat_message_adapter_non_dict(self):
        adapter = ChatMessageAdapter()
        obs = adapter.to_atomic("plain string")
        assert obs.raw_text == "plain string"

    def test_string_adapter_default_logic_type(self):
        adapter = StringAdapter()
        obs = adapter.to_atomic("test")
        assert obs.logic_type == "fact"


# ---------------------------------------------------------------------------
# DNT integration with providers and adapters
# ---------------------------------------------------------------------------


class TestDNTProviderIntegration:
    @pytest.mark.asyncio
    async def test_set_llm_provider_swaps_backend(self):
        dnt = DNT()
        provider_a = _StubProvider([_triplet("A", "B")])
        provider_b = _StubProvider([_triplet("X", "Y")])

        dnt.set_llm_provider(provider_a)
        assert dnt._triplet_extractor._llm is provider_a

        dnt.set_llm_provider(provider_b)
        assert dnt._triplet_extractor._llm is provider_b

    @pytest.mark.asyncio
    async def test_injected_provider_used_during_consolidation(self):
        provider = _StubProvider([_triplet("Alice", "Google")])
        dnt = DNT(llm_provider=provider)

        await dnt.observe("Alice works at Google")
        await dnt.consolidate()

        assert dnt.stats().node_count >= 2

    @pytest.mark.asyncio
    async def test_set_adapter_used_in_observe(self):
        dnt = DNT()
        adapter = ChatMessageAdapter()
        dnt.set_adapter(adapter)

        await dnt.observe({"role": "user", "content": "Hello from chat"})
        assert dnt.stats().buffer_size == 1

        context = await dnt.query("Hello")
        assert "Hello from chat" in context

    @pytest.mark.asyncio
    async def test_provider_from_config_openai(self):
        config = DNTConfig(openai_api_key="sk-test", llm_provider="openai")
        provider = DNT._provider_from_config(config)
        assert isinstance(provider, OpenAIProvider)

    @pytest.mark.asyncio
    async def test_provider_from_config_anthropic(self):
        config = DNTConfig(anthropic_api_key="sk-ant-test", llm_provider="anthropic")
        provider = DNT._provider_from_config(config)
        assert isinstance(provider, AnthropicProvider)

    def test_provider_from_config_no_key_returns_none(self):
        config = DNTConfig(openai_api_key="", anthropic_api_key="")
        provider = DNT._provider_from_config(config)
        assert provider is None
