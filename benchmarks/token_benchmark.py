"""
Token benchmark: RAG (raw text) vs DNT (structured active-path context).

Run with:
    PYTHONPATH=. python benchmarks/token_benchmark.py
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from dnt import DNT, DNTConfig
from dnt.core.models import Triplet

# Approximate tokeniser: 1 token ≈ 4 characters (GPT rule-of-thumb)
def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


OBSERVATIONS = [
    "Alice is the CEO of Acme Corp and reports to the board of directors",
    "Acme Corp was founded in 1985 and currently operates in 40 countries",
    "Alice prefers concise bullet-point reports over lengthy presentations",
    "Bob is the CTO at Acme Corp and works closely with Alice on strategy",
    "The Q3 revenue target for Acme Corp is 50 million dollars",
    "Alice and Bob both graduated from MIT with degrees in computer science",
    "Acme Corp's main competitor is Globex Corporation based in Chicago",
    "Alice dislikes early morning meetings scheduled before 9 am",
    "Bob is leading the cloud migration project due next quarter",
    "Acme Corp is planning to acquire a fintech startup in the next fiscal year",
    "Alice has been with Acme Corp for twelve years",
    "Bob joined Acme Corp three years ago from a Silicon Valley startup",
]

FIXED_TRIPLETS = [
    Triplet(subject="Alice", predicate="is_ceo_of", object="Acme Corp"),
    Triplet(subject="Bob", predicate="is_cto_of", object="Acme Corp"),
    Triplet(subject="Alice", predicate="works_with", object="Bob"),
    Triplet(subject="Acme Corp", predicate="founded_in", object="1985"),
    Triplet(subject="Acme Corp", predicate="competes_with", object="Globex Corporation"),
]


async def run() -> None:
    config = DNTConfig(consolidate_every=4, buffer_size=50, activation_threshold=0.1)
    dnt = DNT(config=config)

    # use fixed triplets so we don't need a real API key
    dnt._triplet_extractor.extract = AsyncMock(return_value=FIXED_TRIPLETS)

    for obs in OBSERVATIONS:
        await dnt.observe(obs)

    # flush any remaining buffer
    await dnt.consolidate()

    query = "Who are the executives at Acme Corp and what are their responsibilities?"

    # --- RAG baseline: concatenate all raw observations ---
    rag_context = "\n".join(OBSERVATIONS)
    rag_tokens = _tokens(rag_context)

    # --- DNT: structured ATP context ---
    dnt_context = await dnt.query(query)
    dnt_tokens = _tokens(dnt_context)

    savings = rag_tokens / max(dnt_tokens, 1)
    stats = dnt.stats()

    print("=" * 60)
    print("  Token Benchmark: RAG vs DNT")
    print("=" * 60)
    print(f"  Query        : {query}")
    print(f"  Observations : {len(OBSERVATIONS)}")
    print(f"  Tree nodes   : {stats.node_count}")
    print(f"  Tree edges   : {stats.edge_count}")
    print()
    print(f"  RAG tokens   : ~{rag_tokens:,}")
    print(f"  DNT tokens   : ~{dnt_tokens:,}")
    print(f"  Savings      : {savings:.1f}x fewer tokens")
    print()
    print("--- DNT context sent to LLM ---")
    print(dnt_context)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run())
