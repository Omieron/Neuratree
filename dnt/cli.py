"""
DNT command-line interface.

Usage:
    dnt dashboard    — start the Streamlit dashboard
    dnt bench        — run the RAG vs DNT token benchmark
    dnt demo         — quick interactive demo (no API key needed)
    dnt test         — run the full test suite
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "dashboard":
        _dashboard()
    elif cmd == "bench":
        _bench()
    elif cmd == "demo":
        asyncio.run(_demo())
    elif cmd == "test":
        _test()
    else:
        _help()


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


def _dashboard() -> None:
    dashboard_path = Path(__file__).parent / "ui" / "dashboard.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
    )


def _bench() -> None:
    from benchmarks.token_benchmark import run
    asyncio.run(run())


def _test() -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        check=True,
    )


async def _demo() -> None:
    from unittest.mock import AsyncMock
    from dnt import DNT, DNTConfig
    from dnt.core.models import Triplet

    print("\n── DNT Demo ─────────────────────────────────")

    config = DNTConfig(consolidate_every=3, activation_threshold=0.05)
    dnt = DNT(config=config)

    # use fixed triplets so no API key is needed
    dnt._triplet_extractor.extract = AsyncMock(
        return_value=[
            Triplet(subject="Alice", predicate="is_ceo_of", object="Acme Corp"),
            Triplet(subject="Bob",   predicate="is_cto_of", object="Acme Corp"),
        ]
    )

    observations = [
        "Alice is the CEO of Acme Corp",
        "Bob is the CTO of Acme Corp",
        "Alice prefers short bullet-point reports",
    ]
    for obs in observations:
        print(f"  observe → {obs}")
        await dnt.observe(obs)

    await dnt.consolidate()
    print()

    query = "Who leads Acme Corp?"
    print(f"  query  → {query}")
    print()
    context = await dnt.query(query)
    print(context)
    print()
    print(dnt.stats())
    print("─────────────────────────────────────────────\n")


def _help() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
