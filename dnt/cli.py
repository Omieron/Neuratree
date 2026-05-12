"""
DNT command-line interface.

Usage:
    dnt dashboard                          — visual dashboard (browser)
    dnt chat --provider openai             — chat with ChatGPT + DNT memory
    dnt chat --provider anthropic          — chat with Claude + DNT memory
    dnt demo                               — quick demo (no API key needed)
    dnt bench                              — RAG vs DNT token benchmark
    dnt test                               — run test suite
"""
from __future__ import annotations

import asyncio
import subprocess
import sys


# ANSI
_R  = "\033[0m"
_B  = "\033[1m"
_DIM = "\033[2m"
_BLUE   = "\033[38;5;33m"
_GREEN  = "\033[38;5;35m"
_YELLOW = "\033[38;5;220m"
_GRAY   = "\033[38;5;240m"
_CYAN   = "\033[38;5;38m"


def main() -> None:
    args = sys.argv[1:]
    cmd  = args[0] if args else "help"

    if cmd == "dashboard":
        _dashboard()
    elif cmd == "chat":
        asyncio.run(_chat(args[1:]))
    elif cmd == "bench":
        _bench()
    elif cmd == "demo":
        asyncio.run(_demo())
    elif cmd == "test":
        _test()
    else:
        _help()


# ── Commands ──────────────────────────────────────────────────────────────────

def _dashboard() -> None:
    import webbrowser, threading
    host, port = "localhost", 8501
    threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    print(f"\n  Dashboard →  http://{host}:{port}\n  Ctrl+C to stop.\n")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "dnt.ui.server:app",
         "--host", host, "--port", str(port)],
        check=True,
    )


async def _chat(args: list[str]) -> None:
    import os

    # parse --provider and --model flags
    provider = "openai"
    model    = None
    i = 0
    while i < len(args):
        if args[i] == "--provider" and i + 1 < len(args):
            provider = args[i + 1]; i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]; i += 2
        else:
            i += 1

    # pick client class and resolve api key + default model
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DNT_OPENAI_API_KEY", "")
        if not api_key:
            print(f"\n  Set OPENAI_API_KEY and try again.\n")
            return
        from dnt.integrations.openai_chat import OpenAIChat
        client = OpenAIChat(api_key=api_key, model=model or "gpt-4o-mini")
        label  = f"openai / {client._model}"

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("DNT_ANTHROPIC_API_KEY", "")
        if not api_key:
            print(f"\n  Set ANTHROPIC_API_KEY and try again.\n")
            return
        from dnt.integrations.claude_chat import ClaudeChat
        client = ClaudeChat(api_key=api_key, model=model or "claude-haiku-4-5-20251001")
        label  = f"anthropic / {client._model}"

    else:
        print(f"\n  Unknown provider '{provider}'. Use openai or anthropic.\n")
        return

    # header
    w = 58
    print(f"\n{_B}{'─' * w}{_R}")
    print(f"{_B}  Neuron Tree Chat  ·  {label}{_R}")
    print(f"{_DIM}{'─' * w}{_R}")
    print(f"{_GRAY}  /stats   show tree stats")
    print(f"  /context show last DNT context")
    print(f"  /quit    exit{_R}")
    print(f"{_DIM}{'─' * w}{_R}\n")

    last_context = ""
    turn = 0

    while True:
        try:
            raw = input(f"{_B}{_BLUE}You:{_R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_GRAY}  Bye.{_R}\n")
            break

        if not raw:
            continue

        # commands
        if raw == "/quit":
            print(f"\n{_GRAY}  Bye.{_R}\n")
            break

        if raw == "/stats":
            s = client.dnt.stats()
            print(f"\n{_GRAY}  nodes={s.node_count}  edges={s.edge_count}  "
                  f"buffer={s.buffer_size}  observed={s.observe_count}{_R}\n")
            continue

        if raw == "/context":
            if last_context:
                print(f"\n{_GRAY}{last_context}{_R}\n")
            else:
                print(f"\n{_GRAY}  No context yet.{_R}\n")
            continue

        # send
        turn += 1
        try:
            answer, stats = await client.send(raw)
        except Exception as exc:
            print(f"\n  {_YELLOW}Error: {exc}{_R}\n")
            continue

        last_context = stats.context

        # answer
        print(f"\n{_B}{_GREEN}Assistant:{_R} {answer}\n")

        # token line
        saved = stats.saved_tokens
        ctx_indicator = (
            f"{_CYAN}context {stats.context_tokens}t{_R}"
            if stats.context_tokens else f"{_GRAY}no context yet{_R}"
        )
        print(
            f"{_DIM}  turn {turn}  ·  "
            f"prompt {stats.prompt_tokens}t  "
            f"completion {stats.completion_tokens}t  ·  "
            f"{ctx_indicator}"
            + (f"  ·  {_GREEN}saved ~{saved}t ({stats.savings_pct}%){_R}" if saved > 0 else _R)
        )
        print()


def _bench() -> None:
    from benchmarks.token_benchmark import run
    asyncio.run(run())


def _test() -> None:
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)


async def _demo() -> None:
    from unittest.mock import AsyncMock
    from dnt import DNT, DNTConfig
    from dnt.core.models import Triplet

    print(f"\n{_B}── DNT Demo ─────────────────────────────────{_R}")

    config = DNTConfig(consolidate_every=3, activation_threshold=0.05)
    dnt = DNT(config=config)

    dnt._triplet_extractor.extract = AsyncMock(return_value=[
        Triplet(subject="Alice", predicate="is_ceo_of", object="Acme Corp"),
        Triplet(subject="Bob",   predicate="is_cto_of", object="Acme Corp"),
    ])

    for obs in [
        "Alice is the CEO of Acme Corp",
        "Bob is the CTO of Acme Corp",
        "Alice prefers short bullet-point reports",
    ]:
        print(f"  observe → {obs}")
        await dnt.observe(obs)

    await dnt.consolidate()
    print()

    q = "Who leads Acme Corp?"
    print(f"  query  → {q}\n")
    print(await dnt.query(q))
    print(f"\n{dnt.stats()}")
    print(f"{_B}─────────────────────────────────────────────{_R}\n")


def _help() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
