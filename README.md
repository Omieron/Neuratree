# Developmental Neuron Tree (DNT)

A biologically-inspired, plug-and-play memory engine for Python projects. DNT gives any application a self-organizing memory layer that learns from observations, compresses context into a graph structure, and returns only the most relevant information at query time — without shipping raw text to an LLM.

```python
dnt = DNT()
await dnt.observe("User asked about AAPL stock")
context = await dnt.query("What kind of stocks is the user interested in?")
```

---

## Why DNT?

Standard RAG pipelines retrieve chunks of raw text and send them wholesale to an LLM, burning tokens on noise. DNT instead maintains a **neuron tree** — a directed graph where nodes are concepts and edges carry Hebbian weights. At query time it traverses only the active paths and returns compact, structured data.

Target: **10x fewer tokens** than a comparable RAG setup.

---

## How It Works

```
Raw input (any format)
        ↓
ProjectAdapter       — normalizes input format
        ↓
AtomicObservation    — standard observation packet
        ↓
WorkingMemoryBuffer  — L1 hot cache (deque)
        ↓  (consolidate() fires every N observations)
NeuronTree           — NetworkX DiGraph, cold long-term memory
        ↓
query()              — searches L1 + tree
        ↓
Hop Traversal        — only active paths selected
        ↓
Compact context sent to the LLM
```

### Consolidation
Every `consolidate_every` observations the buffer is flushed into the tree. This involves extracting subject-predicate-object triplets (via spaCy + OpenAI function calling in Phase 2+), applying Hebbian weight updates (LTP/LTD), and growing the tree via GHSOM when a node's quantization error exceeds its threshold.

### Query
A query string enters the tree at the seed node whose label best matches the query tokens, then hops across edges whose weight meets the `activation_threshold`. The traversal stops at `hop_limit` steps. Only visited nodes contribute to the returned context.

---

## Project Structure

```
dnt/
├── core/
│   ├── models.py           # Pydantic data models
│   ├── tree.py             # NeuronTree (NetworkX DiGraph wrapper)
│   ├── buffer.py           # WorkingMemoryBuffer (L1 cache)
│   └── dnt.py              # Public API
├── learning/
│   ├── triplet.py          # spaCy NER + OpenAI triplet extraction  [Phase 2]
│   ├── hebbian.py          # LTP/LTD weight updates                 [Phase 2]
│   └── ghsom.py            # Hierarchical tree growth               [Phase 2]
├── memory/
│   ├── consolidate.py      # Async consolidation engine             [Phase 3]
│   └── snapshot.py         # export() / from_snapshot()             [Phase 3]
├── llm/
│   ├── base.py             # Abstract LLMProvider                   [Phase 4]
│   └── openai_provider.py  # OpenAI implementation                  [Phase 4]
├── adapters/
│   └── base.py             # Abstract ProjectAdapter                [Phase 4]
├── ui/
│   └── dashboard.py        # Streamlit live visualizer              [Phase 5]
├── config.py               # DNTConfig (Pydantic Settings)
tests/
├── test_core.py
requirements.txt
```

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd Neuratree

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```python
import asyncio
from dnt import DNT, DNTConfig

async def main():
    dnt = DNT()

    # Feed observations
    await dnt.observe("User is interested in tech stocks")
    await dnt.observe("User mentioned AAPL and MSFT")
    await dnt.observe("User prefers long-term holdings")

    # Query for context
    context = await dnt.query("What stocks does the user care about?")
    print(context)

    # Inspect stats
    print(dnt.stats())

asyncio.run(main())
```

**Custom config:**

```python
config = DNTConfig(
    buffer_size=100,
    consolidate_every=20,
    hop_limit=4,
    activation_threshold=0.25,
)
dnt = DNT(config=config)
```

**Export and restore state:**

```python
snapshot = dnt.export()                # serialize to dict
dnt2 = DNT.from_snapshot(snapshot)    # restore from dict
```

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `llm_provider` | `"openai"` | LLM backend |
| `llm_model` | `"gpt-4o-mini"` | Model to use for triplet extraction |
| `openai_api_key` | `""` | API key (or set `DNT_OPENAI_API_KEY` env var) |
| `buffer_size` | `50` | Max L1 buffer capacity |
| `consolidate_every` | `10` | Observations between consolidation runs |
| `max_depth` | `5` | Maximum tree depth |
| `tau1` | `0.5` | GHSOM adaptive QE threshold coefficient |
| `tau2` | `0.01` | GHSOM global stop floor |
| `hebbian_lr` | `0.1` | Hebbian learning rate (η) |
| `decay_factor` | `0.99` | LTD weight decay per tick |
| `hop_limit` | `3` | Max traversal hops per query |
| `activation_threshold` | `0.3` | Min edge weight to follow during traversal |

All parameters can also be set via environment variables prefixed with `DNT_` (e.g. `DNT_BUFFER_SIZE=100`).

---

## Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Core | ✅ Done | Models, NeuronTree, Buffer, DNT API, tests |
| 2 — Learning | 🔜 Next | Triplet extraction, Hebbian LTP/LTD, GHSOM growth |
| 3 — Memory & Tokens | ⏳ Planned | Async consolidation, snapshot, token benchmark |
| 4 — LLM & Adapters | ⏳ Planned | Pluggable LLM providers, ProjectAdapter |
| 5 — UI | ⏳ Planned | Streamlit live neuron tree visualizer |

---

## Design Principles

- **Zero configuration** — works out of the box with sensible defaults
- **Full isolation** — every DNT instance is independent, like a Docker container
- **In-memory by default** — persistence is opt-in via `export()` / `from_snapshot()`
- **Provider agnostic** — the LLM layer is swappable without touching core logic
- **Token efficient** — structural triplet data sent to LLM, not raw text chunks
