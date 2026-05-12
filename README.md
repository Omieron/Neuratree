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
├── cli.py                  # dnt CLI entry point
tests/
├── test_core.py            # 26 tests — core layer
├── test_learning.py        # 19 tests — Hebbian + GHSOM
├── test_memory.py          # 19 tests — consolidation + snapshots
├── test_llm_adapters.py    # 26 tests — LLM providers + adapters
└── test_ui.py              # 34 tests — dashboard helpers
benchmarks/
└── token_benchmark.py      # RAG vs DNT token comparison
Makefile                    # make install / test / run / demo / bench
pyproject.toml
```

---

## Installation

```bash
git clone <repo-url>
cd Neuratree
make install
source .venv/bin/activate
```

That single command creates the virtual environment, installs all dependencies, and patches the `dnt` entry script for Python 3.14 compatibility.

**Optional — Anthropic provider:**

```bash
make install-anthropic
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

## CLI Commands

After `source .venv/bin/activate` every command below works from anywhere:

| Command | What it does |
|---|---|
| `dnt dashboard` | Launch the Streamlit visual dashboard |
| `dnt demo` | Interactive walkthrough — no API key required |
| `dnt bench` | Run the RAG vs DNT token benchmark |
| `dnt test` | Run the full test suite |

Makefile equivalents: `make run`, `make demo`, `make bench`, `make test`.

---

## Dashboard

The dashboard is a live Streamlit UI that lets you feed observations, watch the neuron tree grow, and run queries — all without writing Python.

### Launch

```bash
dnt dashboard
# or
make run
```

Your browser opens at `http://localhost:8501`.

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  SIDEBAR                  │  TABS                                    │
│                           │                                          │
│  ┌─────────────────────┐  │  [ Neuron Tree ] [ Query ] [ Token Stats]│
│  │ Observe             │  │                                          │
│  │ ___________________  │  │  (content changes per tab)              │
│  │ [type observation ] │  │                                          │
│  │ [ Observe ➕ ]      │  │                                          │
│  └─────────────────────┘  │                                          │
│                           │                                          │
│  ┌─────────────────────┐  │                                          │
│  │ [ Consolidate ⚡ ]  │  │                                          │
│  └─────────────────────┘  │                                          │
│                           │                                          │
│  Stats                    │                                          │
│  Nodes: 0                 │                                          │
│  Edges: 0                 │                                          │
│  Buffer: 0/50             │                                          │
│  Depth:  0                │                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Step-by-step walkthrough

**Step 1 — Feed observations (sidebar)**

Type anything in the text box and press **Observe ➕**. Each click pushes one observation into the L1 buffer. The "Buffer" counter in the stats panel ticks up. Nothing hits the tree yet.

```
"User asked about AAPL stock"          → buffer: 1/50
"User prefers long-term holdings"      → buffer: 2/50
"User mentioned MSFT earnings"         → buffer: 3/50
```

**Step 2 — Consolidate (sidebar)**

Press **Consolidate ⚡**. The engine flushes the buffer:
1. spaCy + LLM extract subject-predicate-object triplets from each observation
2. Hebbian learner strengthens co-activated edges (LTP) and decays the rest (LTD)
3. GHSOM grower expands any node whose quantization error exceeds its threshold

The "Nodes", "Edges", and "Depth" counters update immediately.

**Step 3 — Neuron Tree tab**

Switch to the **Neuron Tree** tab to see the interactive pyvis graph.

Node colors indicate depth level:
```
● root node (depth 0)   — blue
● depth 1               — green
● depth 2               — orange
● depth 3+              — red
```

Node size scales with the number of connections. Click any node to see its details in the **Node Inspector** panel below the graph:
- Label, level, parent UUID
- Quantization error (how "loaded" the node is)
- All outgoing edges with their Hebbian weights (0.0 – 1.0)

**Step 4 — Query tab**

Type a question in the query box and press **Query**. The traversal starts at the best-matching seed node and hops across edges that exceed the `activation_threshold`. Active nodes are highlighted in the graph. The ATP context block shows exactly what would be sent to an LLM:

```
[L0] root | mentions → AAPL (w=0.82)
[L1] AAPL | is_a → tech stock (w=0.71)
[L1] AAPL | related_to → MSFT (w=0.64)
```

This is the compressed representation — not raw text.

**Step 5 — Token Stats tab**

Shows a side-by-side bar chart comparing:
- **RAG tokens** — approximate cost of sending raw observation text to an LLM
- **DNT tokens** — cost of the compact ATP context returned by the same query

The savings ratio is displayed above the chart. With real workloads this typically reaches 5–10x.

---

## Running Tests

```bash
dnt test
# or
make test
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Core | ✅ Done | Models, NeuronTree, Buffer, DNT API, tests |
| 2 — Learning | ✅ Done | Triplet extraction, Hebbian LTP/LTD, GHSOM growth |
| 3 — Memory & Tokens | ✅ Done | Async consolidation, snapshot, token benchmark |
| 4 — LLM & Adapters | ✅ Done | Pluggable LLM providers (OpenAI + Anthropic), ProjectAdapter |
| 5 — UI | ✅ Done | Streamlit live neuron tree visualizer + CLI |

---

## Design Principles

- **Zero configuration** — works out of the box with sensible defaults
- **Full isolation** — every DNT instance is independent, like a Docker container
- **In-memory by default** — persistence is opt-in via `export()` / `from_snapshot()`
- **Provider agnostic** — the LLM layer is swappable without touching core logic
- **Token efficient** — structural triplet data sent to LLM, not raw text chunks
