"""
DNT dashboard server — FastAPI backend.
Serves app.html and provides REST endpoints for the UI.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from dnt import DNT, DNTConfig
from dnt.ui.dashboard import build_graph_data, compute_token_savings

app = FastAPI(title="DNT", docs_url=None, redoc_url=None)

_HTML = (Path(__file__).parent / "app.html").read_text(encoding="utf-8")

# ── Singleton DNT instance ────────────────────────────────────────────────────

_dnt: DNT = DNT(config=DNTConfig(
    consolidate_every=999,   # manual only from UI
    buffer_size=50,
    activation_threshold=0.05,
    hebbian_lr=0.3,
))
_observations: List[str] = []
_last_context: str = ""
_active_ids: List[str] = []


# ── Request models ────────────────────────────────────────────────────────────

class ObserveReq(BaseModel):
    text: str

class QueryReq(BaseModel):
    q: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stats_dict() -> Dict[str, Any]:
    s = _dnt.stats()
    buf = _dnt._buffer.peek()
    return {
        "nodes":    s.node_count,
        "edges":    s.edge_count,
        "buffer":   s.buffer_size,
        "capacity": _dnt._config.buffer_size,
        "observed": s.observe_count,
        "buffer_items": [o.raw_text for o in buf[-5:]],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_HTML)


# Silence leftover Streamlit health-check requests from stale browser tabs
@app.get("/_stcore/{path:path}")
async def stcore_stub() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/stats")
async def stats() -> JSONResponse:
    return JSONResponse(_stats_dict())


@app.post("/observe")
async def observe(req: ObserveReq) -> JSONResponse:
    if not req.text.strip():
        raise HTTPException(400, "Empty observation")
    await _dnt.observe(req.text.strip())
    _observations.append(req.text.strip())
    return JSONResponse({"ok": True, **_stats_dict()})


@app.post("/consolidate")
async def consolidate() -> JSONResponse:
    await _dnt.consolidate()
    return JSONResponse({"ok": True, **_stats_dict()})


@app.post("/query")
async def query(req: QueryReq) -> JSONResponse:
    global _last_context, _active_ids
    if not req.q.strip():
        raise HTTPException(400, "Empty query")
    context = await _dnt.query(req.q.strip())
    hits = _dnt._tree.hop_traversal(
        req.q.strip(),
        hop_limit=_dnt._config.hop_limit,
        activation_threshold=_dnt._config.activation_threshold,
    )
    _last_context = context
    _active_ids   = [str(n.id) for n in hits]

    savings: Optional[Dict] = None
    if _observations and _last_context:
        savings = compute_token_savings(_observations, _last_context)

    return JSONResponse({
        "context":    context,
        "hit_count":  len(hits),
        "active_ids": _active_ids,
        "savings":    savings,
    })


@app.get("/graph.html", response_class=HTMLResponse)
async def graph_html() -> HTMLResponse:
    from dnt.ui.dashboard import _build_pyvis_html
    active = set(_active_ids)
    data   = build_graph_data(_dnt._tree, active_ids=active)
    if not data["nodes"]:
        empty = (
            "<html><body style='background:#000;display:flex;align-items:center;"
            "justify-content:center;height:100vh;font-family:system-ui'>"
            "<p style='color:#3A3A3C;font-size:0.85rem'>"
            "Add observations and press Consolidate</p></body></html>"
        )
        return HTMLResponse(empty)
    return HTMLResponse(_build_pyvis_html(data))


@app.get("/graph-data")
async def graph_data() -> JSONResponse:
    """Return node list for the inspector dropdown."""
    raw = _dnt._tree.to_dict()
    nodes = [
        {"id": str(n["id"]), "label": n["label"],
         "level": n["level"], "qe": round(n["quantization_error"], 4)}
        for n in raw["nodes"]
    ]
    return JSONResponse({"nodes": nodes})


@app.get("/node/{node_id}")
async def node_detail(node_id: str) -> JSONResponse:
    from uuid import UUID
    try:
        node = _dnt._tree.get_node(UUID(node_id))
    except ValueError:
        raise HTTPException(400, "Invalid node id")
    if node is None:
        raise HTTPException(404, "Node not found")
    neighbors = _dnt._tree.get_active_neighbors(node.id, threshold=0.0)
    return JSONResponse({
        "label":   node.label,
        "level":   node.level,
        "qe":      round(node.quantization_error, 4),
        "summary": node.summary,
        "connections": [
            {
                "relation": r,
                "target":   n.label,
                "weight":   round(_dnt._tree.get_edge_weight(node.id, n.id) or 0.0, 3),
            }
            for n, r in neighbors
        ],
    })


@app.post("/reset")
async def reset() -> JSONResponse:
    global _dnt, _observations, _last_context, _active_ids
    _dnt          = DNT(config=_dnt._config)
    _observations = []
    _last_context = ""
    _active_ids   = []
    return JSONResponse({"ok": True, **_stats_dict()})
