"""
DNT Dashboard — Streamlit live visualization.

Run with:
    PYTHONPATH=. streamlit run dnt/ui/dashboard.py
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Set
from uuid import UUID

from dnt.core.tree import NeuronTree


# ===========================================================================
# Pure helper functions — no Streamlit dependency, fully testable
# ===========================================================================

_LEVEL_COLORS = ["#4A90D9", "#7BC96F", "#F5A623", "#BD10E0"]
_ACTIVE_COLOR = "#E74C3C"
_NODE_BASE_SIZE = 15
_NODE_MAX_BONUS = 30


def node_color(level: int, active: bool = False) -> str:
    """Return hex color for a node based on its level."""
    if active:
        return _ACTIVE_COLOR
    return _LEVEL_COLORS[min(level, len(_LEVEL_COLORS) - 1)]


def node_size(qe: float) -> int:
    """Return pixel size for a node based on its quantization error."""
    return int(_NODE_BASE_SIZE + min(qe * 10, _NODE_MAX_BONUS))


def build_graph_data(
    tree: NeuronTree,
    active_ids: Optional[Set[str]] = None,
) -> Dict:
    """
    Build serialisable node/edge data for pyvis from a NeuronTree.
    Uses tree.to_dict() so no private attributes are touched.
    """
    active = active_ids or set()
    raw = tree.to_dict()

    nodes = []
    for nd in raw["nodes"]:
        nid = str(nd["id"])
        label = nd["label"]
        level = nd["level"]
        qe = nd["quantization_error"]
        truncated = label[:22] + ("…" if len(label) > 22 else "")
        nodes.append(
            {
                "id": nid,
                "label": truncated,
                "title": (
                    f"<b>{label}</b><br>"
                    f"Level: {level}<br>"
                    f"QE: {qe:.4f}<br>"
                    f"ID: {nid[:8]}…"
                ),
                "color": node_color(level, active=nid in active),
                "size": node_size(qe),
                "level": level,
            }
        )

    edges = []
    for ed in raw["edges"]:
        weight = ed.get("weight", 0.1)
        edges.append(
            {
                "from": ed["source"],
                "to": ed["target"],
                "label": ed.get("relation", ""),
                "width": max(1, int(weight * 8)),
                "title": f"relation: {ed.get('relation', '')}<br>weight: {weight:.3f}",
            }
        )

    return {"nodes": nodes, "edges": edges}


def compute_token_savings(observations: List[str], dnt_context: str) -> Dict:
    """
    Estimate token counts for RAG (raw text) vs DNT (active-path context).
    Rule of thumb: 1 token ≈ 4 characters.
    """

    def _tokens(text: str) -> int:
        return max(1, len(text) // 4)

    rag = _tokens("\n".join(observations))
    dnt = _tokens(dnt_context)
    return {
        "rag_tokens": rag,
        "dnt_tokens": dnt,
        "savings_ratio": round(rag / max(dnt, 1), 2),
        "savings_pct": round((1 - dnt / max(rag, 1)) * 100, 1),
    }


def get_node_detail(tree: NeuronTree, node_id_str: str) -> Dict:
    """Return a detail dict for a node given its string UUID."""
    try:
        node = tree.get_node(UUID(node_id_str))
    except (ValueError, AttributeError):
        return {}
    if node is None:
        return {}

    neighbors = tree.get_active_neighbors(node.id, threshold=0.0)
    return {
        "id": node_id_str,
        "label": node.label,
        "level": node.level,
        "qe": round(node.quantization_error, 4),
        "summary": node.summary,
        "source_ids": node.source_ids,
        "connections": [
            {
                "label": n.label,
                "relation": r,
                "weight": round(tree.get_edge_weight(node.id, n.id) or 0.0, 3),
            }
            for n, r in neighbors
        ],
    }


# ===========================================================================
# Streamlit rendering helpers
# ===========================================================================


def _build_pyvis_html(graph_data: Dict, height: int = 560) -> str:
    from pyvis.network import Network

    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#1E1E2E",
        font_color="#CDD6F4",
        directed=True,
    )
    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "barnesHut": { "gravitationalConstant": -8000, "springLength": 120 },
            "stabilization": { "iterations": 100 }
          },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
            "smooth": { "type": "continuous" },
            "font": { "size": 10, "color": "#A6ADC8" }
          },
          "interaction": { "hover": true, "tooltipDelay": 80 }
        }
        """
    )

    for n in graph_data["nodes"]:
        net.add_node(
            n["id"],
            label=n["label"],
            title=n["title"],
            color=n["color"],
            size=n["size"],
        )
    for e in graph_data["edges"]:
        net.add_edge(
            e["from"],
            e["to"],
            label=e["label"],
            width=e["width"],
            title=e["title"],
        )

    return net.generate_html()


def _run_async(coro):
    """Run an async coroutine from Streamlit's synchronous context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ===========================================================================
# Streamlit app
# ===========================================================================


def main() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    from dnt import DNT, DNTConfig

    st.set_page_config(
        page_title="DNT Dashboard",
        page_icon="🧠",
        layout="wide",
    )
    st.title("🧠 Developmental Neuron Tree")

    # ------------------------------------------------------------------ state
    if "dnt" not in st.session_state:
        st.session_state.dnt = DNT(
            config=DNTConfig(
                consolidate_every=3,
                buffer_size=50,
                activation_threshold=0.05,
                hebbian_lr=0.3,
            )
        )
    if "observations" not in st.session_state:
        st.session_state.observations: List[str] = []
    if "active_ids" not in st.session_state:
        st.session_state.active_ids: Set[str] = set()
    if "last_context" not in st.session_state:
        st.session_state.last_context = ""

    dnt: DNT = st.session_state.dnt

    # ----------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Add Observation")
        new_obs = st.text_area(
            "Observation text",
            placeholder="Alice is CEO of Acme Corp",
            height=90,
            label_visibility="collapsed",
        )
        col_obs, col_con = st.columns(2)
        if col_obs.button("Observe ➕", use_container_width=True) and new_obs.strip():
            _run_async(dnt.observe(new_obs.strip()))
            st.session_state.observations.append(new_obs.strip())
            st.toast("Observation added to buffer")

        if col_con.button("Consolidate ⚡", use_container_width=True):
            _run_async(dnt.consolidate())
            st.toast("Consolidation complete")

        st.divider()
        st.header("Live Stats")
        s = dnt.stats()
        st.metric("Nodes", s.node_count)
        st.metric("Edges", s.edge_count)
        st.metric("Buffer", f"{s.buffer_size} / {dnt._config.buffer_size}")
        st.metric("Total observed", s.observe_count)

        st.divider()
        if st.button("🔄 Reset session", use_container_width=True):
            st.session_state.dnt = DNT(config=dnt._config)
            st.session_state.observations = []
            st.session_state.active_ids = set()
            st.session_state.last_context = ""
            st.rerun()

    # -------------------------------------------------------------------- tabs
    tab_tree, tab_query, tab_stats = st.tabs(
        ["🌳 Neuron Tree", "🔍 Query", "📊 Token Stats"]
    )

    # ============================================================= Tree tab
    with tab_tree:
        s = dnt.stats()
        if s.node_count == 0:
            st.info(
                "No nodes yet.  \n"
                "Add observations in the sidebar and press **Consolidate ⚡**."
            )
        else:
            active = st.session_state.active_ids
            legend_cols = st.columns(len(_LEVEL_COLORS) + 1)
            for i, col in enumerate(legend_cols[: len(_LEVEL_COLORS)]):
                col.markdown(
                    f'<span style="color:{_LEVEL_COLORS[i]}">■</span> Level {i}',
                    unsafe_allow_html=True,
                )
            legend_cols[-1].markdown(
                f'<span style="color:{_ACTIVE_COLOR}">■</span> Query hit',
                unsafe_allow_html=True,
            )

            graph_data = build_graph_data(dnt._tree, active_ids=active)
            try:
                html = _build_pyvis_html(graph_data)
                components.html(html, height=580, scrolling=False)
            except Exception as exc:
                st.warning(f"pyvis render failed: {exc}")
                st.json({"nodes": len(graph_data["nodes"]), "edges": len(graph_data["edges"])})

            st.divider()
            st.subheader("Node Inspector")
            label_to_id = {n["label"]: n["id"] for n in graph_data["nodes"]}
            chosen = st.selectbox("Select node", ["—"] + sorted(label_to_id))
            if chosen != "—":
                detail = get_node_detail(dnt._tree, label_to_id[chosen])
                m1, m2, m3 = st.columns(3)
                m1.metric("Level", detail.get("level", "—"))
                m2.metric("QE", detail.get("qe", "—"))
                m3.metric("Out-edges", len(detail.get("connections", [])))
                if detail.get("summary"):
                    st.info(f"**Summary:** {detail['summary']}")
                conns = detail.get("connections", [])
                if conns:
                    st.table(conns)

    # ============================================================= Query tab
    with tab_query:
        query_input = st.text_input(
            "Ask the tree",
            placeholder="Who works at Acme Corp?",
        )
        if st.button("Run query 🔍") and query_input.strip():
            context = _run_async(dnt.query(query_input.strip()))
            st.session_state.last_context = context

            hits = dnt._tree.hop_traversal(
                query_input.strip(),
                hop_limit=dnt._config.hop_limit,
                activation_threshold=dnt._config.activation_threshold,
            )
            st.session_state.active_ids = {str(n.id) for n in hits}

        if st.session_state.last_context:
            st.subheader("Context returned to LLM")
            st.code(st.session_state.last_context, language="text")
            if st.session_state.active_ids:
                st.caption(
                    f"🔴 {len(st.session_state.active_ids)} node(s) highlighted "
                    "— switch to the **Neuron Tree** tab to see them."
                )

    # ============================================================= Stats tab
    with tab_stats:
        obs = st.session_state.observations
        ctx = st.session_state.last_context
        if not obs or not ctx:
            st.info("Add observations and run a query to see token savings.")
        else:
            sv = compute_token_savings(obs, ctx)
            c1, c2, c3 = st.columns(3)
            c1.metric("RAG tokens (est.)", f"~{sv['rag_tokens']:,}")
            c2.metric("DNT tokens (est.)", f"~{sv['dnt_tokens']:,}")
            c3.metric("Savings", f"{sv['savings_ratio']:.1f}×", f"{sv['savings_pct']}%")

            try:
                import pandas as pd

                chart = pd.DataFrame(
                    {
                        "Method": ["RAG (raw text)", "DNT (active paths)"],
                        "Tokens": [sv["rag_tokens"], sv["dnt_tokens"]],
                    }
                ).set_index("Method")
                st.bar_chart(chart, color=["#4A90D9"])
            except ImportError:
                st.progress(sv["dnt_tokens"] / max(sv["rag_tokens"], 1))

            st.caption(
                f"Estimate based on {len(obs)} observation(s). "
                "1 token ≈ 4 characters (GPT rule-of-thumb)."
            )


if __name__ == "__main__":
    main()
