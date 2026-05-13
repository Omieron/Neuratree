"""
DNT Dashboard — Streamlit live visualization.
Run with:  dnt dashboard  or  make run
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Set
from uuid import UUID

from dnt.core.tree import NeuronTree

_LEVEL_COLORS        = ["#0A84FF", "#30D158", "#FF9F0A", "#BF5AF2"]
_LEVEL_BORDER_COLORS = ["#3A96FF", "#4AE16A", "#FFBB3A", "#CF6AFF"]
_ACTIVE_COLOR        = "#FF453A"
_ACTIVE_BORDER       = "#FF6B5A"
_NODE_BASE           = 20
_NODE_BONUS          = 32

# ─── CSS ──────────────────────────────────────────────────────────────────────
# Every selector targets a known Streamlit testid so we don't fight the
# generated class-name lottery.

_CSS = """
<style>
/* ── System font ── */
html, body, * {
    font-family: -apple-system, BlinkMacSystemFont,
                 "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    box-sizing: border-box;
}

/* ── Remove ALL Streamlit chrome ── */
#MainMenu                                      { display: none !important; }
footer                                         { display: none !important; }
[data-testid="stToolbar"]                      { display: none !important; }
[data-testid="stDecoration"]                   { display: none !important; }
[data-testid="stStatusWidget"]                 { display: none !important; }
[data-testid="stDeployButton"]                 { display: none !important; }
[data-testid="stKeyboardShortcuts"]            { display: none !important; }
[data-testid="stKeyboardShortcutsModal"]       { display: none !important; }
[data-testid="stShortcutsButton"]              { display: none !important; }
[data-modal-container]                         { display: none !important; }
.stActionButton                                { display: none !important; }

/* ── Backgrounds ── */
[data-testid="stAppViewContainer"] { background: #000000 !important; }
[data-testid="stSidebar"]          {
    background: #0A0A0A !important;
    border-right: 1px solid #1C1C1E !important;
}
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 100% !important; }
[data-testid="stSidebar"] section { padding: 1.4rem 1.1rem !important; }

/* ── Dividers ── */
hr { border: none !important; border-top: 1px solid #1C1C1E !important; margin: 1rem 0 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1C1C1E;
    border-radius: 12px;
    padding: 13px 15px !important;
    border: none !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.66rem !important;
    font-weight: 600 !important;
    color: rgba(235,235,245,0.35) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin: 0 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #F5F5F7 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 980px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.36rem 1rem !important;
    background: #1C1C1E !important;
    border: 1px solid #2C2C2E !important;
    color: #F5F5F7 !important;
    transition: background 0.12s !important;
    letter-spacing: 0.01em !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: #2C2C2E !important;
    border-color: #3A3A3C !important;
}
.stButton > button:focus { box-shadow: 0 0 0 3px rgba(10,132,255,0.25) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] { border-bottom: 1px solid #1C1C1E; margin-bottom: 1.6rem; }
button[data-testid="stTab"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: rgba(235,235,245,0.38) !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    padding: 0.55rem 1.1rem !important;
    letter-spacing: 0.01em !important;
}
button[data-testid="stTab"][aria-selected="true"] {
    color: #0A84FF !important;
    border-bottom: 2px solid #0A84FF !important;
    font-weight: 600 !important;
}
button[data-testid="stTab"]:hover { color: rgba(235,235,245,0.7) !important; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: #1C1C1E !important;
    border: 1px solid #2C2C2E !important;
    border-radius: 10px !important;
    color: #F5F5F7 !important;
    font-size: 0.88rem !important;
    caret-color: #0A84FF;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: rgba(235,235,245,0.22) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #0A84FF !important;
    box-shadow: 0 0 0 3px rgba(10,132,255,0.15) !important;
    outline: none !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #1C1C1E !important;
    border: 1px solid #2C2C2E !important;
    border-radius: 10px !important;
    color: #F5F5F7 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1C1C1E !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Code block ── */
[data-testid="stCode"] pre, code {
    background: #1C1C1E !important;
    border: 1px solid #2C2C2E !important;
    border-radius: 10px !important;
    font-family: "SF Mono", "Fira Code", "Monaco", monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.7 !important;
}

/* ── Info / Alert ── */
[data-testid="stAlert"] {
    background: #1C1C1E !important;
    border: 1px solid #2C2C2E !important;
    border-left: 3px solid #2C2C2E !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"] p { color: rgba(235,235,245,0.5) !important; font-size: 0.85rem !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {
    color: rgba(235,235,245,0.3) !important;
    font-size: 0.7rem !important;
}

/* ── Markdown text ── */
[data-testid="stMarkdownContainer"] p {
    color: rgba(235,235,245,0.55);
    font-size: 0.85rem;
    line-height: 1.5;
}
</style>
"""


# ── Pure helpers ──────────────────────────────────────────────────────────────

def node_color(level: int, active: bool = False) -> dict:
    if active:
        return {
            "background": _ACTIVE_COLOR,
            "border":     _ACTIVE_BORDER,
            "highlight":  {"background": "#FF6B5A", "border": "#FF8070"},
            "hover":      {"background": "#FF6B5A", "border": "#FF8070"},
        }
    idx = min(level, len(_LEVEL_COLORS) - 1)
    bg  = _LEVEL_COLORS[idx]
    br  = _LEVEL_BORDER_COLORS[idx]
    return {
        "background": bg,
        "border":     br,
        "highlight":  {"background": br, "border": "#FFFFFF44"},
        "hover":      {"background": br, "border": "#FFFFFF44"},
    }


def node_size(qe: float) -> int:
    return int(_NODE_BASE + min(qe * 10, _NODE_BONUS))


def build_graph_data(
    tree: NeuronTree,
    active_ids: Optional[Set[str]] = None,
) -> Dict:
    active = active_ids or set()
    raw    = tree.to_dict()

    nodes = []
    for nd in raw["nodes"]:
        nid   = str(nd["id"])
        label = nd["label"]
        level = nd["level"]
        qe    = nd["quantization_error"]
        short = label[:28] + ("…" if len(label) > 28 else "")
        nodes.append({
            "id":    nid,
            "label": short,
            "level": level,
            "title": (
                f"<b style='color:#F5F5F7;font-family:system-ui'>{label}</b><br>"
                f"<span style='color:#636366;font-family:system-ui'>"
                f"Level {level} &nbsp;·&nbsp; QE {qe:.3f}</span>"
            ),
            "color": node_color(level, active=nid in active),
            "size":  node_size(qe),
            "font":  {"color": "#F5F5F7", "size": 13, "face": "system-ui",
                      "strokeWidth": 2, "strokeColor": "#0A0A0A"},
        })

    edges = []
    for ed in raw["edges"]:
        w = ed.get("weight", 0.1)
        edges.append({
            "from":  ed["source"],
            "to":    ed["target"],
            "label": ed.get("relation", ""),
            "width": max(1, int(w * 5)),
            "title": (
                f"<span style='color:#636366;font-family:system-ui'>"
                f"{ed.get('relation','')} · {w:.3f}</span>"
            ),
            "color": {"color": "#3A3A3C", "highlight": "#0A84FF", "hover": "#0A84FF", "opacity": 0.85},
        })

    return {"nodes": nodes, "edges": edges}


def compute_token_savings(observations: List[str], dnt_context: str) -> Dict:
    def _t(s: str) -> int:
        return max(1, len(s) // 4)

    rag = _t("\n".join(observations))
    dnt = _t(dnt_context)
    return {
        "rag_tokens":    rag,
        "dnt_tokens":    dnt,
        "savings_ratio": round(rag / max(dnt, 1), 2),
        "savings_pct":   round((1 - dnt / max(rag, 1)) * 100, 1),
    }


def get_node_detail(tree: NeuronTree, node_id_str: str) -> Dict:
    try:
        node = tree.get_node(UUID(node_id_str))
    except (ValueError, AttributeError):
        return {}
    if node is None:
        return {}
    neighbors = tree.get_active_neighbors(node.id, threshold=0.0)
    return {
        "label":   node.label,
        "level":   node.level,
        "qe":      round(node.quantization_error, 4),
        "summary": node.summary,
        "connections": [
            {
                "relation": r,
                "target":   n.label,
                "weight":   round(tree.get_edge_weight(node.id, n.id) or 0.0, 3),
            }
            for n, r in neighbors
        ],
    }


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _build_pyvis_html(graph_data: Dict) -> str:
    from pyvis.network import Network

    net = Network(
        height="100%", width="100%",
        bgcolor="#0A0A0A", font_color="#F5F5F7",
        directed=True,
        cdn_resources="in_line",
    )
    net.set_options("""
    {
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -55,
          "centralGravity": 0.008,
          "springLength": 160,
          "springConstant": 0.06,
          "damping": 0.4,
          "avoidOverlap": 0.8
        },
        "stabilization": {"iterations": 180, "fit": true},
        "minVelocity": 0.5
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5, "type": "arrow"}},
        "smooth": {"enabled": true, "type": "dynamic"},
        "color":  {"inherit": false},
        "font":   {"size": 11, "color": "#48484A", "align": "middle", "strokeWidth": 0},
        "width":  1.5,
        "selectionWidth": 3
      },
      "nodes": {
        "shape": "dot",
        "borderWidth": 2,
        "borderWidthSelected": 3,
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.5)", "size": 10, "x": 0, "y": 3},
        "font": {"size": 13, "face": "system-ui", "color": "#F5F5F7", "strokeWidth": 2, "strokeColor": "#0A0A0A"}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 60,
        "hideEdgesOnDrag": false,
        "navigationButtons": false,
        "keyboard": false
      },
      "layout": {
        "improvedLayout": true
      }
    }
    """)

    for n in graph_data["nodes"]:
        net.add_node(
            n["id"], label=n["label"], title=n["title"],
            color=n["color"], size=n["size"],
        )
    for e in graph_data["edges"]:
        net.add_edge(
            e["from"], e["to"],
            label=e["label"], width=e["width"],
            title=e["title"], color=e["color"],
        )

    html = net.generate_html()
    # Make the vis.js canvas fill the wrapper div properly
    html = html.replace(
        '#mynetwork {',
        '#mynetwork { background: #0A0A0A !important; border: none !important;',
    )
    return html


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _section(label: str) -> None:
    """Render a small uppercase section label."""
    import streamlit as st
    st.markdown(
        f"<p style='font-size:0.63rem;font-weight:600;"
        f"color:rgba(235,235,245,0.28);text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:1.2rem 0 0.45rem'>{label}</p>",
        unsafe_allow_html=True,
    )


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import streamlit as st
    import streamlit.components.v1 as components
    from dnt import DNT, DNTConfig

    st.set_page_config(
        page_title="DNT",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────────
    if "dnt" not in st.session_state:
        st.session_state.dnt = DNT(config=DNTConfig(
            consolidate_every=3,
            buffer_size=50,
            activation_threshold=0.05,
            hebbian_lr=0.3,
        ))
    if "observations" not in st.session_state:
        st.session_state.observations: List[str] = []
    if "active_ids" not in st.session_state:
        st.session_state.active_ids: Set[str] = set()
    if "last_context" not in st.session_state:
        st.session_state.last_context = ""

    dnt: DNT = st.session_state.dnt

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<p style='font-size:1.05rem;font-weight:700;letter-spacing:-0.025em;"
            "color:#F5F5F7;margin:0 0 1px'>Neuron Tree</p>"
            "<p style='font-size:0.7rem;color:rgba(235,235,245,0.25);margin:0'>"
            "Memory Engine</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Stats
        s = dnt.stats()
        c1, c2 = st.columns(2)
        c1.metric("Nodes",    s.node_count)
        c2.metric("Edges",    s.edge_count)
        c1.metric("Buffer",   f"{s.buffer_size} / {dnt._config.buffer_size}")
        c2.metric("Observed", s.observe_count)

        st.divider()

        # Input
        _section("Observe")
        new_obs = st.text_area(
            "obs",
            placeholder="Alice is CEO of Acme Corp",
            height=80,
            label_visibility="collapsed",
        )
        ca, cb = st.columns(2)
        add_clicked = ca.button("Add",         use_container_width=True)
        con_clicked = cb.button("Consolidate", use_container_width=True)

        if add_clicked and new_obs.strip():
            _run_async(dnt.observe(new_obs.strip()))
            st.session_state.observations.append(new_obs.strip())
            st.rerun()

        if con_clicked:
            _run_async(dnt.consolidate())
            st.rerun()

        # Buffer preview
        buf = dnt._buffer.peek()
        if buf:
            _section("Buffer")
            for obs in buf[-4:]:
                txt = obs.raw_text[:54] + ("…" if len(obs.raw_text) > 54 else "")
                st.markdown(
                    f"<p style='font-size:0.71rem;color:rgba(235,235,245,0.3);"
                    f"margin:0 0 5px;line-height:1.45'>{txt}</p>",
                    unsafe_allow_html=True,
                )

        st.divider()
        if st.button("Reset", use_container_width=True):
            st.session_state.dnt          = DNT(config=dnt._config)
            st.session_state.observations = []
            st.session_state.active_ids   = set()
            st.session_state.last_context = ""
            st.rerun()

    # ── Page title ─────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='font-size:1.65rem;font-weight:700;letter-spacing:-0.03em;"
        "color:#F5F5F7;margin:0 0 1.4rem'>Developmental Neuron Tree</h1>",
        unsafe_allow_html=True,
    )

    tab_tree, tab_query, tab_stats = st.tabs(["Neuron Tree", "Query", "Token Stats"])

    # ══════════════════════════════════════════ Neuron Tree
    with tab_tree:
        s = dnt.stats()
        if s.node_count == 0:
            st.info("Tree is empty. Add observations in the sidebar then press Consolidate.")
        else:
            # Legend
            names  = ["Level 0", "Level 1", "Level 2", "Level 3+", "Query hit"]
            colors = _LEVEL_COLORS + [_ACTIVE_COLOR]
            dots   = "".join(
                f"<span style='display:inline-flex;align-items:center;gap:5px;"
                f"margin-right:16px'>"
                f"<span style='width:7px;height:7px;border-radius:50%;"
                f"background:{c}'></span>"
                f"<span style='font-size:0.7rem;color:rgba(235,235,245,0.35)'>{n}</span>"
                f"</span>"
                for n, c in zip(names, colors)
            )
            st.markdown(
                f"<div style='margin-bottom:12px'>{dots}</div>",
                unsafe_allow_html=True,
            )

            graph_data = build_graph_data(dnt._tree, active_ids=st.session_state.active_ids)
            try:
                components.html(_build_pyvis_html(graph_data), height=500, scrolling=False)
            except Exception as exc:
                st.warning(f"Graph error: {exc}")

            st.divider()
            _section("Node Inspector")

            label_map = {n["label"]: n["id"] for n in graph_data["nodes"]}
            chosen    = st.selectbox(
                "node", ["—"] + sorted(label_map),
                label_visibility="collapsed",
            )
            if chosen != "—":
                d     = get_node_detail(dnt._tree, label_map[chosen])
                m1, m2, m3 = st.columns(3)
                m1.metric("Level",       d.get("level", "—"))
                m2.metric("QE",          d.get("qe",    "—"))
                m3.metric("Connections", len(d.get("connections", [])))

                if d.get("summary"):
                    st.caption(d["summary"])

                conns = d.get("connections", [])
                if conns:
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame(conns)[["relation", "target", "weight"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No edges yet.")

    # ══════════════════════════════════════════ Query
    with tab_query:
        cq, cb2 = st.columns([6, 1])
        q_input  = cq.text_input(
            "q",
            placeholder="Who is Alice?   What does Acme Corp do?",
            label_visibility="collapsed",
        )
        search = cb2.button("Search", use_container_width=True)

        if search and q_input.strip():
            ctx  = _run_async(dnt.query(q_input.strip()))
            hits = dnt._tree.hop_traversal(
                q_input.strip(),
                hop_limit=dnt._config.hop_limit,
                activation_threshold=dnt._config.activation_threshold,
            )
            st.session_state.last_context = ctx
            st.session_state.active_ids   = {str(n.id) for n in hits}
            st.rerun()

        if st.session_state.last_context:
            n_hits = len(st.session_state.active_ids)
            if n_hits:
                st.caption(f"{n_hits} node(s) highlighted — switch to Neuron Tree to see them")
            _section("ATP Context")
            st.code(st.session_state.last_context, language="text")
        else:
            st.info("Type a question above and press Search.")

    # ══════════════════════════════════════════ Token Stats
    with tab_stats:
        obs = st.session_state.observations
        ctx = st.session_state.last_context

        if not obs or not ctx:
            st.info("Add observations, consolidate, run a query — token savings appear here.")
        else:
            sv = compute_token_savings(obs, ctx)

            c1, c2, c3 = st.columns(3)
            c1.metric("RAG tokens",  f"~{sv['rag_tokens']:,}", help="Raw text sent to LLM")
            c2.metric("DNT tokens",  f"~{sv['dnt_tokens']:,}", help="ATP context sent to LLM")
            c3.metric("Savings",     f"{sv['savings_ratio']:.1f}x",
                      delta=f"{sv['savings_pct']}% fewer tokens")

            try:
                import pandas as pd
                st.bar_chart(
                    pd.DataFrame({
                        "Method": ["RAG", "DNT"],
                        "Tokens": [sv["rag_tokens"], sv["dnt_tokens"]],
                    }).set_index("Method"),
                    height=220,
                )
            except ImportError:
                pass

            st.caption(f"{len(obs)} observation(s)  ·  1 token ≈ 4 chars")


if __name__ == "__main__":
    main()
