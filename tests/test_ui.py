"""Phase 5 UI helper function tests."""
from uuid import uuid4

import pytest

from dnt.core.models import NeuronEdge, NeuronNode
from dnt.core.tree import NeuronTree
from dnt.ui.dashboard import (
    build_graph_data,
    compute_token_savings,
    get_node_detail,
    node_color,
    node_size,
)

_ACTIVE_BG    = "#FF453A"
_LEVEL_COLORS = ["#0A84FF", "#30D158", "#FF9F0A", "#BF5AF2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tree_with_two_nodes() -> tuple[NeuronTree, NeuronNode, NeuronNode]:
    tree = NeuronTree()
    n1 = NeuronNode(label="Alice", level=0, quantization_error=0.5)
    n2 = NeuronNode(label="Google", level=1, quantization_error=0.1)
    tree.add_node(n1)
    tree.add_node(n2)
    tree.add_edge(NeuronEdge(source=n1.id, target=n2.id, relation="works_at", weight=0.8))
    return tree, n1, n2


# ---------------------------------------------------------------------------
# node_color
# ---------------------------------------------------------------------------


class TestNodeColor:
    def test_active_node_returns_active_color(self):
        assert node_color(0, active=True)["background"] == _ACTIVE_BG
        assert node_color(3, active=True)["background"] == _ACTIVE_BG

    def test_level_0_returns_first_palette_color(self):
        assert node_color(0)["background"] == _LEVEL_COLORS[0]

    def test_level_1(self):
        assert node_color(1)["background"] == _LEVEL_COLORS[1]

    def test_level_beyond_palette_returns_last_color(self):
        assert node_color(99)["background"] == _LEVEL_COLORS[-1]

    def test_non_active_never_returns_active_color(self):
        for level in range(6):
            assert node_color(level, active=False)["background"] != _ACTIVE_BG


# ---------------------------------------------------------------------------
# node_size
# ---------------------------------------------------------------------------


class TestNodeSize:
    def test_zero_qe_returns_base_size(self):
        assert node_size(0.0) == 20

    def test_size_increases_with_qe(self):
        assert node_size(1.0) > node_size(0.0)

    def test_large_qe_caps_bonus(self):
        size_large = node_size(100.0)
        size_max = node_size(10.0)
        assert size_large == size_max  # bonus capped at 30

    def test_size_is_integer(self):
        assert isinstance(node_size(0.5), int)


# ---------------------------------------------------------------------------
# build_graph_data
# ---------------------------------------------------------------------------


class TestBuildGraphData:
    def test_empty_tree_returns_empty_lists(self):
        tree = NeuronTree()
        data = build_graph_data(tree)
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_node_count_matches(self):
        tree, _, _ = _tree_with_two_nodes()
        data = build_graph_data(tree)
        assert len(data["nodes"]) == 2

    def test_edge_count_matches(self):
        tree, _, _ = _tree_with_two_nodes()
        data = build_graph_data(tree)
        assert len(data["edges"]) == 1

    def test_node_has_required_keys(self):
        tree, _, _ = _tree_with_two_nodes()
        node = build_graph_data(tree)["nodes"][0]
        for key in ("id", "label", "title", "color", "size", "level"):
            assert key in node, f"missing key: {key}"

    def test_edge_has_required_keys(self):
        tree, _, _ = _tree_with_two_nodes()
        edge = build_graph_data(tree)["edges"][0]
        for key in ("from", "to", "label", "width", "title"):
            assert key in edge

    def test_active_node_gets_active_color(self):
        tree, n1, _ = _tree_with_two_nodes()
        active_ids = {str(n1.id)}
        data = build_graph_data(tree, active_ids=active_ids)
        active_node = next(n for n in data["nodes"] if n["id"] == str(n1.id))
        assert active_node["color"]["background"] == _ACTIVE_BG

    def test_inactive_node_gets_level_color(self):
        tree, n1, _ = _tree_with_two_nodes()
        data = build_graph_data(tree, active_ids=set())
        node = next(n for n in data["nodes"] if n["id"] == str(n1.id))
        assert node["color"]["background"] == _LEVEL_COLORS[0]

    def test_edge_width_proportional_to_weight(self):
        tree, _, _ = _tree_with_two_nodes()
        edge = build_graph_data(tree)["edges"][0]
        assert edge["width"] >= 1

    def test_long_label_truncated(self):
        tree = NeuronTree()
        tree.add_node(NeuronNode(label="A" * 50))
        node = build_graph_data(tree)["nodes"][0]
        assert len(node["label"]) <= 29  # 28 chars + ellipsis

    def test_no_active_ids_arg_treated_as_empty(self):
        tree, _, _ = _tree_with_two_nodes()
        data = build_graph_data(tree)  # no active_ids arg
        for node in data["nodes"]:
            assert node["color"]["background"] != _ACTIVE_BG


# ---------------------------------------------------------------------------
# compute_token_savings
# ---------------------------------------------------------------------------


class TestComputeTokenSavings:
    def test_empty_dnt_context_ratio_is_high(self):
        obs = ["a" * 400]
        result = compute_token_savings(obs, "tiny")
        assert result["savings_ratio"] > 1.0

    def test_same_text_ratio_is_one(self):
        text = "same text here"
        result = compute_token_savings([text], text)
        assert result["savings_ratio"] == pytest.approx(1.0, abs=0.1)

    def test_returns_all_keys(self):
        result = compute_token_savings(["hello world"], "hi")
        for key in ("rag_tokens", "dnt_tokens", "savings_ratio", "savings_pct"):
            assert key in result

    def test_rag_tokens_larger_than_dnt_tokens(self):
        obs = ["x" * 200, "y" * 200, "z" * 200]
        dnt_ctx = "short context"
        result = compute_token_savings(obs, dnt_ctx)
        assert result["rag_tokens"] > result["dnt_tokens"]

    def test_savings_pct_between_0_and_100(self):
        result = compute_token_savings(["long " * 50], "short")
        assert 0 <= result["savings_pct"] <= 100

    def test_no_observations_doesnt_crash(self):
        result = compute_token_savings([], "some context")
        assert result["rag_tokens"] >= 1

    def test_token_count_min_is_one(self):
        result = compute_token_savings([""], "")
        assert result["rag_tokens"] >= 1
        assert result["dnt_tokens"] >= 1


# ---------------------------------------------------------------------------
# get_node_detail
# ---------------------------------------------------------------------------


class TestGetNodeDetail:
    def test_returns_correct_label(self):
        tree, n1, _ = _tree_with_two_nodes()
        detail = get_node_detail(tree, str(n1.id))
        assert detail["label"] == "Alice"

    def test_returns_level_and_qe(self):
        tree, n1, _ = _tree_with_two_nodes()
        detail = get_node_detail(tree, str(n1.id))
        assert detail["level"] == 0
        assert detail["qe"] == pytest.approx(0.5, abs=0.001)

    def test_connections_included(self):
        tree, n1, n2 = _tree_with_two_nodes()
        detail = get_node_detail(tree, str(n1.id))
        assert len(detail["connections"]) >= 1
        conn = detail["connections"][0]
        assert conn["target"] == "Google"
        assert conn["relation"] == "works_at"

    def test_node_with_incoming_edge_shows_connection(self):
        # Google (n2) has an incoming works_at edge from Alice — bidirectional traversal exposes it
        tree, _, n2 = _tree_with_two_nodes()
        detail = get_node_detail(tree, str(n2.id))
        assert len(detail["connections"]) >= 1
        assert any("Alice" in c["target"] for c in detail["connections"])

    def test_invalid_uuid_returns_empty(self):
        tree, _, _ = _tree_with_two_nodes()
        detail = get_node_detail(tree, "not-a-uuid")
        assert detail == {}

    def test_unknown_uuid_returns_empty(self):
        tree, _, _ = _tree_with_two_nodes()
        detail = get_node_detail(tree, str(uuid4()))
        assert detail == {}

    def test_summary_included_when_set(self):
        tree = NeuronTree()
        node = NeuronNode(label="Test", summary="This is a test node.")
        tree.add_node(node)
        detail = get_node_detail(tree, str(node.id))
        assert detail["summary"] == "This is a test node."

    def test_connection_weight_rounded(self):
        tree, n1, _ = _tree_with_two_nodes()
        detail = get_node_detail(tree, str(n1.id))
        w = detail["connections"][0]["weight"]
        assert isinstance(w, float)
        assert len(str(w).split(".")[-1]) <= 3
