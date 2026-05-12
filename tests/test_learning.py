"""Phase 2 learning module tests."""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from dnt.config import DNTConfig
from dnt.core.models import AtomicObservation, NeuronEdge, NeuronNode, Triplet
from dnt.core.tree import NeuronTree
from dnt.learning.ghsom import GHSOMGrower
from dnt.learning.hebbian import HebbianLearner
from dnt.learning.triplet import TripletExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tree(*labels: str) -> NeuronTree:
    tree = NeuronTree()
    for label in labels:
        tree.add_node(NeuronNode(label=label))
    return tree


def _obs(text: str, logic_type: str = "fact") -> AtomicObservation:
    return AtomicObservation(raw_text=text, logic_type=logic_type)  # type: ignore[arg-type]


def _triplet(subject: str, obj: str, logic_type: str = "fact") -> Triplet:
    return Triplet(
        subject=subject,
        predicate="related_to",
        object=obj,
        logic_type=logic_type,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# TripletExtractor tests
# ---------------------------------------------------------------------------


class TestTripletExtractor:
    def _extractor(self, api_key: str = "") -> TripletExtractor:
        return TripletExtractor(DNTConfig(openai_api_key=api_key))

    # --- heuristic path (no API key) ---

    @pytest.mark.asyncio
    async def test_heuristic_two_entities(self):
        extractor = self._extractor()
        # inject a simple mock nlp that returns two entities
        mock_ent_a = MagicMock()
        mock_ent_a.text = "Alice"
        mock_ent_b = MagicMock()
        mock_ent_b.text = "Google"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent_a, mock_ent_b]
        mock_doc.noun_chunks = []
        mock_nlp = MagicMock(return_value=mock_doc)
        extractor._nlp = mock_nlp

        result = await extractor.extract(_obs("Alice works at Google"))
        assert len(result) == 1
        assert result[0].subject == "Alice"
        assert result[0].object == "Google"

    @pytest.mark.asyncio
    async def test_heuristic_fewer_than_two_entities_returns_empty(self):
        extractor = self._extractor()
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_doc.noun_chunks = []
        mock_nlp = MagicMock(return_value=mock_doc)
        extractor._nlp = mock_nlp

        result = await extractor.extract(_obs("hello world"))
        assert result == []

    # --- OpenAI path ---

    @pytest.mark.asyncio
    async def test_openai_path_called_when_key_set(self):
        extractor = self._extractor(api_key="sk-test")

        # mock entity extraction
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "Alice"
        mock_doc.ents = [mock_ent]
        mock_doc.noun_chunks = []
        extractor._nlp = MagicMock(return_value=mock_doc)

        expected = [Triplet(subject="Alice", predicate="works_at", object="Google")]
        with patch.object(extractor, "_openai_extract", new=AsyncMock(return_value=expected)):
            result = await extractor.extract(_obs("Alice works at Google"))

        assert result == expected

    @pytest.mark.asyncio
    async def test_falls_back_to_heuristic_on_api_error(self):
        extractor = self._extractor(api_key="sk-test")

        mock_doc = MagicMock()
        mock_ent_a = MagicMock()
        mock_ent_a.text = "Alice"
        mock_ent_b = MagicMock()
        mock_ent_b.text = "Google"
        mock_doc.ents = [mock_ent_a, mock_ent_b]
        mock_doc.noun_chunks = []
        extractor._nlp = MagicMock(return_value=mock_doc)

        with patch.object(extractor, "_openai_extract", side_effect=Exception("API down")):
            result = await extractor.extract(_obs("Alice works at Google"))

        # heuristic should still return a triplet
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_logic_type_preserved(self):
        extractor = self._extractor()

        mock_doc = MagicMock()
        mock_ent_a = MagicMock()
        mock_ent_a.text = "dark mode"
        mock_ent_b = MagicMock()
        mock_ent_b.text = "light mode"
        mock_doc.ents = [mock_ent_a, mock_ent_b]
        mock_doc.noun_chunks = []
        extractor._nlp = MagicMock(return_value=mock_doc)

        result = await extractor.extract(_obs("prefers dark mode over light mode", "preference"))
        assert all(t.logic_type == "preference" for t in result)


# ---------------------------------------------------------------------------
# HebbianLearner tests
# ---------------------------------------------------------------------------


class TestHebbianLearner:
    def _learner(self, lr: float = 0.1, decay: float = 0.99) -> HebbianLearner:
        return HebbianLearner(DNTConfig(hebbian_lr=lr, decay_factor=decay))

    def test_ltp_creates_nodes_and_edge(self):
        learner = self._learner()
        tree = NeuronTree()
        learner.update(tree, [_triplet("Alice", "Google")])

        assert tree.find_node_by_label("Alice") is not None
        assert tree.find_node_by_label("Google") is not None
        assert tree.edge_count == 1

    def test_ltp_increases_existing_edge_weight(self):
        learner = self._learner(lr=0.2)
        tree = NeuronTree()
        t = _triplet("Alice", "Google")

        learner.update(tree, [t])
        w1 = tree.get_edge_weight(
            tree.find_node_by_label("Alice").id,
            tree.find_node_by_label("Google").id,
        )

        learner.update(tree, [t])
        w2 = tree.get_edge_weight(
            tree.find_node_by_label("Alice").id,
            tree.find_node_by_label("Google").id,
        )

        # second update should increase (LTD decay is small, LTP should dominate)
        assert w2 is not None and w1 is not None
        assert w2 >= w1

    def test_ltd_decays_edge_weights(self):
        learner = self._learner(lr=0.0, decay=0.5)
        tree = NeuronTree()
        n1 = NeuronNode(label="X")
        n2 = NeuronNode(label="Y")
        tree.add_node(n1)
        tree.add_node(n2)
        tree.add_edge(NeuronEdge(source=n1.id, target=n2.id, relation="test", weight=1.0))

        learner.update(tree, [])  # no triplets — only LTD fires

        w = tree.get_edge_weight(n1.id, n2.id)
        assert w is not None and w < 1.0

    def test_preference_learns_faster_than_fact(self):
        config = DNTConfig(hebbian_lr=0.1, decay_factor=1.0)  # no decay
        learner = HebbianLearner(config)

        tree_fact = NeuronTree()
        tree_pref = NeuronTree()

        learner.update(tree_fact, [_triplet("A", "B", "fact")])
        learner.update(tree_pref, [_triplet("A", "B", "preference")])

        w_fact = tree_fact.get_edge_weight(
            tree_fact.find_node_by_label("A").id,
            tree_fact.find_node_by_label("B").id,
        )
        w_pref = tree_pref.get_edge_weight(
            tree_pref.find_node_by_label("A").id,
            tree_pref.find_node_by_label("B").id,
        )

        assert w_pref is not None and w_fact is not None
        assert w_pref > w_fact

    def test_ltp_raises_quantization_error(self):
        learner = self._learner(lr=0.1)
        tree = NeuronTree()
        learner.update(tree, [_triplet("Alice", "Google")])

        node = tree.find_node_by_label("Alice")
        assert node is not None and node.quantization_error > 0

    def test_weight_capped_at_one(self):
        learner = self._learner(lr=1.0, decay=1.0)
        tree = NeuronTree()
        t = _triplet("A", "B")

        for _ in range(20):
            learner.update(tree, [t])

        a = tree.find_node_by_label("A")
        b = tree.find_node_by_label("B")
        w = tree.get_edge_weight(a.id, b.id)
        assert w is not None and w <= 1.0


# ---------------------------------------------------------------------------
# GHSOMGrower tests
# ---------------------------------------------------------------------------


class TestGHSOMGrower:
    def _grower(self, tau1: float = 0.5, tau2: float = 0.01, max_depth: int = 5) -> GHSOMGrower:
        return GHSOMGrower(DNTConfig(tau1=tau1, tau2=tau2, max_depth=max_depth))

    def test_no_growth_when_qe_below_tau2(self):
        grower = self._grower(tau2=0.5)
        tree = _make_tree("A")
        tree.find_node_by_label("A").quantization_error = 0.1  # below tau2

        new_nodes = grower.grow_if_needed(tree)
        assert new_nodes == 0

    def test_growth_when_qe_exceeds_threshold(self):
        grower = self._grower(tau1=0.5, tau2=0.01)
        tree = _make_tree("A")
        tree.find_node_by_label("A").quantization_error = 1.0  # > 1.0 * 0.5

        new_nodes = grower.grow_if_needed(tree)
        assert new_nodes == 2
        assert tree.node_count == 3

    def test_children_have_incremented_level(self):
        grower = self._grower()
        tree = _make_tree("Root")
        tree.find_node_by_label("Root").quantization_error = 1.0

        grower.grow_if_needed(tree)

        children = [n for n in tree.all_nodes() if n.level == 1]
        assert len(children) == 2

    def test_parent_qe_reset_after_expansion(self):
        grower = self._grower()
        tree = _make_tree("Root")
        root = tree.find_node_by_label("Root")
        root.quantization_error = 1.0
        original_qe = root.quantization_error

        grower.grow_if_needed(tree)

        assert root.quantization_error < original_qe * 0.5

    def test_max_depth_prevents_growth(self):
        grower = self._grower(max_depth=0)
        tree = _make_tree("A")
        tree.find_node_by_label("A").quantization_error = 10.0

        new_nodes = grower.grow_if_needed(tree)
        assert new_nodes == 0

    def test_children_inherit_half_parent_qe(self):
        grower = self._grower()
        tree = _make_tree("Root")
        root = tree.find_node_by_label("Root")
        root.quantization_error = 2.0

        grower.grow_if_needed(tree)

        children = [n for n in tree.all_nodes() if n.level == 1]
        for child in children:
            assert child.quantization_error == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Full consolidation integration test
# ---------------------------------------------------------------------------


class TestConsolidation:
    @pytest.mark.asyncio
    async def test_consolidate_grows_tree(self):
        from dnt.core.dnt import DNT

        config = DNTConfig(
            consolidate_every=2,
            buffer_size=50,
            openai_api_key="",
            tau1=0.01,
            tau2=0.001,
        )
        dnt = DNT(config=config)

        # inject a mock extractor that always returns one triplet
        fixed_triplets = [_triplet("Alice", "Google", "fact")]
        dnt._triplet_extractor.extract = AsyncMock(return_value=fixed_triplets)

        await dnt.observe("observation one")
        await dnt.observe("observation two")  # triggers consolidate()

        # tree should have nodes from the triplet
        assert dnt.stats().node_count >= 2
        assert dnt.stats().edge_count >= 1

    @pytest.mark.asyncio
    async def test_consolidate_empty_buffer_is_noop(self):
        from dnt.core.dnt import DNT

        dnt = DNT()
        before = dnt.stats()
        await dnt.consolidate()
        after = dnt.stats()

        assert before.node_count == after.node_count
        assert before.edge_count == after.edge_count
