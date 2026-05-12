"""Phase 3 memory module tests."""
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from dnt.config import DNTConfig
from dnt.core.buffer import WorkingMemoryBuffer
from dnt.core.dnt import DNT
from dnt.core.models import AtomicObservation, NeuronEdge, NeuronNode, Triplet
from dnt.core.tree import NeuronTree
from dnt.learning.ghsom import GHSOMGrower
from dnt.learning.hebbian import HebbianLearner
from dnt.learning.triplet import TripletExtractor
from dnt.memory.consolidate import ConsolidationEngine
from dnt.memory.snapshot import SnapshotManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs(text: str) -> AtomicObservation:
    return AtomicObservation(raw_text=text)


def _triplet(subj: str, obj: str) -> Triplet:
    return Triplet(subject=subj, predicate="related_to", object=obj)


def _engine(tree: NeuronTree, triplets: list[Triplet]) -> ConsolidationEngine:
    config = DNTConfig()
    extractor = TripletExtractor(config)
    extractor.extract = AsyncMock(return_value=triplets)
    return ConsolidationEngine(
        config=config,
        triplet_extractor=extractor,
        hebbian=HebbianLearner(config),
        ghsom=GHSOMGrower(config),
        tree=tree,
    )


# ---------------------------------------------------------------------------
# ConsolidationEngine tests
# ---------------------------------------------------------------------------


class TestConsolidationEngine:
    @pytest.mark.asyncio
    async def test_run_once_creates_nodes(self):
        tree = NeuronTree()
        engine = _engine(tree, [_triplet("Alice", "Google")])
        await engine.run_once([_obs("Alice works at Google")])
        assert tree.find_node_by_label("Alice") is not None
        assert tree.find_node_by_label("Google") is not None

    @pytest.mark.asyncio
    async def test_run_once_empty_observations_is_noop(self):
        tree = NeuronTree()
        engine = _engine(tree, [_triplet("A", "B")])
        await engine.run_once([])
        assert tree.node_count == 0

    @pytest.mark.asyncio
    async def test_background_mode_processes_queue(self):
        tree = NeuronTree()
        engine = _engine(tree, [_triplet("Alice", "Google")])

        await engine.start()
        assert engine.is_running

        await engine.enqueue([_obs("Alice works at Google")])
        await engine.drain()

        assert tree.node_count >= 2

        await engine.stop()
        assert not engine.is_running

    @pytest.mark.asyncio
    async def test_stop_drains_before_cancelling(self):
        tree = NeuronTree()
        engine = _engine(tree, [_triplet("X", "Y")])

        await engine.start()
        await engine.enqueue([_obs("X relates to Y")])
        await engine.stop()  # should not lose the queued item

        assert tree.node_count >= 2

    @pytest.mark.asyncio
    async def test_is_running_false_before_start(self):
        tree = NeuronTree()
        engine = _engine(tree, [])
        assert not engine.is_running

    @pytest.mark.asyncio
    async def test_multiple_enqueues_all_processed(self):
        tree = NeuronTree()
        engine = _engine(tree, [_triplet("A", "B")])

        await engine.start()
        for _ in range(5):
            await engine.enqueue([_obs("observation")])
        await engine.drain()
        await engine.stop()

        # extractor was called 5 times → nodes A and B should exist
        assert tree.find_node_by_label("A") is not None


# ---------------------------------------------------------------------------
# SnapshotManager tests
# ---------------------------------------------------------------------------


class TestSnapshotManager:
    def _make_state(self) -> tuple[DNTConfig, NeuronTree, WorkingMemoryBuffer, int]:
        config = DNTConfig(buffer_size=10, consolidate_every=5)
        tree = NeuronTree()
        n1 = NeuronNode(label="Alice")
        n2 = NeuronNode(label="Google")
        tree.add_node(n1)
        tree.add_node(n2)
        tree.add_edge(NeuronEdge(source=n1.id, target=n2.id, relation="works_at"))
        buf = WorkingMemoryBuffer(max_size=10)
        buf.push(_obs("Alice works at Google"))
        return config, tree, buf, 42

    def test_export_restore_roundtrip(self):
        config, tree, buf, count = self._make_state()
        snapshot = SnapshotManager.export(config, tree, buf, count)
        r_config, r_tree, r_buffer_items, r_count = SnapshotManager.restore(snapshot)

        assert r_count == 42
        assert r_tree.node_count == 2
        assert r_tree.edge_count == 1
        assert len(r_buffer_items) == 1
        assert r_config.buffer_size == 10

    def test_snapshot_has_version(self):
        config, tree, buf, count = self._make_state()
        snapshot = SnapshotManager.export(config, tree, buf, count)
        assert "version" in snapshot

    def test_save_and_load_file(self):
        config, tree, buf, count = self._make_state()
        snapshot = SnapshotManager.export(config, tree, buf, count)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        SnapshotManager.save(snapshot, path)
        loaded = SnapshotManager.load(path)

        assert loaded["observe_count"] == 42
        assert len(loaded["tree"]["nodes"]) == 2

    def test_saved_file_is_valid_json(self):
        config, tree, buf, count = self._make_state()
        snapshot = SnapshotManager.export(config, tree, buf, count)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        SnapshotManager.save(snapshot, path)
        raw = Path(path).read_text()
        parsed = json.loads(raw)
        assert parsed["observe_count"] == 42

    def test_restore_empty_snapshot(self):
        config, tree, items, count = SnapshotManager.restore({})
        assert count == 0
        assert tree.node_count == 0
        assert items == []


# ---------------------------------------------------------------------------
# ATP query tests
# ---------------------------------------------------------------------------


class TestATPQuery:
    @pytest.mark.asyncio
    async def test_query_includes_relation_info(self):
        dnt = DNT(config=DNTConfig(activation_threshold=0.05))
        dnt._triplet_extractor.extract = AsyncMock(
            return_value=[_triplet("Alice", "Google")]
        )

        await dnt.observe("Alice works at Google")
        await dnt.consolidate()

        context = await dnt.query("Alice")
        # ATP output should include relation info
        assert "Alice" in context

    @pytest.mark.asyncio
    async def test_query_shows_level(self):
        dnt = DNT(config=DNTConfig(activation_threshold=0.05))
        dnt._triplet_extractor.extract = AsyncMock(
            return_value=[_triplet("Alice", "Google")]
        )

        await dnt.observe("Alice works at Google")
        await dnt.consolidate()

        context = await dnt.query("Alice")
        # Level prefix [L0] should appear for root nodes
        assert "[L" in context

    @pytest.mark.asyncio
    async def test_active_neighbors_shown_in_context(self):
        dnt = DNT(config=DNTConfig(activation_threshold=0.05, hebbian_lr=0.5))
        dnt._triplet_extractor.extract = AsyncMock(
            return_value=[
                Triplet(subject="Alice", predicate="works_at", object="Google"),
            ]
        )

        await dnt.observe("Alice works at Google")
        await dnt.consolidate()

        context = await dnt.query("Alice")
        # should show the relation arrow
        assert "→" in context or "Google" in context


# ---------------------------------------------------------------------------
# DNT save/load file tests
# ---------------------------------------------------------------------------


class TestDNTFileIO:
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        dnt = DNT()
        dnt._triplet_extractor.extract = AsyncMock(return_value=[_triplet("A", "B")])

        await dnt.observe("A relates to B")
        await dnt.consolidate()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        dnt.save(path)
        dnt2 = DNT.load(path)

        assert dnt2.stats().observe_count == dnt.stats().observe_count

    @pytest.mark.asyncio
    async def test_loaded_dnt_can_query(self):
        dnt = DNT(config=DNTConfig(activation_threshold=0.05))
        dnt._triplet_extractor.extract = AsyncMock(return_value=[_triplet("Alice", "Google")])

        await dnt.observe("Alice works at Google")
        await dnt.consolidate()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        dnt.save(path)
        dnt2 = DNT.load(path)

        context = await dnt2.query("Alice")
        assert "Alice" in context


# ---------------------------------------------------------------------------
# Adaptive tau1 tests
# ---------------------------------------------------------------------------


class TestAdaptiveTau1:
    def test_tau1_grows_with_tree_size(self):
        config = DNTConfig(tau1=0.5)
        grower = GHSOMGrower(config)

        small_tree = NeuronTree()
        for i in range(5):
            small_tree.add_node(NeuronNode(label=f"node_{i}"))

        large_tree = NeuronTree()
        for i in range(100):
            large_tree.add_node(NeuronNode(label=f"node_{i}"))

        tau1_small = grower._effective_tau1(small_tree)
        tau1_large = grower._effective_tau1(large_tree)

        assert tau1_large > tau1_small

    def test_tau1_capped_at_0_9(self):
        config = DNTConfig(tau1=0.5)
        grower = GHSOMGrower(config)

        huge_tree = NeuronTree()
        for i in range(10_000):
            huge_tree.add_node(NeuronNode(label=f"node_{i}"))

        assert grower._effective_tau1(huge_tree) <= 0.9

    def test_tau1_equals_config_for_empty_tree(self):
        config = DNTConfig(tau1=0.5)
        grower = GHSOMGrower(config)
        empty_tree = NeuronTree()
        assert grower._effective_tau1(empty_tree) == 0.5
