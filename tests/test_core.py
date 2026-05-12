"""Faz 1 temel unit testleri."""
import asyncio
import time
from uuid import uuid4

import pytest
import pytest_asyncio

from dnt.config import DNTConfig
from dnt.core.buffer import WorkingMemoryBuffer
from dnt.core.dnt import DNT
from dnt.core.models import AtomicObservation, NeuronEdge, NeuronNode, Triplet
from dnt.core.tree import NeuronTree


# ---------------------------------------------------------------------------
# Model testleri
# ---------------------------------------------------------------------------


class TestNeuronNode:
    def test_varsayilan_degerler(self):
        node = NeuronNode(label="Apple")
        assert node.label == "Apple"
        assert node.level == 0
        assert node.quantization_error == 0.0
        assert node.vector is None
        assert node.source_ids == []

    def test_benzersiz_id(self):
        a = NeuronNode(label="X")
        b = NeuronNode(label="X")
        assert a.id != b.id


class TestNeuronEdge:
    def test_varsayilan_agirlik(self):
        edge = NeuronEdge(source=uuid4(), target=uuid4(), relation="knows")
        assert edge.weight == 0.1
        assert edge.relation == "knows"

    def test_last_activated_otomatik(self):
        before = time.time()
        edge = NeuronEdge(source=uuid4(), target=uuid4(), relation="test")
        assert edge.last_activated >= before


class TestAtomicObservation:
    def test_varsayilan_logic_type(self):
        obs = AtomicObservation(raw_text="AAPL hissesi yükseldi")
        assert obs.logic_type == "fact"
        assert obs.source == "user"

    def test_logic_type_dogrulama(self):
        obs = AtomicObservation(raw_text="test", logic_type="preference")
        assert obs.logic_type == "preference"


class TestTriplet:
    def test_temel_triplet(self):
        t = Triplet(subject="Alice", predicate="works_at", object="Google")
        assert t.confidence == 1.0
        assert t.logic_type == "fact"


# ---------------------------------------------------------------------------
# WorkingMemoryBuffer testleri
# ---------------------------------------------------------------------------


class TestWorkingMemoryBuffer:
    def _obs(self, text: str) -> AtomicObservation:
        return AtomicObservation(raw_text=text)

    def test_push_ve_len(self):
        buf = WorkingMemoryBuffer(max_size=5)
        buf.push(self._obs("merhaba"))
        assert len(buf) == 1

    def test_max_size_asimaz(self):
        buf = WorkingMemoryBuffer(max_size=3)
        for i in range(10):
            buf.push(self._obs(f"gözlem {i}"))
        assert len(buf) == 3

    def test_search_buluyor(self):
        buf = WorkingMemoryBuffer()
        buf.push(self._obs("AAPL hissesi soruldu"))
        buf.push(self._obs("hava bugün güzel"))
        results = buf.search("AAPL hissesi")
        assert len(results) == 1
        assert "AAPL" in results[0].raw_text

    def test_search_bos_sonuc(self):
        buf = WorkingMemoryBuffer()
        buf.push(self._obs("bitcoin fiyatı"))
        assert buf.search("AAPL") == []

    def test_flush_bosaltiyor(self):
        buf = WorkingMemoryBuffer()
        buf.push(self._obs("test"))
        flushed = buf.flush()
        assert len(flushed) == 1
        assert len(buf) == 0

    def test_peek_temizlemiyor(self):
        buf = WorkingMemoryBuffer()
        buf.push(self._obs("test"))
        buf.peek()
        assert len(buf) == 1


# ---------------------------------------------------------------------------
# NeuronTree testleri
# ---------------------------------------------------------------------------


class TestNeuronTree:
    def test_dugum_ekle(self):
        tree = NeuronTree()
        node = NeuronNode(label="Apple")
        tree.add_node(node)
        assert tree.node_count == 1
        assert tree.get_node(node.id) == node

    def test_kenar_ekle(self):
        tree = NeuronTree()
        n1 = NeuronNode(label="A")
        n2 = NeuronNode(label="B")
        tree.add_node(n1)
        tree.add_node(n2)
        edge = NeuronEdge(source=n1.id, target=n2.id, relation="links_to")
        tree.add_edge(edge)
        assert tree.edge_count == 1

    def test_hop_traversal_bulur(self):
        tree = NeuronTree()
        node = NeuronNode(label="AAPL hisse")
        tree.add_node(node)
        hits = tree.hop_traversal("AAPL", hop_limit=2)
        assert len(hits) >= 1
        assert hits[0].id == node.id

    def test_hop_traversal_bos_agac(self):
        tree = NeuronTree()
        assert tree.hop_traversal("herhangi bir şey") == []

    def test_serialization(self):
        tree = NeuronTree()
        n1 = NeuronNode(label="X")
        n2 = NeuronNode(label="Y")
        tree.add_node(n1)
        tree.add_node(n2)
        tree.add_edge(NeuronEdge(source=n1.id, target=n2.id, relation="test"))

        data = tree.to_dict()
        restored = NeuronTree.from_dict(data)
        assert restored.node_count == 2
        assert restored.edge_count == 1


# ---------------------------------------------------------------------------
# DNT entegrasyon testleri
# ---------------------------------------------------------------------------


class TestDNT:
    @pytest.mark.asyncio
    async def test_observe_string(self):
        dnt = DNT()
        await dnt.observe("Kullanıcı AAPL sordu")
        assert dnt.stats().observe_count == 1
        assert dnt.stats().buffer_size == 1

    @pytest.mark.asyncio
    async def test_observe_dict(self):
        dnt = DNT()
        await dnt.observe({"text": "Kullanıcı portföy sordu", "source": "chat"})
        assert dnt.stats().observe_count == 1

    @pytest.mark.asyncio
    async def test_query_buffer_hit(self):
        dnt = DNT()
        await dnt.observe("Kullanıcı AAPL hissesi sordu")
        context = await dnt.query("AAPL hissesi")
        assert "AAPL" in context

    @pytest.mark.asyncio
    async def test_query_no_hit(self):
        dnt = DNT()
        await dnt.observe("hava güzel bugün")
        context = await dnt.query("AAPL")
        assert "bulunamadı" in context

    @pytest.mark.asyncio
    async def test_consolidate_tetikleniyor(self):
        config = DNTConfig(consolidate_every=3, buffer_size=50)
        dnt = DNT(config=config)
        for i in range(3):
            await dnt.observe(f"gözlem {i}")
        # consolidate() buffer'ı temizler
        assert dnt.stats().buffer_size == 0

    @pytest.mark.asyncio
    async def test_export_import(self):
        dnt = DNT()
        await dnt.observe("test verisi")
        snapshot = dnt.export()

        dnt2 = DNT.from_snapshot(snapshot)
        assert dnt2.stats().observe_count == dnt.stats().observe_count
        assert dnt2.stats().buffer_size == dnt.stats().buffer_size

    @pytest.mark.asyncio
    async def test_stats_doner(self):
        dnt = DNT()
        stats = dnt.stats()
        assert stats.node_count == 0
        assert stats.observe_count == 0

    def test_repr(self):
        dnt = DNT()
        assert "DNT(" in repr(dnt)
