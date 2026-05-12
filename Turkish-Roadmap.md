# Developmental Neuron Tree (DNT)
## Evrensel, Plug & Play Bellek Motoru

---

## Proje Amacı

DNT, herhangi bir Python projesine tek satırla entegre edilebilen, biyolojik beyinden esinlenen bir bellek motorudur. Her proje kendi bağımsız, izole nöron ağacını çalıştırır — tıpkı Docker container mantığı gibi. Aynı kod, farklı hafıza.

```python
# Kullanım bu kadar basit olmalı
dnt = DNT()
await dnt.observe("Kullanıcı AAPL hissesi sordu")
context = await dnt.query("Kullanıcı ne tür hisseler istiyor?")
```

---

## Temel Prensipler

1. **Sıfır konfigürasyon** — Varsayılan ayarlarla direkt çalışır
2. **Tam izolasyon** — Her DNT instance birbirinden habersiz
3. **In-memory** — Kalıcılık opsiyonel, export/import ile
4. **Provider agnostic** — LLM katmanı değiştirilebilir
5. **Token verimli** — RAG'a göre 10x daha az token kullanımı hedefi

---

## Mimari Genel Bakış

```
Ham Veri (her formatta)
        ↓
ProjectAdapter (format dönüştürücü)
        ↓
AtomicObservation (standart paket)
        ↓
WorkingMemoryBuffer — L1 Cache (deque, sıcak hafıza)
        ↓ (consolidate() tetiklenince)
NeuronTree — NetworkX DiGraph (soğuk, kalıcı hafıza)
        ↓
query() → L1 + NeuronTree'ye bakar
        ↓
Hop Traversal → sadece aktif yollar seçilir
        ↓
LLM'e kompakt yapısal veri gönderilir
```

### Veri Akışı Detayı

```
observe(data)
    → adapter.to_atomic(data)          # format normalize
    → buffer.push(observation)         # L1'e yaz
    → counter artır
    → counter % consolidate_every == 0
        → consolidate() tetikle (async, arka planda)

query(soru)
    → buffer.search(soru)              # L1'de ara (sıcak)
    → tree.hop_traversal(soru)         # ağaçta ara (soğuk)
    → ATP: sadece aktif yolları seç
    → kompakt bağlam döndür

consolidate()
    → buffer'daki ham veriyi al
    → triplet_extractor.extract()      # spaCy + OpenAI
    → hebbian.update(tree, triplets)   # bağları güncelle
    → ghsom.grow_if_needed(tree)       # gerekirse büyüt
    → buffer'ı temizle
```

---

## Proje Yapısı

```
dnt/
├── core/
│   ├── __init__.py
│   ├── models.py           # Pydantic veri modelleri
│   ├── tree.py             # NeuronTree (NetworkX)
│   ├── buffer.py           # WorkingMemoryBuffer
│   └── dnt.py              # Public API (DNT sınıfı)
├── learning/
│   ├── __init__.py
│   ├── triplet.py          # spaCy + OpenAI triplet çıkarımı
│   ├── hebbian.py          # Plasticity (LTP/LTD analogları)
│   └── ghsom.py            # Hiyerarşik büyüme mantığı
├── memory/
│   ├── __init__.py
│   ├── consolidate.py      # Async konsolidasyon motoru
│   └── snapshot.py         # export() / from_snapshot()
├── llm/
│   ├── __init__.py
│   ├── base.py             # Abstract LLMProvider
│   └── openai_provider.py  # OpenAI implementasyonu
├── adapters/
│   ├── __init__.py
│   └── base.py             # Abstract ProjectAdapter
├── ui/
│   ├── __init__.py
│   └── dashboard.py        # Streamlit görselleştirme
├── tests/
│   ├── test_core.py
│   ├── test_learning.py
│   └── test_memory.py
├── config.py               # DNTConfig (Pydantic Settings)
├── requirements.txt
└── README.md
```

---

## Veri Modelleri

### NeuronNode
```python
class NeuronNode(BaseModel):
    id: UUID                          # Benzersiz kimlik
    label: str                        # Varlık veya kavram etiketi
    summary: Optional[str]            # LLM tarafından üretilen özet
    quantization_error: float = 0.0   # GHSOM büyüme eşiği için
    vector: Optional[List[float]]     # Kavram embedding'i
    level: int = 0                    # Ağaç derinliği (root=0)
    source_ids: List[str] = []        # Provenance — kaynak takibi
    metadata: Dict[str, Any] = {}
```

### NeuronEdge
```python
class NeuronEdge(BaseModel):
    source: UUID
    target: UUID
    relation: str                     # "works_at", "depends_on" vb.
    weight: float = 0.1               # Hebbian ağırlığı
    last_activated: float             # Zaman damgası
```

### AtomicObservation
```python
class AtomicObservation(BaseModel):
    id: UUID
    raw_text: str                     # Ham giriş
    source: str                       # Nereden geldi
    timestamp: float
    logic_type: Literal["fact", "rule", "preference"]
    metadata: Dict[str, Any] = {}
```

### Triplet
```python
class Triplet(BaseModel):
    subject: str
    predicate: str                    # yılan_kılıfı formatında
    object: str
    logic_type: Literal["fact", "rule", "preference"]
    confidence: float = 1.0
```

---

## Öğrenme Mantığı

### Hebbian Plastisite
- **LTP (Long-Term Potentiation):** İki nöron aynı bağlamda tetiklendiğinde `weight += η * activation`
- **LTD (Long-Term Depression):** Uzun süre kullanılmayan bağlar yavaşça zayıflar `weight *= decay_factor`
- `logic_type` bazlı diferansiyel plastisite: "preference" tipi bilgiler "fact"e göre daha hızlı güncellenir

### GHSOM Büyüme
- Her nöronun `quantization_error` değeri izlenir
- `QE > parent_QE * τ1` eşiği aşılırsa alt ağaç açılır
- `max_depth` parametresi sonsuz büyümeyi engeller
- Global `t2` alt sınırı tüm ağaç için sabit durdurma noktası

### Triplet Çıkarımı (Hibrit)
```
Ham metin
    → spaCy: varlık tespiti (NER) — hızlı, ucuz
    → OpenAI function calling: ilişki tanımlama — doğru
    → Triplet(subject, predicate, object, logic_type)
```

OpenAI function calling schema:
```json
{
  "name": "extract_triplets",
  "parameters": {
    "triplets": [{
      "subject": "string (spaCy listesinden)",
      "predicate": "string (yılan_kılıfı)",
      "object": "string",
      "logic_type": "fact | rule | preference"
    }]
  }
}
```

---

## Token Verimliliği

### Standart RAG vs DNT
```
RAG:
  → tüm ilgili chunk'ları al
  → ham metni LLM'e gönder
  → yüksek token maliyeti, bağlam gürültüsü

DNT:
  → hop traversal ile sadece aktif yolları seç
  → yapısal veri (triplet + ağırlık) gönder, ham metin değil
  → hedef: 10x token tasarrufu
```

### Hop Traversal Mantığı
```
query gelir
    → sorguya semantik olarak en yakın root nörondan başla
    → aktifleşen komşulara hop et (threshold altı bağları atla)
    → maksimum hop_limit adımda dur
    → sadece bu yolun düğümlerini bağlam olarak paketle
```

---

## Konfigürasyon

```python
class DNTConfig(BaseSettings):
    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # Bellek
    buffer_size: int = 50             # L1 max kapasitesi
    consolidate_every: int = 10       # kaç observe'de bir konsolidasyon
    max_depth: int = 5                # maksimum ağaç derinliği

    # GHSOM
    tau1: float = 0.5                 # adaptif QE eşik katsayısı
    tau2: float = 0.01                # global durdurma alt sınırı
    hebbian_lr: float = 0.1          # öğrenme hızı (η)
    decay_factor: float = 0.99        # LTD zayıflatma katsayısı

    # Sorgu
    hop_limit: int = 3                # maksimum traversal adımı
    activation_threshold: float = 0.3 # bağ aktivasyon eşiği
```

---

## Public API

```python
# Başlatma
dnt = DNT()                           # varsayılan config
dnt = DNT(config=DNTConfig(...))      # özel config

# Temel işlemler
await dnt.observe(data: str | dict)   # veri ekle
await dnt.query(soru: str) -> str     # bağlam sorgula
await dnt.consolidate()               # manuel konsolidasyon

# Snapshot
snapshot = dnt.export() -> dict       # state'i dışa aktar
dnt2 = DNT.from_snapshot(snapshot)   # state'i içe aktar

# Adaptör
dnt.set_adapter(adapter: ProjectAdapter)  # format dönüştürücü ekle

# İstatistikler
dnt.stats() -> DNTStats               # token, node, edge sayıları
```

---

## Geliştirme Yol Haritası

### Faz 1 — Çekirdek ✳️ BURADAN BAŞLA
- [ ] `core/models.py` — NeuronNode, NeuronEdge, AtomicObservation, Triplet
- [ ] `core/tree.py` — NeuronTree (NetworkX DiGraph wrapper)
- [ ] `core/buffer.py` — WorkingMemoryBuffer (deque tabanlı L1)
- [ ] `core/dnt.py` — DNT ana sınıfı, observe() + query() iskeleti
- [ ] `config.py` — DNTConfig (Pydantic BaseSettings)
- [ ] `tests/test_core.py` — temel unit testler

**Faz 1 tamamlandığında:** `dnt.observe()` ve `dnt.query()` çalışıyor olmalı (LLM olmadan, basit string matching ile)

### Faz 2 — Öğrenme Mantığı
- [ ] `learning/triplet.py` — spaCy NER + OpenAI function calling
- [ ] `learning/hebbian.py` — LTP/LTD ağırlık güncelleme
- [ ] `learning/ghsom.py` — QE hesaplama + alt ağaç açma
- [ ] `tests/test_learning.py` — triplet doğruluk testleri

**Faz 2 tamamlandığında:** Gerçek veriyle ağaç kendi kendine büyüyor olmalı

### Faz 3 — Bellek & Token Verimliliği
- [ ] `memory/consolidate.py` — async event-driven konsolidasyon
- [ ] `memory/snapshot.py` — export() / from_snapshot()
- [ ] `core/dnt.py` güncelle — hop traversal + ATP query
- [ ] Token benchmark — RAG vs DNT karşılaştırma

**Faz 3 tamamlandığında:** Token tasarrufu ölçülebilir hale gelmeli

### Faz 4 — LLM & Adaptör Katmanı
- [ ] `llm/base.py` — abstract LLMProvider
- [ ] `llm/openai_provider.py` — OpenAI implementasyonu
- [ ] `adapters/base.py` — abstract ProjectAdapter
- [ ] Anthropic provider (opsiyonel)

**Faz 4 tamamlandığında:** LLM provider tek satırla değiştirilebilir

### Faz 5 — UI (Streamlit)
- [ ] `ui/dashboard.py` — canlı nöron ağacı görselleştirme
- [ ] Düğüm detay paneli (QE, label, bağ ağırlıkları)
- [ ] Query visualizer (aktif yol vurgulama)
- [ ] Token tasarrufu istatistik paneli

---

## Bağımlılıklar

```txt
# Core
pydantic>=2.0
pydantic-settings>=2.0
networkx>=3.0
asyncio

# Learning
spacy>=3.7
openai>=1.0

# UI
streamlit>=1.30
pyvis>=0.3

# Test
pytest>=7.0
pytest-asyncio>=0.23
```

---

## Kodlama Standartları

- **Tip güvenliği:** Her fonksiyon tip annotasyonu zorunlu
- **Async:** Tüm I/O işlemleri async/await
- **Tek sorumluluk:** Her modül tek bir iş yapar
- **Türkçe yorumlar:** Kod içi açıklamalar Türkçe
- **Bağımsız katmanlar:** LLM katmanı core'dan bağımsız — birini değiştirmek diğerini bozmamalı
- **Test:** Her faz kendi testleriyle birlikte teslim edilmeli

---

## Notlar

- NetworkX performans sınırı: binlerce düğüm için `rustworkx`'e geçiş opsiyonu açık tutulsun ama şimdilik NetworkX yeterli
- SQLite şu an devre dışı — tüm state in-memory, kalıcılık sadece export/import ile
- GHSOM QE eşiği Faz 2'de sabit (`tau1=0.5`) başlar, Faz 3'te adaptif hale gelir
- `consolidate()` Faz 1'de stub olarak yazılabilir, Faz 2'de gerçek implementasyon
