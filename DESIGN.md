# Engram — Design Document

> Cognitive memory layer for AI agents. Single-file, embeddable, local-first.

---

## 0. Implementation status

**All roadmap versions (v0.1 → v1.0) are shipped and green.** This document reflects both the original design intent and the as-built reality; divergences are noted in §6 and §11.

| Version | Description | Status |
|---------|-------------|--------|
| v0.1 | SQLite schema, `observe()`, `recall()` (cosine), fastembed | ✅ shipped |
| v0.2 | Importance scoring (Ebbinghaus-Hebbian), access log, `decay()` | ✅ shipped |
| v0.3 | Reflection loop, LLM adapters, fact extraction, contradiction detection, bitemporal validity | ✅ shipped |
| v0.4 | Entity extraction, episode→entity edges (Hebbian), BFS spreading-activation recall | ✅ shipped |
| v0.5 | Bitemporal queries — `as_of`, `timeline()` | ✅ shipped |
| v0.6 | MCP server (FastMCP, 6 tools), LangChain + LlamaIndex adapters | ✅ shipped |
| v1.0 | Benchmark suite (`engram-bench`), synthetic recall fixture, README | ✅ shipped |

**Test suite:** 457 tests, all green, plus a SQLCipher module skipped where the
library is absent. `mypy --strict` passes. `ruff` clean.

---

## 1. Problem statement

Existing memory solutions for AI agents are inadequate:

| Solution | Has | Lacks |
|----------|-----|-------|
| Pinecone / Chroma / Qdrant | Vector similarity search | Time, importance, context, graph |
| Mem0 | Fact extraction + vector store | Temporal validity, lightweight design (LLM call per write) |
| Zep / Graphiti | Temporal graph, decay | Heavyweight, server-based, opinionated |
| Letta (MemGPT) | Hierarchical memory tiers | Tied to its own agent runtime |
| LangChain memory | Toy primitives | Not production |
| Neo4j + custom | Flexible graph | You build the cognitive layer yourself |

**Gap:** there is no lightweight, embeddable, cognitively-grounded memory primitive. Engram fills it.

---

## 2. Cognitive model

Taxonomy adapted from Tulving / Squire (real cognitive science):

```
                    ┌─────────────────┐
                    │  Sensory Buffer │  ← last N tokens,
                    │   (~seconds)    │     raw context
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Working Memory  │  ← active session,
                    │   (~minutes)    │     current goals
                    └────────┬────────┘
                             ▼
              ┌──────────────┴──────────────┐
              ▼                             ▼
       ┌────────────┐                ┌────────────┐
       │ Episodic   │                │ Semantic   │
       │ "what      │ ──extraction──▶│  "facts"   │
       │  happened" │                │            │
       └────────────┘                └────────────┘
              │                             │
              └─────────────┬───────────────┘
                            ▼
                    ┌──────────────┐
                    │ Procedural   │  ← "how I usually
                    │  patterns    │     do X"
                    └──────────────┘
```

**Critical distinction**: most systems conflate episodic ("Ivan said X yesterday") with semantic ("Ivan is a CTO"). These are different types with different retrieval and decay logic.

---

## 3. Six novel mechanisms

### 3.1. Bitemporal validity

Every fact carries **two times**:
- `valid_from / valid_to` — when the fact was true in reality
- `recorded_at / superseded_at` — when the system learned/replaced it

Enables queries like:
- *"What did I believe about Ivan in March?"* (valid-time travel)
- *"When did I first learn about the new project?"* (transaction-time travel)

Bitemporal modeling is rare in the AI memory space. This is a real differentiator.

### 3.2. Importance scoring (Ebbinghaus + Hebbian)

Each memory has a dynamic score:

```
importance(m, t) =
    salience(m)                              # subjective weight at encoding
  * exp(-λ * (t - last_access(m)))           # exponential decay (Ebbinghaus)
  + α * log(1 + access_count(m))             # reinforcement from use
  + β * emotional_weight(m)                  # affective tag
```

`λ`, `α`, `β` are configurable. Memories below threshold are pruned during reflection.

### 3.3. Spreading-activation retrieval

Not naive top-K cosine:

1. Find **seed memories** matching the query.
2. **Spread activation** through graph edges with decay.
3. Rank by `α·similarity + β·activation + γ·importance`.

This mimics human recall — one memory triggers related ones. Without it, the agent matches vectors rather than understanding context.

### 3.4. Reflection loop (agent's "sleep")

The most important component. Runs in the background:
- after N new episodes
- after idle (e.g. 5 min of silence)
- on explicit `mem.reflect()`

Steps:
1. **Cluster** recent episodes by entity / topic
2. **Extract facts** — LLM call to produce semantic triples (s, p, o)
3. **Detect contradictions** — same (s, p), different o
4. **Update validity** — close each older same-(s, p) fact's `valid_to` (newest wins; a same-object re-extraction closes silently, as agreement)
5. **Decay** — recompute importance scores
6. **Prune** — drop memories below threshold
7. **Compress** — old episode clusters → summary episodes (lossy, like the brain)

This is **System-2 thinking** for memory. Competitors either skip it or do it synchronously (slow).

### 3.5. Contradiction handling

Never overwrite. Keep both:

```
Fact #142: Ivan works_at "Acme"   [valid: 2023-01 → 2024-06]
Fact #891: Ivan works_at "Globex" [valid: 2024-06 → now]
```

`recall()` returns current by default. With `as_of=date`, returns whatever was valid then.

Settled semantics (2026-07-21):

- The conflict unit is the (subject, predicate) pair, not the entity. Facts
  about one entity under different predicates never interact.
- Reflection auto-adjudicates by recency: inserting an extracted fact closes
  every older active fact with the same (s, p), leaving one active value per
  pair. Only a differing object counts as a contradiction (increments
  `contradictions_resolved`, emits `contradiction_found`); re-extracting the
  same object is agreement and supersedes silently, refreshing provenance and
  confidence through the `superseded_by` chain. Mirrors `contradictions()`,
  which skips same-object pairs.
- Validity stays segmented per row: each supersession closes the old interval
  at `now` and opens a new one, so any `as_of` instant matches exactly one
  row. Walk the `superseded_by` chain to recover how long a value has been
  continuously true.
- Object comparison is exact string equality. Normalizing spellings ("Globex"
  vs "Globex Corp") is the extractor's job; two spellings of one truth under
  the same predicate are deliberately a real contradiction, resolved by
  recency.
- `assert_fact()` (the manual path) never closes anything: coexisting active
  values per (s, p) are allowed there, and `contradictions()` is the surface
  that lets the caller resolve them.

### 3.6. Provenance tracking

Every memory knows where it came from:

```python
mem.why(fact_id="f_891")
# {
#   "fact": "Ivan works_at Globex",
#   "extracted_from": ["episode_e_445", "episode_e_512"],
#   "extracted_by": "reflection_run_2024-06-15",
#   "confidence": 0.87,
#   "model": "claude-opus-4.7"
# }
```

Critical for AI safety and trust — the agent can explain **why** it "knows" something.

---

## 4. API design

```python
from engram import Engram

mem = Engram(
    path="./agent.engram",          # single file, like SQLite
    embedder="bge-small-en-v1.5",   # local model
    reflection_model="claude-haiku" # for background reflection
)

# === WRITE ===
mem.observe(
    "Ivan said he finally moved to Globex",
    actors=["Ivan"],
    salience=0.7,
    tags=["work", "career"],
)

mem.assert_fact(
    subject="Ivan",
    predicate="works_at",
    object="Globex",
    confidence=0.95,
    source="conversation_2024-06-15",
)

# === READ ===
results = mem.recall("where does Ivan work?", k=5)

# Time travel
results = mem.recall(
    "where does Ivan work?",
    as_of=datetime(2024, 3, 1),  # → returns "Acme"
)

# Spreading activation
results = mem.recall(
    "what do I know about Ivan?",
    mode="spreading",
    depth=2,
    decay=0.5,
)

# === REFLECT ===
mem.reflect()         # synchronous
mem.reflect_async()   # background

# === INTROSPECT ===
mem.why(fact_id="f_891")        # provenance
mem.timeline(entity="Ivan")     # full history for an entity
mem.contradictions()            # surface inconsistencies

# === FORGET ===
mem.forget(memory_id="...", reason="user requested")
mem.forget_entity("Ivan")       # GDPR right-to-be-forgotten
```

---

## 5. Storage schema (SQLite, single file)

```sql
-- Raw episodes (what happened)
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    timestamp DATETIME,
    actors JSON,                 -- ["Ivan", "Maria"]
    tags JSON,
    salience REAL,
    emotional_valence REAL,
    embedding BLOB,              -- via sqlite-vec
    summary_of JSON              -- if this is a consolidated summary
);

-- Semantic facts (triples with time)
CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    valid_from DATETIME,
    valid_to DATETIME,           -- NULL = still valid
    recorded_at DATETIME,
    superseded_at DATETIME,
    superseded_by TEXT,          -- FK to facts.id
    confidence REAL,
    derived_from JSON            -- supporting episodes / facts
);

-- Entities (people, places, concepts)
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT,
    type TEXT,                   -- person / place / concept / ...
    aliases JSON,
    first_seen DATETIME,
    last_seen DATETIME
);

-- Graph (edges between everything)
CREATE TABLE edges (
    src_id TEXT,
    dst_id TEXT,
    relation TEXT,               -- "mentions" / "causes" / "part_of"
    weight REAL,                 -- edge strength (Hebbian)
    created_at DATETIME,
    PRIMARY KEY (src_id, dst_id, relation)
);

-- Reflection log (audit trail)
CREATE TABLE reflections (
    id TEXT PRIMARY KEY,
    started_at DATETIME,
    finished_at DATETIME,
    episodes_processed INTEGER,
    facts_extracted INTEGER,
    contradictions_resolved INTEGER,
    model_used TEXT,
    cost_tokens INTEGER
);

-- Access metrics (drive importance)
CREATE TABLE access_log (
    memory_id TEXT,
    accessed_at DATETIME,
    query TEXT,
    rank INTEGER
);
```

Vectors via `sqlite-vec` extension. Graph stored as SQL edges; loaded into in-memory `petgraph` / `NetworkX` for traversal when needed.

---

## 6. Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Core | Python 3.11+ | Broad AI audience, fast iteration |
| Storage | SQLite + sqlite-vec | One file, embedded, zero-config |
| Embeddings | fastembed (ONNX, `bge-small-en-v1.5`, 384-dim) | Local, no heavy Python deps |
| Reflection LLM | Pluggable: Anthropic / OpenAI / Gemini / DeepSeek / Qwen / Kimi / Stub | User picks cost/quality trade-off |
| Graph | SQL edges + pure-Python BFS traversal (no external graph lib) | No extra dependency |
| API | Python lib + MCP server (FastMCP, stdio) | MCP is the agent-ecosystem standard; no REST server |
| CLI | `engram inspect / recall / observe / reflect / timeline / forget / list-agents` | Debugging is critical |
| Working memory | `WorkingMemory` class — LRU scratchpad, Miller 7±2 capacity, optional flush to store | Active reasoning without polluting long-term store |
| Data portability | `export_json()` / `import_json()` / `backup()` | Migration, snapshots |
| Encryption | Optional SQLCipher via `[encryption]` extra; `Engram(key=...)` | Privacy-critical deployments |

---

## 7. Roadmap

All versions shipped. Items are marked ✅.

### v0.1 — Skeleton ✅
- ✅ SQLite schema, basic migrations
- ✅ `observe()`, `recall()` (cosine only)
- ✅ Local embedder via fastembed (`bge-small-en-v1.5`)
- ✅ Smoke tests

### v0.2 — Importance + decay ✅
- ✅ Importance scoring (Ebbinghaus-Hebbian formula)
- ✅ Access log
- ✅ Background `decay()` job, `DecayConfig`

### v0.3 — Reflection loop ✅
- ✅ Fact-extraction prompt
- ✅ Contradiction detection
- ✅ Bitemporal validity intervals (`valid_from/to`, `recorded_at/superseded_at`)
- ✅ `reflect()` / `reflect_async()`
- ✅ `assert_fact()`, `why()`, `contradictions()`
- ✅ LLM adapters: Anthropic, OpenAI, Gemini, DeepSeek, Qwen, Kimi, Stub

### v0.4 — Graph + spreading activation ✅
- ✅ Entity extraction from fact triples
- ✅ Episode→entity edges (Hebbian weight accumulation)
- ✅ BFS spreading-activation recall (`mode="spreading"`, depth, decay)

### v0.5 — Bitemporal queries ✅
- ✅ `recall(as_of=T)` filters by valid timestamp
- ✅ `timeline(entity)` — full fact history for an entity
- ✅ `store.get_facts_as_of()` for point-in-time fact validity

### v0.6 — Integrations ✅
- ✅ MCP server (FastMCP, 6 tools, stdio transport)
- ✅ LangChain adapter: `EngramRetriever` + `EngramChatMessageHistory`
- ✅ LlamaIndex adapter: `EngramMemory`
- ✅ Optional extras in pyproject.toml: `[mcp]`, `[langchain]`, `[llamaindex]`

### v1.0 — Polish ✅
- ✅ `engram-bench` CLI entry, latency suite (p50/p99/throughput)
- ✅ Recall harness (hit@1, hit@5, MRR) over LoCoMo-format data, with a synthetic fixture
- ✅ Cost benchmark (tokens/$ per 1k episodes)
- ✅ README overhaul with real numbers

---

## 8. Benchmarks (measured on M-series, bge-small-en-v1.5)

### Write latency (n=300)

| Operation | p50 | p99 | Throughput |
|-----------|-----|-----|-----------|
| `observe` | 4.1 ms | 4.8 ms | 236 ep/s |

### Search latency as the store grows

`engram-bench scale`. p50, milliseconds, against a pre-computed embedding.

| episodes | cosine | `as_of` | scoped | RSS |
|---|---|---|---|---|
| 1 000 | 0.18 | 0.87 | 0.05 | 342 MB |
| 10 000 | 3.03 | 10.0 | 0.05 | 346 MB |
| 100 000 | 31.8 | 107 | 0.09 | 353 MB |

Exact scan, so unscoped cost is linear in the episodes that survive the
filters. Scoped recall is flat because `agent_id` partitions the vec0 table:
an agent pays for its own episodes, not for the store. `as_of` is the
expensive path at ~3.4x cosine, and the first one that will need attention,
around a million episodes or sooner if a caller leans on time travel.

Memory does not grow with the store; SQLite reads vectors from disk and the
resident footprint is the ONNX runtime.

### Recall quality

Full 500 questions of LongMemEval-S, 246 738 turns ingested one episode per
turn, `bge-small-en-v1.5`:

| mode | session@5 | session@10 | turn@5 | turn@10 |
|---|---|---|---|---|
| `hybrid` (default) | 0.968 | 0.982 | 0.820 | 0.892 |
| `cosine` | 0.956 | 0.978 | 0.772 | 0.862 |

Session recall finds the right conversation; turn recall finds the specific
turn the dataset flagged as holding the answer, 896 of 246 738. The 18-point
gap at k=5 is the interesting part, and it is the number a system quoting a
bare "R@k" is usually not quoting.

The blend weights were swept in one pass (eight configurations, same store
per question): both ends are worse than the middle, and the default moved
from `0.7 / 0.3` to `0.5 / 0.5` on a margin of about five questions in five
hundred.

Raw per-question records: `benchmarks/results/`. Method and per-type
breakdown: `docs/api-reference.md`.

Hybrid became the default on the strength of the turn column. The first run
of this benchmark scored it identically to cosine on all 500 questions, which
is what exposed an implicit-AND bug in the BM25 query; the table is from a
second run against the fixed code, in which cosine reproduced its earlier
figures exactly.

The synthetic fixture in `engram/benchmarks/data/` remains a smoke test for
the retrieval path. Its scores were previously quoted here as "LoCoMo recall
quality", which read as a result on the LoCoMo benchmark and was not one.

### Reflection cost

| Model | Cost per 1k episodes |
|-------|---------------------|
| claude-haiku-4.5 | ~$0.0056 |
| gpt-4o-mini | ~$0.0033 |

---

## 9. Why this is genuinely unique

The novelty is in the **combination + DX**, not any single mechanism in isolation:

1. ✅ Bitemporal validity — almost no one does this
2. ✅ Spreading-activation retrieval — almost no one does this
3. ✅ Provenance as first-class — critical for trust
4. ✅ Embeddable single-file — "SQLite of agent memory"
5. ✅ Cognitive-science taxonomy — not marketing, real grounding
6. ✅ MCP-native — ready for the 2026 agent ecosystem

---

## 10. Open questions

- ✅ **Embedding model default** — resolved: `bge-small-en-v1.5` (384-dim, fast). `bge-base` available as a user override.
- ✅ **Reflection trigger heuristics** — resolved: configurable N (episodes since last reflection) + optional idle-time threshold in `DecayConfig`.
- ✅ **Decay λ** — resolved: default tuned from benchmark; user-configurable via `DecayConfig`.
- ✅ **Multi-agent / shared memory** — resolved ahead of schedule: `agent_id` parameter added in v1. Each `Engram(agent_id=...)` instance scopes reads/writes; cross-agent recall available via `recall(cross_agent=True)`.
- ✅ **Encryption-at-rest** — resolved: optional SQLCipher via `Engram(key="passphrase")` + `pip install 'engdbram[encryption]'`. Supports `rekey()` and encrypted `backup()`. Plain (no-key) databases unchanged.

---

## 11. As-built divergences from spec

Things that differ from the original design, or were built ahead of schedule:

- **FastAPI not implemented.** §6 originally listed `optional FastAPI`. The interface is Python lib + MCP server only. No REST layer was added — MCP covers the agent-ecosystem use case.
- **Graph traversal** — original spec mentioned `petgraph`/`NetworkX`. Actual implementation uses pure-Python BFS over SQL edges loaded into memory. No external graph library dependency.
- **`agent_id` scoping** — originally flagged as a v2 concern. Implemented in v1: `Engram(agent_id="...")` scopes all writes; multiple agents can share one `.engram` file.
- **`WorkingMemory`** — not in original roadmap. Implemented as a Miller 7±2 LRU scratchpad (`engram/working_memory.py`) that can optionally flush evicted items to the long-term store.
- **`compress()`** — §3.4 step 7 described lossy compression of old episode clusters. Implemented as an explicit `mem.compress()` method (not automatic during reflection).
- **`observe_many()`** — batch write not in spec; added for throughput.
- **`export_json()` / `import_json()` / `backup()`** — data portability not in original spec.
- **Additional LLM adapters** — spec listed Anthropic / OpenAI / Ollama. Shipped: Anthropic, OpenAI, Gemini, DeepSeek, Qwen, Kimi, Stub (Ollama uses the OpenAI-compatible adapter).
