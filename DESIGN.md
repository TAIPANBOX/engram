# Engram — Design Document

> Cognitive memory layer for AI agents. Single-file, embeddable, local-first.

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
4. **Update validity** — close older fact's `valid_to`
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
| Core | Python 3.11+ (Rust + bindings later if hot path needs it) | Broad AI audience, fast iteration |
| Storage | SQLite + sqlite-vec | One file, embedded, zero-config |
| Embeddings | fastembed (ONNX) | Local, no heavy Python deps |
| Reflection LLM | Pluggable: Anthropic / OpenAI / Ollama | User picks cost/quality trade-off |
| Graph | SQL edges + in-process traversal | No extra database |
| API | Python lib + optional FastAPI + MCP server | MCP is the agent-ecosystem standard |
| CLI | `engram inspect`, `engram reflect`, `engram timeline` | Debugging is critical |

---

## 7. Roadmap

### v0.1 — Skeleton (week 1-2)
- SQLite schema, basic migrations
- `observe()`, `recall()` (cosine only)
- Local embedder via fastembed
- Smoke tests on 1000 episodes

### v0.2 — Importance + decay (week 3)
- Importance scoring
- Access log
- Background decay job

### v0.3 — Reflection loop (week 4-5)
- Fact-extraction prompt
- Contradiction detection
- Validity intervals
- Async reflection

### v0.4 — Graph + spreading activation (week 6)
- Entity extraction
- Edge building
- Spreading-activation retrieval

### v0.5 — Bitemporal queries (week 7)
- `as_of` parameter
- Timeline API

### v0.6 — Integrations (week 8)
- MCP server
- LangChain / LlamaIndex adapters
- Reference agent using Engram

### v1.0 — Polish (week 9-10)
- LoCoMo benchmark vs Mem0 / Zep
- Latency tests
- Docs + landing page

---

## 8. Benchmarks (must run before v1.0)

- **LoCoMo** — standard long-conversation memory benchmark
- **MemBench**
- **Custom**: cross-session recall accuracy after 50 sessions
- **Latency**: recall p50 / p99 vs competitors
- **Cost**: tokens spent on reflection per 1000 episodes

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

## 10. Open questions to resolve during build

- Embedding model default — `bge-small` (fast, 384-dim) vs `bge-base` (better, 768-dim)?
- Reflection trigger heuristics — what's the right N? Idle-time threshold?
- How aggressive should default decay (`λ`) be? Tune via benchmark.
- Multi-agent / shared memory — v2 concern, but design schema to allow it now (add `agent_id` column).
- Encryption-at-rest — out of scope for v1, but document the threat model.
