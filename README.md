# Engram

> **The SQLite of agent memory.** Embeddable, local-first, cognitively grounded.

[![PyPI version](https://badge.fury.io/py/engram.svg)](https://badge.fury.io/py/engram)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](tests/)

Engram is a **single-file, embeddable memory store for AI agents** that models human-like memory: episodic storage, semantic fact extraction, importance decay, spreading-activation retrieval, and bitemporal validity — all in a local SQLite file. No server. No network. No configuration. Two lines to integrate.

```python
from engram import Engram

mem = Engram(path="./agent.engram")
mem.observe("Ivan moved from Acme to Globex last week", actors=["Ivan"])

for r in mem.recall("where does Ivan work?", k=3):
    print(f"[{r.score:.2f}] {r.episode.content}")
```

---

## Why Engram?

Every other solution forces a trade-off. Engram doesn't.

| Capability | Pinecone / Chroma / Qdrant | Mem0 | Zep / Graphiti | Letta (MemGPT) | LangChain memory | **Engram** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Vector similarity search | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| Semantic fact triples (s, p, o) | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Bitemporal validity** (`as_of` time travel) | ❌ | ❌ | ⚠️ | ❌ | ❌ | ✅ |
| **Spreading-activation retrieval** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Importance decay (Ebbinghaus) | ❌ | ❌ | ✅ | ⚠️ | ❌ | ✅ |
| **Provenance tracking** (`why()`) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Embeddable (no server) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Zero config (single file) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| MCP-native | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| LLM required at write time | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Contradiction detection | ❌ | ⚠️ | ✅ | ⚠️ | ❌ | ✅ |
| Fully local (no cloud) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |

**Key advantages over each competitor:**

- **vs. Pinecone / Chroma / Qdrant** — Vector DBs are just similarity search. Engram adds time, graph, importance, and provenance. They require a separate server; Engram is a file.
- **vs. Mem0** — Mem0 calls an LLM on *every write* (slow, costly). Engram writes instantly; LLM reflection runs async in the background. Mem0 has no temporal validity — it forgets what was true in March.
- **vs. Zep / Graphiti** — Server-based, opinionated runtimes. Engram is a Python library you `pip install`. No Docker, no API keys for the store itself.
- **vs. Letta / MemGPT** — Tied to their own agent runtime. Engram plugs into *any* framework: LangChain, LlamaIndex, raw API, or your own loop.
- **vs. LangChain memory** — LangChain memory is toy-grade: in-process list or a Redis key. No decay, no graph, no temporal queries, not production-ready.

---

## Six mechanisms you won't find elsewhere

### 1. Bitemporal validity

Every fact carries *two* independent timelines:

```
valid_from / valid_to     → when the fact was TRUE in reality
recorded_at / superseded_at → when the system LEARNED it
```

This enables queries that no other memory system supports:

```python
# What did the agent believe about Ivan in March?
mem.recall("Ivan employer", k=5, as_of=datetime(2024, 3, 1, tzinfo=UTC))

# Full fact history — every job Ivan ever had, with dates
mem.timeline("Ivan")
```

Old facts are never deleted — they're closed with `valid_to`. This is standard in financial databases and audit systems, but absent from the entire AI memory space.

### 2. Spreading-activation retrieval

Naive vector search returns "what's mathematically similar." Spreading activation returns "what's contextually connected."

```
query → seed memories (cosine KNN)
              ↓
         graph edges (Hebbian weights)
              ↓
         activated neighbors (with decay)
              ↓
    rank by: α·similarity + β·activation + γ·importance
```

One memory triggers its associates, just like human recall. If Ivan is connected to Project X, a query about Ivan surfaces relevant Project X episodes even without a semantic match.

### 3. Importance scoring (Ebbinghaus + Hebbian)

Each memory has a dynamic importance score:

```
importance(m, t) =
    salience(m)                          # subjective weight at encoding
  × exp(−λ × (t − last_access(m)))      # Ebbinghaus forgetting curve
  + α × log(1 + access_count(m))        # reinforcement from use
  + β × emotional_weight(m)             # affective tag
```

Parameters `λ`, `α`, `β` are configurable. Memories below threshold are pruned during reflection. Important memories survive; noise decays away — automatically.

### 4. Reflection loop (the agent's "sleep")

LLM calls happen *asynchronously*, never blocking writes:

1. Cluster recent episodes by entity / topic
2. Extract semantic triples via LLM (`Ivan works_at Globex`)
3. Detect contradictions (same subject + predicate, different object)
4. Close older facts with `valid_to = now`
5. Recompute importance scores
6. Prune memories below threshold

```python
thread = mem.reflect_async()  # non-blocking, runs in background
thread.join()                 # wait if needed
```

This is System-2 thinking for memory. Competitors either skip it or block the write path.

### 5. Contradiction detection + resolution

Engram never silently overwrites. When a new fact conflicts with an existing one, both are kept with correct temporal bounds:

```
Fact #142: Ivan works_at "Acme"   [valid: 2023-01 → 2024-06]  ← superseded
Fact #891: Ivan works_at "Globex" [valid: 2024-06 →    now  ]  ← current
```

Surface active contradictions at any time:

```python
for a, b in mem.contradictions():
    print(f"CONFLICT: {a.subject} {a.predicate} '{a.object}' vs '{b.object}'")
```

### 6. Provenance tracking

Every fact knows exactly where it came from — which episodes, which reflection run, which model, and with what confidence:

```python
mem.why("fact-uuid")
# {
#   "fact": "Ivan works_at Globex",
#   "extracted_from": ["ep-uuid-1", "ep-uuid-2"],
#   "extracted_by": "reflection-run-2024-06-15",
#   "confidence": 0.87,
#   "model": "claude-haiku-4-5"
# }
```

Critical for AI safety and trust: the agent can *explain* what it knows and why.

---

## Install

```bash
pip install engram

# LLM-powered reflection (optional, pick one):
pip install 'engram[anthropic]'   # Claude
pip install 'engram[openai]'      # OpenAI or any OpenAI-compatible

# Integrations:
pip install 'engram[mcp]'         # MCP server (Claude Desktop, Cursor, etc.)
pip install 'engram[langchain]'   # LangChain retriever + chat history
pip install 'engram[llamaindex]'  # LlamaIndex memory buffer

# Everything:
pip install 'engram[anthropic,mcp,langchain,llamaindex]'
```

**Requirements:** Python 3.11+, no system dependencies (fastembed downloads ONNX model on first use, ~23 MB).

---

## Quickstart

### Basic usage

```python
from engram import Engram

mem = Engram(path="./agent.engram")  # or ":memory:" for ephemeral

# Store observations — instant, no LLM needed
ep_id = mem.observe(
    "Alice presented the Q3 roadmap to the exec team",
    actors=["Alice"],
    tags=["work", "roadmap"],
    salience=0.8,           # 0–1, subjective importance at encoding
    emotional_valence=0.2,  # –1 negative … +1 positive
)

# Semantic recall
results = mem.recall("Alice roadmap", k=5)
for r in results:
    print(f"[score={r.score:.2f}] {r.episode.content}")

# Assert facts directly (no LLM)
mem.assert_fact("Ivan", "works_at", "Globex", confidence=0.95)

# Close cleanly
mem.close()
```

### Async reflection with Claude

```python
from engram import Engram, AnthropicAdapter

mem = Engram(
    path="./agent.engram",
    llm=AnthropicAdapter(model="claude-haiku-4-5-20251001"),
)

mem.observe("Ivan said he finally joined Globex last Monday")
mem.observe("The team shipped v2 of the payment service")

# Trigger reflection (async, non-blocking)
thread = mem.reflect_async()

# Keep doing agent work...
results = mem.recall("Ivan career", k=5)

thread.join()  # wait for reflection to complete
print(f"Reflected: {thread.result.facts_extracted} facts extracted")
```

### Time travel

```python
from datetime import datetime, UTC

# What did the agent know about Ivan in March 2024?
past_results = mem.recall(
    "Ivan employer",
    k=5,
    as_of=datetime(2024, 3, 1, tzinfo=UTC),
)

# Full fact timeline for an entity
for fact in mem.timeline("Ivan"):
    end = fact.valid_to.date() if fact.valid_to else "now"
    print(f"[{fact.valid_from.date()} → {end}]  Ivan {fact.predicate} {fact.object}")
```

### Spreading-activation (graph-based recall)

```python
# Surface contextually connected episodes, not just similar ones
results = mem.recall(
    "what happened with Ivan?",
    k=10,
    mode="spreading",
    depth=2,      # graph traversal depth
    decay=0.5,    # activation decay per hop
)
```

---

## Full API Reference

### `Engram(path, embedder_model, decay_config, llm)`

```python
from engram import Engram, DecayConfig, AnthropicAdapter

mem = Engram(
    path="./agent.engram",              # path to .engram file, or ":memory:"
    embedder_model="bge-small-en-v1.5", # default; local ONNX, ~23 MB
    decay_config=DecayConfig(
        lambda_=0.1,   # Ebbinghaus decay rate. 0.1 ≈ half-life ~7 days.
        alpha=0.2,     # Reinforcement weight per access.
        beta=0.1,      # Emotional valence weight.
        threshold=0.1, # Prune memories below this importance score.
    ),
    llm=AnthropicAdapter(),             # optional; only used by reflect()
)

# Context-manager supported
with Engram(path=":memory:") as mem:
    mem.observe("hello world")
```

---

### `observe(content, *, actors, tags, salience, emotional_valence) → str`

Record a raw episodic observation. Returns the episode id. No LLM call.

```python
ep_id = mem.observe(
    "Alice presented the Q3 roadmap",
    actors=["Alice"],
    tags=["work", "roadmap"],
    salience=0.8,           # subjective importance at encoding (0–1)
    emotional_valence=0.3,  # –1 (negative) … +1 (positive)
)
```

---

### `recall(query, k, *, mode, depth, decay, as_of) → list[SearchResult]`

```python
# Default: cosine similarity (fast)
results = mem.recall("where does Ivan work?", k=5)

# Graph-based spreading-activation
results = mem.recall("Ivan", k=5, mode="spreading", depth=2, decay=0.5)

# Time travel: only episodes that existed at this point
results = mem.recall(
    "Ivan employer",
    k=5,
    as_of=datetime(2024, 3, 1, tzinfo=UTC),
)
```

`SearchResult` fields: `episode`, `score`, `distance`, `importance`.

---

### `assert_fact(subject, predicate, object, *, confidence, source) → str`

Store a semantic triple directly. No LLM required. Returns fact id.

```python
fact_id = mem.assert_fact("Ivan", "works_at", "Globex", confidence=0.95)
fact_id = mem.assert_fact("Alice", "role", "CTO", source="linkedin-profile")
```

---

### `reflect() / reflect_async() → ReflectionRun`

Run the reflection loop (requires `llm`):

```python
run = mem.reflect()            # synchronous
thread = mem.reflect_async()   # background thread; call .join() when ready

print(f"{run.facts_extracted} facts from {run.episodes_processed} episodes")
print(f"Resolved {run.contradictions_resolved} contradictions")
print(f"Cost: {run.cost_tokens} tokens")
```

---

### `timeline(entity) → list[Fact]`

Full fact history for an entity, including superseded facts.

```python
for f in mem.timeline("Ivan"):
    end = f.valid_to.date() if f.valid_to else "now"
    print(f"[{f.valid_from.date()} → {end}]  Ivan {f.predicate} {f.object}")
```

---

### `why(fact_id) → dict`

Explain where a fact came from (provenance).

```python
mem.why(fact_id)
# {
#   "fact": "Ivan works_at Globex",
#   "extracted_from": ["ep-uuid-1", "ep-uuid-2"],
#   "extracted_by": "reflection-run-uuid",
#   "confidence": 0.87,
#   "model": "claude-haiku-4-5-20251001"
# }
```

---

### `contradictions() → list[tuple[Fact, Fact]]`

Surface active facts that share (subject, predicate) but differ in object.

```python
for a, b in mem.contradictions():
    print(f"CONFLICT: {a.subject} {a.predicate} '{a.object}' vs '{b.object}'")
```

---

### `decay() → int`

Recompute importance scores for all episodes (called automatically by `reflect()`).

---

## LLM Adapters

```python
from engram import AnthropicAdapter, OpenAIAdapter

# Claude (default: haiku — fast, cheap)
llm = AnthropicAdapter(model="claude-haiku-4-5-20251001")

# OpenAI
llm = OpenAIAdapter(model="gpt-4o-mini")

# Ollama or any OpenAI-compatible local model
llm = OpenAIAdapter(model="llama3.2", base_url="http://localhost:11434/v1")

mem = Engram(path="./agent.engram", llm=llm)
```

---

## Integrations

### MCP Server

Expose Engram as an MCP tool server — compatible with Claude Desktop, Cursor, and any MCP host:

```bash
python -m engram.mcp_server --path ./agent.engram
# or: ENGRAM_PATH=./agent.engram python -m engram.mcp_server
```

Available MCP tools: `observe`, `recall`, `assert_fact`, `timeline`, `why`, `reflect`.

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "python",
      "args": ["-m", "engram.mcp_server", "--path", "/path/to/agent.engram"]
    }
  }
}
```

---

### LangChain

```python
from engram import Engram
from engram.adapters.langchain import EngramRetriever, EngramChatMessageHistory

mem = Engram(path="./agent.engram")

# Retriever — plug into any RAG chain
retriever = EngramRetriever(engram=mem, k=5)
docs = retriever.invoke("Ivan project")

# Chat history — persists conversation turns across sessions
history = EngramChatMessageHistory(engram=mem)
history.add_user_message("What did Ivan say about Globex?")
history.add_ai_message("Ivan mentioned he joined Globex last week.")
```

---

### LlamaIndex

```python
from engram.adapters.llamaindex import EngramMemory
from llama_index.core.llms import ChatMessage, MessageRole

memory = EngramMemory.from_defaults(engram_path="./agent.engram", k=5)
memory.put(ChatMessage(role=MessageRole.USER, content="Hello!"))

# Semantic recall when query is provided
msgs = memory.get("Ivan Globex")
```

---

## Architecture

```
Engram
├── observe()          → Episode (content + embedding stored immediately)
│                            ↓
│                       vec_episodes  (sqlite-vec ANN index)
│                       episodes      (metadata + importance_score)
│
├── recall()  ─cosine──→ KNN search → SearchResult[]
│             ─spread──→ KNN seeds → BFS activation graph → SearchResult[]
│             ─as_of───→ time-filtered KNN → SearchResult[]
│
├── reflect()          → LLM fact extraction (async, background)
│                            ↓
│                       facts    (bitemporal s/p/o triples)
│                       entities (unique named entities)
│                       edges    (Hebbian-weighted graph)
│
├── timeline(entity)   → facts WHERE subject=? ORDER BY valid_from
├── why(fact_id)       → provenance: derived_from + extracted_by
└── contradictions()   → active facts with same (subject, predicate)
```

### Storage schema

```sql
-- Raw observations
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    timestamp DATETIME,
    actors JSON,          -- ["Ivan", "Maria"]
    tags JSON,
    salience REAL,
    emotional_valence REAL,
    importance_score REAL,
    embedding BLOB        -- via sqlite-vec
);

-- Bitemporal semantic facts
CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    subject TEXT, predicate TEXT, object TEXT,
    valid_from DATETIME,       -- when true in reality
    valid_to DATETIME,         -- NULL = still valid
    recorded_at DATETIME,      -- when system learned it
    superseded_at DATETIME,    -- when overridden
    superseded_by TEXT,        -- FK to facts.id
    confidence REAL,
    derived_from JSON          -- provenance: episode ids
);

-- Entity graph
CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT, type TEXT, ...);
CREATE TABLE edges (
    src_id TEXT, dst_id TEXT, relation TEXT,
    weight REAL,               -- Hebbian-accumulated
    PRIMARY KEY (src_id, dst_id, relation)
);
```

**Single-file design:** the `.engram` file is a plain SQLite database. Copy it, back it up, `rsync` it, or version it in git. No migration scripts, no daemon to restart.

---

## Benchmarks

Measured on Apple M-series, fastembed `bge-small-en-v1.5`, SQLite WAL mode.

### Latency (n=300 episodes in store)

| Operation | p50 | p99 | Throughput |
|---|---|---|---|
| `observe()` | 4.1 ms | 4.8 ms | 236 / s |
| `recall(mode="cosine")` | 4.3 ms | 5.0 ms | 232 / s |
| `recall(mode="spreading")` | 4.4 ms | 5.0 ms | 224 / s |

### LoCoMo Recall Accuracy (5 sessions, 15 questions)

| Metric | Score |
|---|---|
| hit@1 | 33.3% |
| hit@5 | 93.3% |
| MRR | 0.586 |

### Reflection Cost (per 1,000 episodes)

| Model | $/1k episodes |
|---|---|
| gpt-4o-mini | $0.0033 |
| claude-haiku-4.5 | $0.0056 |
| gpt-4o | $0.0542 |
| claude-sonnet-4.6 | $0.0677 |

Reflection is optional and async — you only pay when you need semantic fact extraction.

### Run benchmarks locally

```bash
python -m engram.benchmarks all
python -m engram.benchmarks latency --n 500
python -m engram.benchmarks locomo --data ./my_data.json
python -m engram.benchmarks cost --n 1000 --model gpt-4o-mini
```

---

## Configuration

```python
from engram import DecayConfig

cfg = DecayConfig(
    lambda_=0.1,    # Ebbinghaus decay rate. Higher → faster forgetting.
                    # 0.1 ≈ half-life ~7 days without reinforcement.
    alpha=0.2,      # Reinforcement weight per recall access.
    beta=0.1,       # Emotional valence weight.
    threshold=0.1,  # Prune memories below this importance during reflect().
)
mem = Engram(path="./agent.engram", decay_config=cfg)
```

---

## Development

```bash
git clone https://github.com/taipanbox/engram
cd engram
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

pytest -x           # run tests, stop on first failure
ruff check . --fix  # lint + auto-fix
ruff format .       # format
mypy engram         # type check (strict)
```

### Test coverage

```
tests/
  test_schema.py       schema + SQLite migrations
  test_observe.py      observe() + embeddings
  test_recall.py       cosine recall
  test_smoke.py        end-to-end Engram class
  test_importance.py   decay formula
  test_decay.py        decay background job
  test_store_facts.py  fact CRUD + assert_fact()
  test_reflection.py   reflection loop (stub LLM)
  test_graph.py        entity/edge CRUD + spreading recall
  test_bitemporal.py   as_of + timeline
  test_integrations.py MCP, LangChain, LlamaIndex
  test_benchmarks.py   benchmark infrastructure
```

---

## Roadmap

- [x] v0.1 — SQLite schema, `observe()`, `recall()` (cosine)
- [x] v0.2 — Importance scoring + Ebbinghaus decay
- [x] v0.3 — Reflection loop (async LLM fact extraction)
- [x] v0.4 — Entity graph + spreading-activation retrieval
- [x] v0.5 — Bitemporal queries (`as_of`, `timeline()`)
- [x] v0.6 — MCP server, LangChain + LlamaIndex adapters
- [x] v1.0 — Benchmarks, docs, production polish
- [ ] v1.1 — `forget()` / GDPR right-to-be-forgotten
- [ ] v1.2 — CLI (`engram inspect`, `engram timeline`, `engram reflect`)
- [ ] v1.3 — Multi-agent shared memory (`agent_id` column)
- [ ] v2.0 — Rust hot path for embedding + KNN

---

## Contributing

PRs welcome. Please:

1. Open an issue first for non-trivial changes.
2. Follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`).
3. Run `pytest -x && ruff check . && mypy engram` before submitting.
4. Keep PRs small — one logical change per PR.

---

## License

MIT — see [LICENSE](LICENSE).

Architecture rationale and design decisions are in [DESIGN.md](DESIGN.md).
