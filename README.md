# Engram

> Embeddable cognitive memory layer for AI agents.

**Python 3.11+ · SQLite + sqlite-vec · zero config · MIT**

Engram is a single-file, local-first memory store for AI agents — the **"SQLite of agent memory"**. It models human-like memory with episodic storage, semantic fact extraction, importance decay, spreading-activation retrieval, and bitemporal validity. Drop it into any agent with two lines of code; no server, no network, no heavy dependencies.

---

## Why

| Solution | Has | Lacks |
|---|---|---|
| Pinecone / Chroma | Vector similarity | Time, importance, graph |
| Mem0 | Fact extraction | Temporal validity, lightweight |
| Zep / Graphiti | Temporal graph | Embeddable, zero-config |
| LangChain memory | Simple API | Not production-ready |
| **Engram** | All of the above | Nothing you don't want |

---

## Install

```bash
pip install engram

# LLM-powered reflection (pick one):
pip install 'engram[anthropic]'
pip install 'engram[openai]'

# Integrations (pick any):
pip install 'engram[mcp]'        # MCP server
pip install 'engram[langchain]'  # LangChain retriever + chat history
pip install 'engram[llamaindex]' # LlamaIndex memory buffer
```

---

## Quickstart

```python
from engram import Engram

mem = Engram(path="./agent.engram")

# Store observations
mem.observe("Ivan moved from Acme to Globex last week", actors=["Ivan"])
mem.observe("The team shipped the payment service on Friday")

# Recall by semantic similarity
for r in mem.recall("Ivan career", k=3):
    print(f"[{r.score:.2f}] {r.episode.content}")

# Extract facts with an LLM (optional)
from engram import AnthropicAdapter
mem2 = Engram(path="./agent.engram", llm=AnthropicAdapter())
mem2.reflect()  # runs in background: extract facts, detect contradictions, decay

# Assert facts directly
mem.assert_fact("Ivan", "works_at", "Globex", confidence=0.95)

# Temporal queries
from datetime import datetime, UTC
mem.recall("Ivan employer", k=5, as_of=datetime(2024, 3, 1, tzinfo=UTC))
mem.timeline("Ivan")   # full fact history

# MCP server (for Claude Desktop / any MCP host)
# python -m engram.mcp_server --path ./agent.engram
```

---

## Core API

### `Engram(path, embedder_model, decay_config, llm)`

```python
from engram import Engram, DecayConfig, AnthropicAdapter

mem = Engram(
    path="./agent.engram",        # or ":memory:" for ephemeral
    embedder_model="bge-small-en-v1.5",  # default, local ONNX
    decay_config=DecayConfig(
        lambda_=0.1,   # decay rate (higher → faster forgetting)
        alpha=0.2,     # reinforcement weight per access
        beta=0.1,      # emotional valence weight
        threshold=0.1, # prune below this importance
    ),
    llm=AnthropicAdapter(),       # optional, only used by reflect()
)
```

Context-manager supported: `with Engram() as mem: ...`

---

### `observe(content, *, actors, tags, salience, emotional_valence) → str`

Record a raw episodic observation. Returns the episode id.

```python
ep_id = mem.observe(
    "Alice presented the Q3 roadmap to the exec team",
    actors=["Alice"],
    tags=["work", "roadmap"],
    salience=0.8,           # 0–1, subjective importance at encoding
    emotional_valence=0.3,  # –1 negative … +1 positive
)
```

---

### `recall(query, k, *, mode, depth, decay, as_of) → list[SearchResult]`

Retrieve the top-k most relevant episodes.

```python
# Cosine similarity (default, fast)
results = mem.recall("where does Ivan work?", k=5)

# Graph-based spreading-activation (surfaces contextually linked episodes)
results = mem.recall("Ivan", k=5, mode="spreading", depth=2, decay=0.5)

# Time travel — only episodes existing at this point in time
results = mem.recall("Ivan employer", k=5, as_of=datetime(2024, 3, 1, tzinfo=UTC))
```

`SearchResult` fields: `episode`, `score`, `distance`, `importance`.

---

### `assert_fact(subject, predicate, object, *, confidence, source) → str`

Store a semantic triple directly, without calling an LLM. Returns fact id.

```python
fact_id = mem.assert_fact("Ivan", "works_at", "Globex", confidence=0.95)
```

---

### `reflect() / reflect_async() → ReflectionRun`

Run the reflection loop (LLM required):

1. Fetch un-reflected episodes since last run
2. Extract (subject, predicate, object) triples via LLM
3. Detect and close contradicted facts
4. Rebuild entity graph edges (Hebbian reinforcement)
5. Recompute importance scores (Ebbinghaus decay)
6. Prune episodes below threshold

```python
run = mem.reflect()                    # synchronous
thread = mem.reflect_async()           # background thread
thread.join()
print(f"{run.facts_extracted} facts from {run.episodes_processed} episodes")
```

---

### `timeline(entity) → list[Fact]`

Return the full fact history for an entity, including superseded facts.

```python
for f in mem.timeline("Ivan"):
    validity = f"[{f.valid_from.date()} → {f.valid_to.date() if f.valid_to else 'now'}]"
    print(f"{validity}  Ivan {f.predicate} {f.object}")
```

---

### `why(fact_id) → dict`

Explain where a fact came from.

```python
mem.why(fact_id)
# {
#   "fact": "Ivan works_at Globex",
#   "extracted_from": ["episode-uuid-1", "episode-uuid-2"],
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

Recompute importance scores for all episodes. Called automatically by `reflect()`.

---

## LLM Adapters

```python
from engram import AnthropicAdapter, OpenAIAdapter

# Claude (default: haiku-4.5)
llm = AnthropicAdapter(model="claude-haiku-4-5-20251001")

# OpenAI
llm = OpenAIAdapter(model="gpt-4o-mini")

# Ollama (local)
llm = OpenAIAdapter(model="llama3", base_url="http://localhost:11434/v1")

mem = Engram(llm=llm)
```

---

## Integrations

### MCP Server

Exposes 6 tools to any MCP host (Claude Desktop, Cursor, custom agents):

```bash
python -m engram.mcp_server --path ./agent.engram
# or: ENGRAM_PATH=./agent.engram python -m engram.mcp_server
```

Tools: `observe`, `recall`, `assert_fact`, `timeline`, `why`, `reflect`.

---

### LangChain

```python
from engram import Engram
from engram.adapters.langchain import EngramRetriever, EngramChatMessageHistory

mem = Engram(path="./agent.engram")

# Retriever — plug into any RAG chain
retriever = EngramRetriever(engram=mem, k=5)
docs = retriever.invoke("Ivan project")

# Chat history — persists turns across sessions
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

# Semantic recall when a query is provided
msgs = memory.get("Ivan Globex")
```

---

## Architecture

```
Engram
├── observe()          → Episode (content + embedding)
│                         ↓
│                      vec_episodes (sqlite-vec KNN index)
│                      episodes table (metadata + importance_score)
│
├── recall()  ─cosine──→ KNN search → SearchResult[]
│             ─spread──→ KNN seeds → BFS activation → SearchResult[]
│             ─as_of───→ time-filtered KNN → SearchResult[]
│
├── reflect()          → LLM fact extraction
│                         ↓
│                      facts table (bitemporal s/p/o triples)
│                      entities table (unique named entities)
│                      edges table (Hebbian-weighted graph)
│
├── timeline()         → facts WHERE subject=? ORDER BY valid_from
├── why()              → provenance: derived_from + extracted_by
└── contradictions()   → active facts with same (subject, predicate)
```

**Storage:** single `.engram` SQLite file — copy, backup, or version like any file.

**Embedder:** fastembed `bge-small-en-v1.5` (384-dim, ONNX, ~23MB, runs fully local).

**Graph:** SQL edges with `PRIMARY KEY (src_id, dst_id, relation)` — Hebbian weight accumulated by `ON CONFLICT DO UPDATE SET weight = weight + excluded.weight`.

**Bitemporal facts:** every fact carries `valid_from/valid_to` (reality-time) and `recorded_at/superseded_at` (system-time), enabling both "what was true then?" and "when did I learn this?".

---

## Benchmarks

Measured on Apple M-series (local machine), fastembed `bge-small-en-v1.5`, SQLite WAL mode.

### Latency (n=300 episodes)

| Operation | p50 | p99 | Throughput |
|---|---|---|---|
| `observe()` | 4.1 ms | 4.8 ms | 236/s |
| `recall(mode="cosine")` | 4.3 ms | 5.0 ms | 232/s |
| `recall(mode="spreading")` | 4.4 ms | 5.0 ms | 224/s |

### LoCoMo Recall Accuracy (5 sessions, 15 questions)

| Metric | Score |
|---|---|
| hit@1 | 33.3% |
| hit@5 | 93.3% |
| MRR | 0.586 |

### Reflection Cost Projection (per 1,000 episodes)

| Model | $/1k episodes |
|---|---|
| claude-haiku-4.5 | $0.0056 |
| gpt-4o-mini | $0.0033 |
| claude-sonnet-4.6 | $0.0677 |
| gpt-4o | $0.0542 |

Run benchmarks locally:

```bash
python -m engram.benchmarks all
python -m engram.benchmarks latency --n 500
python -m engram.benchmarks locomo --data ./my_data.json
python -m engram.benchmarks cost --n 1000
```

---

## Configuration

```python
from engram import DecayConfig

cfg = DecayConfig(
    lambda_=0.1,   # Ebbinghaus decay rate. Higher → faster forgetting.
                   # 0.1 ≈ half-life ~7 days without reinforcement.
    alpha=0.2,     # Access-count reinforcement weight.
    beta=0.1,      # Emotional valence weight.
    threshold=0.1, # Prune episodes below this importance score.
)
mem = Engram(decay_config=cfg)
```

---

## Development

```bash
git clone https://github.com/yourname/engram
cd engram
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

pytest -x          # run tests
ruff check . --fix # lint
mypy engram        # type check
```

```
tests/
  test_schema.py         schema + migrations
  test_observe.py        observe() + embeddings
  test_recall.py         cosine recall
  test_smoke.py          end-to-end Engram class
  test_importance.py     decay formula
  test_decay.py          decay job
  test_store_facts.py    fact CRUD
  test_reflection.py     reflection loop
  test_graph.py          entity/edge CRUD + spreading recall
  test_bitemporal.py     as_of + timeline
  test_integrations.py   MCP, LangChain, LlamaIndex
  test_benchmarks.py     benchmark infrastructure
```

---

## License

MIT. See [DESIGN.md](./DESIGN.md) for architecture rationale.
