# Engram

> **The SQLite of agent memory.** Embeddable, local-first, cognitively grounded.

[![PyPI version](https://badge.fury.io/py/engram.svg)](https://badge.fury.io/py/engram)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-219%20passed-green)](tests/)

---

## What is Engram?

AI agents forget everything the moment a conversation ends. They have no sense of what happened yesterday, who they've spoken to before, or what they've already figured out. Giving an agent "memory" with a plain vector database only solves part of the problem — you get similarity search, but you lose time, structure, and understanding.

Engram is a **complete cognitive memory layer** for AI agents, packaged as a single local file. It is to agent memory what SQLite is to relational storage: self-contained, zero-configuration, production-capable, and designed to be embedded inside whatever framework or agent loop you already use.

### The cognitive model

Human memory isn't a flat list of facts — it's a dynamic system. Engram models three layers:

**Episodic memory** — raw observations stored as they happen, with actors, tags, salience, and emotional weight. No LLM required at write time; writes complete in ~4 ms.

**Semantic memory** — structured knowledge extracted from episodes via a background reflection loop: `(subject, predicate, object)` triples with full bitemporal validity. Every fact knows *when it was true* and *when the system learned it* — independently tracked on two timelines. When Ivan switches jobs, the old fact is closed with `valid_to`, not deleted. You can query what the agent believed in March even if the truth has since changed.

**Dynamic importance** — each memory carries a living importance score based on the Ebbinghaus forgetting curve, reinforced by retrieval frequency and emotional weight. Memories below threshold decay and are pruned automatically during reflection. The agent forgets what doesn't matter; critical memories survive.

### Why this matters in practice

- **Debugging**: when the agent says "Ivan works at Globex," you can call `mem.why(fact_id)` to see exactly which episode produced that belief, in which reflection run, by which model, with what confidence.
- **Compliance**: `forget_entity("Ivan")` permanently removes all episodes, facts, and graph edges connected to Ivan — a proper GDPR right-to-be-forgotten implementation.
- **Time travel**: `mem.recall("Ivan employer", as_of=datetime(2024, 3, 1))` returns what the agent knew at that point, not what it knows now. Essential for auditing and debugging.
- **Multi-agent**: multiple agents can share a single `.engram` file. Each writes its own scoped episodes; facts and the entity graph are shared. One agent's reflection enriches another's recall.

```python
from engram import Engram

with Engram(path="./agent.engram") as mem:
    mem.observe("Ivan moved from Acme to Globex last week", actors=["Ivan"])

    for r in mem.recall("where does Ivan work?", k=3):
        print(f"[{r.score:.2f}] {r.episode.content}")
```

That's it. No server to start, no API key for the store, no Docker, no configuration file.

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
| GDPR right-to-be-forgotten | ❌ | ⚠️ | ⚠️ | ❌ | ❌ | ✅ |
| **Multi-agent shared store** | ❌ | ❌ | ⚠️ | ❌ | ❌ | ✅ |
| Embeddable (no server) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Zero config (single file) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| MCP-native | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| LLM required at write time | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Contradiction detection | ❌ | ⚠️ | ✅ | ⚠️ | ❌ | ✅ |
| Fully local (no cloud) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |

**Key advantages over each competitor:**

- **vs. Pinecone / Chroma / Qdrant** — Vector DBs are just similarity search. Engram adds time, graph, importance, and provenance on top. They require a separate server process; Engram is a file you open in two lines.
- **vs. Mem0** — Mem0 calls an LLM on *every write* (slow, costly, requires API key at write time). Engram writes instantly; reflection runs async in the background. Mem0 has no temporal validity — it cannot tell you what was true in March.
- **vs. Zep / Graphiti** — Server-based runtimes with operational overhead. Engram is a Python library you `pip install`. No Docker, no API keys for the store itself, no migration scripts.
- **vs. Letta / MemGPT** — Tied to their own agent runtime and hosting model. Engram plugs into *any* framework: LangChain, LlamaIndex, raw API, or your own loop.
- **vs. LangChain memory** — LangChain memory is toy-grade: an in-process list or a Redis key. No decay, no graph, no temporal queries, not production-ready for long-running agents.

---

## Six mechanisms you won't find elsewhere

### 1. Bitemporal validity

Every fact carries *two* independent timelines:

```
valid_from / valid_to       → when the fact was TRUE in reality
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
         graph edges (Hebbian weights — reinforced by co-access)
              ↓
         activated neighbors (activation × decay per hop)
              ↓
    rank by: α·similarity + β·activation + γ·importance
```

One memory triggers its associates, just like human recall. If Ivan is connected to Project X, a query about Ivan surfaces relevant Project X episodes even without a semantic match.

### 3. Importance scoring (Ebbinghaus + Hebbian)

Each memory has a dynamic importance score recomputed by `decay()`:

```
importance(m, t) =
    salience(m) × exp(−λ × (t − last_access(m)))   # Ebbinghaus forgetting curve
  + α × log(1 + access_count(m))                    # reinforcement from retrieval
  + β × emotional_weight(m)                          # affective tag
```

Parameters `λ`, `α`, `β` are configurable via `DecayConfig`. Memories below threshold are pruned during reflection. Important memories survive; noise decays away automatically.

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
print(f"{thread.result.facts_extracted} facts, {thread.result.cost_tokens} tokens")
```

This is System-2 thinking for memory. Competitors either skip it entirely or block the write path.

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
#   "model": "claude-haiku-4-5-20251001"
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

**Requirements:** Python 3.11+, no system dependencies. `fastembed` downloads the ONNX embedding model (~23 MB) on first use; all subsequent calls are local.

---

## Quickstart

### Basic usage

```python
from engram import Engram

mem = Engram(path="./agent.engram")  # or ":memory:" for ephemeral

# Store an observation — instant, no LLM needed
ep_id = mem.observe(
    "Alice presented the Q3 roadmap to the exec team",
    actors=["Alice"],
    tags=["work", "roadmap"],
    salience=0.8,           # 0–1, subjective importance at encoding
    emotional_valence=0.2,  # –1 (negative) … +1 (positive)
)

# Semantic recall
results = mem.recall("Alice roadmap", k=5)
for r in results:
    print(f"[score={r.score:.2f}] {r.episode.content}")

# Assert facts directly (no LLM)
mem.assert_fact("Ivan", "works_at", "Globex", confidence=0.95)

mem.close()
```

### Bulk import with observe_many

When loading historical context, `observe_many()` runs a single ONNX inference pass for the whole batch and commits all rows in one transaction — about 2× faster than calling `observe()` in a loop:

```python
from engram import Engram, ObserveInput

items = [
    ObserveInput(
        content="Alice joined Globex as CTO",
        actors=["Alice"],
        tags=["hr"],
        salience=0.9,
    ),
    ObserveInput(content="Q3 planning session concluded", tags=["planning"]),
    ObserveInput(content="Ivan submitted the architecture proposal", actors=["Ivan"]),
]

with Engram(path="./agent.engram") as mem:
    ids = mem.observe_many(items)
    print(f"Inserted {len(ids)} episodes")
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

# Trigger reflection in the background
thread = mem.reflect_async()

# Keep doing agent work…
results = mem.recall("Ivan career", k=5)

thread.join()
run = thread.result
print(f"Facts: {run.facts_extracted}  Contradictions resolved: {run.contradictions_resolved}")
print(f"Tokens used: {run.cost_tokens}")
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
    depth=2,    # graph traversal depth
    decay=0.5,  # activation decay per hop
)
```

### Multi-agent shared store

Multiple agents can read and write to the same `.engram` file. Episodes are scoped per agent; facts and the entity graph are shared.

```python
from engram import Engram

# Each agent has its own episode scope
planner = Engram(path="./team.engram", agent_id="planner")
coder   = Engram(path="./team.engram", agent_id="coder")

planner.observe("Decided to migrate to PostgreSQL", tags=["arch"])
coder.observe("Started migration branch: feat/pg-migration", tags=["dev"])

# Each agent recalls only its own episodes by default
planner_results = planner.recall("migration", k=5)

# Cross-agent search when needed
all_results = planner.recall("migration", k=10, cross_agent=True)

# Inspect who's written to the shared file
with Engram(path="./team.engram") as global_view:
    print(global_view.list_agents())  # ['coder', 'planner']

planner.close()
coder.close()
```

### GDPR right-to-be-forgotten

```python
# Permanently erase a single episode
mem.forget(episode_id)

# Erase everything about a person: episodes, facts, graph edges
result = mem.forget_entity("Ivan")
print(f"Deleted {result.episodes_deleted} episodes, {result.facts_deleted} facts")
```

---

## CLI

Engram ships a command-line interface for inspecting and operating stores without writing code:

```
engram inspect     <path>
engram recall      <path> <query> [--k K] [--mode cosine|spreading] [--as-of DATE]
                                  [--agent-id ID] [--cross-agent]
engram timeline    <path> <entity>
engram observe     <path> <content> [--actors NAME...] [--tags TAG...]
                                    [--salience F] [--valence F] [--agent-id ID]
engram reflect     <path> [--llm anthropic|openai] [--model MODEL] [--agent-id ID]
engram forget      <path> (--episode ID | --entity NAME) [--agent-id ID]
engram list-agents <path>
```

```bash
# Inspect a store
engram inspect ./agent.engram

# Store: ./agent.engram  (1.4 MB)
#   Episodes:       1842   (vec index: 1842)
#   Facts:           234   (active: 198, superseded: 36)
#   Entities:         41
#   Reflections:      12   (last: 2025-05-11 09:14 UTC)

# Recall
engram recall ./agent.engram "Ivan employer" --k 3

# Recall as of a past date
engram recall ./agent.engram "Ivan employer" --as-of 2024-03-01

# Observe from the command line
engram observe ./agent.engram "Alice promoted to VP Engineering" --actors Alice --tags hr

# Run reflection
engram reflect ./agent.engram --llm anthropic --model claude-haiku-4-5-20251001

# Forget an entity (GDPR)
engram forget ./agent.engram --entity Ivan

# Multi-agent: list all agents
engram list-agents ./team.engram

# Recall scoped to one agent
engram recall ./team.engram "migration" --agent-id coder
```

---

## Full API Reference

### `Engram(path, *, embedder_model, decay_config, llm, agent_id)`

```python
from engram import Engram, DecayConfig, AnthropicAdapter

mem = Engram(
    path="./agent.engram",   # path to .engram file, or ":memory:" for in-process
    embedder_model="BAAI/bge-small-en-v1.5",  # default; local ONNX, ~23 MB
    decay_config=DecayConfig(
        lambda_=0.1,   # Ebbinghaus decay rate. 0.1 ≈ half-life ~7 days.
        alpha=0.2,     # Reinforcement weight per recall access.
        beta=0.1,      # Emotional valence weight.
        threshold=0.1, # Prune memories below this importance during reflect().
    ),
    llm=AnthropicAdapter(),  # optional; only used by reflect()
    agent_id="my-agent",     # optional; scopes writes and reads to this agent
)

# Context-manager supported
with Engram(path=":memory:") as mem:
    mem.observe("hello world")
```

---

### `observe(content, *, actors, tags, salience, emotional_valence) → str`

Record a raw episodic observation. Returns the episode id. No LLM call. ~4 ms.

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

### `observe_many(items) → list[str]`

Batch variant of `observe()`. Accepts a list of `ObserveInput` instances, runs a single ONNX inference pass and inserts all rows in one SQL transaction. ~2× faster than a loop at 100+ episodes.

```python
from engram import ObserveInput

ids = mem.observe_many([
    ObserveInput(content="Alice joined as CTO", actors=["Alice"], salience=0.9),
    ObserveInput(content="Q3 planning complete", tags=["planning"]),
])
```

`ObserveInput` fields: `content` (required), `actors`, `tags`, `salience` (default 0.5), `emotional_valence` (default 0.0).

---

### `recall(query, k, *, mode, depth, decay, as_of, cross_agent) → list[SearchResult]`

```python
# Default: cosine similarity
results = mem.recall("where does Ivan work?", k=5)

# Graph-based spreading-activation
results = mem.recall("Ivan", k=5, mode="spreading", depth=2, decay=0.5)

# Time travel: only episodes that existed at this point
results = mem.recall(
    "Ivan employer",
    k=5,
    as_of=datetime(2024, 3, 1, tzinfo=UTC),
)

# Cross-agent: bypass agent_id scope
results = mem.recall("migration", k=10, cross_agent=True)
```

`SearchResult` fields: `episode`, `score` (0–1, higher is better), `distance` (raw L2), `importance`.

---

### `assert_fact(subject, predicate, object, *, confidence, source) → str`

Store a semantic triple directly. No LLM required. Returns the fact id.

```python
fact_id = mem.assert_fact("Ivan", "works_at", "Globex", confidence=0.95)
fact_id = mem.assert_fact("Alice", "role", "CTO", source="linkedin-profile")
```

---

### `reflect() / reflect_async() → ReflectionRun / ReflectionThread`

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

Full fact history for an entity, including superseded facts, in chronological order.

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

### `forget(episode_id) → None`

Permanently erase a single episode from all storage structures (vector index, access log, graph edges). Raises `KeyError` if the episode does not exist.

```python
mem.forget(ep_id)
```

---

### `forget_entity(entity_name) → ForgetResult`

GDPR right-to-be-forgotten: permanently delete all data about a named entity across all agents. Removes episodes where the entity appears in `actors`, all facts where it is subject or object, and all graph edges connected to it.

```python
result = mem.forget_entity("Ivan")
print(f"Deleted {result.episodes_deleted} episodes, {result.facts_deleted} facts")
```

---

### `list_agents() → list[str]`

Return all distinct `agent_id` values that have written to this store. Episodes written without an `agent_id` are not included.

```python
with Engram(path="./team.engram") as mem:
    print(mem.list_agents())  # ['coder', 'planner', 'reviewer']
```

---

### `decay() → int`

Recompute importance scores for all episodes using the Ebbinghaus formula. Called automatically by `reflect()`. Returns the number of episodes updated.

Uses a single SQL `GROUP BY` fetch and a single `executemany` update — O(1) SQL round-trips regardless of episode count.

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

# Semantic recall when a query is provided
msgs = memory.get("Ivan Globex")
```

---

## Architecture

```
Engram
├── observe() / observe_many()  → Episode (content + embedding stored immediately)
│                                      ↓
│                                 vec_episodes  (sqlite-vec ANN index)
│                                 episodes      (metadata, agent_id, importance_score)
│
├── recall()  ─cosine──────────→ KNN search → SearchResult[]
│             ─spreading────────→ KNN seeds → BFS activation graph → SearchResult[]
│             ─as_of────────────→ time-filtered KNN → SearchResult[]
│             ─cross_agent──────→ bypass agent_id scope
│
├── reflect() / reflect_async() → LLM fact extraction (async, background)
│                                      ↓
│                                 facts    (bitemporal s/p/o triples)
│                                 entities (unique named entities)
│                                 edges    (Hebbian-weighted graph)
│
├── timeline(entity)   → facts WHERE subject=? ORDER BY valid_from
├── why(fact_id)       → provenance: derived_from + extracted_by
├── contradictions()   → active facts with same (subject, predicate)
├── forget()           → hard-delete one episode (all structures)
├── forget_entity()    → GDPR: hard-delete all data about a named entity
└── list_agents()      → distinct agent_ids in the store
```

### Storage schema

```sql
-- Raw observations (one row per observed event, scoped by agent_id)
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    timestamp DATETIME,
    actors JSON,               -- ["Ivan", "Alice"]
    tags JSON,
    salience REAL,
    emotional_valence REAL,
    importance_score REAL,
    agent_id TEXT DEFAULT NULL -- NULL = unscoped / backward-compatible
);

-- ANN vector index (sqlite-vec virtual table, mirrors episodes rowid)
CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding float[384]);

-- Bitemporal semantic facts (shared across all agents)
CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    subject TEXT, predicate TEXT, object TEXT,
    valid_from DATETIME,       -- when true in reality
    valid_to DATETIME,         -- NULL = still valid
    recorded_at DATETIME,      -- when system learned it
    superseded_at DATETIME,
    superseded_by TEXT,        -- FK to facts.id
    confidence REAL,
    derived_from JSON,         -- provenance: episode ids
    extracted_by TEXT          -- FK to reflections.id
);

-- Entity graph (shared across all agents)
CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT, type TEXT, aliases JSON, ...);
CREATE TABLE edges (
    src_id TEXT, dst_id TEXT, relation TEXT,
    weight REAL,               -- Hebbian-accumulated on co-access
    PRIMARY KEY (src_id, dst_id, relation)
);

-- Retrieval history (scoped by agent_id)
CREATE TABLE access_log (
    memory_id TEXT, accessed_at DATETIME, query TEXT, rank INTEGER,
    agent_id TEXT DEFAULT NULL
);

-- Reflection audit log (scoped by agent_id)
CREATE TABLE reflections (
    id TEXT PRIMARY KEY, started_at DATETIME, finished_at DATETIME,
    episodes_processed INTEGER, facts_extracted INTEGER,
    contradictions_resolved INTEGER, model_used TEXT, cost_tokens INTEGER,
    agent_id TEXT DEFAULT NULL
);
```

**Single-file design:** the `.engram` file is a standard SQLite database. Copy it, back it up with `rsync`, or open it with any SQLite browser. No migration daemon, no schema registry, no lock files.

**Zero-dependency writes:** every `observe()` call hits only Python + SQLite. The ONNX runtime for embeddings is already in-process. No network, no external API call.

**Backward compatibility:** stores created before v1.3 (without `agent_id`) open without modification. The migration silently adds the column with `DEFAULT NULL`, preserving all existing data.

---

## Benchmarks

Measured on Apple M-series, fastembed `BAAI/bge-small-en-v1.5`, SQLite WAL mode.

### Write latency (n=300 episodes in store)

| Operation | p50 | p99 | Notes |
|---|---|---|---|
| `observe()` | 4.1 ms | 4.8 ms | Embedding dominates (~3.5 ms ONNX) |
| `observe_many()` 100 eps | 2.0 ms/ep | — | Single ONNX pass + single transaction |
| `observe_many()` 500 eps | 1.6 ms/ep | — | Batch efficiency increases with N |

### Read latency (n=300 episodes)

| Operation | p50 | p99 |
|---|---|---|
| `recall(mode="cosine")` | 4.3 ms | 5.0 ms |
| `recall(mode="spreading")` | 4.4 ms | 5.0 ms |
| `recall(as_of=...)` | 4.5 ms | 5.2 ms |

### Decay (n=1000 episodes)

| Implementation | Latency |
|---|---|
| v1.x: N individual SQL round-trips | ~52 ms |
| **v2.0: batch GROUP BY + executemany** | **~2.5 ms** |

The batch rewrite eliminates 5 000 SQL calls and replaces them with 3.

### LoCoMo Recall Accuracy (5 sessions, 15 questions)

| Metric | Score |
|---|---|
| hit@1 | 33.3% |
| hit@5 | 93.3% |
| MRR | 0.586 |

### Reflection cost (per 1 000 episodes)

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

### Test coverage (219 tests)

```
tests/
  test_schema.py       schema + SQLite migrations (incl. pre-v1.3 backward compat)
  test_observe.py      observe() + embeddings
  test_recall.py       cosine recall
  test_smoke.py        end-to-end Engram class
  test_importance.py   decay formula
  test_decay.py        decay background job
  test_store_facts.py  fact CRUD + assert_fact()
  test_reflection.py   reflection loop (stub LLM), cost_tokens, reflect_async
  test_graph.py        entity/edge CRUD + spreading recall
  test_bitemporal.py   as_of + timeline
  test_forget.py       forget(), forget_entity(), GDPR cascade
  test_cli.py          all CLI subcommands + --agent-id + --cross-agent
  test_multiagent.py   agent_id scoping, shared facts, cross-agent recall
  test_performance.py  observe_many correctness + batch decay + LRU cache
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
- [x] v1.1 — `forget()` / GDPR right-to-be-forgotten
- [x] v1.2 — CLI (`engram inspect`, `recall`, `timeline`, `observe`, `reflect`, `forget`, `list-agents`)
- [x] v1.3 — Multi-agent shared memory (`agent_id` column, `cross_agent` recall, `list_agents()`)
- [x] v2.0 — Batch decay (21× speedup), `observe_many()` (2× speedup), embedding LRU cache

---

## Contributing

PRs welcome. Please:

1. Open an issue first for non-trivial changes.
2. Follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`).
3. Run `pytest -x && ruff check . && mypy engram` before submitting.
4. Keep PRs small — one logical change per PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

---

## License

MIT — see [LICENSE](LICENSE).

Architecture rationale and design decisions: [DESIGN.md](DESIGN.md).
