# Engram API Reference

[Back to README](../README.md)

This is the exhaustive method-by-method reference for the `Engram` and `AsyncEngram` classes, the LLM adapters, the LangChain / LlamaIndex / MCP integrations, the on-disk architecture and storage schema, benchmark numbers, and the `DecayConfig` tuning knobs. For the narrative introduction, install instructions, and quickstart, see the [README](../README.md).

---

## Full API Reference

### `Engram(path, *, embedder_model, decay_config, llm, agent_id, key, events_path)`

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
    llm=AnthropicAdapter(),  # optional; used by reflect() and compress()
    agent_id="my-agent",     # optional; scopes writes and reads to this agent
    key="passphrase",        # optional; enables SQLCipher encryption-at-rest
    events_path="./agent.events.ndjson",  # optional; see Agent Passport events below
)

# Context-manager supported
with Engram(path=":memory:") as mem:
    mem.observe("hello world")
```

> **Encryption-at-rest:** pass `key="..."` to encrypt the database via
> SQLCipher (`pip install 'engdbram[encryption]'`). Plain (no-key) stores
> are unchanged. Use `mem.rekey(new_key)` to change or remove the key.

> **Agent Passport events:** pass `events_path="..."` (or set
> `ENGRAM_EVENTS_PATH`) to opt in to an NDJSON event log conforming to the
> [Agent Passport](https://github.com/TAIPANBOX/agent-passport) `agent-event`
> envelope (SPEC.md §6). Off by default - no file is written unless
> configured. `observe()`, `assert_fact()`, `forget()`, and `forget_fact()`
> each emit a `memory_written`/`memory_forgotten` event (`info`); `reflect()`
> emits one `reflection_run` event (`info`) plus one `contradiction_found`
> event (`medium`) per fact it supersedes with a differing object (a
> same-object re-extraction supersedes silently, as agreement). Events with no `agent_id` set on
> the instance are skipped, never fabricated. A local file append is not a
> network call, so this does not violate Engram's write-time invariant - see
> `engram/events.py` for the full reasoning. Failures to write an event are
> logged as a warning and never raised into the memory operation. Each event
> also carries a SPEC.md §6.5 `prev_hash` chain (one file, one chain,
> resumed across restarts) - verify with `agent-conform -chain <file>`.

---

### `observe(content, *, actors, tags, salience, emotional_valence, timestamp) → str`

Record a raw episodic observation. Returns the episode id. No LLM call. ~4 ms.

```python
ep_id = mem.observe(
    "Alice presented the Q3 roadmap",
    actors=["Alice"],
    tags=["work", "roadmap"],
    salience=0.8,           # subjective importance at encoding (0–1)
    emotional_valence=0.3,  # –1 (negative) … +1 (positive)
)

# Back-filling history: date it when it happened, not when it was written,
# so recall(as_of=...) places it in the period it belongs to.
ep_id = mem.observe("Alice joined Globex", timestamp=datetime(2024, 3, 1, tzinfo=UTC))
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

### `recall(query, k, *, mode, depth, decay, vector_weight, fts_weight, as_of, cross_agent, candidate_limit) → list[SearchResult]`

```python
# Default: cosine similarity
results = mem.recall("where does Ivan work?", k=5)

# Hybrid: BM25 keyword + cosine vector, blended (also honors as_of)
results = mem.recall("Ivan Globex transfer", k=5, mode="hybrid")
results = mem.recall("exact term", k=5, mode="hybrid",
                     vector_weight=0.3, fts_weight=0.7)

# Graph-based spreading-activation
results = mem.recall("Ivan", k=5, mode="spreading", depth=2, decay=0.5)

# Time travel: only episodes that existed at this point (works in all modes)
results = mem.recall(
    "Ivan employer",
    k=5,
    as_of=datetime(2024, 3, 1, tzinfo=UTC),
)

# Cross-agent: bypass agent_id scope
results = mem.recall("migration", k=10, cross_agent=True)

# Tune the candidate pool for hybrid search
results = mem.recall("Q3 revenue", k=5, mode="hybrid", candidate_limit=64)  # per-source pool
```

`k` is capped at 4096, the vec0 KNN limit; above it `recall` raises `ValueError`.

`k_inner` is accepted but ignored since v2.3, and warns. It sized an over-fetch
that compensated for `as_of` and `agent_id` being applied after the vector
search. They are now a metadata column and a partition key on the index itself,
evaluated during the scan, so `k` counts only rows that already passed them and
there is no inner limit left to tune.

`SearchResult` fields: `episode`, `score` (0–1, higher is better - derived
from the L2 distance of unit-norm embeddings, so monotone in cosine),
`distance` (raw L2 from sqlite-vec), `importance`.

User-supplied query strings are safe to pass even when they contain FTS5
operators (`*`, `(`, `OR`, `NOT`, `-`, `"`); tokens are escaped and wrapped
as phrases before reaching SQLite.

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

### `timeline(entity, *, as_of=None) → list[Fact]`

Fact history for an entity, in chronological order. By default returns
everything (including superseded facts) so callers can see how beliefs
evolved. Pass `as_of=...` to get only facts whose validity interval
contains that timestamp - the public entry point to the bitemporal
fact path.

```python
# Full history, including superseded facts
for f in mem.timeline("Ivan"):
    end = f.valid_to.date() if f.valid_to else "now"
    print(f"[{f.valid_from.date()} → {end}]  Ivan {f.predicate} {f.object}")

# What did the agent believe about Ivan in March 2024?
for f in mem.timeline("Ivan", as_of=datetime(2024, 3, 1, tzinfo=UTC)):
    print(f"valid: Ivan {f.predicate} {f.object}")
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

Permanently erase a single episode from all storage structures (vector index, FTS index, access log, graph edges). Raises `KeyError` if the episode does not exist.

```python
mem.forget(ep_id)
```

---

### `forget_fact(fact_id) → None`

Permanently erase a single semantic fact (a triple returned by `assert_fact()` or produced by `reflect()`). Raises `KeyError` if the fact does not exist.

```python
mem.forget_fact(fact_id)
```

---

### `forget_entity(entity_name) → ForgetResult`

GDPR right-to-be-forgotten: permanently delete all data about a named entity across all agents. Removes episodes where the entity appears in `actors`, all facts where it is subject or object, and all graph edges connected to it.

```python
result = mem.forget_entity("Ivan")
print(f"Deleted {result.episodes_deleted} episodes, {result.facts_deleted} facts")
```

---

### `compress(*, max_episodes, importance_threshold, batch_size) → CompressionRun`

Compress low-importance episodes into LLM-generated summary episodes. Requires an `llm` adapter.

```python
result = mem.compress(
    max_episodes=1000,        # no-op if store has fewer episodes than this
    importance_threshold=0.3, # compress episodes with importance_score < threshold
    batch_size=20,            # episodes grouped per LLM call
)
# CompressionRun fields: episodes_removed, summaries_created, model_used, cost_tokens
print(f"Removed {result.episodes_removed} → {result.summaries_created} summaries")
```

---

### `backup(dest) → None`

Hot backup using SQLite's built-in online backup API. Safe to call while the store is open and actively written to.

```python
mem.backup("./agent_backup.engram")  # str or Path
```

---

### `rekey(new_key) → None`

Change the SQLCipher passphrase of an encrypted store, or pass `None` to
remove encryption entirely. Only valid on databases originally opened
with `Engram(..., key=...)`. To encrypt a plain database, dump it with
`export_json()` and re-import into a fresh `Engram(key=...)`.

```python
mem = Engram(path="./agent.engram", key="old-pass")
mem.rekey("new-pass")     # rotate
mem.rekey(None)           # drop encryption
```

---

### `export_json(dest) → dict`

Export the full store (episodes, facts, entities, edges) to a JSON file. Returns the document dict.

```python
doc = mem.export_json("./agent_dump.json")
print(doc["counts"])  # {'episodes': 842, 'facts': 134, 'entities': 41, 'edges': 97}
```

---

### `import_json(src, *, merge) → dict`

Import from a JSON file produced by `export_json()`. Returns counts of inserted rows per table.

```python
counts = mem.import_json("./agent_dump.json")           # raises on duplicate ids
counts = mem.import_json("./agent_dump.json", merge=True)  # skip duplicates silently
```

---

### `decay() → int`

Recompute importance scores using the Ebbinghaus formula. Scoped to this instance's `agent_id` when set (all episodes when unscoped). Called automatically by `reflect()`. Returns the number of episodes updated.

Uses a single SQL `GROUP BY` fetch and a single `executemany` update - O(1) SQL round-trips regardless of episode count.

---

### `list_agents() → list[str]`

Return all distinct `agent_id` values that have written to this store.

```python
with Engram(path="./team.engram") as mem:
    print(mem.list_agents())  # ['coder', 'planner', 'reviewer']
```

---

### `WorkingMemory(capacity, engram)`

LRU scratchpad with optional long-term spillover.

```python
from engram import WorkingMemory, WorkingMemoryItem

wm = WorkingMemory(
    capacity=7,    # max slots (default 7, per Miller's 7±2 law)
    engram=mem,    # optional; evicted items written via observe()
)

wm.set("key", "content", priority=1)  # kwargs stored in item.metadata
item: WorkingMemoryItem = wm.get("key")   # promotes to MRU; None if missing
item = wm.peek("key")                     # no LRU change
wm.delete("key")                          # remove one item
wm.flush()                                # write all to long-term store + clear
wm.clear()                                # discard without writing

len(wm)         # current size
"key" in wm     # membership test
wm.items()      # list[WorkingMemoryItem] from LRU to MRU
wm.capacity     # int
```

`WorkingMemoryItem` fields: `key`, `content`, `metadata` (dict), `created_at`, `accessed_at`.

---

### `AsyncEngram(path, *, embedder_model, decay_config, llm, agent_id, key)`

Async-compatible wrapper with the same interface as `Engram`. Every method
is `async def` and dispatches to the synchronous implementation via
`asyncio.to_thread` - the event loop is never blocked by ONNX inference or
SQLite I/O. The surface is at parity with the sync API: `recall` accepts
`k_inner`/`candidate_limit`, `timeline` accepts `as_of=`.

```python
from engram import AsyncEngram

async with AsyncEngram(path="./agent.engram") as mem:
    ep_id = await mem.observe("Hello world")
    results = await mem.recall("hello", k=3, mode="hybrid", candidate_limit=64)
    bitemporal = await mem.timeline("Alice", as_of=datetime(2024, 3, 1, tzinfo=UTC))
    fact_id = await mem.assert_fact("Alice", "role", "CTO")
    await mem.decay()
    await mem.backup("./backup.engram")
    doc = await mem.export_json("./dump.json")
    counts = await mem.import_json("./dump.json", merge=True)
    await mem.forget(ep_id)
    await mem.forget_fact(fact_id)
    result = await mem.forget_entity("Bob")
```

---

## LLM Adapters

Both `reflect()` and `compress()` use the LLM adapter:

```python
from engram import (
    AnthropicAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    DeepSeekAdapter,
    QwenAdapter,
    KimiAdapter,
    StubLLMAdapter,  # tests / offline development
)

# Claude (default: haiku - fast, cheap)
llm = AnthropicAdapter(model="claude-haiku-4-5-20251001")

# Claude routed through a proxy (e.g. TokenFuse) instead of the API directly
llm = AnthropicAdapter(model="claude-haiku-4-5-20251001",
                        base_url="http://localhost:8080", api_key="proxy-key")

# OpenAI
llm = OpenAIAdapter(model="gpt-4o-mini")

# Ollama or any OpenAI-compatible local model
llm = OpenAIAdapter(model="llama3.2", base_url="http://localhost:11434/v1")

# Google Gemini (reads GOOGLE_API_KEY by default)
llm = GeminiAdapter(model="gemini-2.0-flash")

# OpenAI-compatible providers pre-wired with the right base URL
llm = DeepSeekAdapter(model="deepseek-chat")     # DEEPSEEK_API_KEY
llm = QwenAdapter(model="qwen-max")              # DASHSCOPE_API_KEY
llm = KimiAdapter(model="moonshot-v1-8k")        # MOONSHOT_API_KEY

mem = Engram(path="./agent.engram", llm=llm)
```

**Prompt-injection hardening.** Episode bodies sent to an LLM during
`reflect()` are wrapped in `<observation>` blocks and the system prompt
instructs the model to ignore directives inside them. Every extraction
runs at `temperature=0`, and any LLM-derived `confidence` is capped at
`0.95` so a successful injection cannot persist a fabricated fact as
absolute truth. Facts you assert directly via `mem.assert_fact(...,
confidence=1.0)` are not capped - the cap is specifically for facts
mined from user-controlled text.

---

## Integrations

### MCP Server

`pip install 'engdbram[mcp]'` adds an `engram-mcp` console command that exposes
a store as an [MCP](https://modelcontextprotocol.io) tool server over stdio -
no network listener, compatible with Claude Desktop, Claude Code, Cursor, and
any other MCP host:

```bash
engram-mcp --db ./agent.engram
engram-mcp --db ./agent.engram --agent-id my-agent   # scope the default agent
engram-mcp --db ./agent.engram --events ./agent.events.ndjson  # opt in to Agent Passport events

# equivalently, via environment variables
ENGRAM_MCP_DB=./agent.engram ENGRAM_MCP_AGENT_ID=my-agent ENGRAM_MCP_EVENTS=./agent.events.ndjson engram-mcp
```

If `--db`/`ENGRAM_MCP_DB` is omitted, the server falls back to an in-memory
store that is discarded on exit - fine for a quick trial, not for anything
you want to keep.

Five tools are registered:

| Tool | Purpose |
|---|---|
| `remember` | Store a memory. `kind="episodic"` (default) takes `content`; `kind="semantic"` takes `subject`/`predicate`/`object` instead - the structured triple, not free text. Both accept an optional per-call `agent_id` to override the server's default agent scope. |
| `recall` | Retrieve memories relevant to a natural-language `query`, with `mode="cosine"` (default), `"hybrid"`, or `"spreading"`, and an optional `limit` and `agent_id`. |
| `why` | Explain a memory's provenance by id - extraction chain and confidence for a fact, encoding/access metadata for an episode. |
| `forget` | Permanently delete one memory (episodic or semantic) by id. Irreversible. |
| `stats` | Store-wide counts (episodic, semantic, procedural - the last is always 0, see below), fact validity breakdown, entity/reflection counts, db file size. |

A few things worth calling out:

- **`kind="procedural"` is deliberately rejected.** Engram's store implements
  episodic and semantic memory only; there is no procedural memory layer to
  write to, so `remember(kind="procedural", ...)` raises a clear error instead
  of silently doing nothing.
- **`reflect()` is not exposed as a tool.** It can call an external LLM to
  extract facts from episodes, which is out of scope for a zero-config memory
  server and would break the "no network calls at write time" guarantee every
  other tool here upholds. Run reflection out-of-band - `engram reflect` (CLI)
  or `mem.reflect_async()` (library) - against the same `.engram` file.
- **`agent_id` is an opaque string.** Agent Passport `agent://...` URIs are accepted as-is and used verbatim as the scoping key - Engram does not parse, resolve, or validate them.

Add to your MCP client's config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram-mcp",
      "args": ["--db", "/path/to/agent.engram"]
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

# Retriever - plug into any RAG chain
retriever = EngramRetriever(engram=mem, k=5)
docs = retriever.invoke("Ivan project")

# Chat history - persists conversation turns across sessions
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
├── observe() / observe_many()  → Episode (content + embedding + FTS stored immediately)
│                                      ↓
│                                 vec_episodes  (sqlite-vec ANN index)
│                                 fts_episodes  (FTS5 full-text index)
│                                 episodes      (metadata, agent_id, importance_score)
│
├── recall()  ─cosine──────────→ KNN search → SearchResult[]
│             ─hybrid───────────→ FTS5 BM25 + KNN → blended score → SearchResult[]
│             ─spreading────────→ KNN seeds → BFS activation graph → SearchResult[]
│             ─as_of────────────→ time-filtered KNN → SearchResult[]
│             ─cross_agent──────→ bypass agent_id scope
│
├── WorkingMemory               → LRU scratchpad, capacity 7±2
│                                 eviction → observe() into long-term store
│
├── AsyncEngram                 → async def wrappers via run_in_executor
│
├── reflect() / reflect_async() → LLM fact extraction (async, background)
│                                      ↓
│                                 facts    (bitemporal s/p/o triples)
│                                 entities (unique named entities)
│                                 edges    (Hebbian-weighted graph)
│
├── compress()                  → LLM summarisation of low-importance episodes
│                                 originals hard-deleted → summary episode stored
│
├── timeline(entity)   → facts WHERE subject=? ORDER BY valid_from
├── why(fact_id)       → provenance: derived_from + extracted_by
├── contradictions()   → active facts with same (subject, predicate), different object
├── forget()           → hard-delete one episode (all structures)
├── forget_fact()      → hard-delete one semantic fact
├── forget_entity()    → GDPR: hard-delete all data about a named entity
├── backup(dest)       → SQLite online backup API (safe while open)
├── export_json(dest)  → portable JSON dump (episodes, facts, entities, edges)
├── import_json(src)   → restore from JSON dump, merge mode available
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
    summary_of JSON,           -- episode ids this row summarises (compress())
    importance_score REAL,
    agent_id TEXT DEFAULT NULL -- NULL = unscoped / backward-compatible
);

-- ANN vector index (sqlite-vec virtual table, mirrors episodes rowid)
CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding float[384]);

-- Full-text search index (FTS5 content table, mirrors episodes rowid)
CREATE VIRTUAL TABLE fts_episodes USING fts5(content, content='episodes', content_rowid='rowid');

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

**Single-file design:** the `.engram` file is a standard SQLite database. Copy it, back it up with `rsync` or `mem.backup()`, or open it with any SQLite browser. No migration daemon, no schema registry, no lock files.

**Zero-dependency writes:** every `observe()` call hits only Python + SQLite. The ONNX runtime for embeddings is already in-process. No network, no external API call.

**Backward compatibility:** stores created before v1.3 (without `agent_id`) open without modification. The migration silently adds missing columns with `DEFAULT NULL`, preserving all existing data.

---

## Benchmarks

Measured on Apple M-series, fastembed `BAAI/bge-small-en-v1.5`, SQLite WAL mode.

### Write latency (n=300 episodes in store, embedding included)

| Operation | p50 | p99 | Notes |
|---|---|---|---|
| `observe()` | 4.1 ms | 4.8 ms | Embedding dominates (~3.5 ms ONNX) |
| `observe_many()` 100 eps | 2.0 ms/ep | - | Single ONNX pass + single transaction |
| `observe_many()` 500 eps | 1.6 ms/ep | - | Batch efficiency increases with N |

### Read latency as the store grows

`engram-bench scale --sizes 1000,10000,100000`. Search columns are p50/p95 in
milliseconds against a pre-computed query embedding, so they show the part
that grows with the store; `recall()` keeps the ~4 ms embedding in, because
that is what a caller waits for. RSS is a fresh process opening the finished
store and recalling from it.

| episodes | file | RSS | cosine | hybrid | `as_of` | scoped | `recall()` |
|---|---|---|---|---|---|---|---|
| 1 000 | 2.4 MB | 342 MB | 0.18 / 0.18 | 0.45 / 0.53 | 0.87 / 1.03 | **0.05 / 0.05** | 4.4 / 4.9 |
| 10 000 | 19.8 MB | 346 MB | 3.03 / 3.25 | 3.59 / 3.68 | 10.0 / 10.5 | **0.05 / 0.07** | 7.2 / 7.7 |
| 100 000 | 193.6 MB | 353 MB | 31.8 / 32.5 | 35.8 / 36.9 | 107 / 107 | **0.09 / 0.11** | 35.7 / 36.6 |

Read it as four facts:

- **Unscoped search is linear**, ten times the episodes for ten times the
  cost, because the scan is exact. Around 100 000 episodes it stops being
  free at ~32 ms, still small next to any LLM call in the same loop, and at a
  million it would be the slowest thing an agent does.
- **Scoped search does not grow at all.** `agent_id` is a vec0 partition key,
  so an agent holding five episodes pays for five however large the shared
  store gets: 350x faster than unscoped at 100 000 episodes.
- **`as_of` costs about 3.4x cosine** and hits the wall first, at ~107 ms per
  query at 100 000 episodes. The `ts` metadata filter is evaluated per row
  during the scan.
- **Memory is flat.** Vectors live on disk and are read by SQLite, so the
  resident footprint is the ONNX runtime, not the store.

Spreading activation is absent because its edges come from `reflect()`, which
needs an LLM; on a store built by `observe()` alone the graph is empty and the
number would describe the seed KNN rather than the mode.

### Decay (n=1000 episodes)

| Implementation | Latency |
|---|---|
| v1.x: N individual SQL round-trips | ~52 ms |
| **v2.0+: batch GROUP BY + executemany** | **~2.5 ms** |

The batch rewrite eliminates 5 000 SQL calls and replaces them with 3.

### Per-commit write (WAL vs DELETE journal)

| Journal mode | Latency per commit | Notes |
|---|---|---|
| DELETE (SQLite default) | ~0.31 ms | Exclusive lock + random-write sync |
| **WAL (v2.0.1+)** | **~0.07 ms** | Sequential append, no exclusive lock |

WAL mode is enabled automatically for all file-based stores. Readers (`recall`, `timeline`) and writers (`observe`, `reflect_async`) now run concurrently without blocking each other.

### Recall accuracy

Measured on the full 500 questions of
[LongMemEval-S](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
(`longmemeval_s_cleaned.json`, ICLR 2025, MIT). Each question carries its own
history of 30 to 60 sessions; every turn is ingested as one episode, 246 738
in total, and the question is asked against that store. `bge-small-en-v1.5`,
`mode="cosine"`, no LLM anywhere in the loop.

| | k=5 | k=10 |
|---|---|---|
| **session recall** | 0.956 | 0.978 |
| **turn recall** | 0.772 | 0.862 |

The dataset marks both the sessions holding the evidence and the individual
turns, so there are two honest numbers and they are not interchangeable:

- **session recall** counts a hit when any retrieved episode came from a
  session listed in `answer_session_ids`. A session can run a dozen turns, so
  this says the right conversation was found.
- **turn recall** counts a hit only when a retrieved episode is one of the
  896 turns flagged `has_answer` out of 246 738. This says the answer itself
  was put in front of the agent, and it is 18 points lower at k=5.

A memory system quoting one unqualified "R@k" is almost certainly quoting the
first. Ask which.

By question type, session recall at k=5: `single-session-assistant` 1.000,
`knowledge-update` 0.974, `multi-session` 0.970, `single-session-user` 0.957,
`temporal-reasoning` 0.925, `single-session-preference` 0.900.

Reproduce it (the dataset is 265 MB and is not vendored):

```bash
engram-bench longmemeval --data ./longmemeval_s_cleaned.json --k 5,10 \
    --checkpoint ./lme.jsonl --resume
```

Ingestion is the whole cost: 5.6 hours on eight dedicated cores, against 15 ms
per query. `--checkpoint` writes one record per question so a run that stops
can be resumed and scored from where it got to. The records behind the table
above are in [`benchmarks/results/`](../benchmarks/results/).

**`mode="hybrid"` is deliberately absent.** The same run scored it and got
figures identical to cosine to three decimals on all 500 questions, which is
what exposed a bug: the BM25 query joined its terms with an implicit AND, so
it matched nothing for any question longer than a few words and the blend
reduced to cosine. That is fixed, but no hybrid number is published until a
run measures the fixed code.

`engram-bench locomo` also scores hit@k and MRR over any file in LoCoMo's
format. The fixture bundled with it is synthetic, five hand-written sessions
whose keywords were chosen next to the text they match, so it is a smoke test
for the retrieval path and its scores are not quoted anywhere.

### Reflection cost (per 1 000 episodes)

| Model | $/1k episodes |
|---|---|
| gpt-4o-mini | $0.0033 |
| claude-haiku-4.5 | $0.0056 |
| gpt-4o | $0.0542 |
| claude-sonnet-4.6 | $0.0677 |

Reflection is optional and async - you only pay when you need semantic fact extraction.

### Run benchmarks locally

```bash
# Both spellings work
python -m engram.benchmarks all
engram-bench scale --sizes 1000,10000,100000
engram-bench latency --n 500
engram-bench locomo --data ./my_data.json
engram-bench cost --n 1000 --model gpt-4o-mini
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

