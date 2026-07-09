# Engram

> **The SQLite of agent memory.** Embeddable, local-first, cognitively grounded.

[![PyPI version](https://badge.fury.io/py/engdbram.svg)](https://badge.fury.io/py/engdbram)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-392%20passed-green)](tests/)

---

## The problem

Every AI agent starts from zero. Ask it something it answered last week — it has no idea. Show it a document it already processed — it processes it again. Tell it Ivan moved to a new company — it still thinks Ivan works at the old one.

This happens because agents have no persistent memory. When the conversation ends, everything is gone.

The usual fix is to throw a vector database at the problem. Store text, embed it, search by similarity. That helps — but it's not enough. You still can't ask *"what did the agent think in March?"* or *"where did this belief come from?"* or *"show me everything the agent knows about Ivan."* A vector search finds similar text. It doesn't understand time, relationships, or importance.

**Engram is memory done properly.**

---

## What Engram does

Engram gives your agent a persistent memory that works like a file — one `.engram` file on disk, no server required. You `pip install` it and start using it in two lines:

```python
from engram import Engram

with Engram(path="./agent.engram") as mem:
    # Remember something
    mem.observe("Ivan moved from Acme to Globex last week", actors=["Ivan"])

    # Recall it later — even in a completely different session
    for r in mem.recall("where does Ivan work?", k=3):
        print(f"[{r.score:.2f}] {r.episode.content}")
```

No server to start. No API key for the store. No Docker. No configuration file.

Here is what Engram gives you that a plain vector database does not:

**Remembers raw events** — every observation is stored with who was involved, what tags apply, and how important it felt at the time. Search finds the right memories even when the query is phrased differently.

**Understands facts** — a background process (no LLM needed at write time) reads your observations and extracts structured knowledge: *Ivan works at Globex*, *Alice is the CTO*. These facts can be queried directly, updated when things change, and traced back to their source.

**Knows what happened when** — if Ivan changes jobs, the old fact is not deleted. It is closed with an end date. You can ask what the agent believed in March even if the truth has changed since.

**Forgets wisely** — memories that haven't been accessed in a while gradually become less important. Memories that matter (accessed often, emotionally significant) stay sharp. The agent doesn't accumulate noise forever.

**Explains itself** — for any fact, you can ask where it came from: which observation triggered it, which LLM run extracted it, with what confidence.

**Works with multiple agents** — several agents can share a single `.engram` file. Each has its own private observations; extracted facts and the relationship graph are shared between them.

---

## What is Engram, technically?

Engram is a **cognitive memory layer** for AI agents — a single local file (`agent.engram`) built on SQLite. It models three kinds of memory that mirror how human memory works:

**Episodic memory** — raw observations stored as they happen, with actors, tags, salience, and emotional weight. No LLM required at write time; writes complete in ~4 ms.

**Semantic memory** — structured knowledge extracted from episodes via a background reflection loop: `(subject, predicate, object)` triples with full bitemporal validity. Every fact tracks *when it was true in reality* and *when the system learned it* — independently on two timelines. When Ivan switches jobs, the old fact is closed with `valid_to`, not deleted. You can query what the agent believed in March even if the truth has since changed.

**Dynamic importance** — each memory carries a living importance score based on the Ebbinghaus forgetting curve, reinforced by retrieval frequency and emotional weight. Memories below threshold decay and are pruned automatically during reflection. The agent forgets what doesn't matter; critical memories survive.

### What you can actually do with it

- **Debug beliefs**: when the agent says "Ivan works at Globex," call `mem.why(fact_id)` to see exactly which episode produced that belief, which reflection run extracted it, which model, and with what confidence.
- **Erase a person**: `forget_entity("Ivan")` permanently removes all episodes, facts, and graph edges connected to Ivan — a proper GDPR right-to-be-forgotten.
- **Query the past**: `mem.recall("Ivan employer", as_of=datetime(2024, 3, 1))` returns what the agent knew at that exact point in time, not what it knows now.
- **Run multiple agents**: a planner and a coder can share one file — each sees its own episodes, both benefit from shared extracted facts.
- **Plug into any MCP client**: run `engram-mcp --db ./agent.engram` and Claude Desktop, Claude Code, or Cursor can `remember`, `recall`, `why`, and `forget` against the same store with zero integration code — see [MCP Server](#mcp-server).

---

## Why not just use a vector database?

Vector databases (Pinecone, Chroma, Qdrant) store text and find similar text. That is useful, but it is a fraction of what memory requires.

They cannot tell you *when* something was true. They cannot explain *why* the agent believes something. They have no concept of facts becoming outdated, of contradictions, or of some memories mattering more than others. And they run as separate servers — you need Docker, a network connection, and an API call just to write a sentence.

Engram is not a replacement for a vector database — it includes one, built in, with no separate process. On top of it, Engram adds time, structure, importance, and provenance that vector DBs do not have.

Every other solution forces a trade-off. Engram doesn't.

| Capability | Pinecone / Chroma / Qdrant | Mem0 | Zep / Graphiti | Letta (MemGPT) | LangChain memory | **Engram** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Vector similarity search | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| **Hybrid BM25 + vector recall** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Semantic fact triples (s, p, o) | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Bitemporal validity** (`as_of` time travel) | ❌ | ❌ | ⚠️ | ❌ | ❌ | ✅ |
| **Spreading-activation retrieval** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Importance decay (Ebbinghaus) | ❌ | ❌ | ✅ | ⚠️ | ❌ | ✅ |
| **Working memory (7±2 scratchpad)** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Memory compression via LLM** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Async API** | ❌ | ❌ | ⚠️ | ❌ | ❌ | ✅ |
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

## How Engram works

### Memory that doesn't forget the wrong things

Most tools either remember everything forever (noise accumulates) or forget everything when the session ends (nothing persists). Engram does neither.

Every memory gets an importance score. Memories you access often, or that carry emotional weight, stay sharp. Memories that sit untouched gradually fade. When the agent runs its background reflection pass, low-importance memories are pruned automatically. The result is a store that stays useful instead of bloating.

This is modelled on the [Ebbinghaus forgetting curve](https://en.wikipedia.org/wiki/Forgetting_curve) — the same pattern that describes how humans forget — combined with Hebbian reinforcement from repeated retrieval.

### Facts that know when they were true

When you just store text and search it, you lose track of time. "Ivan works at Acme" and "Ivan works at Globex" are just two strings — you don't know which is current, or what changed.

Engram extracts structured facts from your observations — triples like *(Ivan, works_at, Globex)* — and tracks two independent timelines for each:

- **When it was true in reality** (`valid_from` / `valid_to`)
- **When the system learned it** (`recorded_at` / `superseded_at`)

When Ivan changes jobs, the old fact is not deleted — it is closed with an end date. The new fact is added alongside it. You can query what the agent believed at any point in the past:

```python
# What did the agent think about Ivan's employer in March?
mem.recall("Ivan employer", k=5, as_of=datetime(2024, 3, 1, tzinfo=UTC))

# Full fact history — every job Ivan ever had, with dates
mem.timeline("Ivan")
```

This two-timeline approach is standard in financial databases and audit systems. In the AI memory space, Engram is the only tool that implements it.

### Three ways to search

Engram ships three retrieval modes behind the same API:

**`mode="cosine"`** (default) — pure semantic vector search. Finds memories that mean the same thing as your query, even if the words are different.

**`mode="hybrid"`** — combines keyword search (BM25) with semantic search, then blends the scores. Best when you need both exact term matching and conceptual understanding. The blend is configurable:

```python
# BM25 keyword + cosine vector, weighted blend
results = mem.recall("Alice CTO Globex", k=5, mode="hybrid")

# More weight on exact keywords, less on semantics
results = mem.recall("quarterly budget", k=5, mode="hybrid",
                     vector_weight=0.3, fts_weight=0.7)
```

**`mode="spreading"`** — follows relationship edges between memories. If Ivan is connected to Project X in the graph, a query about Ivan can surface Project X episodes even if they share no words or meaning. One memory activates its associates, like human associative recall.

Technically: spreading activation runs BFS over Hebbian-weighted graph edges, ranking results by `α·cosine_similarity + β·graph_activation + γ·importance_score`.

### A scratchpad for the agent's current task

Engram also provides `WorkingMemory` — a small, fast, in-memory scratchpad for whatever the agent is actively thinking about. It holds a fixed number of items (default 7, matching the average human working memory capacity). When it fills up, the least-recently-used item is dropped — and if you pass an Engram store, it is automatically saved to long-term memory before being evicted:

```python
from engram import WorkingMemory

wm = WorkingMemory(capacity=5, engram=mem)  # evicted items → long-term store
wm.set("task", "Summarise the quarterly report")
wm.set("context", "Revenue grew 12% YoY — needs explanation")

item = wm.get("task")      # read + promote to most-recently-used
item = wm.peek("context")  # read without changing eviction order

wm.flush()  # write everything to long-term store + clear
```

### Background reflection (the agent's "sleep")

LLM calls in Engram never block writes. The reflection loop runs asynchronously — while the agent keeps working:

1. Group recent observations by entity or topic
2. Call the LLM to extract structured facts (`Ivan works_at Globex`)
3. Detect contradictions — same subject and predicate, different value
4. Close outdated facts with an end date
5. Recompute importance scores
6. Prune memories below threshold

```python
thread = mem.reflect_async()  # starts in background, returns immediately
thread.join()                 # wait only when you need the results
print(f"{thread.result.facts_extracted} facts, {thread.result.cost_tokens} tokens")
```

### Compressing old memories

When a store grows large, `compress()` groups low-importance observations into batches and asks the LLM to summarise each batch into a single paragraph. The originals are hard-deleted; the summary is stored in their place, with a `summary_of` pointer to what it replaced:

```python
result = mem.compress(
    max_episodes=1000,        # only compress when store exceeds this
    importance_threshold=0.3, # target: episodes below this importance score
    batch_size=20,            # observations per LLM call
)
print(f"Removed {result.episodes_removed} episodes → {result.summaries_created} summaries")
```

Compression is lossy by design. Run `reflect()` first to extract facts from episodes before compressing them — facts survive compression, raw text does not.

---

## Under the hood — technical details

### Bitemporal validity

Every fact carries *two* independent timelines:

```
valid_from / valid_to       → when the fact was TRUE in reality
recorded_at / superseded_at → when the system LEARNED it
```

### Hybrid BM25 + cosine recall

Three retrieval modes unified in one API:

```
mode="cosine"    → pure vector similarity (semantic)
mode="hybrid"    → FTS5 BM25 + cosine, normalised and blended
mode="spreading" → cosine KNN seeds → BFS over Hebbian graph
```

### Importance scoring formula

```
importance(m, t) =
    salience(m) × exp(−λ × (t − last_access(m)))   # Ebbinghaus forgetting curve
  + α × log(1 + access_count(m))                    # Hebbian reinforcement
  + β × emotional_weight(m)                          # affective weight
```

Parameters `λ`, `α`, `β` are configurable via `DecayConfig`.

### Spreading-activation graph traversal

```
query → seed memories (cosine KNN)
              ↓
         graph edges (Hebbian weights — reinforced by co-access)
              ↓
         activated neighbors (activation × decay per hop)
              ↓
    rank by: α·similarity + β·activation + γ·importance
```

### Working memory — Miller's 7±2 law

Fixed-capacity LRU cache backed by `collections.OrderedDict`. Evicted items optionally written to long-term store via `observe()`. Capacity default of 7 matches the average human working memory span (Miller, 1956).

---

## Install

```bash
pip install engdbram

# LLM-powered reflection (optional, pick one):
pip install 'engdbram[anthropic]'   # Claude
pip install 'engdbram[openai]'      # OpenAI or any OpenAI-compatible

# Integrations:
pip install 'engdbram[mcp]'         # MCP server (Claude Desktop, Cursor, etc.)
pip install 'engdbram[langchain]'   # LangChain retriever + chat history
pip install 'engdbram[llamaindex]'  # LlamaIndex memory buffer

# Everything:
pip install 'engdbram[anthropic,mcp,langchain,llamaindex]'
```

> The distribution name on PyPI is **`engdbram`** (the `engram` name is squatted). The import name is still `engram`, so application code is unaffected: `from engram import Engram`.

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

### Async API

```python
import asyncio
from engram import AsyncEngram, ObserveInput

async def main():
    async with AsyncEngram(path="./agent.engram") as mem:
        # All methods are async — event loop never blocked by ONNX or SQLite
        await mem.observe("Alice joined Globex as CTO", actors=["Alice"])
        await mem.observe_many([
            ObserveInput(content="Q3 planning complete", tags=["planning"]),
            ObserveInput(content="Ivan submitted architecture proposal", actors=["Ivan"]),
        ])

        results = await mem.recall("who joined Globex?", k=3)
        for r in results:
            print(f"[{r.score:.2f}] {r.episode.content}")

        await mem.assert_fact("Alice", "role", "CTO")
        facts = await mem.timeline("Alice")

asyncio.run(main())
```

### Working memory scratchpad

```python
from engram import Engram, WorkingMemory

with Engram(path="./agent.engram") as mem:
    # 5-slot scratchpad; evicted items automatically saved to long-term memory
    wm = WorkingMemory(capacity=5, engram=mem)

    wm.set("goal", "Draft the board presentation")
    wm.set("context", "Q3 revenue up 12%, but CAC increased")
    wm.set("constraint", "Must fit 10 slides, no more")

    task = wm.get("goal")        # promotes to most-recently-used
    note = wm.peek("constraint") # reads without changing LRU order

    print(f"Current slots: {len(wm)} / {wm.capacity}")
    wm.flush()  # write everything to long-term store + clear
```

### Hybrid recall

```python
with Engram(path="./agent.engram") as mem:
    # BM25 keyword match + cosine vector search, blended
    results = mem.recall("Alice quarterly roadmap", k=5, mode="hybrid")

    # Tune the blend weights
    results = mem.recall(
        "exact phrase match needed",
        k=5,
        mode="hybrid",
        vector_weight=0.3,  # less semantic
        fts_weight=0.7,     # more keyword
    )
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

### Memory compression

```python
from engram import Engram, AnthropicAdapter

mem = Engram(
    path="./agent.engram",
    llm=AnthropicAdapter(model="claude-haiku-4-5-20251001"),
)

# Compress episodes with low importance into LLM summaries
result = mem.compress(
    max_episodes=500,         # no-op if store is smaller than this
    importance_threshold=0.3, # episodes below this score are candidates
    batch_size=20,            # episodes per LLM call
)
print(f"Compressed {result.episodes_removed} episodes → {result.summaries_created} summaries")
print(f"Tokens used: {result.cost_tokens}")

mem.close()
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

### Backup and export

```python
# Hot backup — safe to call while the store is open
mem.backup("./agent_backup.engram")

# Portable JSON export (episodes, facts, entities, edges)
doc = mem.export_json("./agent_dump.json")
print(f"Exported {doc['counts']['episodes']} episodes, {doc['counts']['facts']} facts")

# Import into another store
with Engram(path="./new_store.engram") as dst:
    counts = dst.import_json("./agent_dump.json")
    # merge=True skips duplicate ids instead of raising
    counts = dst.import_json("./agent_dump.json", merge=True)
```

### GDPR right-to-be-forgotten

```python
# Permanently erase a single episode
mem.forget(episode_id)

# Permanently erase a single semantic fact
mem.forget_fact(fact_id)

# Erase everything about a person: episodes, facts, graph edges
result = mem.forget_entity("Ivan")
print(f"Deleted {result.episodes_deleted} episodes, {result.facts_deleted} facts")
```

---

## CLI

Engram ships a command-line interface for inspecting and operating stores without writing code:

```
engram inspect     <path> [--agent-id ID]
engram recall      <path> <query> [--k K] [--mode cosine|spreading|hybrid] [--as-of DATE]
                                  [--agent-id ID] [--cross-agent]
engram timeline    <path> <entity>
engram observe     <path> <content> [--actors NAME...] [--tags TAG...]
                                    [--salience F] [--valence F] [--agent-id ID]
engram reflect     <path> [--llm anthropic|openai] [--model MODEL]
                          [--base-url URL] [--agent-id ID]
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

# Recall (cosine, hybrid, or spreading)
engram recall ./agent.engram "Ivan employer" --k 3
engram recall ./agent.engram "Ivan employer" --mode hybrid --k 5

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

### `Engram(path, *, embedder_model, decay_config, llm, agent_id, key)`

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
)

# Context-manager supported
with Engram(path=":memory:") as mem:
    mem.observe("hello world")
```

> **Encryption-at-rest:** pass `key="..."` to encrypt the database via
> SQLCipher (`pip install 'engdbram[encryption]'`). Plain (no-key) stores
> are unchanged. Use `mem.rekey(new_key)` to change or remove the key.

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

### `recall(query, k, *, mode, depth, decay, vector_weight, fts_weight, as_of, cross_agent, k_inner, candidate_limit) → list[SearchResult]`

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

# Tune the candidate pool sizes for harder bitemporal / hybrid queries
results = mem.recall("Ivan employer", k=5, as_of=t, k_inner=200)        # vector inner KNN size
results = mem.recall("Q3 revenue", k=5, mode="hybrid", candidate_limit=64)  # per-source pool
```

`SearchResult` fields: `episode`, `score` (0–1, higher is better — derived
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
contains that timestamp — the public entry point to the bitemporal
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

Uses a single SQL `GROUP BY` fetch and a single `executemany` update — O(1) SQL round-trips regardless of episode count.

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
`asyncio.to_thread` — the event loop is never blocked by ONNX inference or
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

# Claude (default: haiku — fast, cheap)
llm = AnthropicAdapter(model="claude-haiku-4-5-20251001")

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
confidence=1.0)` are not capped — the cap is specifically for facts
mined from user-controlled text.

---

## Integrations

### MCP Server

`pip install 'engdbram[mcp]'` adds an `engram-mcp` console command that exposes
a store as an [MCP](https://modelcontextprotocol.io) tool server over stdio —
no network listener, compatible with Claude Desktop, Claude Code, Cursor, and
any other MCP host:

```bash
engram-mcp --db ./agent.engram
engram-mcp --db ./agent.engram --agent-id my-agent   # scope the default agent

# equivalently, via environment variables
ENGRAM_MCP_DB=./agent.engram ENGRAM_MCP_AGENT_ID=my-agent engram-mcp
```

If `--db`/`ENGRAM_MCP_DB` is omitted, the server falls back to an in-memory
store that is discarded on exit — fine for a quick trial, not for anything
you want to keep.

Five tools are registered:

| Tool | Purpose |
|---|---|
| `remember` | Store a memory. `kind="episodic"` (default) takes `content`; `kind="semantic"` takes `subject`/`predicate`/`object` instead — the structured triple, not free text. Both accept an optional per-call `agent_id` to override the server's default agent scope. |
| `recall` | Retrieve memories relevant to a natural-language `query`, with `mode="cosine"` (default), `"hybrid"`, or `"spreading"`, and an optional `limit` and `agent_id`. |
| `why` | Explain a memory's provenance by id — extraction chain and confidence for a fact, encoding/access metadata for an episode. |
| `forget` | Permanently delete one memory (episodic or semantic) by id. Irreversible. |
| `stats` | Store-wide counts (episodic, semantic, procedural — the last is always 0, see below), fact validity breakdown, entity/reflection counts, db file size. |

A few things worth calling out:

- **`kind="procedural"` is deliberately rejected.** Engram's store implements
  episodic and semantic memory only; there is no procedural memory layer to
  write to, so `remember(kind="procedural", ...)` raises a clear error instead
  of silently doing nothing.
- **`reflect()` is not exposed as a tool.** It can call an external LLM to
  extract facts from episodes, which is out of scope for a zero-config memory
  server and would break the "no network calls at write time" guarantee every
  other tool here upholds. Run reflection out-of-band — `engram reflect` (CLI)
  or `mem.reflect_async()` (library) — against the same `.engram` file.
- **`agent_id` is an opaque string.** Agent Passport `agent://...` URIs are accepted as-is and used verbatim as the scoping key — Engram does not parse, resolve, or validate them.

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
├── contradictions()   → active facts with same (subject, predicate)
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
| `recall(mode="hybrid")` | 4.6 ms | 5.3 ms |
| `recall(mode="spreading")` | 4.4 ms | 5.0 ms |
| `recall(as_of=...)` | 4.5 ms | 5.2 ms |

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
# Both spellings work
python -m engram.benchmarks all
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

### Test coverage (392 tests)

```
tests/
  test_schema.py         schema + SQLite migrations (incl. backward compat)
  test_observe.py        observe() + embeddings
  test_recall.py         cosine recall
  test_hybrid_recall.py  hybrid BM25 + cosine recall, FTS index population
  test_smoke.py          end-to-end Engram class
  test_importance.py     decay formula
  test_decay.py          decay background job + access log
  test_store_facts.py    fact CRUD + assert_fact()
  test_reflection.py     reflection loop (stub LLM), cost_tokens, reflect_async
  test_graph.py          entity/edge CRUD + spreading recall
  test_bitemporal.py     as_of + timeline (incl. naive-datetime boundaries)
  test_forget.py         forget(), forget_entity(), GDPR cascade
  test_cli.py            all CLI subcommands + --agent-id + --cross-agent
  test_multiagent.py     agent_id scoping, shared facts, cross-agent recall
  test_performance.py    observe_many correctness + batch decay + LRU cache
  test_export.py         export_json / import_json round-trip + merge mode
  test_backup.py         backup() — hot copy, openable as Engram
  test_working_memory.py WorkingMemory LRU, eviction, flush, spillover
  test_async_engram.py   AsyncEngram — all async methods + concurrency safety
  test_compress.py       compress() — LLM summarisation, batching, no-op paths
  test_encryption.py     SQLCipher encryption-at-rest + rekey()
  test_llm_adapters.py   all LLM adapters + response-parsing edge cases
  test_integrations.py   LangChain, LlamaIndex
  test_mcp_server.py     MCP server (engram-mcp): remember/recall/why/forget/stats,
                          agent pooling, procedural rejection, reflect() not exposed
  test_benchmarks.py     benchmark infrastructure
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
- [x] v1.3 — Multi-agent shared memory (`agent_id`, `cross_agent`, `list_agents()`)
- [x] v2.0 — Batch decay (21×), `observe_many()` (2×), embedding LRU cache
- [x] v2.0.1 — WAL journal mode + 32 MB page cache (4× faster commits, concurrent reads/writes)
- [x] v2.1 — Hybrid recall (FTS5 BM25 + cosine), `WorkingMemory`, `AsyncEngram`, `compress()`, `backup()`, `export_json` / `import_json`
- [x] v2.1.1 — GitHub Actions CI, `DATA_FLOW.md`, tunable `k_inner` / `candidate_limit`, adapter history hydration, PyPI distribution renamed to `engdbram`
- [x] v2.1.2 — Multi-agent isolation hardening (per-agent `prune`, FTS cleanup), hybrid `as_of`, FTS5 query safety, embedder normalization, prompt-injection hardening, async API parity (`timeline(as_of=)`, `recall(k_inner=, candidate_limit=)`), tag-triggered PyPI publishing via OIDC
- [x] v2.2.0 — Correctness pass (bitemporal `as_of` UTC coercion, `contradictions()` identical-fact fix, `import_json` re-embed + FTS, reflection abort rollback, LLM response-parse guards), thread-safety (`Store` + embedder locks for `reflect_async` / `AsyncEngram`), per-agent edge & decay scoping, PEP 561 `py.typed`, CI matrix (Python 3.13 + encryption job), release gated on tests, `migrate()` fails loudly on embedder-dimension mismatch, `pip-audit` CI job
- [x] Since v2.2.0 (on `main`, not yet tagged) — MCP server tool surface (`engram-mcp`): stdio transport, optional `[mcp]` extra, `remember`/`recall`/`why`/`forget`/`stats` tools with structured semantic params and per-call `agent_id`, `reflect()` deliberately not exposed; public `forget_fact()` API (sync + async) for erasing a single semantic fact

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

- Architecture rationale and design decisions: [DESIGN.md](DESIGN.md)
- Release notes by version: [CHANGELOG.md](CHANGELOG.md)
- Read / write paths and on-disk guarantees: [DATA_FLOW.md](DATA_FLOW.md)
