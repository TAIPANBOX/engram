# Engram Data Flow & Database Architecture

This document describes the cognitive data flow and the underlying SQLite database operations of **Engram** — an embeddable, local-first memory layer for AI agents.

---

## 1. Cognitive Memory Architecture (Data Flow)

Engram coordinates memory across two primary loops:
1. **System-1 (Fast/Episodic Memory)**: Real-time observation logging, vector embeddings, full-text search indexing, retrieval-access tracking, and short-term working memory scratchpads.
2. **System-2 (Slow/Semantic Memory)**: Background reflection (sleep), LLM-based fact extraction (subject, predicate, object triples), Hebbian-reinforced entity-relationship graph building, bitemporal alignment of conflicting beliefs, and importance decay pruning.

### Sequence Flow Diagram

The following sequence diagram shows the lifecyle of observations, retrievals, and consolidation into semantic facts and Hebbian edges:

```mermaid
sequenceDiagram
    autonumber
    actor Agent as AI Agent / Client
    participant Core as Engram Core (core.py)
    participant Emb as Embedder (fastembed)
    participant DB as SQLite DB & extensions
    participant WM as WorkingMemory (RAM)
    participant LLM as Reflection LLM (System-2)

    Note over Agent, WM: Phase 1: Ingestion & Working Memory (observe & WM)
    Agent->>WM: wm.set(key, content)
    activate WM
    Note over WM: Miller's Law (7±2 capacity)<br/>LRU Eviction
    WM-->>Core: Flush evicted item (observe)
    deactivate WM

    Agent->>Core: observe(content, actors, tags)
    activate Core
    Core->>Emb: embed(content)
    Emb-->>Core: 384-dim Vector (float32)
    Core->>DB: INSERT INTO episodes, vec_episodes, fts_episodes
    DB-->>Core: RowID / Episode ID
    Core-->>Agent: return Episode ID
    deactivate Core

    Note over Agent, DB: Phase 2: Retrieval & Access Tracking (recall & access_log)
    Agent->>Core: recall(query, mode="hybrid")
    activate Core
    Core->>DB: Vector Distance (sqlite-vec) + Text Search (FTS5 BM25)
    DB-->>Core: Top-K candidates
    Core->>DB: Buffer access in RAM (flush every 50 to access_log)
    Core-->>Agent: return list[SearchResult]
    deactivate Core

    Note over Agent, LLM: Phase 3: Reflection, Facts & Graph (reflect & decay)
    Agent->>Core: reflect()
    activate Core
    Core->>DB: Query episodes since last reflection
    DB-->>Core: Unprocessed episodes
    Core->>LLM: Extraction Prompt (JSON schema)
    LLM-->>Core: Extracted facts (s, p, o, confidence)
    
    Core->>DB: Assert facts & find_or_create_entity()
    Note over Core, DB: Bitemporal alignment:<br/>if conflict found (s, p),<br/>close old fact (valid_to = NOW)
    DB-->>Core: Resolve contradictions & save timeline
    
    Core->>DB: Create/reinforce edges (episodic ➔ entity, entity ➔ entity)
    Note over Core, DB: Hebbian Learning:<br/>weight = weight + excluded.weight
    
    Core->>Core: decay() -> recompute importance (Ebbinghaus)
    Core->>DB: Prune low-importance episodes & update database
    Core-->>Agent: return ReflectionRun audit trail
    deactivate Core
```

---

## 2. Core Phases & SQLite Database Operations

### Phase 1: Ingestion & Working Memory
* **Working Memory (`WorkingMemory`)**: Serves as a fast RAM buffer. Holds active context or transient goals using an LRU cache with a capacity bounded by Miller's Law ($7 \pm 2$). When an item is evicted, it can optionally spill over (flush) to the long-term store by calling `observe()`.
* **Observations (`observe`)**: Inserts new episodic observations into the SQLite `episodes` table.
* **Vector Embeddings**: Computes a 384-dimensional vector utilizing local ONNX inference (`fastembed`) and saves it in `vec_episodes` (a virtual `sqlite-vec` index).
* **Text Indexing**: Syncs the raw text content with `fts_episodes` (a virtual `FTS5` table) for traditional BM25 searches.

### Phase 2: Retrieval & Access Tracking
* **Cosine Recall**: Performs a vector nearest-neighbor search (`MATCH` query) against `vec_episodes` using L2 distance.
* **Hybrid Recall**: Pulls candidate matches from both FTS5 text search and vector ANN search, normalizes their scores, and ranks them according to weighted fractions (`vector_weight` and `fts_weight`).
* **Access Log Buffer**: Logs retrievals in `access_log` using a memory-buffered queue of 50 items. This avoids writing to disk on every query and maintains high read performance.

### Phase 3: Reflection & Maintenance
* **Fact Extraction**: During a `reflect()` run, an external LLM models raw episodes into semantic facts: `(subject, predicate, object)` triples.
* **Hebbian Graph Building**: Establishes links in the `edges` table. Connecting entities to each other and to the source episodes. Weight accumulates Hebbian-style (`ON CONFLICT DO UPDATE SET weight = weight + excluded.weight`).
* **Bitemporal Contradiction Handling**: When a new fact contradictions an existing fact, the old fact is not overwritten. The database updates its `valid_to` and `superseded_at` timestamps, establishing a permanent historical timeline.
* **Importance Decay**: Triggers the Hebbian-Ebbinghaus formula:
  $$\text{importance} = \text{salience} \cdot e^{-\lambda \Delta t} + \alpha \ln(1 + \text{access\_count}) + \beta \cdot \text{emotional\_valence}$$
  Episodes whose importance score falls below a configured threshold are deleted from the database.

---

## 3. SQLite Database Schema Quick Reference

```sql
-- Raw episodic observations
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    timestamp DATETIME,
    actors JSON,
    tags JSON,
    salience REAL,
    emotional_valence REAL,
    summary_of JSON,
    importance_score REAL,
    agent_id TEXT
);

-- Bitemporal semantic facts
CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    valid_from DATETIME,
    valid_to DATETIME,
    recorded_at DATETIME,
    superseded_at DATETIME,
    superseded_by TEXT,
    confidence REAL,
    derived_from JSON,
    extracted_by TEXT
);

-- Extracted Hebbian entities
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT,
    type TEXT,
    aliases JSON,
    first_seen DATETIME,
    last_seen DATETIME
);

-- Associative memory edges
CREATE TABLE edges (
    src_id TEXT,
    dst_id TEXT,
    relation TEXT,
    weight REAL,
    created_at DATETIME,
    PRIMARY KEY (src_id, dst_id, relation)
);
```
