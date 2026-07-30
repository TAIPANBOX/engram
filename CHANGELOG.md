# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.0] - 2026-07-30

### Changed
- **`recall()` defaults to `mode="hybrid"`** instead of `"cosine"`, in the
  library, the CLI, the MCP server tool and the LangChain retriever. On the
  full 500 questions of LongMemEval-S, scored against the same store per
  question, hybrid beats cosine at every k and by most where it matters to an
  agent: 0.820 against 0.772 on retrieving the specific turn that holds the
  answer at k=5. It is also faster, 12 ms against 15 ms per query, because
  BM25 narrows the candidate set. The two modes disagreed on 57 of the 500
  questions. Pass `mode="cosine"` for the old behaviour; cosine leads on
  exactly one cut, `multi-session` at k=10.
- **The vector index is rebuilt on first open.** `vec_episodes` now declares
  `agent_id` as a vec0 partition key and `ts` as a metadata column, so scoped
  and `as_of` recall are resolved inside the KNN scan instead of being filtered
  after it. Migration is automatic and runs once per store, in one O(N) pass
  that holds the vectors in memory while the table is replaced.
- **A rebuilt `.engram` file requires `sqlite-vec` 0.1.6 or newer** and cannot
  be opened by an older install. The dependency floor moved to 0.1.9
  accordingly. Copy a store before upgrading if it has to stay readable by an
  older environment.
- `Engram.observe()` and `AsyncEngram.observe()` accept `timestamp=`, for
  back-filling history at the time it happened rather than the time it was
  written.

### Fixed
- Scoped recall could return nothing at all. vec0 resolved its `k` against the
  whole index and `agent_id` was applied afterwards in the join, so the filter
  could only cut into an already-chosen global top-k: in a shared store where
  one agent held most of the episodes, `recall(k=5)` from another agent
  returned zero rows rather than that agent's five nearest. `as_of` had the
  same shape, and a date far enough back emptied the window. Both filters are
  now vec0 constraints. Existing tests missed this because their stores held
  two episodes, where the global top-k trivially contains everything.
- `compress()` no longer erases the period it summarises. The summary episode
  is stamped with the newest timestamp among its sources instead of the moment
  it was written, so `recall(as_of=...)` into a compressed window finds the
  summary standing in for the originals it hard-deleted.
- `forget_entity()` is atomic. It ran as a sequence of self-committing
  deletes, so an interruption could leave an entity half-erased with nothing
  recording that the erasure never finished.
- `recall(k=...)` above the vec0 limit of 4096 raises `ValueError` naming the
  limit, instead of surfacing a raw `sqlite3.OperationalError`.

### Added
- **Recall accuracy on LongMemEval-S**, full 500 questions, 246 738 turns.
  Hybrid: session recall 0.968 at k=5 and 0.982 at k=10, turn recall 0.820 and
  0.892. Cosine: 0.956 / 0.978 and 0.772 / 0.862.
  `engram-bench longmemeval` runs it, `--checkpoint` makes a six-hour run
  resumable, and the per-question records behind the table are committed in
  `benchmarks/results/`. First published recall number that comes from a
  public dataset rather than from a fixture written alongside its own answers.
- `engram-bench scale` sweeps store sizes and reports search latency, file
  size and resident memory at each, replacing read-latency numbers that were
  measured at 300 episodes and were mostly the query embedding. Published
  results at 1k/10k/100k are in the API reference: unscoped search is linear
  (0.18 / 3.03 / 31.8 ms p50), scoped search is flat (0.05 / 0.05 / 0.09 ms)
  because `agent_id` partitions the index, `as_of` costs about 3.4x cosine,
  and memory does not grow with the store.

### Removed
- The published recall-accuracy numbers (`hit@1 33.3%`, `hit@5 93.3%`,
  `MRR 0.586`). They were produced by the fixture bundled in
  `engram/benchmarks/data/`, which is five hand-written sessions and fifteen
  questions whose keywords were chosen alongside the text they match, and were
  presented under the heading "LoCoMo recall quality", which read as a result
  on the LoCoMo benchmark and was not one. The harness still scores any file
  in that format via `engram-bench locomo --data`, and now says plainly when
  it is running on the synthetic fixture. Numbers return when there is a run
  against a public dataset behind them.

### Deprecated
- `recall(k_inner=...)` is ignored and warns. It sized an over-fetch that
  compensated for filters running outside the vector search; there is no inner
  limit left to tune.

### Fixed
- `reflect()` no longer counts a same-object re-extraction as a resolved
  contradiction and no longer emits a `contradiction_found` event for it. The
  older row is still superseded (fresh provenance, intact `superseded_by`
  chain), but silently: same (subject, predicate, object) is agreement, which
  is what `contradictions()` already said. Only a differing object counts and
  emits.

## [2.2.1] - 2026-07-15

### Security
- `forget()`, `delete_episode()`, and `get_episode()` are now scoped by
  `agent_id`: in a pooled multi-agent database one agent can no longer read or
  delete another agent's episodes by id. Facts and entities stay shared by
  design.

### Fixed
- A malformed fact mid-`reflect()` no longer leaves a dangling reflection run
  or duplicated facts: the whole run is wrapped in a store transaction and
  rolls back atomically on failure.
- Concurrent `reflect()` calls on one instance are serialized, eliminating
  double-processing of the same episodes (the LLM call stays outside the lock).

### Changed
- License switched from MIT to Apache-2.0 (LICENSE file and packaging
  metadata; this is the first PyPI release whose metadata carries Apache-2.0).
- LICENSE copyright holder unified to TAIPANBOX.

## [2.2.0] - 2026-07-01

### Added
- `migrate()` now fails loudly with `ValueError` when an existing store is
  opened with an embedder of a different dimension, instead of silently
  corrupting vector search (the vec0 dimension is immutable after creation).
- CI: `security` job running `pip-audit` against the installed environment.
- PEP 561 `py.typed` marker — downstream consumers now get Engram's type hints.
- CI: Python 3.13 added to the test matrix; a dedicated `encryption` job
  installs `libsqlcipher-dev` and exercises the SQLCipher path. (macOS is not
  in the matrix: the hosted macOS Python lacks loadable SQLite extension
  support that sqlite-vec requires.)
- `release.yml` now lint/type-checks and runs the full test suite before
  publishing, so a tag on a broken commit cannot reach PyPI.

### Fixed
- Bitemporal `as_of` / `since` comparisons are now correct for naive (tz-less)
  datetimes. Timestamps are compared as TEXT in SQLite; a naive `as_of`
  serialised without the `+00:00` offset and flipped the boundary comparison,
  silently returning the wrong point-in-time facts/episodes. All stored and
  queried datetimes are now coerced to a canonical UTC isoformat.
- `contradictions()` no longer reports two facts with identical
  `(subject, predicate, object)` as a conflict — that is agreement, not a
  contradiction.
- `import_json()` now re-embeds episode content and repopulates the FTS index
  instead of writing zero vectors, so imported episodes are actually findable
  via vector and hybrid recall.
- A reflection run that aborts mid-extraction (e.g. an LLM/API error) is now
  rolled back instead of lingering unfinished; the incremental window keys off
  the last *completed* run, so episodes are no longer silently skipped after a
  failure.
- Anthropic/OpenAI response parsing degrades to "no facts" on structurally
  surprising responses (empty content, a leading `tool_use` block) instead of
  raising `IndexError`/`AttributeError` mid-reflection.
- The shared SQLite connection is now serialised behind a re-entrant lock
  (`Store`) and the embedder cache behind its own lock, making `reflect_async`
  and `AsyncEngram`'s thread-pool dispatch safe against interleaved writes.
- Hybrid recall no longer collapses a lone candidate's normalised score to
  `0.0`; an only/equal-scored hit now maps to `1.0`.
- Importance decay clamps elapsed time at 0 so clock skew can't flip the
  exponent positive and blow the score up.
- Spreading-activation edges are now scoped per agent: in a shared store,
  activation no longer hops through another agent's episodes (at `depth>=3`
  this could perturb an agent's own ranking). Entities and facts stay shared,
  so cross-agent semantic memory and global `forget_entity()` are unchanged.
- Importance decay is now scoped to the calling agent, symmetric with the
  agent-scoped prune: one agent's `reflect()`/`decay()` no longer recomputes
  another agent's scores with the wrong `DecayConfig`.

### Changed
- New nullable `agent_id` column on the `edges` table (auto-migrated; existing
  stores backfill to `NULL`). JSON export/import now round-trips it.

## [2.1.2] - 2026-05-29

### Fixed
- `prune_episodes` is now scoped to the calling agent's `agent_id` in shared
  multi-agent stores, no longer deletes other agents' low-importance episodes,
  cleans the FTS index, and removes orphaned dst edges.
- Hybrid recall (`mode="hybrid"`) now honors `as_of`; previously the parameter
  was silently dropped.
- FTS5 queries no longer crash on bare `*`, `(`, `OR`, `NOT`, `-`, or `"` in
  user input — each token is escaped and wrapped as a phrase.
- Spreading-activation retrieval no longer double-counts the seed cosine score
  (the spurious `beta * cosine` on top of `alpha * cosine`) and now respects
  the Hebbian edge weight, clamped to `[0, 1]`.
- Embedder L2-normalizes all returned vectors so the "cosine" score is
  metric-correct for every fastembed model, not just bge-small/bge-base; the
  dim is now probed at runtime rather than silently defaulting to 384 for
  unknown models.
- Reflection is hardened against prompt injection: episode bodies are wrapped
  in `<observation>` blocks, the system prompt instructs the model to treat
  them as inert data, all extractions run at `temperature=0`, and any
  LLM-derived confidence is capped at `0.95` (also clamped in
  `reflection.reflect()` for stubs that bypass JSON parsing).
- LangChain `EngramChatMessageHistory` and LlamaIndex `EngramMemory` no longer
  silently swallow hydration errors; failures are logged and hydration aborts
  cleanly.
- mypy strict pass works whether or not the optional `sqlcipher3` package is
  installed (CI was previously red).

### Added
- `Engram.timeline(entity, as_of=…)` routes to `get_facts_as_of`, exposing the
  bitemporal fact path through the public API for the first time.
- `AsyncEngram.recall` accepts `k_inner` and `candidate_limit`; `AsyncEngram.timeline`
  accepts `as_of` — async surface is now at parity with sync.
- Tests covering adapter message-history hydration, FTS5 escape edge cases,
  hybrid + `as_of`, prompt-injection confidence clamping, the spreading-
  activation double-count regression, and edge-weight respect.

### Changed
- `AsyncEngram` uses `asyncio.to_thread(...)` instead of the deprecated
  `asyncio.get_event_loop().run_in_executor(None, ...)`.
- README / DESIGN.md / GETTING_STARTED.md / error messages reference the
  `engdbram` PyPI distribution name consistently. Import name is unchanged
  (`from engram import Engram`).

## [2.1.1] - 2026-05-21

### Added
- GitHub Actions CI workflow (lint, typecheck, tests on Python 3.11 / 3.12).
- `DATA_FLOW.md` documenting the read/write paths and on-disk guarantees.
- `Engram.recall()` exposes `k_inner` and `candidate_limit` as tunables.
- LangChain `EngramChatMessageHistory` and LlamaIndex `EngramMemory` rebuild
  their in-memory conversation buffer from persisted episodes on construction.

### Changed
- PyPI distribution name renamed `engram-ai` → `engdbram` (the `engram` name
  is squatted upstream). Import name unchanged.

## [2.1.0] and earlier

See `git log` for the pre-2.1.1 history.
