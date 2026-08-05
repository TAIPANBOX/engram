# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- **`export_json()` is scoped to the calling instance's `agent_id`.** It was
  the read path 2.2.0 and 2.2.1 left open. `Engram(path="./team.engram",
  agent_id="coder").export_json("dump.json")` wrote every other agent's raw
  episode text into a plaintext file, while `get_episode()`, `recall()`,
  `forget()`, `prune_episodes()`, `decay()` and `get_neighbors()` had all been
  closed against exactly that, and the export path was never covered by a test
  that used a scoped instance at all.

  Episodes and edges carry an `agent_id` and are now filtered to it. Facts and
  entities are not, and that is the deliberate half: they have no `agent_id`
  column, 2.2.0 and 2.2.1 both record that they stay shared across the agents
  in one file, and every fact and entity read path is already cross-agent. An
  export that hid them would have turned a decision into a bug and hidden
  nothing, since the same instance can read them through `timeline()` and
  `contradictions()`. The rule the fix follows is narrower than "scope
  everything": an export must not carry more than the instance can already
  read. An unscoped instance still exports the whole store.
- **`backup()` refuses on an agent-scoped instance**, with a `ValueError`
  naming both alternatives. It copies the whole file through SQLite's online
  backup API, so unlike an export it cannot be filtered: the choice was
  between a copy that hands over every agent's data and a refusal. Proceeding
  with a warning was the other candidate and loses on the only question that
  matters here, because the file is delivered either way. Open the store
  without an `agent_id` to back up the file, or use `export_json()` for one
  agent's own data. This is a behaviour change for anyone who called
  `backup()` from a scoped instance; nothing in this repo did, and the CLI
  exposes no backup command.

  Not changed, and stated here rather than left to be rediscovered:
  `import_json()` keeps the `agent_id` recorded in the document instead of
  re-stamping rows with the importing instance's scope. That is what makes a
  migration preserve who wrote what, and 2.2.0 added edge `agent_id`
  round-tripping for the same reason, but it does mean a scoped instance can
  write rows outside its own scope from a hand-edited document, which the read
  side no longer permits. The asymmetry is known, not overlooked.

### Fixed
- **The three write paths that most needed an audit trail emitted nothing.**
  `observe_many()`, `forget_entity()` and the deletion half of `compress()`
  wrote to the store without writing to the event log, while the single-item
  paths beside each of them emitted. None of the three had a test in
  `tests/test_events.py`, which is why all three could be missed at once.

  `observe_many()` now emits one `memory_written` per episode, the same
  envelope `observe()` emits. One event per memory rather than one summary
  event per batch: the envelope carries one `memory_id` (SPEC.md §6.2), so a
  batched payload would be a second data shape under an existing type that
  every consumer would have to learn, and a single line holding an unbounded
  id array would outgrow the 1 MiB window the chain reader uses to resume a
  file. The batch is written in one append instead of one per event, through
  a new `EventLog.emit_many`. Measured locally rather than in a committed
  benchmark, on 246 738 payloads (the LongMemEval-S turn count) in the
  500-item batches `engram/benchmarks/longmemeval.py` loads with, Python
  3.14.6: 4.97 s and 81.7 MiB of NDJSON, against 15.14 s for the same events
  one `emit()` call each. Events stay off unless a path is configured, so
  nothing pays this by default.

  `forget_entity()`, the documented right-to-be-forgotten path, now emits one
  `memory_forgotten` per erased memory, episodes and facts alike, each naming
  the entity whose erasure caused it. Per memory rather than one event for the
  run, because a count cannot be reconciled against anything: the point of an
  erasure record is that an auditor can pair it with the `memory_written` that
  created that memory. The ids are read inside the same transaction that
  erases them, so the log lists what was actually deleted rather than what was
  there a moment earlier.

  `compress()` hard-deleted every source episode through the store, bypassing
  the event-emitting `forget()`, while emitting a `memory_written` for each
  summary it created. A run therefore appeared in the log as pure creation,
  with the deletions invisible, which is the one direction an audit log must
  not be wrong in. Each deleted source now emits a `memory_forgotten` naming
  the summary that replaced it, which is the only link left between the two
  halves once the original is gone.

  No new event type. Compression is genuinely "N memories replaced by one" and
  none of the four types engram is registered for names that, but a creation
  plus N deletions that point at it describes the run without inventing a
  fifth type, and the registry in agent-passport SPEC.md §6.2 is not this
  repo's to change.
- **Episodes written before v2.1 are indexed for full-text search on the next
  open.** They never were. The backfill that was supposed to do it filtered on
  `rowid NOT IN (SELECT rowid FROM fts_episodes)`, and `fts_episodes` is an
  external-content table, so an unqualified scan of it reads through to
  `episodes` and returns every rowid whether or not it is indexed. The
  predicate excluded everything; the INSERT reported 0 rows on every store it
  has ever run on. Nothing looked wrong from the SQL, and the suite did not
  catch it because it checked the index only after `observe()` and
  `observe_many()`, which populate it by a different path and always worked.

  Since 2.3.0 made `hybrid` the default this has been a silent ranking loss
  rather than an error. An unindexed episode is not missing: it scores 0.0 on
  the BM25 half of every recall and sits below anything written after the
  upgrade. The store looks healthy and quietly answers with its newest
  memories, which is the failure mode that does not get reported.

  The repair is `INSERT INTO fts_episodes(fts_episodes) VALUES('rebuild')`, the
  FTS5 idiom for exactly this. It reindexes the whole table, so it is gated on
  `PRAGMA user_version` (previously unused, so 0 on every existing store) and
  runs once rather than on every open. Counting the index to decide instead is
  the same trap: `SELECT count(*) FROM fts_episodes` returns the episode count
  even when nothing at all is indexed, and `tests/test_schema.py` now asserts
  that, so the next person to reach for it finds out from a test rather than
  from a store that ranks its oldest memories last.

  Measured locally rather than in a committed benchmark, on 100 000 synthetic
  episodes under Python 3.14.6 with SQLite 3.53.4: the rebuild takes 0.20 s and
  the `PRAGMA user_version` read that now guards it takes 5.4 microseconds.

## [2.4.1] - 2026-07-30

### Added
- The LongMemEval checkpoint records the rank of every evidence and gold-session
  hit, and each question's evidence count, alongside the pass/fail flags. Any
  smaller k, and the strict "every flagged turn retrieved" reading, are now
  recomputable from a finished run instead of requiring another six-hour pass.
  Storing the verdict and discarding the observation is what made the first run
  unrepeatable except by repeating it.
- `engram-bench longmemeval` prints turn recall three ways when the checkpoint
  carries ranks: any flagged turn retrieved, the same over questions that flag
  any turn at all, and every flagged turn retrieved.

### Fixed
- The published turn-recall figure is now stated with the caveat it needed. 21
  of the 500 questions flag no turn, so they cannot be hit and lower the number
  for a reason unrelated to retrieval: over the 479 answerable questions the
  figure is 0.866 at k=5, not 0.830. And 59% of questions flag more than one
  turn, so "at least one retrieved" is an upper bound on what the model was
  actually handed, which the docs now say rather than imply.
- The headline recall table said `hybrid` (default) over the figures for
  `0.7 / 0.3`, which stopped being the default in 2.4.0 itself. Both rows were
  real numbers from real runs and the sweep table below carried the right ones
  under their own label, so nothing looked wrong, but a reader comparing Engram
  to another system would have taken a configuration the library no longer
  ships. Corrected in the README, the API reference, DESIGN.md, the
  `Engram.recall` docstring and the pipeline diagram, all sourced from one run.

## [2.4.0] - 2026-07-30

### Changed
- **The hybrid blend defaults to `0.5 / 0.5`** instead of `0.7 / 0.3`. The old
  weights shipped without anyone measuring them; `engram-bench longmemeval
  --sweep` now scores every configuration in one pass, and on all 500 questions
  of LongMemEval-S the equal blend leads on session@5, turn@5, session@10 and
  turn@10 at once. The margin is one point of turn@5, five questions in five
  hundred, and the two sit on the same plateau. The result worth having is the
  shape: BM25 alone (0.750 turn@5) and vector alone (0.772) are both clearly
  worse than the middle (0.830).

### Added
- `engram-bench longmemeval --sweep` scores cosine plus hybrid across seven
  blend points in a single pass. Ingest is shared, so the whole sweep costs
  about a minute on top of a six-hour run. Records in `benchmarks/results/`.

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
