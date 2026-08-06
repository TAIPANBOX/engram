"""Expose Engram as an MCP (Model Context Protocol) server.

This module lets any MCP-capable agent (Claude Desktop, Claude Code, or any
other MCP client) get persistent, provenance-tracked memory with zero
integration code: point the client at ``engram-mcp --db agent.engram`` and
five tools become available: ``remember``, ``recall``, ``why``, ``forget``,
and ``stats``.

Run with::

    engram-mcp --db ./agent.engram --agent-id agent://acme.example/my-agent
    ENGRAM_MCP_DB=./agent.engram python -m engram.mcp_server

Pass ``--events ./events.ndjson`` (or set ``ENGRAM_MCP_EVENTS``) to opt in to
an Agent Passport NDJSON event log alongside the store -- see
:mod:`engram.events`. Omit it and nothing is written; this is off by default.

Design notes
------------
* **Transport**: stdio only — the MCP default for local/subprocess agent
  integrations. No network listener is opened.
* **Optional dependency**: the ``mcp`` SDK (``pip install 'engdbram[mcp]'``)
  is imported lazily inside :func:`_build_server`, never at module import
  time, so ``import engram`` (and even ``import engram.mcp_server``) never
  requires it to be installed.
* **Thin wrapper**: every tool below delegates to the existing public
  :class:`engram.core.Engram` API (``observe``, ``assert_fact``, ``recall``,
  ``why``, ``forget``, ``forget_fact``). No retrieval, decay, or extraction
  logic is reimplemented here — see the module-level ``_remember``/
  ``_recall``/``_why``/``_forget``/``_stats`` functions, which are plain,
  independently testable functions that the FastMCP tool wrappers merely call.
* ``reflect()`` is intentionally **not** exposed as a tool. It may call an
  external LLM to extract facts from episodes, which is out of scope for a
  zero-config memory server and would violate Engram's "no network calls at
  write time" invariant for every other operation here.
* **Prompt-injection posture**: every tool description below is a static
  string literal, fixed at import time. None of them are ever built from
  memory content, query text, or any other request/response data, so stored
  memories cannot smuggle instructions into the tool metadata an MCP client
  reads.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from engram.core import Engram

# ----------------------------------------------------------------------
# Multi-agent instance pool
# ----------------------------------------------------------------------


class _EngramPool:
    """Lazily-constructed, per-agent :class:`Engram` instances over one db file.

    :class:`Engram` scopes ``agent_id`` at *construction* time (see
    ``Engram.__init__``): writes and default-scoped reads are fixed to
    whatever ``agent_id`` the instance was built with. There is no per-call
    way to redirect a write to a different agent. Supporting the
    ``agent_id`` override on ``remember``/``recall``/``stats`` therefore
    means keeping one :class:`Engram` instance per distinct agent actually
    requested, all pointed at the same database file — the same pattern
    used across multiple ``Engram(path=..., agent_id=...)`` instances
    sharing a file elsewhere in this codebase (see ``tests/test_multiagent.py``).

    Instances are cached so the embedding model is loaded at most once per
    distinct agent_id seen by this process, and closed together in
    :meth:`close`.

    Not a global: an instance of this class is created once in :func:`main`
    and threaded through via closures, never stored at module scope.

    Caveat: with ``db_path == ":memory:"``, each additional agent_id would
    open its own *disconnected* in-memory database rather than sharing
    state, since SQLite's ``:memory:`` databases are private per connection.
    Single-agent use of ``:memory:`` (the default when no ``--db`` is given)
    is fine; deliberately mixing ``:memory:`` with multiple agent ids is not
    supported and is left to the caller to avoid.
    """

    def __init__(
        self,
        db_path: str,
        default_agent_id: str | None,
        events_path: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.default_agent_id = default_agent_id
        self.events_path = events_path
        self._instances: dict[str | None, Engram] = {}

    def get(self, agent_id: str | None = None) -> Engram:
        """Return the Engram instance for *agent_id*, creating it if needed.

        ``None`` means "use the server's default agent scope".
        """
        key = agent_id if agent_id is not None else self.default_agent_id
        mem = self._instances.get(key)
        if mem is None:
            mem = Engram(path=self.db_path, agent_id=key, events_path=self.events_path)
            self._instances[key] = mem
        return mem

    def close(self) -> None:
        """Close every pooled Engram instance."""
        for mem in self._instances.values():
            mem.close()
        self._instances.clear()


# ----------------------------------------------------------------------
# Tool implementations (plain functions — independently unit-testable,
# no FastMCP dependency at all)
# ----------------------------------------------------------------------

_VALID_KINDS = ("episodic", "semantic", "procedural")


def _remember(
    pool: _EngramPool,
    content: str | None = None,
    kind: str = "episodic",
    agent_id: str | None = None,
    subject: str | None = None,
    predicate: str | None = None,
    object: str | None = None,  # noqa: A002
) -> dict[str, Any]:
    """Store a new memory. See ``_REMEMBER_DESCRIPTION`` for the contract."""
    mem = pool.get(agent_id)

    if kind == "episodic":
        if subject is not None or predicate is not None or object is not None:
            raise ValueError(
                "kind='episodic' takes only content; subject/predicate/object "
                "must not be provided (use kind='semantic' for a structured fact)"
            )
        if not content:
            raise ValueError("kind='episodic' requires non-empty content")
        memory_id = mem.observe(content)
    elif kind == "semantic":
        if content is not None:
            raise ValueError(
                "kind='semantic' takes subject/predicate/object; content must "
                "not be provided (use kind='episodic' for free-form text)"
            )
        if not subject or not predicate or not object:
            raise ValueError(
                "kind='semantic' requires all three of subject, predicate, "
                "and object to be non-empty strings"
            )
        memory_id = mem.assert_fact(subject, predicate, object)
    elif kind == "procedural":
        raise ValueError(
            "kind='procedural' is not supported: Engram's store currently "
            "implements only 'episodic' (observe) and 'semantic' (assert_fact) "
            "memory. This is a known gap in the underlying engine, not a bug "
            "in this tool — see the MCP phase report."
        )
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {_VALID_KINDS}")

    return {
        "id": memory_id,
        "kind": kind,
        "agent_id": agent_id if agent_id is not None else pool.default_agent_id,
    }


def _recall(
    pool: _EngramPool,
    query: str,
    limit: int = 5,
    agent_id: str | None = None,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """Retrieve memories relevant to *query*. See ``_RECALL_DESCRIPTION``."""
    mem = pool.get(agent_id)
    results = mem.recall(query, k=limit, mode=mode)
    return [
        {
            "id": r.episode.id,
            "content": r.episode.content,
            "score": r.score,
            "importance": r.importance,
            "timestamp": r.episode.timestamp.isoformat(),
            "actors": r.episode.actors,
            "tags": r.episode.tags,
        }
        for r in results
    ]


def _why(pool: _EngramPool, memory_id: str) -> dict[str, Any]:
    """Explain the provenance of a memory. See ``_WHY_DESCRIPTION``.

    Looks the id up as a fact first (facts are cheap point lookups and are
    not agent-scoped -- shared across agents by design, see DESIGN.md 11),
    then as an episode. The fact lookup always uses the default-agent
    instance's store unscoped: ``get_fact`` never filters by agent, so any
    pooled instance sees the same fact rows. Episodes are private per agent,
    though, so the episode lookup passes ``pool.default_agent_id`` to
    ``get_episode``: this stops a ``why()`` call from reading another
    agent's episode content. When the server has no default agent id
    configured (``pool.default_agent_id is None``), the episode lookup
    stays unscoped too, matching this server's single-tenant default.
    """
    mem = pool.get(pool.default_agent_id)
    store = mem._store  # read-only introspection, same pattern as engram.cli

    fact = store.get_fact(memory_id)
    if fact is not None:
        provenance = mem.why(memory_id)
        return {
            "kind": "semantic",
            "id": fact.id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
            "confidence": fact.confidence,
            "valid_from": fact.valid_from.isoformat(),
            "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
            "recorded_at": fact.recorded_at.isoformat(),
            "extracted_from": provenance["extracted_from"],
            "extracted_by_reflection_run": provenance["extracted_by"],
            "extraction_model": provenance["model"],
        }

    episode = store.get_episode(memory_id, agent_id=pool.default_agent_id)
    if episode is not None:
        access_count, last_accessed = store.get_access_stats(memory_id)
        return {
            "kind": "episodic",
            "id": episode.id,
            "content": episode.content,
            "timestamp": episode.timestamp.isoformat(),
            "actors": episode.actors,
            "tags": episode.tags,
            "salience": episode.salience,
            "emotional_valence": episode.emotional_valence,
            "importance_score": episode.importance_score,
            "summary_of": episode.summary_of,
            "agent_id": episode.agent_id,
            "access_count": access_count,
            "last_accessed": last_accessed.isoformat() if last_accessed else None,
            "note": (
                "Episodic memories are raw observations, not LLM-derived facts, "
                "so they have no extraction chain -- this is encoding and "
                "access metadata instead."
            ),
        }

    raise KeyError(f"memory not found: {memory_id!r}")


def _forget(pool: _EngramPool, memory_id: str) -> dict[str, Any]:
    """Permanently delete a memory by id. See ``_FORGET_DESCRIPTION``."""
    mem = pool.get(pool.default_agent_id)

    try:
        mem.forget(memory_id)
    except KeyError:
        pass
    else:
        return {"id": memory_id, "kind": "episodic", "deleted": True}

    try:
        mem.forget_fact(memory_id)
    except KeyError:
        pass
    else:
        return {"id": memory_id, "kind": "semantic", "deleted": True}

    raise KeyError(f"memory not found: {memory_id!r}")


def _stats(pool: _EngramPool, agent_id: str | None = None) -> dict[str, Any]:
    """Return store statistics. See ``_STATS_DESCRIPTION``."""
    mem = pool.get(agent_id)
    store = mem._store  # same read-only introspection as engram.cli's inspect
    effective_agent = agent_id if agent_id is not None else pool.default_agent_id

    episodes = store.episode_count()
    facts_total = store.fact_count()
    facts_active = store.active_fact_count()

    db_size_bytes: int | None = None
    if pool.db_path != ":memory:":
        p = Path(pool.db_path)
        if p.exists():
            db_size_bytes = p.stat().st_size

    return {
        "agent_id": effective_agent,
        "counts": {
            "episodic": episodes,
            "semantic": facts_active,
            # procedural memory does not exist in this Engram version; see
            # the module docstring and the MCP phase report.
            "procedural": 0,
        },
        "vector_index_size": store.vec_count(),
        "facts_total": facts_total,
        "facts_active": facts_active,
        "facts_superseded": facts_total - facts_active,
        "entities": store.entity_count(),
        "reflections": store.reflection_count(),
        # Which numbers above are about THIS agent and which are about the
        # file. Both kinds are correct and they read identically, which is
        # how a caller passing agent_id came away thinking the whole store
        # was theirs: episodes were scoped and everything beside them was
        # not. The scoped ones are now scoped (see Store.vec_count and
        # Store.reflection_count); facts and entities carry no agent_id and
        # are shared by design (CHANGELOG 2.2.0, 2.2.1), so they are named
        # here rather than quietly left to be misread.
        "scope": {
            "scoped_to_agent": ["episodic", "vector_index_size", "reflections"],
            "shared_across_agents": [
                "semantic",
                "facts_total",
                "facts_active",
                "facts_superseded",
                "entities",
            ],
        },
        "db_path": pool.db_path,
        "db_size_bytes": db_size_bytes,
    }


# ----------------------------------------------------------------------
# Static tool descriptions (fixed string literals -- never interpolated
# from memory content or other request data; see module docstring)
# ----------------------------------------------------------------------

_REMEMBER_DESCRIPTION = (
    "Store a new memory in Engram. kind='episodic' (default) stores "
    "free-form text describing an event or observation: pass content "
    "(stored verbatim) and do NOT pass subject/predicate/object. "
    "kind='semantic' stores a structured fact triple: pass subject, "
    "predicate, and object (all three required) and do NOT pass content. "
    "Providing the wrong parameters for a kind is an error. "
    "kind='procedural' is not supported by this Engram version and raises "
    "an error. agent_id optionally scopes the write to a specific agent; "
    "omit it to use the server's default agent."
)

_RECALL_DESCRIPTION = (
    "Retrieve memories relevant to a natural-language query, ranked by "
    "relevance. mode='hybrid' (default) blends embedding similarity with "
    "keyword (BM25) search, which finds the specific turn holding an answer "
    "more often than either alone; mode='cosine' ranks by embedding "
    "similarity only; mode='spreading' additionally follows graph edges "
    "between related memories (spreading activation) to surface indirectly "
    "relevant context. Returns a list of memories, each with its id, content, "
    "and relevance score, most relevant first."
)

_WHY_DESCRIPTION = (
    "Explain the provenance of a memory by id. For a semantic fact: "
    "returns the full subject/predicate/object triple, its confidence, "
    "validity window, the source material it was extracted from, and the "
    "reflection run and model that extracted it. For an episodic memory: "
    "returns its content, encoding metadata (actors, tags, salience), and "
    "access history -- episodic memories are raw observations rather than "
    "LLM-derived facts, so they have no extraction chain."
)

_FORGET_DESCRIPTION = (
    "Permanently delete a single memory (episodic or semantic) by id. This is irreversible."
)

_STATS_DESCRIPTION = (
    "Return store statistics: memory counts per kind (episodic, semantic, "
    "procedural), fact validity breakdown (active vs superseded), entity "
    "and reflection-run counts, and the database file size in bytes. "
    "The response's 'scope' object says which of those numbers are about "
    "the requested agent (episodic, vector_index_size, reflections) and "
    "which are about the whole file, shared across every agent in it "
    "(semantic, facts_total, facts_active, facts_superseded, entities). "
    "Do not report a shared number as if it belonged to one agent."
)


# ----------------------------------------------------------------------
# FastMCP wiring
# ----------------------------------------------------------------------


def _build_server(pool: _EngramPool) -> Any:
    """Create and return a FastMCP server with Engram's tools registered.

    Separated from :func:`main` so the server object can be built and
    introspected in tests without going through stdio. The ``mcp`` SDK is
    imported here, not at module scope, so this is the only code path in
    the module that requires it to be installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError("The MCP SDK is not installed. Run: pip install 'engdbram[mcp]'") from exc

    mcp: FastMCP[None] = FastMCP(
        "engram",
        instructions=(
            "Persistent cognitive memory for AI agents. Use 'remember' to store "
            "episodic events or semantic facts, 'recall' to retrieve relevant "
            "memories, 'why' to inspect a memory's provenance, 'forget' to "
            "erase one, and 'stats' for store-wide counts."
        ),
    )

    @mcp.tool(description=_REMEMBER_DESCRIPTION)
    def remember(
        content: str | None = None,
        kind: str = "episodic",
        agent_id: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,  # noqa: A002
    ) -> dict[str, Any]:
        return _remember(pool, content, kind, agent_id, subject, predicate, object)

    @mcp.tool(description=_RECALL_DESCRIPTION)
    def recall(
        query: str,
        limit: int = 5,
        agent_id: str | None = None,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        return _recall(pool, query, limit, agent_id, mode)

    @mcp.tool(description=_WHY_DESCRIPTION)
    def why(memory_id: str) -> dict[str, Any]:
        return _why(pool, memory_id)

    @mcp.tool(description=_FORGET_DESCRIPTION)
    def forget(memory_id: str) -> dict[str, Any]:
        return _forget(pool, memory_id)

    @mcp.tool(description=_STATS_DESCRIPTION)
    def stats(agent_id: str | None = None) -> dict[str, Any]:
        return _stats(pool, agent_id)

    return mcp


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engram-mcp",
        description="Run Engram as an MCP server over stdio.",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("ENGRAM_MCP_DB"),
        metavar="PATH",
        help="path to the Engram database file (env: ENGRAM_MCP_DB; default: in-memory)",
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("ENGRAM_MCP_AGENT_ID"),
        metavar="AGENT_ID",
        dest="agent_id",
        help=(
            "default agent scope for this server. Any string is accepted and "
            "used as an opaque scoping key, but with --events on, an id that "
            "is not an 'agent://<trust-domain>/<name>' URI produces event "
            "lines the shared envelope's own schema rejects, and is warned "
            "about once (env: ENGRAM_MCP_AGENT_ID)"
        ),
    )
    parser.add_argument(
        "--events",
        default=os.environ.get("ENGRAM_MCP_EVENTS"),
        metavar="PATH",
        dest="events",
        help=(
            "opt-in path to an Agent Passport NDJSON event log (see "
            "engram.events); omit to disable event emission entirely "
            "(env: ENGRAM_MCP_EVENTS)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point registered as the ``engram-mcp`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    db_path = args.db or ":memory:"
    if not args.db:
        # stdout is the stdio transport channel -- diagnostics must go to stderr.
        print(
            "warning: no --db/ENGRAM_MCP_DB given; using an in-memory store "
            "that is discarded when this process exits.",
            file=sys.stderr,
        )

    pool = _EngramPool(db_path, args.agent_id, events_path=args.events)
    try:
        server = _build_server(pool)
        server.run(transport="stdio")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
