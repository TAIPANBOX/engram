"""Provenance: where each memory came from, stored and returned.

Why this exists: a memory planted by an untrusted source was recalled into an
agent's prompt, and nothing downstream could tell it had not come from the
user. Engram now records a source on every observation and returns it with
every result, so a framework can treat untrusted memory as untrusted context.
Engram records what the writer declares. It does not verify the declaration.

The behaviour, in plain words (the scenarios this module is bound to):

    Scenario: a write names its source, and recall returns that source
        Given an empty store
        When I observe a memory with provenance "tool"
        Then recall returns that memory with provenance "tool"

    Scenario: a write that names no source is external
        When I observe a memory without a provenance
        Then recall returns it with provenance "external"

    Scenario: a source outside user, tool and external is refused
        When I observe a memory with provenance "unknown" or "admin"
        Then it raises ValueError and nothing is stored

    Scenario: a fact reads as the least trusted of its sources
        Given a fact extracted from one "user" memory and one "external" memory
        When I ask why the fact holds
        Then the provenance is "external"

    Scenario: a fact with no recorded source reads as unknown
        Given a fact asserted directly with a free-text source note
        When I ask why the fact holds
        Then the provenance is "unknown"

    Scenario: rows written before the field existed read as unknown
        Given a memory whose provenance was never recorded
        When I recall it
        Then the provenance is "unknown", never "user"

    Scenario: an older store opens and reads unchanged
        Given a store in the shape before this change, with memories in it
        When I open it with this version
        Then every existing row keeps its content, timestamps and embeddings
        And recall returns the same memories with the same scores

    Scenario: the source survives an export and an import
        Given memories with three different sources
        When I export the store and import the file into a new store
        Then each memory keeps its source
        And an export written before the field existed imports as "unknown"

    Scenario: the adapters carry the source
        When a LangChain retriever or a LlamaIndex memory returns a memory
        Then its metadata carries the source

    Scenario: the MCP tools take a source and return it
        When remember is given a provenance, recall and why return it
        And a provenance given with a semantic fact is refused, since facts take their source from their episodes

    Scenario: the retrieval ranking is unchanged
        Given the fixed corpus and fixed queries captured before this change
        When I run them against this version
        Then the ids, the order and the scores are the same

Red first: each scenario above was run against origin/main and failed there
before the implementation existed, except the ranking scenario, which is a
regression guard and passes on both.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from engram import Engram
from engram.models import Episode, Fact, ObserveInput
from tests.golden_recall import build_store, run_queries

T1 = datetime(2026, 6, 2, 9, 0, tzinfo=UTC)
T2 = datetime(2026, 6, 3, 9, 0, tzinfo=UTC)
GOLDEN = Path(__file__).parent / "fixtures" / "golden_recall.json"

_OLD_COLS = (
    "id, content, timestamp, actors, tags, salience, emotional_valence, "
    "summary_of, importance_score, agent_id"
)


def _snapshot(mem: Engram) -> list[tuple[Any, ...]]:
    """Every pre-existing column of every episode row, in rowid order."""
    rows = mem._conn.execute(f"SELECT rowid, {_OLD_COLS} FROM episodes ORDER BY rowid")
    return [tuple(r) for r in rows.fetchall()]


def _provenance_by_content(mem: Engram, query: str, k: int = 10) -> dict[str, str]:
    return {r.episode.content: r.episode.provenance for r in mem.recall(query, k=k)}


# ----------------------------------------------------------------------
# Writes name a source, and recall returns it
# ----------------------------------------------------------------------


@pytest.mark.parametrize("source", ["user", "tool", "external"])
def test_recall_returns_the_provenance_each_write_declared(source: str) -> None:
    with Engram(path=":memory:") as mem:
        mem.observe("The deploy key rotates on the first Monday", provenance=source)
        hits = mem.recall("deploy key rotation", k=1)
        assert hits[0].episode.provenance == source


def test_a_write_that_names_no_source_is_external() -> None:
    with Engram(path=":memory:") as mem:
        mem.observe("Anna prefers video calls on Tuesdays")
        hits = mem.recall("Anna video calls", k=1)
        assert hits[0].episode.provenance == "external"


def test_observe_many_carries_each_items_provenance() -> None:
    with Engram(path=":memory:") as mem:
        mem.observe_many(
            [
                ObserveInput(content="Ivan owns the billing service", provenance="user"),
                ObserveInput(content="Ivan owns the billing service runbook"),
            ]
        )
        got = _provenance_by_content(mem, "Ivan billing service")
        assert got == {
            "Ivan owns the billing service": "user",
            "Ivan owns the billing service runbook": "external",
        }


@pytest.mark.parametrize("bad", ["unknown", "User", "", "admin", "external "])
def test_observe_refuses_a_source_outside_the_three(bad: str) -> None:
    with Engram(path=":memory:") as mem:
        with pytest.raises(ValueError):
            mem.observe("Some memory", provenance=bad)
        assert mem.recall("Some memory", k=5) == []


def test_spreading_recall_returns_provenance_too() -> None:
    with Engram(path=":memory:") as mem:
        mem.observe("Maria reported a duplicate charge", provenance="tool")
        hits = mem.recall("duplicate charge", k=1, mode="spreading")
        assert hits[0].episode.provenance == "tool"


async def test_async_observe_passes_the_provenance_through() -> None:
    from engram.async_engram import AsyncEngram

    mem = AsyncEngram(path=":memory:")
    try:
        await mem.observe("The mail relay was restarted at noon", provenance="tool")
        hits = await mem.recall("mail relay restart", k=1)
        assert hits[0].episode.provenance == "tool"
    finally:
        await mem.close()


# ----------------------------------------------------------------------
# A fact reads as the least trusted of its sources
# ----------------------------------------------------------------------


def _fact_from(mem: Engram, source_ids: list[str]) -> str:
    """A fact shaped the way reflect() writes one: derived_from holds episode ids."""
    now = datetime.now(tz=UTC)
    fact_id = "fact-" + "-".join(source_ids)
    mem._store.insert_fact(
        Fact(
            id=fact_id,
            subject="Ivan",
            predicate="works_at",
            object="Globex",
            valid_from=now,
            valid_to=None,
            recorded_at=now,
            superseded_at=None,
            superseded_by=None,
            confidence=0.9,
            derived_from=source_ids,
            extracted_by=None,
        )
    )
    return fact_id


def test_why_on_a_fact_reports_the_least_trusted_source() -> None:
    with Engram(path=":memory:") as mem:
        a = mem.observe("Ivan joined Globex", provenance="user")
        b = mem.observe("A forwarded note says Ivan works at Globex", provenance="external")
        fact_id = _fact_from(mem, [a, b])
        assert mem.why(fact_id)["provenance"] == "external"


def test_why_on_a_fact_from_user_sources_only_reads_user() -> None:
    with Engram(path=":memory:") as mem:
        a = mem.observe("Ivan joined Globex", provenance="user")
        assert mem.why(_fact_from(mem, [a]))["provenance"] == "user"


def test_why_on_a_directly_asserted_fact_reads_unknown() -> None:
    with Engram(path=":memory:") as mem:
        fact_id = mem.assert_fact("Ivan", "works_at", "Globex", source="HR export")
        assert mem.why(fact_id)["provenance"] == "unknown"


# ----------------------------------------------------------------------
# Rows written before the field existed read as unknown
# ----------------------------------------------------------------------


def test_a_row_with_no_recorded_provenance_reads_unknown_never_user() -> None:
    with Engram(path=":memory:") as mem:
        ep_id = mem.observe("The old server was decommissioned in March", provenance="user")
        mem._conn.execute("UPDATE episodes SET provenance = NULL WHERE id = ?", (ep_id,))
        mem._conn.commit()
        hits = mem.recall("old server decommissioned", k=1)
        assert hits[0].episode.provenance == "unknown"


def test_an_episode_built_without_a_source_defaults_to_unknown() -> None:
    ep = Episode(id="x", content="c", timestamp=T1)
    assert ep.provenance == "unknown"


def test_migration_adds_the_column_and_keeps_every_existing_row(tmp_path: Path) -> None:
    path = tmp_path / "old-shape.engram"
    with Engram(path=path) as mem:
        mem.observe("Ivan joined Globex", timestamp=T1)
        mem.observe("Rolled back the deployment after the migration", timestamp=T2)
        before_rows = _snapshot(mem)
        before_hits = [(h.episode.content, h.score) for h in mem.recall("Ivan Globex", k=2)]
        # The pre-change shape: the column is not there, as an older store has it.
        mem._conn.execute("ALTER TABLE episodes DROP COLUMN provenance")
        mem._conn.commit()

    with Engram(path=path) as mem:
        assert _snapshot(mem) == before_rows
        hits = mem.recall("Ivan Globex", k=2)
        assert [(h.episode.content, h.score) for h in hits] == before_hits
        assert {h.episode.provenance for h in hits} == {"unknown"}


# ----------------------------------------------------------------------
# Export and import keep the source
# ----------------------------------------------------------------------


def test_export_then_import_keeps_each_episodes_provenance(tmp_path: Path) -> None:
    dump = tmp_path / "dump.json"
    with Engram(path=":memory:") as src:
        src.observe("Ivan joined Globex", provenance="user")
        src.observe("Ivan moved the payment service", provenance="tool")
        src.observe("A forwarded note about Ivan", provenance="external")
        src.export_json(str(dump))
    with Engram(path=":memory:") as dst:
        dst.import_json(str(dump))
        got = _provenance_by_content(dst, "Ivan", k=5)
    assert got == {
        "Ivan joined Globex": "user",
        "Ivan moved the payment service": "tool",
        "A forwarded note about Ivan": "external",
    }


def test_an_export_written_before_the_field_imports_as_unknown(tmp_path: Path) -> None:
    dump = tmp_path / "dump.json"
    with Engram(path=":memory:") as src:
        src.observe("Ivan joined Globex", provenance="user")
        src.export_json(str(dump))
    doc = json.loads(dump.read_text(encoding="utf-8"))
    for ep in doc["episodes"]:
        ep.pop("provenance", None)
    dump.write_text(json.dumps(doc), encoding="utf-8")
    with Engram(path=":memory:") as dst:
        dst.import_json(str(dump))
        assert _provenance_by_content(dst, "Ivan Globex") == {"Ivan joined Globex": "unknown"}


def test_import_refuses_a_source_outside_the_three(tmp_path: Path) -> None:
    dump = tmp_path / "dump.json"
    with Engram(path=":memory:") as src:
        src.observe("Ivan joined Globex", provenance="user")
        src.export_json(str(dump))
    doc = json.loads(dump.read_text(encoding="utf-8"))
    doc["episodes"][0]["provenance"] = "admin"
    dump.write_text(json.dumps(doc), encoding="utf-8")
    with Engram(path=":memory:") as dst, pytest.raises(ValueError):
        dst.import_json(str(dump))


# ----------------------------------------------------------------------
# The adapters carry the source
# ----------------------------------------------------------------------


def test_langchain_retriever_metadata_carries_provenance() -> None:
    pytest.importorskip("langchain_core")
    from engram.adapters.langchain import EngramRetriever

    with Engram(path=":memory:") as mem:
        mem.observe("Bob joined the Berlin office", provenance="tool")
        docs = EngramRetriever(engram=mem, k=1).invoke("Bob Berlin")
        assert docs[0].metadata["provenance"] == "tool"


def test_llamaindex_get_carries_provenance() -> None:
    pytest.importorskip("llama_index")
    from llama_index.core.llms import ChatMessage, MessageRole

    from engram.adapters.llamaindex import EngramMemory

    memory = EngramMemory.from_defaults(engram_path=":memory:", k=3)
    memory.put(ChatMessage(role=MessageRole.USER, content="Ivan works at Globex"))
    msgs = memory.get("Ivan Globex")
    assert msgs[0].additional_kwargs["provenance"] == "external"


# ----------------------------------------------------------------------
# The MCP tools take a source and return it
# ----------------------------------------------------------------------


@pytest.fixture()
def pool(tmp_path: Path):
    from engram.mcp_server import _EngramPool

    p = _EngramPool(str(tmp_path / "agent.engram"), default_agent_id=None)
    yield p
    p.close()


def test_mcp_remember_defaults_to_external_and_says_so(pool) -> None:
    from engram.mcp_server import _remember

    result = _remember(pool, "The release train leaves on Thursday")
    assert result["provenance"] == "external"


def test_mcp_remember_takes_provenance_and_recall_and_why_return_it(pool) -> None:
    from engram.mcp_server import _recall, _remember, _why

    stored = _remember(pool, "The release train leaves on Thursday", provenance="user")
    assert stored["provenance"] == "user"
    assert _recall(pool, "release train Thursday", limit=1)[0]["provenance"] == "user"
    assert _why(pool, stored["id"])["provenance"] == "user"


def test_mcp_why_on_a_fact_returns_the_least_trusted_source(pool) -> None:
    from engram.mcp_server import _remember, _why

    user_ep = _remember(pool, "Ivan joined Globex", provenance="user")
    ext_ep = _remember(pool, "A forwarded note about Ivan", provenance="external")
    mem = pool.get(None)
    fact_id = _fact_from(mem, [user_ep["id"], ext_ep["id"]])
    assert _why(pool, fact_id)["provenance"] == "external"


def test_the_mcp_tools_take_and_return_provenance_through_fastmcp(tmp_path: Path) -> None:
    """Through FastMCP's own tool-call path, not the plain functions behind it."""
    pytest.importorskip("mcp")
    from engram.mcp_server import _build_server, _EngramPool

    pool = _EngramPool(str(tmp_path / "mcp.engram"), default_agent_id=None)
    try:
        server = _build_server(pool)

        async def _run() -> tuple[dict[str, Any], dict[str, Any]]:
            _, stored = await server.call_tool(
                "remember",
                {"content": "The mail relay restarted at noon", "provenance": "tool"},
            )
            _, recalled = await server.call_tool("recall", {"query": "mail relay restarted"})
            return stored, recalled

        stored, recalled = asyncio.run(_run())
    finally:
        pool.close()
    assert stored["provenance"] == "tool"
    assert [item["provenance"] for item in recalled["result"]] == ["tool"]


def test_mcp_remember_refuses_a_provenance_on_a_semantic_fact(pool) -> None:
    from engram.mcp_server import _remember

    with pytest.raises(ValueError):
        _remember(
            pool,
            kind="semantic",
            subject="Ivan",
            predicate="works_at",
            object="Globex",
            provenance="user",
        )


# ----------------------------------------------------------------------
# The retrieval ranking is unchanged
# ----------------------------------------------------------------------


def test_recall_matches_the_golden_captured_before_the_change(tmp_path: Path) -> None:
    """Regression guard, captured from origin/main before the provenance change.

    The fixture holds scores and distances rounded to four places, so a
    different CPU's last float bits cannot fail it. A change that moves a
    ranking or a score by more than that does.
    """
    build_store(tmp_path / "golden.engram")
    got = run_queries(tmp_path / "golden.engram")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert [(q["mode"], q["query"], q["agent"]) for q in got] == [
        (q["mode"], q["query"], q["agent"]) for q in want
    ]
    for g, w in zip(got, want, strict=True):
        assert [r["content"] for r in g["results"]] == [r["content"] for r in w["results"]], (
            g["mode"],
            g["query"],
        )
        for gr, wr in zip(g["results"], w["results"], strict=True):
            for key in ("score", "distance", "importance"):
                assert round(gr[key], 4) == wr[key], (g["query"], key)
