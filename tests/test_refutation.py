"""Refuting a fact: it closes, what was derived from it is marked, nothing is erased.

The behaviour these tests hold is written as scenarios in
``features/human-priority-and-refutation.feature``, one test per scenario, and
``scripts/features-are-bound.sh`` keeps the two in step. Each test was run
against the unchanged code first and failed there.

A refuted fact keeps its row. It gains a ``superseded_at`` (and a ``valid_to``
equal to it), so ``timeline()`` and ``as_of`` still tell what was believed
before, and it records who refuted it. A fact derived from it is not closed:
it may stand on other grounds, so it is marked with the id of the refuted fact
and left for its owner to judge. The mark travels through derivations, so a
fact derived from a marked fact carries it too.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engram import AsyncEngram, Engram, Fact

A = "agent://acme.example/a"
B = "agent://acme.example/b"


def _fact(mem: Engram, obj: str = "Globex", **kwargs: object) -> str:
    return mem.assert_fact("Ivan", "works_at", obj, **kwargs)  # type: ignore[arg-type]


def _get(mem: Engram, fact_id: str) -> Fact:
    fact = mem._store.get_fact(fact_id)
    assert fact is not None
    return fact


def test_refuting_a_fact_closes_it_and_keeps_it_in_the_history() -> None:
    with Engram(path=":memory:") as mem:
        fact_id = _fact(mem)
        believed_from = _get(mem, fact_id).valid_from
        time.sleep(0.01)

        result = mem.refute_fact(fact_id)

        assert result.fact_id == fact_id
        assert mem._store.get_active_facts("Ivan", "works_at") == []
        history = mem.timeline("Ivan")
        assert [f.id for f in history] == [fact_id]
        fact = history[0]
        assert fact.superseded_at is not None
        assert fact.valid_to == fact.superseded_at
        assert fact.superseded_by is None
        assert fact.refuted_by == "external"
        assert [f.id for f in mem.timeline("Ivan", as_of=believed_from)] == [fact_id]
        assert mem.timeline("Ivan", as_of=fact.superseded_at) == []
        assert mem.why(fact_id)["refuted_by"] == "external"


def test_refuting_marks_every_fact_derived_from_it_and_closes_none_of_them() -> None:
    with Engram(path=":memory:") as mem:
        first = _fact(mem)
        second = mem.assert_fact("Ivan", "lives_near", "Globex campus", derived_from=[first])
        third = mem.assert_fact("Ivan", "commutes_by", "tram", derived_from=[second])
        unrelated = mem.assert_fact("Maria", "role", "CTO")

        result = mem.refute_fact(first)

        assert set(result.undermined) == {second, third}
        for fact_id in (second, third):
            fact = _get(mem, fact_id)
            assert fact.undermined_by == [first]
            assert fact.valid_to is None and fact.superseded_at is None
            assert mem.why(fact_id)["undermined_by"] == [first]
        assert _get(mem, unrelated).undermined_by == []
        assert mem._store.fact_count() == 4


def test_a_fact_derived_from_a_refuted_fact_is_born_marked() -> None:
    with Engram(path=":memory:") as mem:
        first = _fact(mem)
        mem.refute_fact(first)

        second = mem.assert_fact("Ivan", "lives_near", "Globex campus", derived_from=[first])
        third = mem.assert_fact("Ivan", "commutes_by", "tram", derived_from=[second])

        assert _get(mem, second).undermined_by == [first]
        assert _get(mem, third).undermined_by == [first]


def test_refuting_twice_changes_nothing() -> None:
    with Engram(path=":memory:") as mem:
        first = _fact(mem)
        second = mem.assert_fact("Ivan", "lives_near", "Globex campus", derived_from=[first])
        mem.refute_fact(first)
        closed_at = _get(mem, first).superseded_at

        # A second refutation by someone else changes nothing either, who made
        # the first included.
        again = mem.refute_fact(first, provenance="tool")

        assert again.undermined == []
        assert _get(mem, first).superseded_at == closed_at
        assert _get(mem, first).refuted_by == "external"
        assert _get(mem, second).undermined_by == [first]


def test_a_models_refutation_of_a_persons_fact_is_refused() -> None:
    with Engram(path=":memory:") as mem:
        person = _fact(mem, provenance="user")
        child = mem.assert_fact("Ivan", "lives_near", "Globex campus", derived_from=[person])

        for attempt in (
            {},
            {"provenance": "tool"},
            {"provenance": "external"},
            {"provenance": "user", "quarantined": True},
        ):
            with pytest.raises(PermissionError, match="person"):
                mem.refute_fact(person, **attempt)  # type: ignore[arg-type]

        held = _get(mem, person)
        assert held.valid_to is None and held.refuted_by is None
        assert _get(mem, child).undermined_by == []

        mem.refute_fact(person, provenance="user")
        assert _get(mem, person).refuted_by == "user"
        assert _get(mem, child).undermined_by == [person]

    # What a model wrote, a model may refute.
    with Engram(path=":memory:") as mem:
        model = _fact(mem, "Initech")
        mem.refute_fact(model, provenance="tool")
        assert _get(mem, model).refuted_by == "tool"


def test_an_agent_cannot_refute_another_agents_fact(tmp_path: Path) -> None:
    path = str(tmp_path / "shared.engram")
    with Engram(path=path, agent_id=A) as mem:
        shared = _fact(mem, shared=True)
        private = _fact(mem, "Initech", shared=False)

    with Engram(path=path, agent_id=B) as mem:
        with pytest.raises(PermissionError):
            mem.refute_fact(shared)
        with pytest.raises(KeyError):
            mem.refute_fact(private)

    with Engram(path=path) as operator:
        operator.refute_fact(shared)
        operator.refute_fact(private)
        assert _get(operator, shared).refuted_by == "external"
        assert _get(operator, private).refuted_by == "external"


def test_the_cascade_reaches_a_private_fact_without_naming_it(tmp_path: Path) -> None:
    path = str(tmp_path / "shared.engram")
    with Engram(path=path, agent_id=A) as mem:
        premise = _fact(mem, shared=True)

    with Engram(path=path, agent_id=B) as mem:
        private = mem.assert_fact(
            "Ivan", "lives_near", "Globex campus", shared=False, derived_from=[premise]
        )

    with Engram(path=path, agent_id=A) as mem:
        result = mem.refute_fact(premise)
        assert private not in result.undermined
        assert result.undermined == []

    with Engram(path=path, agent_id=B) as mem:
        assert _get(mem, private).undermined_by == [premise]


def test_refuting_a_missing_fact_raises() -> None:
    with Engram(path=":memory:") as mem, pytest.raises(KeyError):
        mem.refute_fact("no-such-fact")


def test_a_refuted_fact_is_no_longer_a_contradiction() -> None:
    with Engram(path=":memory:") as mem:
        _fact(mem, "Globex")
        other = _fact(mem, "Initech")
        assert len(mem.contradictions()) == 1

        mem.refute_fact(other)

        assert mem.contradictions() == []


def test_asserting_a_fact_derived_from_an_unknown_fact_is_refused() -> None:
    with Engram(path=":memory:") as mem:
        with pytest.raises(KeyError):
            mem.assert_fact("Ivan", "works_at", "Globex", derived_from=["no-such-fact"])
        assert mem._store.fact_count() == 0


async def test_the_async_wrapper_refutes_too() -> None:
    mem = AsyncEngram(path=":memory:")
    try:
        first = await mem.assert_fact("Ivan", "works_at", "Globex")
        second = await mem.assert_fact("Ivan", "lives_near", "Globex campus", derived_from=[first])

        result = await mem.refute_fact(first)

        assert result.undermined == [second]
        facts = {f.id: f for f in await mem.timeline("Ivan")}
        assert facts[first].refuted_by == "external"
        assert facts[second].undermined_by == [first]
    finally:
        await mem.close()


def test_the_memory_server_why_tool_returns_the_refutation_and_the_marks(tmp_path: Path) -> None:
    from engram.mcp_server import _EngramPool, _remember, _why

    pool = _EngramPool(str(tmp_path / "server.engram"), A)
    try:
        premise = str(
            _remember(pool, kind="semantic", subject="Ivan", predicate="p", object="o")["id"]
        )
        mem = pool.get()
        child = mem.assert_fact("Ivan", "q", "r", derived_from=[premise])
        mem.refute_fact(premise)

        refuted = _why(pool, premise)
        marked = _why(pool, child)
    finally:
        pool.close()

    assert refuted["refuted_by"] == "external" and refuted["undermined_by"] == []
    assert marked["refuted_by"] is None and marked["undermined_by"] == [premise]


def test_the_cli_timeline_shows_a_refuted_fact_and_a_marked_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from engram.cli import main

    path = str(tmp_path / "cli.engram")
    with Engram(path=path) as mem:
        premise = mem.assert_fact("Ivan", "works_at", "Globex")
        mem.assert_fact("Ivan", "commutes_to", "campus", derived_from=[premise])
        mem.assert_fact("Ivan", "lives_in", "Berlin")
        replaced = mem.assert_fact("Ivan", "drives", "a tram")
        newer = mem.assert_fact("Ivan", "drives", "a bus")
        mem._store.close_fact(replaced, valid_to=datetime.now(tz=UTC), superseded_by=newer)
        mem.refute_fact(premise, provenance="tool")

    main(["timeline", path, "Ivan"])
    lines = {
        line.split("Ivan ")[1].split("  (conf")[0]: line
        for line in capsys.readouterr().out.splitlines()
        if "Ivan " in line
    }

    assert "refuted by tool" in lines["works_at Globex"]
    assert "rests on 1 refuted fact" in lines["commutes_to campus"]
    assert "refuted" not in lines["lives_in Berlin"] and "rests on" not in lines["lives_in Berlin"]
    # A fact that was only replaced still reads as superseded, not as refuted.
    assert "superseded" in lines["drives a tram"] and "refuted" not in lines["drives a tram"]


def test_a_cycle_or_a_diamond_in_derived_from_is_walked_once() -> None:
    mem = Engram(path=":memory:")
    root = _fact(mem)
    b = mem.assert_fact("Ivan", "p1", "x", derived_from=[root])
    c = mem.assert_fact("Ivan", "p2", "x", derived_from=[b])
    e = mem.assert_fact("Ivan", "p3", "x", derived_from=[root])
    f = mem.assert_fact("Ivan", "p4", "x", derived_from=[b, e])
    # A hand-edited file: b now also claims to derive from c, closing a loop
    # between two facts that both sit below the refuted one.
    mem._conn.execute("UPDATE facts SET derived_from = ? WHERE id = ?", (json.dumps([root, c]), b))
    mem._conn.commit()

    outcome: dict[str, object] = {}
    walker = threading.Thread(
        target=lambda: outcome.update(result=mem.refute_fact(root)), daemon=True
    )
    walker.start()
    walker.join(timeout=10)

    # Not closed on purpose when the walk never ends: the walker would hold the
    # store's lock and close() would wait on it for ever.
    assert not walker.is_alive(), "the walk over derived_from never ended"
    result = outcome["result"]
    assert set(result.undermined) == {b, c, e, f}  # type: ignore[attr-defined]
    for fact_id in (b, c, e, f):
        assert _get(mem, fact_id).undermined_by == [root]
    mem.close()
