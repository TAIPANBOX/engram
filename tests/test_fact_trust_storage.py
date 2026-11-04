"""Storage for fact trust: the migration, the export and the docs-versus-code check.

The behaviour these tests hold is written as scenarios in
``features/human-priority-and-refutation.feature``, one test per scenario, and
``scripts/features-are-bound.sh`` keeps the two in step. Each test was run
against the unchanged code first and failed there.

Three nullable columns arrive on ``facts``: ``provenance``, ``refuted_by`` and
``undermined_by``. NULL is what a row written before them holds, and it reads
as no recorded source, not refuted and not marked, so a store from before this
change opens with every row as it was. The step is stamped in
``PRAGMA user_version`` (generation 2) and is safe to run again.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from engram import Engram
from engram.schema import _SCHEMA_VERSION

_NEW_COLUMNS = ("provenance", "refuted_by", "undermined_by")
_REPO = Path(__file__).resolve().parents[1]


def _load_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _strip_to_generation_1(path: str) -> None:
    """Return a store to how a 3.0.0 file looked: no trust columns, marker at 1.

    The current code cannot write such a file, so this is the only way to open
    the migration over rows that were already there.
    """
    if sqlite3.sqlite_version_info < (3, 35):
        pytest.skip("ALTER TABLE DROP COLUMN needs SQLite 3.35")
    conn = sqlite3.connect(path)
    _load_vec(conn)
    for column in _NEW_COLUMNS:
        conn.execute(f"ALTER TABLE facts DROP COLUMN {column}")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()


def _fact_columns(path: str) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()]
    finally:
        conn.close()


def _user_version(path: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def test_a_store_from_before_the_trust_columns_opens_with_every_row_intact(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "old.engram")
    with Engram(path=path) as mem:
        globex = mem.assert_fact("Ivan", "works_at", "Globex", confidence=0.8, source="HR export")
        initech = mem.assert_fact("Ivan", "works_at", "Initech")
    _strip_to_generation_1(path)
    assert not set(_NEW_COLUMNS) & set(_fact_columns(path))

    with Engram(path=path) as mem:
        facts = {f.id: f for f in mem.timeline("Ivan")}
        assert set(facts) == {globex, initech}
        assert facts[globex].object == "Globex"
        assert facts[globex].confidence == pytest.approx(0.8)
        assert facts[globex].derived_from == ["HR export"]
        for fact in facts.values():
            assert fact.provenance is None
            assert fact.refuted_by is None
            assert fact.undermined_by == []
        assert mem.why(globex)["provenance"] == "unknown"
        assert mem.why(globex)["refuted_by"] is None

    assert set(_NEW_COLUMNS) <= set(_fact_columns(path))
    assert _user_version(path) == _SCHEMA_VERSION == 2


def test_the_trust_columns_step_is_gated_on_the_schema_marker(tmp_path: Path) -> None:
    path = str(tmp_path / "gated.engram")
    with Engram(path=path) as mem:
        fact_id = mem.assert_fact("Ivan", "works_at", "Globex", provenance="user")
        mem.refute_fact(fact_id, provenance="user")

    for _ in range(3):
        with Engram(path=path) as mem:
            fact = mem._store.get_fact(fact_id)
            assert fact is not None
            assert (fact.provenance, fact.refuted_by) == ("user", "user")
    columns = _fact_columns(path)
    assert [columns.count(c) for c in _NEW_COLUMNS] == [1, 1, 1]
    assert _user_version(path) == _SCHEMA_VERSION

    # Both directions, as the FTS marker's test does: a marker that is never
    # read repeats the work, and one that is never written repeats it forever.
    _strip_to_generation_1(path)
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    conn.commit()
    conn.close()
    Engram(path=path).close()
    assert not set(_NEW_COLUMNS) & set(_fact_columns(path)), (
        "the marker said the step was done and it ran anyway"
    )

    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION - 1}")
    conn.commit()
    conn.close()
    Engram(path=path).close()
    assert set(_NEW_COLUMNS) <= set(_fact_columns(path)), (
        "the marker was behind and the step did not run"
    )
    assert _user_version(path) == _SCHEMA_VERSION


def _fact_trust(mem: Engram) -> dict[str, tuple[Any, Any, list[str]]]:
    return {
        f.object: (f.provenance, f.refuted_by, f.undermined_by)
        for f in mem._store.get_all_facts("Ivan")
    }


def test_export_then_import_keeps_provenance_refutation_and_marks(tmp_path: Path) -> None:
    dump = str(tmp_path / "dump.json")
    with Engram(path=":memory:") as src:
        person = src.assert_fact("Ivan", "works_at", "Globex", provenance="user")
        model = src.assert_fact("Ivan", "works_at", "Initech", provenance="tool")
        src.assert_fact("Ivan", "lives_near", "Globex campus", derived_from=[model])
        src.refute_fact(model, provenance="user")
        before = _fact_trust(src)
        src.export_json(dump)

    assert before["Globex"] == ("user", None, [])
    assert before["Initech"] == ("tool", "user", [])
    assert before["Globex campus"] == ("unknown", None, [model])

    with Engram(path=":memory:") as dst:
        dst.import_json(dump)
        assert _fact_trust(dst) == before
        held = dst._store.get_fact(person)
        assert held is not None and held.provenance == "user"


def test_an_export_from_before_these_fields_imports_and_refuses_a_bad_source(
    tmp_path: Path,
) -> None:
    dump = tmp_path / "dump.json"
    with Engram(path=":memory:") as src:
        src.assert_fact("Ivan", "works_at", "Globex", provenance="user")
        src.export_json(str(dump))

    doc: dict[str, Any] = json.loads(dump.read_text())
    for fact in doc["facts"]:
        for key in _NEW_COLUMNS:
            del fact[key]
    old = tmp_path / "old.json"
    old.write_text(json.dumps(doc))
    with Engram(path=":memory:") as dst:
        assert dst.import_json(str(old))["facts"] == 1
        assert _fact_trust(dst) == {"Globex": (None, None, [])}

    for key in ("provenance", "refuted_by"):
        bad = json.loads(dump.read_text())
        bad["facts"][0][key] = "admin"
        bad_path = tmp_path / f"bad-{key}.json"
        bad_path.write_text(json.dumps(bad))
        with Engram(path=":memory:") as dst, pytest.raises(ValueError, match=key):
            dst.import_json(str(bad_path))
        with Engram(path=":memory:") as dst:
            assert dst._store.fact_count() == 0


def test_the_procedural_refusal_points_at_a_document_that_exists() -> None:
    from engram.mcp_server import _EngramPool, _remember

    pool = _EngramPool(":memory:", None)
    try:
        with pytest.raises(ValueError, match="procedural") as refused:
            _remember(pool, "always check the logs first", kind="procedural")
    finally:
        pool.close()

    message = str(refused.value)
    assert "phase report" not in message
    assert "docs/api-reference.md" in message
    assert (_REPO / "docs" / "api-reference.md").is_file()
