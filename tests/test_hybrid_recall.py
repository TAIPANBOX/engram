"""Tests for hybrid (BM25 + cosine) recall mode."""

from __future__ import annotations

import pytest

from engram import Engram, ObserveInput


@pytest.fixture()
def mem(tmp_path):
    path = str(tmp_path / "hybrid.engram")
    with Engram(path=path) as m:
        m.observe_many(
            [
                ObserveInput(content="Alice joined Globex as Chief Technology Officer"),
                ObserveInput(content="Q3 budget was approved at five hundred thousand dollars"),
                ObserveInput(content="Ivan transferred from Acme Corporation to Globex"),
                ObserveInput(content="The quarterly board meeting discussed merger strategy"),
                ObserveInput(content="Bob presented the annual revenue forecast to stakeholders"),
            ]
        )
        yield m


def test_hybrid_returns_results(mem: Engram) -> None:
    results = mem.recall("Alice technology officer", k=3, mode="hybrid")
    assert len(results) > 0


def test_hybrid_scores_in_range(mem: Engram) -> None:
    results = mem.recall("Globex company transfer", k=5, mode="hybrid")
    for r in results:
        assert 0.0 <= r.score <= 1.0


def test_hybrid_respects_k(mem: Engram) -> None:
    results = mem.recall("quarterly budget meeting", k=2, mode="hybrid")
    assert len(results) <= 2


def test_hybrid_top_result_relevant(mem: Engram) -> None:
    results = mem.recall("Alice CTO role technology", k=3, mode="hybrid")
    contents = [r.episode.content for r in results]
    assert any("Alice" in c for c in contents)


def test_hybrid_fts_exact_match_boosted(mem: Engram) -> None:
    # "Acme" appears in exactly one episode — should surface it
    results = mem.recall("Acme Corporation Ivan", k=3, mode="hybrid")
    assert len(results) > 0
    top_content = results[0].episode.content
    assert "Ivan" in top_content or "Acme" in top_content


def test_hybrid_custom_weights(mem: Engram) -> None:
    # fts-only weighting (vector_weight=0, fts_weight=1) still returns results
    results = mem.recall("budget dollars", k=3, mode="hybrid", vector_weight=0.0, fts_weight=1.0)
    assert len(results) > 0


def test_hybrid_all_vector_weight(mem: Engram) -> None:
    results_hybrid = mem.recall(
        "board merger strategy", k=3, mode="hybrid", vector_weight=1.0, fts_weight=0.0
    )
    results_cosine = mem.recall("board merger strategy", k=3, mode="cosine")
    # Both modes should surface results; scores are computed differently but ids overlap
    hybrid_ids = {r.episode.id for r in results_hybrid}
    cosine_ids = {r.episode.id for r in results_cosine}
    assert len(hybrid_ids & cosine_ids) > 0


def test_hybrid_fts_index_populated_on_insert(tmp_path) -> None:
    """Verify that observe() populates the FTS table."""
    path = str(tmp_path / "fts_check.engram")
    with Engram(path=path) as mem:
        mem.observe("Unique term xyzzy42 in this episode")
        results = mem.recall("xyzzy42", k=1, mode="hybrid")
    assert len(results) == 1
    assert "xyzzy42" in results[0].episode.content


def test_hybrid_observe_many_fts_populated(tmp_path) -> None:
    path = str(tmp_path / "fts_batch.engram")
    with Engram(path=path) as mem:
        mem.observe_many(
            [
                ObserveInput(content="Batch term alpha42 first"),
                ObserveInput(content="Batch term beta99 second"),
            ]
        )
        r1 = mem.recall("alpha42", k=1, mode="hybrid")
        r2 = mem.recall("beta99", k=1, mode="hybrid")
    assert "alpha42" in r1[0].episode.content
    assert "beta99" in r2[0].episode.content


def test_hybrid_mode_string(mem: Engram) -> None:
    # mode="hybrid" must not raise; "cosine" and "spreading" must still work
    mem.recall("test", k=1, mode="cosine")
    mem.recall("test", k=1, mode="spreading")
    mem.recall("test", k=1, mode="hybrid")


def test_hybrid_honors_as_of(tmp_path) -> None:
    """Hybrid mode must respect as_of and exclude future episodes."""
    from datetime import UTC, datetime, timedelta

    path = str(tmp_path / "hybrid_asof.engram")
    with Engram(path=path) as mem:
        old_id = mem.observe("ancient marker zzzhybrid")
        new_id = mem.observe("recent marker zzzhybrid")
        # Backdate the first one so as_of can split them deterministically.
        t_old = datetime(2026, 1, 1, tzinfo=UTC)
        t_new = datetime(2026, 6, 1, tzinfo=UTC)
        mem._store._conn.execute(
            "UPDATE episodes SET timestamp = ? WHERE id = ?", (t_old.isoformat(), old_id)
        )
        mem._store._conn.execute(
            "UPDATE episodes SET timestamp = ? WHERE id = ?", (t_new.isoformat(), new_id)
        )
        mem._store._conn.commit()

        # Without as_of, both surface.
        ids_all = {r.episode.id for r in mem.recall("zzzhybrid", k=5, mode="hybrid")}
        assert {old_id, new_id}.issubset(ids_all)

        # With as_of mid-window, only the old one.
        ids_then = {
            r.episode.id
            for r in mem.recall("zzzhybrid", k=5, mode="hybrid", as_of=t_old + timedelta(days=1))
        }
        assert ids_then == {old_id}


def test_hybrid_handles_fts5_special_chars(tmp_path) -> None:
    """User query containing FTS5 operators must not crash."""
    path = str(tmp_path / "fts_escape.engram")
    with Engram(path=path) as mem:
        mem.observe("nothing special here")
        # These previously raised sqlite3.OperationalError via raw MATCH.
        for q in ["a*b", "(quoted)", "x OR y", "foo - bar", 'q"uote']:
            mem.recall(q, k=3, mode="hybrid")
