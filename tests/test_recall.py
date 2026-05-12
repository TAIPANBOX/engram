"""Tests for Engram.recall()."""

import pytest

from engram import Engram, SearchResult


@pytest.fixture()
def mem() -> Engram:
    with Engram(path=":memory:") as m:
        yield m


def test_recall_empty_store(mem: Engram) -> None:
    results = mem.recall("anything")
    assert results == []


def test_recall_finds_inserted_episode(mem: Engram) -> None:
    mem.observe("The company signed a deal with Acme Corp")
    results = mem.recall("Acme Corp deal")
    assert len(results) == 1
    assert "Acme" in results[0].episode.content


def test_recall_top_result_is_semantically_similar(mem: Engram) -> None:
    mem.observe("Ivan moved to Berlin last summer")
    mem.observe("The quarterly revenue report was released")
    mem.observe("Python 3.12 introduced the new type syntax")

    results = mem.recall("Where did Ivan relocate to?", k=1)
    assert len(results) == 1
    assert "Ivan" in results[0].episode.content
    assert results[0].score > 0.0


def test_recall_respects_k_limit(mem: Engram) -> None:
    for i in range(10):
        mem.observe(f"Meeting notes from session {i}")
    results = mem.recall("meeting", k=3)
    assert len(results) <= 3


def test_recall_returns_search_result_type(mem: Engram) -> None:
    mem.observe("Product launch scheduled for Q3")
    results = mem.recall("product launch")
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(isinstance(r.score, float) for r in results)
    assert all(isinstance(r.distance, float) for r in results)
