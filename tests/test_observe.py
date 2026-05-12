"""Tests for Engram.observe()."""

import pytest

from engram import Engram


@pytest.fixture()
def mem() -> Engram:
    with Engram(path=":memory:") as m:
        yield m


def test_observe_returns_string_id(mem: Engram) -> None:
    ep_id = mem.observe("Alice joined the project today")
    assert isinstance(ep_id, str)
    assert len(ep_id) > 0


def test_observe_stores_episode(mem: Engram) -> None:
    ep_id = mem.observe("Bob fixed the critical bug", actors=["Bob"], tags=["engineering"])
    stored = mem._store.get_episode(ep_id)
    assert stored is not None
    assert stored.content == "Bob fixed the critical bug"
    assert "Bob" in stored.actors
    assert "engineering" in stored.tags


def test_observe_stores_embedding_in_vec_table(mem: Engram) -> None:
    mem.observe("Charlie deployed to production")
    assert mem._store.vec_count() == 1


def test_observe_multiple_episodes(mem: Engram) -> None:
    ids = [mem.observe(f"Event number {i}") for i in range(5)]
    assert len(set(ids)) == 5  # all IDs are unique
    assert mem._store.episode_count() == 5
    assert mem._store.vec_count() == 5
