"""Tests for WorkingMemory scratchpad."""

from __future__ import annotations

import pytest

from engram import Engram, WorkingMemory, WorkingMemoryItem

# ------------------------------------------------------------------
# Basic set / get / delete
# ------------------------------------------------------------------


def test_set_and_get() -> None:
    wm = WorkingMemory(capacity=5)
    wm.set("task", "Summarise the report")
    item = wm.get("task")
    assert item is not None
    assert item.content == "Summarise the report"
    assert item.key == "task"


def test_get_missing_returns_none() -> None:
    wm = WorkingMemory(capacity=5)
    assert wm.get("nonexistent") is None


def test_peek_does_not_update_lru() -> None:
    wm = WorkingMemory(capacity=2)
    wm.set("a", "first")
    wm.set("b", "second")
    # peek "a" — should not promote it
    wm.peek("a")
    # now fill capacity: "a" is LRU and should be evicted
    wm.set("c", "third")
    assert wm.peek("a") is None
    assert wm.peek("b") is not None
    assert wm.peek("c") is not None


def test_delete_existing() -> None:
    wm = WorkingMemory(capacity=5)
    wm.set("x", "value")
    assert wm.delete("x") is True
    assert wm.get("x") is None


def test_delete_missing() -> None:
    wm = WorkingMemory(capacity=5)
    assert wm.delete("ghost") is False


def test_len() -> None:
    wm = WorkingMemory(capacity=5)
    assert len(wm) == 0
    wm.set("a", "1")
    wm.set("b", "2")
    assert len(wm) == 2


def test_contains() -> None:
    wm = WorkingMemory(capacity=5)
    wm.set("k", "v")
    assert "k" in wm
    assert "missing" not in wm


# ------------------------------------------------------------------
# Capacity and LRU eviction
# ------------------------------------------------------------------


def test_capacity_enforced() -> None:
    wm = WorkingMemory(capacity=3)
    wm.set("a", "1")
    wm.set("b", "2")
    wm.set("c", "3")
    assert len(wm) == 3
    wm.set("d", "4")
    assert len(wm) == 3


def test_lru_evicted() -> None:
    wm = WorkingMemory(capacity=3)
    wm.set("a", "1")
    wm.set("b", "2")
    wm.set("c", "3")
    # access "a" to make it MRU; "b" is now LRU
    wm.get("a")
    wm.set("d", "4")
    assert wm.get("b") is None  # evicted
    assert wm.get("a") is not None
    assert wm.get("c") is not None
    assert wm.get("d") is not None


def test_update_existing_promotes_to_mru() -> None:
    wm = WorkingMemory(capacity=2)
    wm.set("a", "v1")
    wm.set("b", "v2")
    # re-set "a" to promote it; "b" becomes LRU
    wm.set("a", "v1_updated")
    wm.set("c", "v3")  # should evict "b"
    assert wm.get("b") is None
    assert wm.get("a") is not None
    assert wm.get("a").content == "v1_updated"  # type: ignore[union-attr]


def test_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        WorkingMemory(capacity=0)


# ------------------------------------------------------------------
# items() ordering
# ------------------------------------------------------------------


def test_items_lru_to_mru_order() -> None:
    wm = WorkingMemory(capacity=5)
    wm.set("first", "1")
    wm.set("second", "2")
    wm.set("third", "3")
    keys = [i.key for i in wm.items()]
    assert keys == ["first", "second", "third"]


# ------------------------------------------------------------------
# clear / flush
# ------------------------------------------------------------------


def test_clear_empties_store() -> None:
    wm = WorkingMemory(capacity=5)
    wm.set("a", "1")
    wm.set("b", "2")
    wm.clear()
    assert len(wm) == 0


def test_flush_without_engram_discards_and_clears() -> None:
    wm = WorkingMemory(capacity=5)
    wm.set("a", "1")
    wm.set("b", "2")
    count = wm.flush()
    assert count == 2
    assert len(wm) == 0


def test_flush_with_engram_writes_to_longterm(tmp_path) -> None:
    path = str(tmp_path / "wm.engram")
    with Engram(path=path) as mem:
        wm = WorkingMemory(capacity=5, engram=mem)
        wm.set("task", "Quarterly report summary")
        wm.set("context", "Revenue grew 12% YoY")
        count = wm.flush()
        assert count == 2
        assert len(wm) == 0
        assert mem._store.episode_count() == 2


def test_eviction_flushes_to_longterm(tmp_path) -> None:
    path = str(tmp_path / "wm_evict.engram")
    with Engram(path=path) as mem:
        wm = WorkingMemory(capacity=2, engram=mem)
        wm.set("a", "first item")
        wm.set("b", "second item")
        # third insert should evict "a" to long-term
        wm.set("c", "third item")
        assert mem._store.episode_count() == 1
        results = mem.recall("first item", k=1)
        assert len(results) == 1
        assert "first item" in results[0].episode.content


# ------------------------------------------------------------------
# metadata
# ------------------------------------------------------------------


def test_metadata_stored() -> None:
    wm = WorkingMemory(capacity=5)
    wm.set("note", "content here", priority=1, source="user")
    item = wm.get("note")
    assert item is not None
    assert item.metadata["priority"] == 1
    assert item.metadata["source"] == "user"


def test_repr() -> None:
    wm = WorkingMemory(capacity=7)
    wm.set("a", "x")
    assert "WorkingMemory" in repr(wm)
    assert "capacity=7" in repr(wm)
    assert "size=1" in repr(wm)


# ------------------------------------------------------------------
# WorkingMemoryItem type
# ------------------------------------------------------------------


def test_working_memory_item_type() -> None:
    wm = WorkingMemory(capacity=5)
    wm.set("t", "text")
    item = wm.get("t")
    assert isinstance(item, WorkingMemoryItem)
