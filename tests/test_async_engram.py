"""Tests for AsyncEngram — async wrapper over Engram."""

from __future__ import annotations

import pytest

from engram import AsyncEngram, ObserveInput

pytestmark = pytest.mark.asyncio


async def test_async_observe_returns_id(tmp_path) -> None:
    path = str(tmp_path / "async.engram")
    async with AsyncEngram(path=path) as mem:
        ep_id = await mem.observe("Alice joined the project")
    assert isinstance(ep_id, str) and len(ep_id) == 36


async def test_async_recall_returns_results(tmp_path) -> None:
    path = str(tmp_path / "async.engram")
    async with AsyncEngram(path=path) as mem:
        await mem.observe("Alice joined Globex as CTO")
        results = await mem.recall("Alice CTO")
    assert len(results) == 1
    assert "Alice" in results[0].episode.content


async def test_async_recall_hybrid(tmp_path) -> None:
    path = str(tmp_path / "async_hybrid.engram")
    async with AsyncEngram(path=path) as mem:
        await mem.observe("Ivan transferred from Acme to Globex")
        results = await mem.recall("Acme transfer", mode="hybrid")
    assert len(results) > 0


async def test_async_observe_many(tmp_path) -> None:
    path = str(tmp_path / "async_many.engram")
    async with AsyncEngram(path=path) as mem:
        ids = await mem.observe_many(
            [
                ObserveInput(content="Event one"),
                ObserveInput(content="Event two"),
                ObserveInput(content="Event three"),
            ]
        )
    assert len(ids) == 3


async def test_async_assert_fact_and_timeline(tmp_path) -> None:
    path = str(tmp_path / "async_facts.engram")
    async with AsyncEngram(path=path) as mem:
        await mem.assert_fact("Alice", "role", "CTO", confidence=0.9)
        facts = await mem.timeline("Alice")
    assert len(facts) == 1
    assert facts[0].object == "CTO"


async def test_async_decay(tmp_path) -> None:
    path = str(tmp_path / "async_decay.engram")
    async with AsyncEngram(path=path) as mem:
        await mem.observe("Some event")
        updated = await mem.decay()
    assert updated == 1


async def test_async_backup(tmp_path) -> None:
    path = str(tmp_path / "async_src.engram")
    dest = str(tmp_path / "async_backup.engram")
    async with AsyncEngram(path=path) as mem:
        await mem.observe("Backup test event")
        await mem.backup(dest)
    assert (tmp_path / "async_backup.engram").exists()


async def test_async_export_import(tmp_path) -> None:
    src_path = str(tmp_path / "async_src.engram")
    dst_path = str(tmp_path / "async_dst.engram")
    dump = str(tmp_path / "dump.json")
    async with AsyncEngram(path=src_path) as mem:
        await mem.observe("Export test")
        doc = await mem.export_json(dump)
    assert doc["counts"]["episodes"] == 1
    async with AsyncEngram(path=dst_path) as mem:
        counts = await mem.import_json(dump)
    assert counts["episodes"] == 1


async def test_async_forget(tmp_path) -> None:
    path = str(tmp_path / "async_forget.engram")
    async with AsyncEngram(path=path) as mem:
        ep_id = await mem.observe("Temporary event")
        await mem.forget(ep_id)
        results = await mem.recall("temporary event")
    assert len(results) == 0


async def test_async_context_manager(tmp_path) -> None:
    path = str(tmp_path / "async_ctx.engram")
    async with AsyncEngram(path=path) as mem:
        assert mem._store is not None


async def test_async_k_parameter(tmp_path) -> None:
    path = str(tmp_path / "async_k.engram")
    async with AsyncEngram(path=path) as mem:
        for i in range(5):
            await mem.observe(f"Event number {i}")
        results = await mem.recall("event", k=3)
    assert len(results) <= 3
