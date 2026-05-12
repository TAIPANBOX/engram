"""Tests for mem.compress() — LLM-driven memory summarisation."""

from __future__ import annotations

from engram import CompressionRun, Engram, ObserveInput, StubLLMAdapter


def _make_store(tmp_path, n_episodes: int = 5, importance: float = 0.1) -> tuple[Engram, str]:
    path = str(tmp_path / "compress.engram")
    stub = StubLLMAdapter(summary="Summary of several events.")
    mem = Engram(path=path, llm=stub)
    mem.observe_many([ObserveInput(content=f"Event number {i}") for i in range(n_episodes)])
    # Force low importance so they become compression candidates
    for ep in mem._store.get_episodes_below_importance(1.1):  # all episodes
        mem._store.update_importance(ep.id, importance)
    return mem, path


# ------------------------------------------------------------------
# No-op cases
# ------------------------------------------------------------------


def test_compress_no_llm_is_noop(tmp_path) -> None:
    path = str(tmp_path / "nollm.engram")
    with Engram(path=path) as mem:
        for i in range(5):
            mem.observe(f"Event {i}")
        result = mem.compress(max_episodes=0)
    assert result.episodes_removed == 0
    assert result.summaries_created == 0


def test_compress_below_threshold_is_noop(tmp_path) -> None:
    path = str(tmp_path / "small.engram")
    stub = StubLLMAdapter(summary="summary")
    with Engram(path=path, llm=stub) as mem:
        for i in range(3):
            mem.observe(f"Event {i}")
        # max_episodes=10 — store has only 3, so no compression
        result = mem.compress(max_episodes=10)
    assert result.episodes_removed == 0
    assert result.summaries_created == 0


def test_compress_no_candidates_is_noop(tmp_path) -> None:
    path = str(tmp_path / "highimp.engram")
    stub = StubLLMAdapter(summary="summary")
    with Engram(path=path, llm=stub) as mem:
        for i in range(5):
            mem.observe(f"Event {i}")
        # All episodes have default importance 1.0; threshold 0.3 → no candidates
        result = mem.compress(max_episodes=0, importance_threshold=0.3)
    assert result.episodes_removed == 0
    assert result.summaries_created == 0


# ------------------------------------------------------------------
# Core compression behaviour
# ------------------------------------------------------------------


def test_compress_removes_originals(tmp_path) -> None:
    mem, _ = _make_store(tmp_path, n_episodes=5)
    with mem:
        before = mem._store.episode_count()
        result = mem.compress(max_episodes=0, importance_threshold=1.1, batch_size=5)
        after = mem._store.episode_count()
    assert result.episodes_removed == 5
    assert result.summaries_created == 1
    # net: removed 5, added 1 summary → -4
    assert after == before - 5 + 1


def test_compress_creates_summary_episode(tmp_path) -> None:
    mem, _ = _make_store(tmp_path, n_episodes=4)
    with mem:
        mem.compress(max_episodes=0, importance_threshold=1.1, batch_size=4)
        results = mem.recall("summary of events", k=5)
    assert any("Summary" in r.episode.content for r in results)


def test_compress_summary_has_correct_tag(tmp_path) -> None:
    mem, _ = _make_store(tmp_path, n_episodes=3)
    with mem:
        mem.compress(max_episodes=0, importance_threshold=1.1, batch_size=3)
        all_eps = mem._store.get_episodes_since(None)
    summary_eps = [ep for ep in all_eps if "summary" in ep.tags]
    assert len(summary_eps) == 1


def test_compress_summary_of_field_set(tmp_path) -> None:
    mem, _ = _make_store(tmp_path, n_episodes=3)
    with mem:
        mem.compress(max_episodes=0, importance_threshold=1.1, batch_size=3)
        remaining = mem._store.get_episodes_since(None)
    summary = next(ep for ep in remaining if "summary" in ep.tags)
    assert len(summary.summary_of) == 3


def test_compress_batching(tmp_path) -> None:
    mem, _ = _make_store(tmp_path, n_episodes=6)
    with mem:
        result = mem.compress(max_episodes=0, importance_threshold=1.1, batch_size=2)
    assert result.summaries_created == 3
    assert result.episodes_removed == 6


def test_compress_returns_compression_run(tmp_path) -> None:
    mem, _ = _make_store(tmp_path, n_episodes=4)
    with mem:
        result = mem.compress(max_episodes=0, importance_threshold=1.1, batch_size=4)
    assert isinstance(result, CompressionRun)
    assert result.model_used == "stub"


def test_compress_mixed_importance(tmp_path) -> None:
    path = str(tmp_path / "mixed.engram")
    stub = StubLLMAdapter(summary="Compressed.")
    with Engram(path=path, llm=stub) as mem:
        mem.observe_many([
            ObserveInput(content=f"Low importance event {i}") for i in range(4)
        ])
        # Set two as low, two as high
        eps = mem._store.get_episodes_since(None)
        for i, ep in enumerate(eps):
            mem._store.update_importance(ep.id, 0.1 if i < 2 else 0.9)
        result = mem.compress(max_episodes=0, importance_threshold=0.3, batch_size=10)
    # Only 2 low-importance episodes should be candidates
    assert result.episodes_removed == 2
    assert result.summaries_created == 1
