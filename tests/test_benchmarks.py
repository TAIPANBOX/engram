"""Tests for the benchmark infrastructure (fast, uses small n, no real LLMs)."""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_DATA = Path(__file__).parent.parent / "engram" / "benchmarks" / "data" / "locomo_sample.json"


# ------------------------------------------------------------------
# Latency benchmark
# ------------------------------------------------------------------


def test_latency_stats_observe_returns_valid_shape() -> None:
    from engram.benchmarks.latency import bench_observe

    stats = bench_observe(n=10, warmup=2)
    assert stats.p50_ms > 0
    assert stats.p99_ms >= stats.p50_ms
    assert stats.mean_ms > 0
    assert stats.throughput_per_sec > 0


def test_latency_stats_recall_cosine() -> None:
    from engram.benchmarks.latency import bench_recall

    stats = bench_recall(n_episodes=20, n_queries=10, k=3, mode="cosine", warmup=2)
    assert stats.p50_ms > 0
    assert stats.p99_ms >= stats.p50_ms


def test_latency_stats_recall_spreading() -> None:
    from engram.benchmarks.latency import bench_recall

    stats = bench_recall(n_episodes=20, n_queries=5, k=3, mode="spreading", warmup=1)
    assert stats.p50_ms > 0


def test_latency_suite_returns_three_keys() -> None:
    from engram.benchmarks.latency import run_latency_suite

    results = run_latency_suite(n=10, k=3)
    assert set(results.keys()) == {"observe", "recall_cosine", "recall_spreading"}


def test_latency_stats_str_format() -> None:
    from engram.benchmarks.latency import LatencyStats

    s = LatencyStats(p50_ms=5.1, p99_ms=12.3, mean_ms=6.0, throughput_per_sec=150.0)
    text = str(s)
    assert "p50=" in text
    assert "p99=" in text
    assert "tput=" in text


# ------------------------------------------------------------------
# LoCoMo benchmark
# ------------------------------------------------------------------


def test_locomo_sample_file_exists() -> None:
    assert SAMPLE_DATA.exists(), "locomo_sample.json not found"


def test_locomo_run_on_sample_data() -> None:
    from engram.benchmarks.locomo import run_locomo

    result = run_locomo(SAMPLE_DATA, k=5)
    assert result.n_sessions == 5
    assert result.n_episodes == 15  # 5 sessions x 3 turns
    assert result.n_questions == 15
    assert 0.0 <= result.hit_rate_at_1 <= 1.0
    assert 0.0 <= result.hit_rate_at_5 <= 1.0
    assert result.hit_rate_at_5 >= result.hit_rate_at_1
    assert 0.0 <= result.mrr <= 1.0


def test_locomo_hit_rate_at_5_reasonable() -> None:
    """Engram should recall > 50% of simple keyword questions at k=5."""
    from engram.benchmarks.locomo import run_locomo

    result = run_locomo(SAMPLE_DATA, k=5)
    assert result.hit_rate_at_5 >= 0.5, f"hit@5={result.hit_rate_at_5:.1%} — below expected 50%"


def test_locomo_accepts_dict_source() -> None:
    """run_locomo accepts a pre-loaded dict, not just a file path."""
    import json

    from engram.benchmarks.locomo import run_locomo

    with SAMPLE_DATA.open() as f:
        data = json.load(f)
    result = run_locomo(data, k=3)
    assert result.n_questions == 15


def test_locomo_result_str() -> None:
    from engram.benchmarks.locomo import LoCoMoResult

    r = LoCoMoResult(
        n_sessions=5,
        n_episodes=15,
        n_questions=15,
        hit_rate_at_1=0.6,
        hit_rate_at_5=0.8,
        mrr=0.7,
    )
    text = str(r)
    assert "hit@1" in text
    assert "MRR" in text


# ------------------------------------------------------------------
# Cost benchmark
# ------------------------------------------------------------------


def test_cost_bench_returns_valid_result() -> None:
    from engram.benchmarks.cost import run_cost_bench

    result = run_cost_bench(n_episodes=10, facts_per_batch=2)
    assert result.n_episodes == 10
    assert result.n_reflect_runs >= 1
    assert result.facts_extracted > 0
    assert result.est_input_tokens > 0
    assert result.est_output_tokens > 0
    assert result.tokens_per_1000_episodes > 0


def test_cost_bench_cost_usd() -> None:
    from engram.benchmarks.cost import run_cost_bench

    result = run_cost_bench(n_episodes=10)
    cost = result.cost_usd("claude-haiku-4.5")
    assert cost >= 0.0


def test_cost_bench_projected_cost() -> None:
    from engram.benchmarks.cost import run_cost_bench

    result = run_cost_bench(n_episodes=20)
    proj = result.projected_cost_per_1000_episodes("gpt-4o-mini")
    assert proj >= 0.0


def test_cost_bench_str_format() -> None:
    from engram.benchmarks.cost import run_cost_bench

    result = run_cost_bench(n_episodes=10)
    text = str(result)
    assert "episodes=" in text
    assert "tokens" in text.lower()


# ------------------------------------------------------------------
# CLI smoke tests
# ------------------------------------------------------------------


def test_cli_locomo_small(capsys: pytest.CaptureFixture[str]) -> None:
    from engram.benchmarks.cli import main

    main(["locomo", "--data", str(SAMPLE_DATA), "--k", "3"])
    out = capsys.readouterr().out
    assert "hit@1" in out
    assert "MRR" in out


def test_cli_cost_small(capsys: pytest.CaptureFixture[str]) -> None:
    from engram.benchmarks.cli import main

    main(["cost", "--n", "10"])
    out = capsys.readouterr().out
    assert "episodes" in out.lower()
    assert "tokens" in out.lower()


def test_cli_no_subcommand_exits(capsys: pytest.CaptureFixture[str]) -> None:
    from engram.benchmarks.cli import main

    with pytest.raises(SystemExit):
        main([])


# ------------------------------------------------------------------
# Scaling benchmark
# ------------------------------------------------------------------


def test_scale_point_measures_every_query_path() -> None:
    from engram.benchmarks.scale import bench_scale_point

    point = bench_scale_point(n=60, k=5, n_queries=6, warmup=2)

    assert point.n_episodes == 60
    assert point.file_mb > 0, "WAL must be checkpointed before the file is measured"
    assert point.build_s > 0
    assert set(point.latencies) == {
        "cosine",
        "hybrid",
        "as_of",
        "scoped (5 of n)",
        "recall() end to end",
    }
    for stats in point.latencies.values():
        assert stats.p50_ms > 0
        assert stats.p95_ms >= stats.p50_ms


def test_scale_queries_are_absent_from_the_corpus() -> None:
    """Querying with corpus strings measures the embedder's LRU cache instead
    of the store: hits while the corpus fits, misses once it does not."""
    from engram.benchmarks.latency import _make_texts
    from engram.benchmarks.scale import _make_queries

    queries = _make_queries(200)
    assert len(set(queries)) == 200, "repeated queries would be served from cache"
    assert not set(queries) & set(_make_texts(2000))


def test_scoped_recall_does_not_scale_with_the_store() -> None:
    """The partition key is the whole point: an agent holding five episodes
    pays for five, not for everything the other agent wrote."""
    from engram.benchmarks.scale import bench_scale_point

    small = bench_scale_point(n=100, k=5, n_queries=6, warmup=2)
    large = bench_scale_point(n=1500, k=5, n_queries=6, warmup=2)

    scoped_growth = large.latencies["scoped (5 of n)"].p50_ms / (
        small.latencies["scoped (5 of n)"].p50_ms or 1e-9
    )
    unscoped_growth = large.latencies["cosine"].p50_ms / (small.latencies["cosine"].p50_ms or 1e-9)

    assert scoped_growth < 3, f"scoped recall grew {scoped_growth:.1f}x for 15x the store"
    assert unscoped_growth > scoped_growth
