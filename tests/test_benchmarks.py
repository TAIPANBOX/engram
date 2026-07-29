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


# ------------------------------------------------------------------
# LongMemEval harness
# ------------------------------------------------------------------


def _fake_longmemeval(tmp_path, n_per_type: int = 4):
    """A LongMemEval-shaped file small enough to run in a test.

    Grouped by question type, like the real one, so a sampler that slices from
    the front instead of stratifying shows up here.
    """
    import json

    questions = []
    for qtype in ("type-a", "type-b"):
        for i in range(n_per_type):
            qid = f"{qtype}-{i}"
            questions.append(
                {
                    "question_id": qid,
                    "question_type": qtype,
                    "question": f"where did the {qid} meeting happen",
                    "question_date": "2024/03/01 (Fri) 10:00",
                    "answer": "Berlin",
                    "answer_session_ids": [f"s-{qid}-1"],
                    "haystack_session_ids": [f"s-{qid}-0", f"s-{qid}-1"],
                    "haystack_dates": ["2024/01/02 (Tue) 09:00", "2024/02/03 (Sat) 11:00"],
                    "haystack_sessions": [
                        [{"role": "user", "content": f"unrelated chatter about lunch {qid}"}],
                        [
                            {
                                "role": "user",
                                "content": f"the {qid} meeting happened in Berlin",
                                "has_answer": True,
                            }
                        ],
                    ],
                }
            )
    path = tmp_path / "fake_lme.json"
    path.write_text(json.dumps(questions), encoding="utf-8")
    return path


def test_longmemeval_scores_session_and_turn_recall(tmp_path) -> None:
    from engram.benchmarks.longmemeval import run_longmemeval

    result = run_longmemeval(_fake_longmemeval(tmp_path), k_values=(1, 2))

    assert result.n_questions == 8
    assert result.n_episodes == 16
    assert not result.sampled
    # The evidence turn names the question, so it should top a k=1 recall.
    assert result.session_recall[1] == 1.0
    assert result.turn_recall[1] == 1.0
    assert set(result.session_recall_by_type) == {"type-a", "type-b"}


def test_longmemeval_sample_is_stratified_and_reproducible(tmp_path) -> None:
    """The real file is ordered by question type, so a front slice would
    measure one category and report it as a subsample of all of them."""
    import json

    from engram.benchmarks.longmemeval import _stratified_sample

    questions = json.loads(_fake_longmemeval(tmp_path, n_per_type=10).read_text())
    picked = _stratified_sample(questions, 10, seed=0)

    types = [q["question_type"] for q in picked]
    assert types.count("type-a") == 5
    assert types.count("type-b") == 5
    assert [q["question_id"] for q in _stratified_sample(questions, 10, seed=0)] == [
        q["question_id"] for q in picked
    ]


def test_longmemeval_evidence_tag_is_not_searchable(tmp_path) -> None:
    """The has_answer flag is carried as a tag. Tags are not embedded and not
    in the FTS index, so marking evidence cannot help retrieval find it."""
    import json

    from engram import Engram
    from engram.benchmarks.longmemeval import _ingest

    question = json.loads(_fake_longmemeval(tmp_path).read_text())[0]
    with Engram() as mem:
        _ingest(mem, question)
        assert mem.recall("__evidence__", k=5, mode="hybrid") != []
        for r in mem.recall("__evidence__", k=5, mode="hybrid"):
            assert "__evidence__" not in r.episode.content
