"""CLI entry point for Engram benchmarks.

Usage:
    python -m engram.benchmarks latency [--n N] [--k K]
    python -m engram.benchmarks locomo [--data FILE] [--k K]
    python -m engram.benchmarks cost [--n N]
    python -m engram.benchmarks all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _header(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _run_latency(n: int, k: int) -> None:
    _header(f"Latency Benchmark  (n={n}, k={k})")
    print("  Warming up fastembed model…", flush=True)
    from engram.benchmarks.latency import run_latency_suite

    results = run_latency_suite(n=n, k=k)
    print()
    header = f"  {'Operation':<24} {'p50 ms':>8} {'p99 ms':>8} {'mean ms':>8} {'tput/s':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, stats in results.items():
        print(
            f"  {name:<24} {stats.p50_ms:>8.1f} {stats.p99_ms:>8.1f} "
            f"{stats.mean_ms:>8.1f} {stats.throughput_per_sec:>8.0f}"
        )


def _run_locomo(data_path: str | None, k: int) -> None:
    bundled = data_path is None
    if data_path is None:
        data_path = str(Path(__file__).parent / "data" / "locomo_sample.json")
    _header(f"Recall Accuracy  (k={k}, data={Path(data_path).name})")
    if bundled:
        print("  NOTE: the bundled fixture is synthetic, written alongside the")
        print("  questions it answers. These scores show the retrieval path is")
        print("  wired up. They are not a result on LoCoMo or any other")
        print("  benchmark; pass --data to measure something real.")
        print()
    print("  Ingesting sessions and running QA eval…", flush=True)
    from engram.benchmarks.locomo import run_locomo

    result = run_locomo(data_path, k=k)
    print()
    print(f"  Sessions:     {result.n_sessions}")
    print(f"  Episodes:     {result.n_episodes}")
    print(f"  Questions:    {result.n_questions}")
    print()
    print(f"  hit@1:        {result.hit_rate_at_1:.1%}")
    print(f"  hit@{k}:        {result.hit_rate_at_5:.1%}")
    print(f"  MRR:          {result.mrr:.3f}")


def _run_scale(sizes: str, k: int, n_queries: int) -> None:
    parsed = tuple(int(s) for s in sizes.split(",") if s.strip())
    _header(f"Scaling  (sizes={parsed}, k={k}, {n_queries} queries each)")
    print("  Exact vector scan, so cost is linear in the episodes that survive")
    print("  the filters. Every column but the last excludes query embedding,")
    print("  which is a flat ~4 ms and would otherwise hide the curve. RSS is")
    print("  a fresh process opening the finished store and recalling from it.")
    print()
    from engram.benchmarks.scale import bench_scale_point

    # Printed per size rather than at the end: the largest store takes minutes
    # to build, and a run that shows nothing until it is done looks hung.
    head = ""
    for n in sorted(parsed):
        point = bench_scale_point(n, k=k, n_queries=n_queries)
        if not head:
            head = f"  {'episodes':>9} {'file MB':>8} {'RSS MB':>7} {'build s':>8}"
            head += "".join(f" {m + ' p50/p95':>22}" for m in point.latencies)
            print(head)
            print("  " + "-" * (len(head) - 2))
        rss = f"{point.rss_mb:.0f}" if point.rss_mb is not None else "n/a"
        row = f"  {point.n_episodes:>9,} {point.file_mb:>8.1f} {rss:>7} {point.build_s:>8.1f}"
        for stats in point.latencies.values():
            row += f" {stats.p50_ms:>10.2f} /{stats.p95_ms:>10.2f}"
        print(row, flush=True)


def _run_longmemeval(
    data_path: str, k_values: str, mode: str, sample: int | None, seed: int
) -> None:
    ks = tuple(int(x) for x in k_values.split(",") if x.strip())
    _header(f"LongMemEval-S  (mode={mode}, k={ks})")
    from engram.benchmarks.longmemeval import run_longmemeval

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total} questions", flush=True)

    result = run_longmemeval(
        data_path, k_values=ks, mode=mode, sample=sample, seed=seed, progress=progress
    )

    print()
    scope = f"stratified sample, seed {seed}" if result.sampled else "full set"
    print(f"  Questions:    {result.n_questions:,}  ({scope})")
    print(f"  Episodes:     {result.n_episodes:,}  (one per turn)")
    print(f"  Ingest:       {result.ingest_s:,.0f} s")
    print(f"  Query:        {result.query_s * 1000 / max(result.n_questions, 1):,.0f} ms/question")
    print()
    print("  Evidence found inside the top k:")
    for k in ks:
        print(
            f"    k={k:<4} session {result.session_recall[k]:.3f}"
            f"    turn {result.turn_recall[k]:.3f}"
        )
    print()
    print("  Session recall by question type:")
    for qtype, scores in result.session_recall_by_type.items():
        cells = "  ".join(f"k={k} {scores[k]:.3f}" for k in ks)
        print(f"    {qtype:<26} {cells}")


def _run_cost(n: int) -> None:
    _header(f"Cost Benchmark  (n={n} episodes)")
    print("  Simulating reflection passes…", flush=True)
    from engram.benchmarks.cost import _MODEL_PRICES, run_cost_bench

    result = run_cost_bench(n_episodes=n)
    print()
    print(f"  Episodes processed:    {result.n_episodes:,}")
    print(f"  Reflection runs:       {result.n_reflect_runs:,}")
    print(f"  Facts extracted:       {result.facts_extracted:,}")
    print(f"  Est. input tokens:     {result.est_input_tokens:,}")
    print(f"  Est. output tokens:    {result.est_output_tokens:,}")
    print(f"  Tokens / 1k episodes:  {result.tokens_per_1000_episodes:,.0f}")
    print()
    print(f"  {'Model':<22} {'$/1M in':>8} {'$/1M out':>9} {'$/1k ep':>10}")
    print("  " + "-" * 54)
    for model, (inp_p, out_p) in _MODEL_PRICES.items():
        cost = result.projected_cost_per_1000_episodes(model)
        print(f"  {model:<22} {inp_p:>8.2f} {out_p:>9.2f} ${cost:>9.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="engram-bench",
        description="Engram benchmark suite",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    lat = sub.add_parser("latency", help="observe/recall latency (p50, p99)")
    lat.add_argument("--n", type=int, default=200, help="iterations (default 200)")
    lat.add_argument("--k", type=int, default=5, help="recall top-k (default 5)")

    loc = sub.add_parser("locomo", help="LoCoMo recall accuracy benchmark")
    loc.add_argument("--data", default=None, metavar="FILE", help="path to LoCoMo JSON")
    loc.add_argument("--k", type=int, default=5, help="recall top-k (default 5)")

    scale = sub.add_parser("scale", help="recall latency as the store grows")
    scale.add_argument("--sizes", default="1000,10000,100000", help="store sizes, comma separated")
    scale.add_argument("--k", type=int, default=10, help="recall top-k (default 10)")
    scale.add_argument("--queries", type=int, default=50, help="queries per size (default 50)")

    lme = sub.add_parser("longmemeval", help="retrieval recall on LongMemEval-S")
    lme.add_argument("--data", required=True, metavar="FILE", help="longmemeval_s json")
    lme.add_argument("--k", default="5,10", help="comma-separated k values (default 5,10)")
    lme.add_argument("--mode", default="cosine", help="recall mode (default cosine)")
    lme.add_argument("--sample", type=int, default=None, help="stratified subsample of this size")
    lme.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")

    cost = sub.add_parser("cost", help="reflection token cost projection")
    cost.add_argument("--n", type=int, default=200, help="episodes to simulate (default 200)")

    sub.add_parser("all", help="run latency + locomo + cost with defaults")

    args = parser.parse_args(argv)

    if args.cmd == "latency":
        _run_latency(args.n, args.k)
    elif args.cmd == "locomo":
        _run_locomo(args.data, args.k)
    elif args.cmd == "scale":
        _run_scale(args.sizes, args.k, args.queries)
    elif args.cmd == "longmemeval":
        _run_longmemeval(args.data, args.k, args.mode, args.sample, args.seed)
    elif args.cmd == "cost":
        _run_cost(args.n)
    elif args.cmd == "all":
        _run_latency(200, 5)
        _run_locomo(None, 5)
        _run_cost(200)
    else:
        parser.print_help()
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
