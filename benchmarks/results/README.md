# Benchmark results

Raw per-question records behind the numbers published in
[`docs/api-reference.md`](../../docs/api-reference.md#recall-accuracy), so a
reader can recompute them instead of taking them on trust.

Not part of the installed package (`[tool.hatch.build.targets.wheel]` ships
only `engram/`).

## `longmemeval_s_2026-07-30.jsonl`

One JSON object per question from a full 500-question run of
[LongMemEval-S](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
(`longmemeval_s_cleaned.json`), 246 738 turns ingested as one episode each,
`bge-small-en-v1.5`, on an 8-core dedicated cloud instance. Ingestion took
5.6 hours; the queries took 15 ms each.

Each record holds the question id and type, how many episodes its history
became, the ingest and query times, and the hit flags per mode and per k.

Recompute the aggregate:

```python
import json
from engram.benchmarks.longmemeval import _aggregate

recs = [json.loads(l) for l in open("longmemeval_s_2026-07-30.jsonl")]
for mode, r in _aggregate(recs, (5, 10), ("cosine", "hybrid"), False).items():
    print(mode, r.session_recall, r.turn_recall)
```

The `hybrid` column in this file is not a hybrid measurement. The run itself
exposed why: hybrid was identical to cosine on all 500 questions because the
BM25 query joined its terms with an implicit AND and matched nothing. Fixed
in the same branch; the file is kept as the evidence that produced the fix.
