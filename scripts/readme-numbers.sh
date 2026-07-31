#!/usr/bin/env bash
# Enforces invariant 2 of CLAUDE.md: every number this project publishes is one
# somebody measured, and it still says what the measurement says.
#
# This exists because engram's own release history is a sequence of fixes to
# published numbers. 2.4.1 shipped for no other reason than that the headline
# table still described the previous blend weights one release after they
# stopped being the default, and nobody could have caught that from the table
# alone. A number in a README is a claim with no owner. The only way one stays
# true is if something refuses the push when it stops being true.
#
# What it checks, and why each is checkable rather than trusted:
#
#   1. The recall table, recomputed from benchmarks/results/*.jsonl. The raw
#      per-question records are committed precisely so a reader can recompute
#      instead of trusting, and this makes the repo do it on every push.
#   2. The corpus size, likewise summed from those records rather than quoted.
#   3. The test count, from the suite itself.
#   4. The version, agreeing across pyproject.toml, engram/__init__.py and the
#      README badge. Three places state it, so two of them can drift.
#
# It does not check prose. Prose needs a reader; numbers need a script.
#
# This file is the ONE copy of this check. The local hook and CI both call it.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

PY="python3"
[ -x .venv/bin/python ] && PY=.venv/bin/python

"$PY" - <<'PYEOF'
import json
import pathlib
import re
import subprocess
import sys
import tomllib

problems = []


def note(msg):
    problems.append(msg)


readme = pathlib.Path("README.md").read_text()

# ---------------------------------------------------------------- 1 and 2
# The recall table, recomputed. The sweep file carries every weighting that was
# scored in one pass, so the default's row is not a separate run that could have
# drifted from the others.
sweep = pathlib.Path(
    "benchmarks/results/longmemeval_s_2026-07-30_weight_sweep.jsonl"
)
if not sweep.exists():
    note(f"{sweep} is missing, so the recall table was not checked at all")
else:
    recs = [json.loads(l) for l in sweep.read_text().splitlines() if l.strip()]
    if not recs:
        note(f"{sweep} is empty, which means this check measured nothing")
        recs = []

    def agg(mode):
        out = {}
        for scope in ("session", "turn"):
            for k in ("5", "10"):
                hits = sum(1 for r in recs if r["hits"][mode][scope][k])
                out[f"{scope}@{k}"] = round(hits / len(recs), 3)
        out["ms"] = round(sum(r["query_s"][mode] for r in recs) / len(recs) * 1000)
        return out

    # The README's default row must be the weighting the README itself names as
    # the default, so a weight change that forgets the table cannot pass.
    m = re.search(r"\*\*`hybrid`\*\* \(default, `([\d.]+) / ([\d.]+)`\)", readme)
    if not m:
        note("could not find the hybrid default row in the README recall table")
    else:
        v, f = m.group(1), m.group(2)
        mode = f"hybrid v{v}/f{f}"
        if recs and mode not in recs[0]["hits"]:
            note(
                f"the README calls {v}/{f} the default, and the sweep has no "
                f"'{mode}' column, so the published row cannot be reproduced"
            )
        elif recs:
            rows = {
                mode: re.search(
                    r"\*\*`hybrid`\*\* \(default[^|]*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|",
                    readme,
                ),
                "cosine": re.search(
                    r"\|\s*`cosine`\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|",
                    readme,
                ),
            }
            for name, mt in rows.items():
                if not mt:
                    note(f"could not read the {name} row of the README recall table")
                    continue
                got = agg(name)
                published = [
                    x.strip().strip("*").strip() for x in mt.groups()
                ]
                expected = [
                    f"{got['session@5']:.3f}",
                    f"{got['session@10']:.3f}",
                    f"{got['turn@5']:.3f}",
                    f"{got['turn@10']:.3f}",
                    str(got["ms"]),
                ]
                labels = ["session@5", "session@10", "turn@5", "turn@10", "ms/query"]
                for lab, pub, exp in zip(labels, published, expected):
                    if pub != exp:
                        note(
                            f"README {name} {lab} says {pub}, the records give {exp}"
                        )

    if recs:
        episodes = sum(r["n_episodes"] for r in recs)
        # The README writes it with a thin space between thousands.
        if not re.search(rf"{episodes // 1000}[\s  ]?{episodes % 1000:03d}", readme):
            note(f"README does not state the corpus size the records sum to: {episodes}")
        if not re.search(rf"All {len(recs)} questions", readme):
            note(f"README does not say the run covered {len(recs)} questions")

# ---------------------------------------------------------------------- 3
# The test count, from the suite, not from anything that quotes it.
#
# THIS NUMBER DEPENDS ON THE ENVIRONMENT, and the first version of this check
# did not know that. tests/test_encryption.py skips at module level when
# sqlcipher3 is absent, so its 7 tests are not collected at all: 466 on a plain
# install, 473 with the encryption extra. Comparing a measured environment to a
# single fixed number fails on whichever machine is not the one the number came
# from, and it did, in CI, on the very run that first wired this into a
# workflow.
#
# So the README states both, and this asks which environment it is in rather
# than assuming. Anything that measures the environment must either be
# environment-independent or know what it is running on.
proc = subprocess.run(
    [sys.executable, "-m", "pytest", "--collect-only", "-q"],
    capture_output=True,
    text=True,
)
m = re.search(r"(\d+) tests collected", proc.stdout)
if not m:
    note(
        "the suite reported no test count, which means this check measured "
        "nothing. Install pytest rather than letting the check pass quietly."
    )
else:
    total = int(m.group(1))
    quoted = re.search(
        r"### Test coverage \((\d+) tests, (\d+) with the encryption extra\)",
        readme,
    )
    if not quoted:
        note(
            "could not read the test-coverage heading in the README. It must "
            "state both counts, because the number depends on whether the "
            "encryption extra is installed."
        )
    else:
        base, with_extra = int(quoted.group(1)), int(quoted.group(2))
        try:
            import sqlcipher3  # noqa: F401

            expected, which = with_extra, "with the encryption extra"
        except ImportError:
            expected, which = base, "on a plain install"
        if expected != total:
            note(
                f"README says {expected} tests {which} and the suite collects "
                f"{total} here"
            )
        if with_extra <= base:
            note(
                f"the README claims {with_extra} tests with the encryption "
                f"extra and {base} without, which cannot be right: the extra "
                f"only adds tests"
            )

# ---------------------------------------------------------------------- 4
# The version, in all three places that state it.
pyproject = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
declared = pyproject["project"]["version"]

init = pathlib.Path("engram/__init__.py").read_text()
m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
if not m:
    note("engram/__init__.py has no __version__")
elif m.group(1) != declared:
    note(
        f"engram/__init__.py reports {m.group(1)} and pyproject.toml declares "
        f"{declared}, so an installed package lies about its own version"
    )

m = re.search(r"status-v([0-9][^-]*)-", readme)
if not m:
    note("could not find the status badge in the README")
elif m.group(1) != declared:
    note(f"README badge says v{m.group(1)} and pyproject.toml declares {declared}")

# ---------------------------------------------------------------------------
if problems:
    for p in problems:
        print(f"FAIL: {p}")
    print()
    print("A number in a README is a claim with no owner. If a default, a weight")
    print("or an algorithm changed, the numbers describing it are part of the")
    print("same change, not a follow-up. See CLAUDE.md invariant 2.")
    sys.exit(1)

print("OK: the recall table, the corpus size, the test count and the version all")
print("    match what the repository actually contains.")
PYEOF
