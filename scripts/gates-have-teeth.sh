#!/usr/bin/env bash
# Checks that the gates in `scripts/` still FAIL on the faults they exist to
# catch, still PASS on what they must not catch, and REFUSE to report success
# when they measured nothing at all.
#
# WHY
#
# Every gate here parses text, and a text parser does not break loudly: it
# stops matching and reports success. The mutants that proved each one existed
# as prose, in commit messages and in the `*(gate: ...)*` markers in CLAUDE.md,
# which is a record of what was true once. Nothing ran them again.
#
# A gate that has quietly stopped catching anything looks exactly like a gate
# with nothing to catch, and stays that way until the fault it guards ships.
#
# WHY THE THIRD PROPERTY IS SEPARATE FROM THE FIRST
#
# All three of these gates already refuse when their subject is absent: the
# recall sweep missing, the suite reporting no test count, the embedder failing
# to warm. Every one of those sentences was true, was established by hand once,
# and nothing re-ran them. Two of the three were observed doing it for real on
# 2026-08-09, on a machine with no pytest and no rfc8785, which is how this
# harness learned that this repo's gates need an environment before they can
# measure anything. A check that cannot tell "did not fail" from
# "did not run" is the most expensive recurring mistake in this estate's
# tooling, and it lives in tooling rather than product code because tooling is
# where a silent pass looks like a result.
#
# HOW IT MUTATES WITHOUT LEAVING A MESS
#
# It edits tracked files in place, so it refuses to start unless the tree is
# clean, restores with `git checkout` after every case, restores again from a
# trap on any exit path including a kill, and asserts the tree is clean before
# reporting success.
#
#
# A GATE THAT IS ALREADY FAILING CANNOT BE JUDGED
#
# A case expecting a gate to FAIL proves nothing if the gate was failing before
# the mutation. So every fail-case runs the gate on the UNMUTATED tree first
# and reports UNJUDGEABLE rather than a pass. Found on 2026-08-09 in it-rat,
# where one gate was legitimately red and a case against it would have been
# indistinguishable from a working one.
#
# A MUTATION THAT DID NOT APPLY PROVES NOTHING
#
# Every edit asserts it changed the file. A case whose edit applied nothing is
# a failure here, not a pass. That is not hypothetical: five such mutations
# were caught across idryx and tokenfuse on 2026-08-09, and three of the five
# had been verified BY HAND against the same gate minutes earlier. The hand
# version and the harness version differ only in how many layers of quoting sit
# between the text and python, which is exactly the difference nobody sees.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

# The gates here run pytest and warm the embedder, so they need the package
# installed. Without it they report "measured nothing", correctly, and every
# case below would fail for that reason rather than for the one it is about.
if ! python3 -c 'import rfc8785, pytest' 2>/dev/null; then
	printf 'this repo\x27s gates need the package installed to measure anything.\n'
	printf 'they say so themselves rather than passing quietly, which is the\n'
	printf 'property this harness checks, so it cannot run without it either:\n'
	printf '    pip install -e ".[dev]"\n'
	exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
	printf 'this script mutates tracked files, so it needs a clean tree.\n'
	printf 'commit or stash first; it restores with `git checkout` and cannot\n'
	printf 'tell your edits from its own.\n'
	exit 1
fi

# Untracked files too: a mutation may RENAME a tracked file, and `git checkout`
# restores the original while leaving the new name behind. And the INDEX, since
# a gate may read `git ls-files` rather than the disk, so a mutation has to move
# the file in both. Safe because this
# script refuses to start unless the tree is clean, so anything untracked
# during a run was created by the run. `-x` is deliberately absent: ignored
# build output is not ours to delete.
restore() {
	git reset -q --hard HEAD 2>/dev/null
	git clean -fdq 2>/dev/null
}
baseline_dir="$(mktemp -d)"

# One trap for both, because a second `trap ... EXIT` REPLACES the first
# rather than adding to it. Writing them separately disarmed `restore` on
# every interrupt path, which would leave a mutated tree behind on Ctrl-C.
cleanup() {
	restore
	rm -rf "$baseline_dir"
}
trap cleanup EXIT INT TERM


failures=0
cases=0

# run_case <name> <expect: fail|pass> <gate> <python edit> [required output]
#
# The needle separates "it failed" from "it failed for the reason this case is
# about". Without it, a case expecting failure is satisfied by any failure,
# including one this harness caused itself.
run_case() {
	local name="$1" expect="$2" gate="$3" edit="$4" needle="${5:-}"
	cases=$((cases + 1))

	# A gate that is ALREADY failing cannot be judged: a fail-case against it
	# passes while proving nothing, which is this harness committing the very
	# fault it exists to catch. Added estate-wide on 2026-08-09 after it-rat,
	# where `demo-bundle-current.sh` was red on a clean tree because the
	# published demo had fallen behind genaryx. Any case written against it
	# would have gone green having measured nothing.
	#
	# The result is cached per GATE rather than per case: the tree is restored
	# between cases, so a gate's verdict on the clean tree cannot change within
	# one run, and some of these gates compile or run a whole suite.
	if [ "$expect" = fail ]; then
		local key base_out
		key="$baseline_dir/$(printf '%s' "$gate" | cksum | tr -d ' ')"
		if [ ! -f "$key" ]; then
			if eval "$gate" >/dev/null 2>&1; then printf 'green' >"$key"; else printf 'red' >"$key"; fi
		fi
		base_out="$(cat "$key")"
		if [ "$base_out" = red ]; then
			printf 'UNJUDGEABLE  %s\n             the gate is already failing on a clean tree, so a\n             failure after the mutation would prove nothing\n' "$name"
			failures=$((failures + 1))
			return
		fi
	fi

	if ! python3 -c "$edit"; then
		printf 'BROKEN  %s\n        its mutation did not apply, so this case proved nothing\n' "$name"
		failures=$((failures + 1))
		restore
		return
	fi

	local out rc
	out=$(eval "$gate" 2>&1)
	rc=$?
	restore

	# Exit code first, then wording. Checking the needle before the expectation
	# turns "it did not fail at all" into "it failed for the wrong reason",
	# which sends the reader to look at prose when the gate is toothless.
	if [ "$expect" = fail ] && [ "$rc" -ne 0 ] && [ -n "$needle" ] &&
		! printf '%s' "$out" | grep -qF -- "$needle"; then
		printf 'WRONG REASON  %s\n              it failed, but not saying: %s\n' "$name" "$needle"
		failures=$((failures + 1))
		return
	fi
	if [ "$expect" = fail ] && [ "$rc" -eq 0 ]; then
		printf 'TOOTHLESS  %s\n           the gate passed on a fault it exists to catch\n' "$name"
		failures=$((failures + 1))
	elif [ "$expect" = pass ] && [ "$rc" -ne 0 ]; then
		printf 'OVEREAGER  %s\n           the gate failed on something it must not catch\n' "$name"
		failures=$((failures + 1))
		printf '%s\n' "$out" | head -4 | sed 's/^/           /'
	else
		printf 'ok  %-58s (%s)\n' "$name" "$expect"
	fi
}

py() { printf 'def edit(p, a, b):\n    s = open(p).read()\n    assert a in s, "pattern not found in " + p\n    open(p, "w").write(s.replace(a, b, 1))\n%s\n' "$1"; }

echo "=== faults each gate must catch ==="

# invariant: no network client at import time. A module-level import of httpx
# is the exact shape, and httpx is already an indirect dependency, so the
# import resolves and the mutation does not fail on a missing package.
run_case "no-network-at-write: a network client imported at module level" fail \
	'./scripts/no-network-at-write.sh' \
	"$(py 'edit("engram/cli.py", "import sys", "import sys\nimport httpx")')" \
	"at module level"

run_case "readme-numbers: a stale test count" fail \
	'./scripts/readme-numbers.sh' \
	"$(py 'import re
s = open("README.md").read()
m = re.search(r"### Test coverage \((\d+) tests, (\d+) with the encryption extra\)", s)
assert m, "no test-coverage heading in README.md"
open("README.md","w").write(s.replace(m.group(0), "### Test coverage (%d tests, %d with the encryption extra)" % (int(m.group(1))+7, int(m.group(2))+7), 1))')" \
	"tests"

# The DERIVED number, which is the one both places got wrong: the README said
# 15 points and the API reference said 18, while the table above each gave a
# gap of 14. A number computed from two published numbers is as unowned as they
# are, and this gate is the only thing that owns it.
run_case "readme-numbers: a derived recall gap that no longer follows" fail \
	'./scripts/readme-numbers.sh' \
	"$(py 'edit("docs/api-reference.md", "it is 14 points lower", "it is 19 points lower")')" \
	"points lower"

run_case "local-first: the stated first-use download stops matching" fail \
	'./scripts/local-first.sh' \
	"$(py 'edit("README.md", "downloads the ONNX embedding model (~64 MB)", "downloads the ONNX embedding model (~12 MB)")')" \
	"first-use download"

echo
echo "=== and what they must NOT catch ==="

# An import INSIDE a function is the whole point of the invariant: the network
# client may exist, it may not be reached at import time. A gate that fired on
# this would be deleted by whoever is unblocking CI.
run_case "no-network-at-write: a network client imported inside a function" pass \
	'./scripts/no-network-at-write.sh' \
	"$(py 'edit("engram/cli.py", "def main(", "def _unused_lazy_import():\n    import httpx\n\n    return httpx\n\n\ndef main(")')"

echo
echo "=== and the one this estate learned the hard way ==="
echo "    a gate whose subject is gone must SAY so, not report OK on nothing"

run_case "readme-numbers: the recall sweep is gone" fail \
	'./scripts/readme-numbers.sh' \
	"$(py 'import subprocess
subprocess.run(["git", "rm", "-q", "benchmarks/results/longmemeval_s_2026-07-30_weight_sweep.jsonl"], check=True)')" \
	"was not checked at all"

run_case "readme-numbers: the sweep is present but empty" fail \
	'./scripts/readme-numbers.sh' \
	"$(py 'open("benchmarks/results/longmemeval_s_2026-07-30_weight_sweep.jsonl", "w").write("")')" \
	"measured nothing"

run_case "local-first: the README stops stating a download size at all" fail \
	'./scripts/local-first.sh' \
	"$(py 'edit("README.md", "downloads the ONNX embedding model (~64 MB)", "downloads the ONNX embedding model")')" \
	"no longer states the first-use download size"

echo
if [ -n "$(git status --porcelain)" ]; then
	printf 'FAIL: this script left the tree dirty, so it cannot be trusted about anything above\n'
	git status --porcelain | head -5
	exit 1
fi

if [ "$failures" -gt 0 ]; then
	printf '%d of %d cases failed.\n' "$failures" "$cases"
	printf 'A gate that has quietly stopped catching anything looks exactly like a gate\n'
	printf 'with nothing to catch, and stays that way until the fault it guards ships.\n'
	exit 1
fi

printf 'OK: %d cases. Every gate fails on its own fault, passes on a non-fault,\n' "$cases"
printf '    and refuses to report success when it measured nothing.\n'
