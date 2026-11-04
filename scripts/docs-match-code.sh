#!/usr/bin/env bash
# Checks that the documents say what the code does about procedural memory:
# nothing. The store implements episodic and semantic memory, and
# `remember(kind="procedural")` is refused.
#
# WHY
#
# For months CLAUDE.md, README.md and SECURITY.md described engram as modelling
# "episodic, semantic and procedural memory" while `engram/mcp_server.py`
# refused the third kind with an error that pointed at a document that does not
# exist. Nothing compared the two: a reader of the README learned one thing and
# a caller of the tool learned the other. Both statements were written
# carefully, and neither could notice the other.
#
# HOW
#
# Two halves, and the gate fails if either is gone.
#
#   code:  engram/mcp_server.py still has the branch that refuses the
#          procedural kind. If somebody builds procedural memory, or removes
#          the refusal, the documents below become wrong in the other
#          direction, so this fails and says which sentence to revisit.
#   docs:  the word "procedural" appears in a document no more often than the
#          table below allows. Every allowed mention is a sentence that says it
#          is NOT built (or, in the README's test table, that a test holds the
#          refusal). A new mention fails until it is added to the table, which
#          is a deliberate act and a reviewable line, not an accident of
#          copying a diagram.
#
# Counting mentions is blunt on purpose. A smarter check would have to
# understand sentences, and the one that failed here was an alt text on an
# image and a text node inside an SVG, neither of which has a sentence.
#
# WHAT THIS DOES NOT DO
#
# It does not read the sentences. A document can still say something false
# without using the word. It catches the claim this repository actually made.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

# path -> the most mentions of the word it may carry. Absent means none.
# CHANGELOG.md and features/ are not scanned: history and scenarios may name it.
allowed() {
	case "$1" in
	README.md) echo 1 ;;
	CLAUDE.md) echo 6 ;;
	DESIGN.md) echo 3 ;;
	docs/api-reference.md) echo 4 ;;
	*) echo 0 ;;
	esac
}

fail=0
scanned=0

# Tracked files only: an untracked scratch file is not a document of this repo.
while IFS= read -r f; do
	[ -f "$f" ] || continue
	scanned=$((scanned + 1))
	n=$(grep -oi 'procedural' "$f" | wc -l | tr -d ' ')
	max=$(allowed "$f")
	if [ "$n" -gt "$max" ]; then
		printf 'CLAIM  %s\n       says "procedural" %d times, allowed %d\n' "$f" "$n" "$max"
		grep -ni 'procedural' "$f" | head -5 | sed 's/^/       /' | cut -c1-160
		printf '       the store does not implement procedural memory; say so, or build it and\n'
		printf '       change this table with the claim\n'
		fail=$((fail + 1))
	fi
done < <(git ls-files 'README.md' 'CLAUDE.md' 'DESIGN.md' 'DATA_FLOW.md' 'GETTING_STARTED.md' \
	'SECURITY.md' 'CONTRIBUTING.md' 'docs/*.md' 'docs/*.svg' 'assets/*.svg' \
	'pyproject.toml' 'components.json')

# The subject must exist, or "no document claims it" is true of nothing.
if [ ! -f README.md ] || [ "$scanned" -eq 0 ]; then
	echo "measured nothing: README.md or the document set is missing, which is a" >&2
	echo "failure of this script and not a clean bill of health." >&2
	exit 1
fi

if ! grep -q 'elif kind == "procedural":' engram/mcp_server.py 2>/dev/null; then
	printf 'CODE   engram/mcp_server.py no longer refuses kind="procedural"\n'
	printf '       either procedural memory was built, and every document above that says\n'
	printf '       it is not built is now wrong, or the refusal was removed; revisit both\n'
	fail=$((fail + 1))
fi

echo
printf 'docs-match-code: %d documents, %d broken\n' "$scanned" "$fail"
[ "$fail" -eq 0 ]
