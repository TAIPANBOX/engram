#!/usr/bin/env bash
# Enforces invariant 1 of CLAUDE.md: no network call at write time.
#
# Storing a memory is local, always. Only reflect() may reach an external LLM,
# and only when the caller asks for it. The structural form of that promise is
# that no provider SDK or HTTP client is imported at module level anywhere in
# the package: a top-level import would load the SDK for everybody who imports
# engram, turn an optional capability into a hard requirement, and put a network
# dependency behind an `import engram`.
#
# The current code already satisfies this. engram/llm.py imports anthropic and
# openai inside the functions that use them, and engram/core.py imports
# engram.llm lazily rather than at module scope. This script keeps it that way.
#
# Uses Python's AST, not a regexp, because indentation is the entire
# distinction and a regexp would be fooled by an import inside a try block at
# module scope.
#
# This file is the ONE copy of this check. The local hook and CI both call it.
# Two copies of one check always diverge, so do not inline it anywhere.

set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import ast
import pathlib
import sys

# Anything that can open a socket. engram.llm itself is deliberately NOT here:
# it is our own module, it is safe to import, and what matters is that the SDKs
# inside it stay lazy.
NETWORK = {
    "anthropic",
    "openai",
    "ollama",
    "httpx",
    "requests",
    "aiohttp",
    "urllib3",
    "http",
    "socket",
}

fail = False

for path in sorted(pathlib.Path("engram").rglob("*.py")):
    tree = ast.parse(path.read_text(), filename=str(path))

    # Module-level statements only. An import nested in a function is exactly
    # what this invariant asks for, so we do not walk into those.
    for node in tree.body:
        candidates = [node]
        if isinstance(node, ast.Try):
            candidates = list(node.body) + list(node.orelse) + list(node.finalbody)
        if isinstance(node, ast.If):
            # `if TYPE_CHECKING:` blocks never execute at runtime, but anything
            # else guarded by an `if` at module scope does.
            test = node.test
            type_checking = (
                isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            ) or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if type_checking:
                continue
            candidates = list(node.body) + list(node.orelse)

        for stmt in candidates:
            names = []
            if isinstance(stmt, ast.Import):
                names = [a.name for a in stmt.names]
            elif isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.level == 0:
                names = [stmt.module]

            for name in names:
                root = name.split(".")[0]
                if root in NETWORK:
                    print(
                        f"FAIL: {path}:{stmt.lineno} imports '{root}' at module level"
                    )
                    fail = True

if fail:
    print()
    print("Writing a memory must never require the network. Import the provider")
    print("SDK inside the function that needs it, the way engram/llm.py already")
    print("does. See CLAUDE.md invariant 1.")
    sys.exit(1)

print("OK: no network client is imported at module level.")
PY
