#!/usr/bin/env bash
# Enforces invariant 4 of CLAUDE.md: local-first is a constraint, not a
# preference.
#
# The neighbouring check, no-network-at-write.sh, holds the SOURCE side: no
# provider SDK is imported at module level. This one holds the INSTALL side,
# which is a different promise and can break without a single import changing:
# somebody adds a dependency that wants a server, or the embedding path starts
# reaching out on every call, and nothing in the source looks wrong.
#
# Three things are checked.
#
#   1. The default install is an allow-list. `pip install engram` may pull
#      sqlite-vec, fastembed and rfc8785, and nothing else. Every LLM provider
#      lives behind an extra. An allow-list rather than a denylist on purpose:
#      a new dependency should have to be argued for, not merely fail to match
#      a list of things somebody thought of in 2026.
#
#   2. Once the embedding model is present, a full observe-and-recall cycle
#      opens ZERO sockets. This is the sentence that matters, and it is checked
#      by making socket creation raise rather than by watching traffic, so a
#      call that would have connected fails loudly instead of passing quietly
#      on a machine that happens to have no route.
#
#   3. The README's stated first-use download matches the bytes actually
#      fetched. It did not: the README said ~23 MB and the real fetch is 64 MB,
#      the whole of it `model_optimized.onnx`. That number is the first thing a
#      stranger reads about what installing costs them.
#
# WHAT THIS CHECK ESTABLISHED, AND WHY INVARIANT 4 CHANGED. Run with a cold
# cache and no network, `Engram(...).observe(...)` fails: fastembed fetches the
# ONNX model on first use. The invariant used to say that an embedding path
# requiring a network round trip "breaks the one sentence that describes this
# project", which read as an absolute and was not one. The README already said
# so plainly and the invariant had not caught up. It now states the real
# promise, which is the one this script holds: one fetch, then never again.
#
# COST. Parts 2 and 3 need the model on disk. On a machine that has ever run
# the suite it is already there. On a cold machine this script downloads it
# once, says so, and does not pretend it did nothing. That is why this runs in
# CI rather than in a pre-push hook: CI is the machine that can afford it.
#
# This file is the ONE copy of this check.

set -uo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
import os
import pathlib
import re
import shutil
import sys
import tempfile
import tomllib

problems = []

# ------------------------------------------------------- 1. the default install
LOCAL_ONLY = {"sqlite-vec", "fastembed", "rfc8785"}

meta = tomllib.load(open("pyproject.toml", "rb"))["project"]
defaults = meta.get("dependencies", [])
extras = meta.get("optional-dependencies", {})


def name_of(spec):
    return re.split(r"[<>=!~\[\s;]", spec.strip(), maxsplit=1)[0].lower()


declared = {name_of(d) for d in defaults}
unexpected = declared - LOCAL_ONLY
if unexpected:
    problems.append(
        f"the default install would pull {', '.join(sorted(unexpected))}, which is "
        f"not in the local-only allow-list ({', '.join(sorted(LOCAL_ONLY))}). If "
        f"this dependency really is local and required, add it to LOCAL_ONLY in "
        f"this script and say in the commit why every install now carries it. If "
        f"it is optional, it belongs in an extra."
    )

missing = LOCAL_ONLY - declared
if missing:
    problems.append(
        f"{', '.join(sorted(missing))} is in the allow-list but not in the default "
        f"dependencies. The allow-list has gone stale, which means it is no longer "
        f"measuring the install it claims to measure."
    )

# Every provider SDK must be reachable only through an extra.
PROVIDERS = {"anthropic", "openai", "google-genai", "ollama", "mcp"}
in_extras = {name_of(s) for specs in extras.values() for s in specs}
for p in sorted(PROVIDERS & declared):
    problems.append(
        f"{p} is a DEFAULT dependency. A provider SDK in the default install "
        f"turns an optional capability into a hard requirement for everybody."
    )

# ------------------------------------------------ 2. and 3. need the real model
# Warm the cache first, with the network allowed. Whether this downloads or
# finds the model already there is the difference between a cold and a warm
# machine, and it is stated rather than hidden.
sys.path.insert(0, ".")
cache_root = pathlib.Path(os.environ.get("FASTEMBED_CACHE_PATH", tempfile.gettempdir())) / "fastembed_cache"
was_present = cache_root.exists() and any(cache_root.glob("models--*"))
if not was_present:
    print("    (embedding model not on this machine, fetching it once)")

try:
    import engram

    warm = engram.Engram(":memory:", agent_id="gate-warm")
    warm.observe("warming the embedder so the offline half can mean something")
    warm.close()
except Exception as exc:  # noqa: BLE001
    problems.append(
        f"could not warm the embedder, so parts 2 and 3 measured nothing: "
        f"{type(exc).__name__}: {exc}"
    )
    for p in problems:
        print(f"FAIL: {p}")
    sys.exit(1)

# ---------------------------------------------------------- 2. zero sockets now
import socket  # noqa: E402


class Blocked(RuntimeError):
    pass


class NoSocket(socket.socket):
    def __init__(self, *a, **k):
        raise Blocked("this path opened a socket")


socket.socket = NoSocket
socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(Blocked("create_connection"))
socket.getaddrinfo = lambda *a, **k: (_ for _ in ()).throw(Blocked("a DNS lookup"))

workdir = tempfile.mkdtemp()
try:
    store = engram.Engram(os.path.join(workdir, "gate.db"), agent_id="gate-offline")
    store.observe("the gateway binds to loopback by default")
    hits = store.recall("what does the gateway bind to")
    store.close()
    if not hits:
        problems.append(
            "the offline cycle ran without opening a socket but recalled nothing, "
            "so it did not exercise the embedding path and proves less than it looks"
        )
except Blocked as exc:
    problems.append(
        f"a write-and-recall cycle tried to reach the network ({exc}) on a machine "
        f"where the model is already present. Local-first means the fetch happens "
        f"once, at first use, and never again."
    )
except Exception as exc:  # noqa: BLE001
    problems.append(f"the offline cycle failed: {type(exc).__name__}: {exc}")
finally:
    socket.socket = socket.SocketType
    shutil.rmtree(workdir, ignore_errors=True)

# ------------------------------------------------- 3. the number in the README
model_dirs = sorted(cache_root.glob("models--*")) if cache_root.exists() else []
if not model_dirs:
    problems.append(
        f"no model found under {cache_root} after warming, so the download size "
        f"could not be measured. This check cannot confirm the README's number."
    )
else:
    # Sum unique blobs. The snapshot entries are symlinks into blobs/, so
    # walking everything would count the model twice.
    seen = {}
    for f in model_dirs[0].rglob("*"):
        if f.is_file() and not f.is_symlink():
            seen[f.resolve()] = f.stat().st_size
    measured_mb = sum(seen.values()) / 1_048_576

    readme = pathlib.Path("README.md").read_text()
    m = re.search(r"downloads the ONNX embedding model \(~?([\d.]+)\s*MB\)", readme)
    if not m:
        problems.append(
            "README no longer states the first-use download size in the form this "
            "check reads. The sentence is how a stranger learns what installing "
            "costs them; keep it, and keep it in a form that can be checked."
        )
    else:
        claimed = float(m.group(1))
        if abs(claimed - measured_mb) > max(2.0, measured_mb * 0.15):
            problems.append(
                f"README says the first-use download is ~{claimed:g} MB. Measured "
                f"{measured_mb:.1f} MB in {model_dirs[0].name}. This is the first "
                f"thing a stranger reads about what installing costs them."
            )

if problems:
    for p in problems:
        print(f"FAIL: {p}")
    print()
    print("Local-first is what this project is. See CLAUDE.md invariant 4.")
    sys.exit(1)

print(f"OK: default install is {', '.join(sorted(declared))}, all local;")
print("    a full observe-and-recall cycle opens no socket once the model is present;")
print(f"    README's stated first-use download matches the {measured_mb:.0f} MB actually fetched.")
PY
