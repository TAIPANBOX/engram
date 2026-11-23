#!/usr/bin/env bash
# Audit the installed environment against the PyPI advisory database.
#
# One file, called by the `security` job in ci.yml on every push and pull
# request and by vulnscan.yml once a week, so the ignore list and the reason
# behind each entry live in one place and the two runs cannot drift apart.
#
# Expects the environment the security job builds: Engram installed editable
# with the dev, langchain and llamaindex extras, and pip-audit beside it.
#
# The ignores:
#   PYSEC-2026-597: nltk, pulled transitively by the optional llamaindex extra
#     (llama-index-core), not by core Engram. No fixed version was published
#     when this was added; drop the ignore once nltk ships a patched release.
#   PYSEC-2026-3740 (@decided 2026-09-02: the ignore is added): the same nltk,
#     the same optional extra. A path traversal in nltk's model save/load
#     helpers; Engram's own code imports nltk nowhere and never calls those
#     helpers, and no patched nltk was published when this was added. Same rule:
#     drop the ignore the day nltk ships a fix.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v pip-audit >/dev/null 2>&1; then
  echo "FAIL: pip-audit is not installed, so nothing was audited"
  exit 1
fi

exec pip-audit --skip-editable \
  --ignore-vuln PYSEC-2026-597 \
  --ignore-vuln PYSEC-2026-3740
