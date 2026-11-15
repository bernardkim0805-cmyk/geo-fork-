#!/usr/bin/env bash
# Usage: ./run.sh questions.yaml answers.json
# Reads causal_graph.json from this repo. LLM settings (later) come from
# BRINE_URL / BRINE_MODEL / BRINE_API_KEY in the environment.
set -euo pipefail

# Resolve paths the caller passed against the caller's cwd, before we cd into
# the repo. Without this, `/path/to/run.sh q.yaml out.json` run from another
# directory looks for q.yaml inside the repo and fails.
abspath() {
  local dir; dir="$(dirname "$1")"
  mkdir -p "$dir"
  echo "$(cd "$dir" && pwd)/$(basename "$1")"
}
QUESTIONS="${1:+$(abspath "$1")}"
OUT="${2:+$(abspath "$2")}"

cd "$(dirname "$0")"
# No arguments: fall back to the repo's own files, as before.
QUESTIONS="${QUESTIONS:-questions.yaml}"
OUT="${OUT:-answers.json}"
PY="${PYTHON:-python3}"

# Use a local venv with pinned deps. Created once if missing.
if [ ! -x .venv/bin/python ]; then
  echo "[run.sh] creating .venv and installing pinned requirements" >&2
  "$PY" -m venv .venv
  .venv/bin/pip install --quiet -r requirements.txt
fi

exec .venv/bin/python -m lane2.cli "$QUESTIONS" "$OUT"
