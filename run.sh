#!/usr/bin/env bash
# Usage: ./run.sh questions.yaml answers.json
# Reads causal_graph.json from this repo. LLM settings (later) come from
# BRINE_URL / BRINE_MODEL / BRINE_API_KEY in the environment.
set -euo pipefail
cd "$(dirname "$0")"

QUESTIONS="${1:-questions.yaml}"
OUT="${2:-answers.json}"
PY="${PYTHON:-python3}"

# Use a local venv with pinned deps. Created once if missing.
if [ ! -x .venv/bin/python ]; then
  echo "[run.sh] creating .venv and installing pinned requirements" >&2
  "$PY" -m venv .venv
  .venv/bin/pip install --quiet -r requirements.txt
fi

exec .venv/bin/python -m lane2.cli "$QUESTIONS" "$OUT"
