#!/usr/bin/env bash
# Usage: ./run.sh questions.yaml answers.json
# Reads causal_graph.json from this repo. LLM settings come from
# BRINE_URL / BRINE_MODEL / BRINE_API_KEY in the environment.
#
# Two pipelines live in this repo. Pick one with PIPELINE (default: lane2):
#   PIPELINE=lane2 ./run.sh questions.yaml answers.json   walking skeleton in lane2/ (no LLM calls yet)
#   PIPELINE=agent ./run.sh questions.yaml answers.json   LLM agent in agent/ (2 LLM calls per question)
set -euo pipefail
cd "$(dirname "$0")"

# Local convenience only: .env is gitignored and never committed.
[ -f .env ] && source .env

QUESTIONS="${1:-questions.yaml}"
OUT="${2:-answers.json}"
PY="${PYTHON:-python3}"
PIPELINE="${PIPELINE:-lane2}"

# Use a local venv with pinned deps. Created once if missing.
if [ ! -x .venv/bin/python ]; then
  echo "[run.sh] creating .venv and installing pinned requirements" >&2
  "$PY" -m venv .venv
  .venv/bin/pip install --quiet -r requirements.txt
fi

case "$PIPELINE" in
  lane2) exec .venv/bin/python -m lane2.cli "$QUESTIONS" "$OUT" ;;
  agent) exec .venv/bin/python -m agent.run "$QUESTIONS" "$OUT" ;;
  *) echo "[run.sh] unknown PIPELINE='$PIPELINE' (use lane2 or agent)" >&2; exit 2 ;;
esac
