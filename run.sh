#!/usr/bin/env bash
# Usage: ./run.sh questions.yaml answers.json
# Needs BRINE_URL, BRINE_MODEL, BRINE_API_KEY in the environment.
set -euo pipefail
cd "$(dirname "$0")"
# Local convenience only: .env is gitignored and never committed.
[ -f .env ] && source .env
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --quiet -r requirements.txt
fi
.venv/bin/python -m agent.run "${1:-questions.yaml}" "${2:-answers.json}"
