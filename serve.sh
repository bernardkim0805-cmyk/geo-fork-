#!/usr/bin/env bash
# Usage: ./serve.sh [--port 8765]
# Dev only: opens the pipeline viewer with a Run button. Not used for judging.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  echo "[serve.sh] creating .venv and installing pinned requirements" >&2
  "$PY" -m venv .venv
  .venv/bin/pip install --quiet -r requirements.txt
fi
exec .venv/bin/python -m lane2.serve "$@"
