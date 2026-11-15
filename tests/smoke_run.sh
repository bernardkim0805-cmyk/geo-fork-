#!/usr/bin/env bash
# Smoke test for the judged entry point: runs ./run.sh exactly as an examiner
# would (from an unrelated directory, relative paths, no dev flags) on one
# question, then checks the output. Run before every push to main.
#
#   tests/smoke_run.sh                # ~2-3 min, mostly LLM latency
#   tests/smoke_run.sh --fresh        # also test a clean install: copies the
#                                     # repo without .venv, so run.sh must
#                                     # rebuild it from requirements.txt
#   tests/smoke_run.sh --id c7        # pick the question (default: first one)
#
# Checks: exit code 0, answers.json written, passes tests/validate_answers.py,
# and the LLM was really called (an unreachable LLM must not pass silently).
# Does NOT check whether answers are right; that's eval_decide.py's job.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FRESH=0
QID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --fresh) FRESH=1 ;;
    --id) QID="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Which repo copy to run. --fresh: a copy with no venv, caches or old answers.
TARGET="$REPO"
if [ "$FRESH" = 1 ]; then
  TARGET="$WORK/repo"
  rsync -a --exclude .venv --exclude __pycache__ --exclude 'answers*.json' \
        --exclude .git "$REPO/" "$TARGET/"
  echo "[smoke] fresh copy at $TARGET (no .venv)" >&2
fi

# Needs any python with PyYAML for the setup steps; the repo venv has it.
PYH="$REPO/.venv/bin/python"
[ -x "$PYH" ] || PYH="python3"

# Fail before the slow part if the LLM isn't configured.
(cd "$REPO" && "$PYH" -c "import sys; from lane2 import llm; sys.exit(0 if llm.configured() else 1)") \
  || fail "BRINE_URL / BRINE_MODEL / BRINE_API_KEY not set (env or .env)"

# One-question file in a directory that is NOT the repo.
CALLER="$WORK/caller"
mkdir -p "$CALLER"
"$PYH" - "$REPO/questions.yaml" "$CALLER/one.yaml" "$QID" <<'PY'
import sys, yaml
src, dst, qid = sys.argv[1], sys.argv[2], sys.argv[3]
data = yaml.safe_load(open(src, encoding="utf-8"))
items = data["questions"] if isinstance(data, dict) else data
pick = [q for q in items if str(q["id"]) == qid] if qid else items[:1]
if not pick:
    sys.exit(f"no question with id {qid!r} in {src}")
yaml.safe_dump({"questions": pick}, open(dst, "w", encoding="utf-8"), sort_keys=False)
print(f"[smoke] question: {pick[0]['id']}", file=sys.stderr)
PY

# The examiner's call: cd somewhere else, relative paths, output in a subdir.
echo "[smoke] running run.sh from $CALLER" >&2
START=$(date +%s)
set +e
(cd "$CALLER" && "$TARGET/run.sh" one.yaml out/answers.json) > "$WORK/run.log" 2>&1
RC=$?
set -e
echo "[smoke] run.sh finished in $(( $(date +%s) - START ))s, exit $RC" >&2
if [ "$RC" -ne 0 ]; then
  tail -20 "$WORK/run.log" >&2
  fail "run.sh exited $RC"
fi

ANS="$CALLER/out/answers.json"
[ -s "$ANS" ] || fail "run.sh exited 0 but wrote no $ANS"

# Format + edge ids + every question answered.
(cd "$REPO" && "$PYH" tests/validate_answers.py "$ANS" "$CALLER/one.yaml") \
  || fail "answers.json failed tests/validate_answers.py"

# The run must have reached the LLM; a fallback-only run would mask an outage.
"$PYH" - "$CALLER/out/answers.trace.json" <<'PY' || fail "LLM was never called (0 tokens); check BRINE endpoint"
import json, sys
traces = json.load(open(sys.argv[1], encoding="utf-8"))
errs = [t["id"] for t in traces if "error" in t]
if errs:
    sys.exit(f"[smoke] pipeline error on: {', '.join(errs)}")
tok = sum(t.get("tokens", {}).get("total", 0) for t in traces)
print(f"[smoke] LLM tokens used: {tok}", file=sys.stderr)
sys.exit(0 if tok > 0 else 1)
PY

echo "SMOKE PASS" >&2
