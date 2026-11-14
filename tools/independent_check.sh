#!/usr/bin/env bash
# Clean-checkout test on ARM64 Linux, the way the judges will run it.
# Run from the repo root on your Mac:   ./tools/independent_check.sh [branch]
#
# - clones the branch fresh FROM GITHUB (so only what is pushed is tested)
# - sets up with run.sh exactly as written, in a python:3.12-slim arm64 container
# - runs lane2 and agent on the dev set and the held-out set
# - validates the format and scores against tests/*gold.yaml
# The BRINE settings come from your .env and go in as environment variables only.
# They are never printed or written to a file.
set -euo pipefail
BRANCH="${1:-actual-parse}"
REPO="https://github.com/bernardkim0805-cmyk/geo-fork-.git"
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; source .env; set +a; }
for v in BRINE_URL BRINE_MODEL BRINE_API_KEY; do
  [ -n "${!v:-}" ] || { echo "missing $v (put it in .env)"; exit 2; }
done
mkdir -p outputs/clean_check
docker run --rm --platform linux/arm64 \
  -e BRINE_URL -e BRINE_MODEL -e BRINE_API_KEY -e BRANCH="$BRANCH" -e REPO="$REPO" \
  -v "$PWD/tests:/local_tests:ro" -v "$PWD/outputs/clean_check:/out" \
  python:3.12-slim bash -c '
set -uo pipefail
apt-get update -qq >/dev/null && apt-get install -y -qq git >/dev/null
git clone -q -b "$BRANCH" "$REPO" /w && cd /w
echo "== commit under test: $(git log --oneline -1)"
# scoring files only (not part of the submission): use local copies if not pushed yet
for f in score.py dev_gold.yaml heldout_gold.yaml heldout_questions.yaml; do
  [ -f tests/$f ] || cp /local_tests/$f tests/
done
for P in lane2 agent; do
  for SET in dev_questions.yaml tests/heldout_questions.yaml; do
    name=$P-$(basename $SET .yaml)
    echo; echo "########## $name"
    start=$(date +%s)
    PIPELINE=$P ./run.sh $SET /out/$name.json > /out/$name.log 2>&1; rc=$?
    echo "exit code $rc, $(( $(date +%s) - start ))s"; grep -iE "token|error|warning" /out/$name.log | tail -3
    [ -f /out/$name.json ] || { echo "NO OUTPUT FILE"; tail -20 /out/$name.log; continue; }
    .venv/bin/python tests/validate_answers.py /out/$name.json $SET | tail -2
    case $SET in dev*) G=tests/dev_gold.yaml;; *) G=tests/heldout_gold.yaml;; esac
    .venv/bin/python tests/score.py /out/$name.json $G
  done
done
' 2>&1 | tee outputs/clean_check/report.txt
echo; echo "Full report: outputs/clean_check/report.txt"
