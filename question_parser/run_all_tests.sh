#!/usr/bin/env bash
# Runs every test suite. Needs data/graph.json, data/questions.md, data/questions_varied.md,
# data/questions_runon.md. Expect: 0 wrong everywhere, 0 attacks fooled, all checks ok.
set -e
cd "$(dirname "$0")"
echo "== bank 1";          python run_bank.py | tail -2
echo "== varied + messy";  python messy_test.py | grep -v "^  " | tail -15
echo "== run-on";          python run_runon.py
echo "== rules 10k";     python stress_test_v2.py 0 10000 | grep -E "TOTAL|false passes"
echo "== attacks";         python attack_test.py | tail -1; python attack_round2.py | tail -1
echo "== audit";           python audit_test.py
echo "== edge cases";      python edge_test.py
