"""Check answers.json against the README format and the graph.

    python tests/validate_answers.py answers.json [questions.yaml]

Fails (exit 1) on any malformed entry, unknown edge id, or missing question.
>5% unparseable entries halves the automatic score, so run this every time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DIRECTIONS = {"increase", "decrease", "no_effect", "mixed", "unclear"}
CERTAINTIES = {"high", "moderate", "low", "very_low"}


def main() -> int:
    answers = json.loads(Path(sys.argv[1]).read_text())
    graph = json.loads((ROOT / "causal_graph.json").read_text())
    edge_ids = {e["id"] for e in graph["edges"]}
    errors = []

    if "team" not in answers or not isinstance(answers.get("answers"), list):
        print("top level must have 'team' and an 'answers' list"); return 1

    for a in answers["answers"]:
        aid = a.get("id", "?")
        err = lambda m: errors.append(f"{aid}: {m}")
        for k in ("id", "status", "answer", "confidence"):
            if k not in a:
                err(f"missing {k}")
        if not isinstance(a.get("confidence"), (int, float)) or not 0 <= a.get("confidence", -1) <= 1:
            err("confidence must be a number in [0, 1]")
        if a.get("status") == "answered":
            if a.get("direction") not in DIRECTIONS: err(f"bad direction {a.get('direction')!r}")
            if a.get("certainty") not in CERTAINTIES: err(f"bad certainty {a.get('certainty')!r}")
            if not a.get("edge_ids"): err("answered but no edge_ids")
            bad = [e for e in a.get("edge_ids", []) if e not in edge_ids]
            if bad: err(f"unknown edge ids {bad[:3]}")
        elif a.get("status") == "no_evidence":
            if a.get("direction") is not None: err("no_evidence must have direction null")
            if not a.get("searched_for"): err("no_evidence needs searched_for")
        else:
            err(f"bad status {a.get('status')!r}")
        for x in a.get("extras", []) or []:
            if not (ROOT / x).exists(): err(f"extras path missing: {x}")

    if len(sys.argv) > 2:
        qs = yaml.safe_load(Path(sys.argv[2]).read_text())
        qs = qs["questions"] if isinstance(qs, dict) else qs
        got = {a.get("id") for a in answers["answers"]}
        for q in qs:
            if str(q["id"]) not in got:
                errors.append(f"{q['id']}: no answer (scored wrong at full confidence)")

    n = len(answers["answers"])
    for e in errors:
        print("  ", e)
    print(f"{n} answers, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
