"""Entry point: python -m lane2.cli QUESTIONS.yaml ANSWERS.json [--debug]

Never lets one bad question kill the run: a missing answer is scored as
wrong at full confidence, so any crash becomes a low-confidence no_evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

from .decide import decide
from .graph import Graph
from .ground import ground
from .parse import parse
from .render import render
from .retrieve import retrieve
from .schema import Question

TEAM = "geo-fork"


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = data["questions"] if isinstance(data, dict) else data
    return [Question(id=str(q["id"]), text=str(q["question"])) for q in items]


def answer_one(q: Question, graph: Graph, debug: bool) -> dict:
    query = parse(q)
    g = ground(query, graph)
    ev = retrieve(g, graph)
    d = decide(query, g, ev, graph)
    out = render(q, query, d)
    if debug:
        print(f"[{q.id}] parse={query.parse_method} X={query.intervention!r} Y={query.outcome!r} "
              f"X_nodes={len(g.intervention_nodes)} Y_nodes={len(g.outcome_nodes)} "
              f"edges={len(ev.edge_ids)} -> {d.status}/{d.direction} conf={d.confidence}", file=sys.stderr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", type=Path)
    ap.add_argument("answers", type=Path)
    ap.add_argument("--debug", action="store_true", help="print per-question trace to stderr")
    args = ap.parse_args(argv)

    graph = Graph()
    answers = []
    for q in load_questions(args.questions):
        try:
            answers.append(answer_one(q, graph, args.debug))
        except Exception:
            traceback.print_exc()
            answers.append({"id": q.id, "status": "no_evidence",
                            "answer": "Internal error while answering this question.",
                            "direction": None, "confidence": 0.1, "certainty": None,
                            "edge_ids": [], "searched_for": "pipeline error before search completed"})

    args.answers.write_text(json.dumps({"team": TEAM, "answers": answers}, indent=2), encoding="utf-8")
    print(f"wrote {len(answers)} answers to {args.answers}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
