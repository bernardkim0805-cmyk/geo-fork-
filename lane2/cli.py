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
    return [Question(id=str(q["id"]), text=str(q["question"]), kind=str(q.get("kind") or ""))
            for q in items]


def answer_one(q: Question, graph: Graph, debug: bool) -> tuple[dict, dict]:
    query = parse(q, debug=debug)
    g = ground(query, graph, debug=debug)
    ev = retrieve(g, graph)
    d = decide(query, g, ev, graph)
    out = render(q, query, d)
    if debug:
        print(f"[{q.id}] parse={query.parse_method} X={query.intervention!r} Y={query.outcome!r} "
              f"X_nodes={len(g.intervention_nodes)} Y_nodes={len(g.outcome_nodes)} "
              f"edges={len(ev.edge_ids)} -> {d.status}/{d.direction} conf={d.confidence}", file=sys.stderr)
    trace = {
        "id": q.id,
        "question": q.text,
        "kind": q.kind,
        "parse": {"method": query.parse_method, "intervention": query.intervention,
                  "outcome": query.outcome, "tokens": query.tokens},
        "tokens": {"parse": query.tokens, "ground": g.tokens, "total": query.tokens + g.tokens},
        "ground": {
            "intervention_terms": g.intervention_terms,
            "outcome_terms": g.outcome_terms,
            "tokens": g.tokens,
            "intervention_nodes": [{"id": n, "name": graph.nodes[n]["name"]} for n in g.intervention_nodes[:10]],
            "outcome_nodes": [{"id": n, "name": graph.nodes[n]["name"]} for n in g.outcome_nodes[:10]],
        },
        "retrieve": {"edge_count": len(ev.edge_ids), "edge_ids": ev.edge_ids[:20]},
        "decide": {"status": d.status, "direction": d.direction, "certainty": d.certainty,
                   "confidence": d.confidence, "tally": d.tally},
        "answer": out,
    }
    return out, trace


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", type=Path)
    ap.add_argument("answers", type=Path)
    ap.add_argument("--debug", action="store_true", help="print per-question trace to stderr")
    ap.add_argument("--trace", type=Path, default=None, help="write per-question trace JSON for the UI")
    ap.add_argument("--only", default=None,
                    help="comma-separated question ids to run (e.g. b3,c7); others are skipped")
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        missing = wanted - {q.id for q in questions}
        if missing:
            ap.error(f"unknown question id(s): {', '.join(sorted(missing))}")
        questions = [q for q in questions if q.id in wanted]

    graph = Graph()
    answers, traces = [], []
    for q in questions:
        try:
            out, trace = answer_one(q, graph, args.debug)
            answers.append(out)
            traces.append(trace)
        except Exception:
            traceback.print_exc()
            err = {"id": q.id, "status": "no_evidence",
                   "answer": "Internal error while answering this question.",
                   "direction": None, "confidence": 0.1, "certainty": None,
                   "edge_ids": [], "searched_for": "pipeline error before search completed"}
            answers.append(err)
            traces.append({"id": q.id, "question": q.text, "error": traceback.format_exc()})

    args.answers.write_text(json.dumps({"team": TEAM, "answers": answers}, indent=2), encoding="utf-8")
    print(f"wrote {len(answers)} answers to {args.answers}", file=sys.stderr)

    trace_path = args.trace or args.answers.with_suffix(".trace.json")
    trace_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"wrote trace to {trace_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
