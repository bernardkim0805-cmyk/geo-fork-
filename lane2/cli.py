"""Entry point: python -m lane2.cli QUESTIONS.yaml ANSWERS.json [--debug]

Never lets one bad question kill the run: a missing answer is scored as
wrong at full confidence, so any crash becomes a low-confidence no_evidence.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import traceback
from pathlib import Path

import yaml

from . import llm
from . import paper_retrieve
from . import roles
from .decide import CONFIG as DECIDE_CONFIG, LLM_STATS, decide, role_items
from .graph import Graph
from .live import Live
from .parse import parse
from .render import render
from .retrieve import retrieve
from .schema import Evidence, Grounding, Query, Question
from .uncertainty import band, score_uncertainty

TEAM = "geo-fork"


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = data["questions"] if isinstance(data, dict) else data
    return [Question(id=str(q["id"]), text=str(q["question"]), kind=str(q.get("kind") or ""))
            for q in items]


def _edge_detail(graph: Graph, eid: str, groups: dict | None = None) -> dict:
    """Flatten one edge into what the UI needs to draw and label it."""
    e = graph.edges[eid]
    subj, obj = graph.nodes.get(e["subject"], {}), graph.nodes.get(e["object"], {})
    return {
        "id": eid,
        "subject": {"id": e["subject"], "name": subj.get("name"),
                    "type": subj.get("entity_type")},
        "object": {"id": e["object"], "name": obj.get("name"),
                   "type": obj.get("entity_type"),
                   "qualifier": obj.get("state_or_change_qualifier")},
        "predicate": e.get("predicate"),
        "negated": e.get("negated"),
        "certainty": e.get("certainty_grade"),
        "causal_language": e.get("causal_language"),
        "context": e.get("ecosystem_context"),
        "strength": (e.get("strength") or {}).get("qualitative_descriptor"),
        "evidence": ((e.get("evidential_basis") or {}).get("evidence_types") or [None])[0],
        "sentence": (e.get("original_sentence") or "")[:300],
        "doc": e.get("source_document"),
        # which outcome phrase this edge answers; an edge can answer several
        "answers": sorted(p for p, nodes in (groups or {}).items()
                          if e["object"] in set(nodes)),
    }


def _llm_tokens() -> int:
    return llm.USAGE["prompt_tokens"] + llm.USAGE["completion_tokens"]


def decide_counted(query, g, ev, graph, edge_q=None):
    """decide() plus the LLM tokens its role-check / D2 calls spent (0 when cached)."""
    before = _llm_tokens()
    d = decide(query, g, ev, graph, edge_q)
    return d, _llm_tokens() - before


def _vote_detail(graph: Graph, v) -> dict:
    """One edge's vote, with the node names and qualifiers that produced it."""
    e = graph.edges.get(v.edge_id) or {}
    subj, obj = graph.nodes.get(e.get("subject"), {}), graph.nodes.get(e.get("object"), {})
    return {
        "id": v.edge_id, "doc": v.doc, "direction": v.direction, "weight": v.weight,
        "factors": v.extra, "reason": v.reason, "grade": v.grade,
        "head_to_head": v.head_to_head, "associative": v.associative,
        "flipped": v.flipped_subject,
        "subject": subj.get("name"), "subject_q": subj.get("state_or_change_qualifier"),
        "predicate": e.get("predicate"), "negated": e.get("negated"),
        "object": obj.get("name"), "object_q": obj.get("state_or_change_qualifier"),
        "sentence": (e.get("original_sentence") or "")[:300],
    }


def retrieve_pair(x: str, y: str, raw: str, graph: Graph, mode: str = "effect",
                  say=None) -> dict:
    """Ground and retrieve one cause->effect pair (no judging yet)."""
    say = say or (lambda *a, **k: None)
    sub = Query(x, y, raw=raw, parse_method="pair", mode=mode)
    ev = retrieve(sub, graph, on_grounded=lambda g: say("retrieve"))
    return {"intervention": x, "outcome": y, "query": sub, "grounding": ev.grounding,
            "evidence": ev}


def decide_pair(pr: dict, graph: Graph, debug: bool) -> dict:
    """Judge one retrieved pair on its own."""
    g, ev = pr["grounding"], pr["evidence"]
    d, dtok = decide_counted(pr["query"], g, ev, graph)
    if debug:
        print(f"    [pair] {pr['intervention']!r} -> {pr['outcome']!r}: "
              f"nodes={len(g.intervention_nodes)}x{len(g.outcome_nodes)} "
              f"edges={len(ev.edge_ids)} -> {d.status}/{d.direction}", file=sys.stderr)
    pr.update(decision=d, decide_tokens=dtok)
    return pr


def run_pair(x: str, y: str, raw: str, graph: Graph, debug: bool,
             mode: str = "effect", say=None) -> dict:
    """Ground, retrieve and judge one cause->effect pair on its own."""
    return decide_pair(retrieve_pair(x, y, raw, graph, mode, say), graph, debug)


def edge_queries(pairs: list[dict]) -> dict:
    """Edge id -> the sub-question that ranked it highest. The combined pass reads
    each edge against its own X and Y, not the question's first ones."""
    best: dict = {}
    for p in pairs:
        ev = p["evidence"]
        for eid in ev.edge_ids:
            s = ev.scores.get(eid, 0.0)
            if eid not in best or s > best[eid][0]:
                best[eid] = (s, p["query"])
    return {eid: q for eid, (s, q) in best.items()}


def _dedupe(seqs) -> list:
    out, seen = [], set()
    for seq in seqs:
        for v in seq:
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def combine_groundings(gs: list, query: Query) -> Grounding:
    """Union of the pairs' groundings, for the single answers.json entry."""
    if not gs:
        return Grounding(intervention_terms=query.interventions,
                         outcome_terms=query.outcomes)
    groups: dict[str, list[str]] = {}
    for g in gs:
        for phrase, nodes in g.outcome_groups.items():
            groups.setdefault(phrase, [])
            groups[phrase] = _dedupe([groups[phrase], nodes])
    return Grounding(
        intervention_nodes=_dedupe(g.intervention_nodes for g in gs),
        outcome_nodes=_dedupe(g.outcome_nodes for g in gs),
        intervention_terms=_dedupe(g.intervention_terms for g in gs),
        outcome_terms=_dedupe(g.outcome_terms for g in gs),
        tokens=sum(g.tokens for g in gs),
        contexts=_dedupe(g.contexts for g in gs),
        stats={"pairs": len(gs),
               "pairs_grounded": sum(1 for g in gs
                                     if g.intervention_nodes and g.outcome_nodes)},
        outcome_groups=groups,
    )


def combine_evidence(evs: list) -> Evidence:
    """Union of the pairs' edges, best-ranked first across all of them."""
    if not evs:
        return Evidence()
    scores: dict[str, float] = {}
    found_by: dict[str, list[str]] = {}
    for ev in evs:
        for eid in ev.edge_ids:
            s = ev.scores.get(eid, 0.0)
            if eid not in scores or s > scores[eid]:
                scores[eid] = s
            found_by.setdefault(eid, ev.found_by.get(eid, []))
    ranked = sorted(scores, key=lambda e: (-scores[e], e))
    return Evidence(
        edge_ids=ranked,
        stats={"pairs": len(evs), "total": len(ranked),
               "passes": evs[0].stats.get("passes", []),
               "contexts": _dedupe(ev.stats.get("contexts", []) for ev in evs)},
        found_by=found_by,
        out_of_context=_dedupe(ev.out_of_context for ev in evs),
        reverse_edge_ids=_dedupe(ev.reverse_edge_ids for ev in evs),
        scores=scores,
    )


def pair_sentences(pairs: list[dict], base: str) -> str:
    """Spell out each pair, since a general term and a specific example can
    disagree and the single direction field cannot say so."""
    if len(pairs) < 2:
        return base
    lines = []
    for p in pairs:
        d = p["decision"]
        n = len(d.edge_ids)
        if d.status != "answered":
            lines.append(f"{p['intervention']} on {p['outcome']}: no evidence found")
        else:
            lines.append(f"{p['intervention']} on {p['outcome']}: {d.direction}"
                         f" ({n} edge{'s' if n != 1 else ''}, certainty {d.certainty})")
    return base + " Per sub-question: " + "; ".join(lines) + "."


def answer_one(q: Question, graph: Graph, debug: bool,
               check_uncertainty: bool = False, live: Live | None = None) -> tuple[dict, dict]:
    def say(stage, **info):  # progress for the live viewer; no-op without --live
        if live:
            live.stage(q.id, stage, **info)

    say("parse")
    query = parse(q, debug=debug)
    parse_trace = {"method": query.parse_method, "intervention": query.intervention,
                   "outcome": query.outcome, "tokens": query.tokens,
                   "interventions": query.interventions, "outcomes": query.outcomes,
                   "pairs": [list(p) for p in query.pairs],
                   "mode": query.mode, "presumed": query.presumed}
    d2_before = dict(LLM_STATS)
    roles_before = dict(roles.STATS)
    # Advisory only: scores the parse, never changes it. Off unless --uncertainty.
    if check_uncertainty:
        u_score, u_reason, u_tokens, u_cached = score_uncertainty(q.text, query)
    else:
        u_score, u_reason, u_tokens, u_cached = None, "uncertainty check disabled", 0, False
    # One search per cause->effect pair. "herbicides such as glyphosate" is two
    # questions: a finding can hold for glyphosate and not for herbicides
    # generally, and a single merged search cannot tell them apart.
    say("ground", parse=parse_trace, pair=None)
    pairs = []
    for i, (x, y) in enumerate(query.pairs):
        say("ground", pair=i)
        pairs.append(retrieve_pair(x, y, query.raw, graph, query.mode, say))
    # Retrieval by meaning: the model reads paper titles, then the chosen papers'
    # edges, and its picks replace what the keyword passes found for each pair.
    # RETRIEVE_LLM=only (default) | union (add to keyword edges) | off (keyword only).
    llm_mode = os.environ.get("RETRIEVE_LLM", "only").lower()
    llm_info = None
    if llm_mode in ("union", "only") and pairs:
        say("retrieve")
        llm_info = paper_retrieve.apply(query, pairs, graph, llm_mode)
    ltok = llm_info["tokens"]["total"] if llm_info else 0
    # D5: one batched role-check call for the whole question; every decide below
    # (each pair, then the combined pass) reads from its cache.
    rtok = _llm_tokens()
    if DECIDE_CONFIG["role_check"]:
        roles.prefetch([it for p in pairs for it in role_items(
            p["evidence"].edge_ids, graph, p["query"])])
    rtok = _llm_tokens() - rtok
    for i, pr in enumerate(pairs):
        decide_pair(pr, graph, debug)
        if live:
            live.pair_done(q.id, {"intervention": pr["intervention"], "outcome": pr["outcome"],
                                  "nodes": [len(pr["grounding"].intervention_nodes),
                                            len(pr["grounding"].outcome_nodes)],
                                  "edges": len(pr["evidence"].edge_ids),
                                  "status": pr["decision"].status,
                                  "direction": pr["decision"].direction})
    say("decide", pair=None)
    g = combine_groundings([p["grounding"] for p in pairs], query)
    ev = combine_evidence([p["evidence"] for p in pairs])
    d, dtok = decide_counted(query, g, ev, graph, edge_queries(pairs))
    dtok += sum(p["decide_tokens"] for p in pairs) + rtok
    d2 = {k: LLM_STATS[k] - d2_before[k] for k in LLM_STATS}
    role_stats = {k: roles.STATS[k] - roles_before[k] for k in roles.STATS}
    out = render(q, query, d)
    if pairs:
        out["answer"] = pair_sentences(pairs, out.get("answer", ""))
    if debug:
        print(f"[{q.id}] parse={query.parse_method} X={query.intervention!r} Y={query.outcome!r} "
              f"X_nodes={len(g.intervention_nodes)} Y_nodes={len(g.outcome_nodes)} "
              f"edges={len(ev.edge_ids)} -> {d.status}/{d.direction} conf={d.confidence}"
              + (f" unc={u_score}" if u_score is not None else ""), file=sys.stderr)
    trace = {
        "id": q.id,
        "question": q.text,
        "kind": q.kind,
        "parse": parse_trace,
        # one entry per cause->effect sub-question, judged on its own
        "pairs": [{
            "intervention": p["intervention"], "outcome": p["outcome"],
            "nodes": [len(p["grounding"].intervention_nodes),
                      len(p["grounding"].outcome_nodes)],
            "anchor": [p["grounding"].stats.get("intervention", {}).get("anchor"),
                       p["grounding"].stats.get("outcome", {}).get("anchor")],
            "edges": len(p["evidence"].edge_ids),
            "status": p["decision"].status, "direction": p["decision"].direction,
            "certainty": p["decision"].certainty,
            "confidence": p["decision"].confidence,
            "tally": p["decision"].tally,
        } for p in pairs],
        # Shady's dev-only parse check (branch actual-parse). Always present so
        # the UI never has to test for the key; score is null when off or failed.
        "uncertainty": {"score": u_score, "reason": u_reason,
                        "cached": u_cached, "band": band(u_score)},
        "tokens": {"parse": query.tokens, "ground": g.tokens,
                   "retrieve_llm": ltok,
                   "decide": dtok, "uncertainty": u_tokens,
                   "total": query.tokens + g.tokens + ltok + dtok + u_tokens},
        # what the LLM retrieval pass picked; null when RETRIEVE_LLM is off
        "llm_retrieve": llm_info,
        "ground": {
            "intervention_terms": g.intervention_terms,
            "outcome_terms": g.outcome_terms,
            "tokens": g.tokens,
            "contexts": g.contexts,
            # true totals; the node lists below are truncated for display
            "intervention_count": len(g.intervention_nodes),
            "outcome_count": len(g.outcome_nodes),
            "stats": g.stats,
            "intervention_nodes": [{"id": n, "name": graph.nodes[n]["name"]} for n in g.intervention_nodes[:10]],
            "outcome_nodes": [{"id": n, "name": graph.nodes[n]["name"]} for n in g.outcome_nodes[:10]],
        },
        "retrieve": {
            "edge_count": len(ev.edge_ids),
            "edge_ids": ev.edge_ids[:20],
            "edges": [_edge_detail(graph, eid, g.outcome_groups)
                      for eid in ev.edge_ids[:40]],
            "stats": ev.stats,
            "found_by": ev.found_by,
            "out_of_context": ev.out_of_context,
            "reverse": [_edge_detail(graph, eid) for eid in ev.reverse_edge_ids[:10]],
        },
        "decide": {"status": d.status, "direction": d.direction, "certainty": d.certainty,
                   "confidence": d.confidence, "tally": d.tally,
                   "notes": d.notes, "searched_for": d.searched_for,
                   "cited": len(d.edge_ids),
                   "votes": [_vote_detail(graph, v) for v in d.votes],
                   "explain": d.explain,
                   # D2: removal-subject LLM check, this question only
                   "d2": d2,
                   # D5: batched role check, this question only
                   "roles": role_stats},
        "answer": out,
    }
    return out, trace


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", type=Path)
    ap.add_argument("answers", type=Path)
    ap.add_argument("--debug", action="store_true", help="print per-question trace to stderr")
    ap.add_argument("--trace", type=Path, default=None, help="write per-question trace JSON for the UI")
    ap.add_argument("--uncertainty", action="store_true",
                    help="DEV ONLY: score each parse 0-100 with the LLM (one extra call per question)")
    ap.add_argument("--live", action="store_true",
                    help="write ANSWERS.live.json after every stage so ui.html can show progress")
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
    live = Live(args.answers.with_suffix(".live.json"), questions) if args.live else None
    answers, traces = [], []
    for q in questions:
        try:
            out, trace = answer_one(q, graph, args.debug, args.uncertainty, live)
            answers.append(out)
            traces.append(trace)
            if live:
                live.finish(q.id, trace)
        except Exception:
            traceback.print_exc()
            err = {"id": q.id, "status": "no_evidence",
                   "answer": "Internal error while answering this question.",
                   "direction": None, "confidence": 0.1, "certainty": None,
                   "edge_ids": [], "searched_for": "pipeline error before search completed"}
            answers.append(err)
            traces.append({"id": q.id, "question": q.text, "error": traceback.format_exc()})
            if live:
                live.finish(q.id, traces[-1], error=True)

    args.answers.write_text(json.dumps({"team": TEAM, "answers": answers}, indent=2), encoding="utf-8")
    print(f"wrote {len(answers)} answers to {args.answers}", file=sys.stderr)

    trace_path = args.trace or args.answers.with_suffix(".trace.json")
    trace_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"wrote trace to {trace_path}", file=sys.stderr)
    if live:
        live.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
