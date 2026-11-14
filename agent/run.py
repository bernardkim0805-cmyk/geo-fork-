"""Answer questions from causal_graph.json.

Usage:  python -m agent.run questions.yaml answers.json

Per question:
  1. LLM call "plan": turn the question into search terms, split into the HEAD of each phrase
     (what is measured or done) and its MODIFIERS. Code then checks the plan against the graph
     (agent/plancheck.py) and drops terms that drift to another concept.
  2. Code: find edges whose SUBJECT matches the cause side and whose OBJECT matches the outcome
     side. Edges that match only the other way round are counted, never kept.
  3. LLM call "filter": keep the edges that really bear on the question.
  4. Code: drop edges about second-order properties (variability, similarity to a reference),
     normalise polarity (depth vs level, mortality vs survival), dedupe, then one vote per paper.
  5. Template answer + a trace file in outputs/traces/ that shows every step.
"""
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import yaml

from agent import llm
from agent.graph import Graph, aggregate, confidence_no_evidence, flip
from agent.plancheck import PlanChecker

TEAM = "Team Geo"
MAX_CANDIDATES = 80
TRACE_DIR = "outputs/traces"

PLAN_SYSTEM = """You turn a question about an ecological restoration evidence graph into search terms.
The graph has edges: SUBJECT (an intervention or condition) -> OBJECT (a measured outcome).
Node text looks like: "Increased native forb cover | entity: native forb | attribute: cover".
Return JSON only:
{"cause":   {"label": str, "head": str, "head_terms": [str], "modifier_terms": [str], "exclude": [str]},
 "outcome": {"label": str, "head": str, "head_terms": [str], "modifier_terms": [str], "exclude": [str]}}
Rules:
- label: the phrase exactly as the question says it.
- head: the head noun of the phrase, the thing measured or done ("peatland restoration" ->
  restoration; "water table position" -> position; "seedling survival" -> survival; "mowing" -> mowing).
- head_terms: lowercase word stems: the original phrase first, then the head, then 2-6 synonyms
  of the SAME variable or process in this context (position -> level, height, depth; survival ->
  mortality; restoration -> rewetting, rehabilitation). Stems match the start of a word
  ("herbicid" matches herbicides). Add specific examples (herbicide -> glyphosate, picloram).
- modifier_terms: stems for the other words of the phrase and their synonyms ("peatland" ->
  peatland, peat, bog, fen; "water table" -> water table, groundwater). Empty for a one-word phrase.
- For an intervention, head_terms also list the names papers use for the same action
  (peatland rewetting -> rewetting, restoration, reclamation, ditch blocking; herbicide -> glyphosate).
- Never add actions or interventions that change a VARIABLE (for a water table: not rewetting,
  drainage, ditch blocking), never things it causes, never broader umbrella terms (hydrology),
  never its opposite.
- Use the term "native" (exactly) when the phrase is about native species; exclude: stems that
  would wrongly match (for natives: invasive, exotic, non-native)."""

FILTER_SYSTEM = """You check candidate edges from an evidence graph against a question.
You are given the intervention and outcome being asked about, then lines "index: SUBJECT -> OBJECT".
Be strict: when unsure, leave the edge out. A line tagged "[not stated: ...]" names context words
from the question that the graph text does not mention; keep the edge only if the question's
meaning still clearly applies (a planted white pine IS a seedling; a bog IS a peatland).
Keep an edge only if SUBJECT is an instance of the question's intervention/condition AND
OBJECT measures the question's outcome (or its direct opposite).
Watch for traps: native vs non-native/invasive are DIFFERENT outcomes; a subject that is
the absence/control of the intervention does not count; REVERSED edges: if the SUBJECT is an
action or process that brings about the question's cause (rewetting when the cause asked about
is water table position), the edge runs the wrong way for this question: reject it.
"flip" lists kept edges whose OBJECT measures the OPPOSITE quantity of the asked outcome
(e.g. mortality when the question asks about survival, bare ground when it asks about
plant cover). Do not flip for any other reason.
Return JSON only: {"keep": [int], "flip": [int]}"""


def plan(question):
    return llm.chat_json(PLAN_SYSTEM, question, max_tokens=400)


def filter_edges(graph, question, p, edges, partial_ids=()):
    """`partial_ids`: candidates whose context words (modifiers) the graph never states; the
    filter is told so it can judge from the question whether the edge is still about them."""
    unstated = ", ".join(p["cause"].get("modifier_terms", [])[:2] + p["outcome"].get("modifier_terms", [])[:2])
    lines = []
    for i, e in enumerate(edges):
        s, o = graph.nodes[e["subject"]], graph.nodes[e["object"]]
        tag = f"  [not stated: {unstated}]" if e["id"] in partial_ids and unstated else ""
        lines.append(f"{i}: {s['name']} -> {o['name']}{tag}")
    user = (f"Question: {question}\nIntervention: {p['cause'].get('label')}\n"
            f"Outcome: {p['outcome'].get('label')}\n\nCandidates:\n" + "\n".join(lines))
    out = llm.chat_json(FILTER_SYSTEM, user, max_tokens=60 + 4 * len(edges))
    keep = [edges[i] for i in out.get("keep", []) if isinstance(i, int) and 0 <= i < len(edges)]
    flips = {edges[i]["id"] for i in out.get("flip", [])
             if isinstance(i, int) and 0 <= i < len(edges)}
    return keep, flips & {e["id"] for e in keep}


def searched_for(p, had_candidates):
    c, o = p["cause"], p["outcome"]
    tail = (" Candidate edges matched these terms but none survived the relevance, attribute and "
            "orientation checks." if had_candidates else " No edge matched both sides.")
    return (f"edges whose subject node is '{c.get('phrase') or c.get('label')}' "
            f"(head terms: {', '.join(c.get('head_terms', []))}; context: {', '.join(c.get('modifier_terms', [])) or 'none'}) "
            f"and whose object node measures '{o.get('phrase') or o.get('label')}' "
            f"(head terms: {', '.join(o.get('head_terms', []))}; context: {', '.join(o.get('modifier_terms', [])) or 'none'})."
            + tail)


WORDS = {"increase": "increases", "decrease": "decreases", "no_effect": "has no clear effect on",
         "mixed": "has mixed effects on", "unclear": "has an unclear effect on"}


def write_answer(graph, p, agg, kept):
    c, o = p["cause"]["label"], p["outcome"]["label"]
    pc = agg["paper_counts"]
    tally = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(pc.items()))
    best = sorted(kept, key=lambda e: ["very_low", "low", "moderate", "high"].index(
        e.get("certainty_grade") or "low"), reverse=True)[0]
    doc = graph.docs.get(best["source_document"], {})
    cite = f"{(doc.get('authors') or ['?'])[0].split(',')[0]} {doc.get('year', '')}".strip()
    quote = best["original_sentence"].strip().replace("\n", " ")
    if len(quote) > 220:
        quote = quote[:217] + "..."
    return (f"{c} {WORDS[agg['direction']]} {o}, based on {agg['n_edges']} distinct claims from "
            f"{agg['n_papers']} papers (one vote per paper: {tally}). "
            f'Example ({cite}): "{quote}"')


def answer_one(graph, checker, q):
    qid, question = q["id"], " ".join(str(q["question"]).split())
    trace = {"id": qid, "question": question}
    try:
        p = checker.validate(plan(question), question)
        trace["plan"] = p
        cands, partial_ids, reversed_n = graph.candidates(p)
        trace["n_candidates"] = len(cands)
        trace["modifier_unmatched"] = partial_ids
        trace["reversed_rejected"] = reversed_n
        cands = graph.spread_by_paper(cands, MAX_CANDIDATES)
        kept, llm_flips = filter_edges(graph, question, p, cands, set(partial_ids)) if cands else ([], set())
        trace["llm_rejected"] = [e["id"] for e in cands if e not in kept]

        # fix 2.6: only the asked-about attribute counts
        excluded, kept2 = [], []
        for e in kept:
            reason = graph.second_order(e, question)
            if reason:
                excluded.append({"id": e["id"], "object": graph.nodes[e["object"]]["name"], "reason": reason})
            else:
                kept2.append(e)
        trace["excluded"] = excluded

        # fix 2.5: polarity against the asked attribute; the LLM's flip only where the table is silent
        asked = graph.asked_class(p["outcome"])
        trace["asked_attribute"] = list(asked) if asked else None
        dirs, info = {}, {}
        for e in kept2:
            raw, final, inverted = graph.normalise_direction(e, asked)
            by_llm = False
            if not inverted and e["id"] in llm_flips and graph.attribute_class(graph.nodes[e["object"]]["name"]) is None:
                final, by_llm = flip(raw), True
            dirs[e["id"]] = final
            info[e["id"]] = (raw, final, inverted, by_llm)

        # fix 2.4: one claim per (paper, subject, object, direction)
        kept3, dups = graph.dedupe(kept2, dirs)
        trace["duplicates_removed"] = dups
        trace["kept"] = [e["id"] for e in kept3]
        trace["papers"] = sorted({e["source_document"] for e in kept3})

        if not kept3:
            entry = {
                "id": qid, "status": "no_evidence",
                "answer": (f"The graph has no claim in which {p['cause'].get('phrase') or p['cause']['label']} "
                           f"is the cause and {p['outcome'].get('phrase') or p['outcome']['label']} is the measured outcome."
                           + (f" {reversed_n} edge(s) link the two the other way round and were not used."
                              if reversed_n else "")),
                "direction": None,
                "confidence": confidence_no_evidence(bool(cands)),
                "certainty": None, "edge_ids": [], "searched_for": searched_for(p, bool(cands))}
        else:
            agg = aggregate(graph, kept3, dirs)
            trace["aggregate"] = agg
            entry = {
                "id": qid, "status": "answered",
                "answer": write_answer(graph, p, agg, kept3),
                "direction": agg["direction"], "confidence": agg["confidence"],
                "certainty": agg["certainty"], "edge_ids": [e["id"] for e in kept3],
                "searched_for": None}
        trace["edges"] = [{
            "id": e["id"], "direction_raw": info[e["id"]][0], "direction": info[e["id"]][1],
            "inverted": info[e["id"]][2], "flipped_by_llm": info[e["id"]][3],
            "subject": graph.nodes[e["subject"]]["name"],
            "predicate": e["predicate"],
            "object": graph.nodes[e["object"]]["name"],
            "paper": e["source_document"], "grade": e.get("certainty_grade"),
            "sentence": e["original_sentence"]} for e in kept3]
    except Exception as err:
        # Never drop an entry: a missing id is scored wrong at full confidence.
        traceback.print_exc()
        trace["error"] = repr(err)
        entry = {"id": qid, "status": "answered",
                 "answer": "The system failed on this question; no reliable answer.",
                 "direction": "unclear", "confidence": 0.1, "certainty": "very_low",
                 "edge_ids": [], "searched_for": None}

    os.makedirs(TRACE_DIR, exist_ok=True)
    path = f"{TRACE_DIR}/{qid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    entry["extras"] = [path]
    print(f"[{qid}] {entry['status']} {entry.get('direction')} conf={entry['confidence']} "
          f"edges={len(entry['edge_ids'])} papers={len(trace.get('papers', []))}", file=sys.stderr)
    return entry


def main():
    qfile, out = (sys.argv[1:3] + ["questions.yaml", "answers.json"][len(sys.argv) - 1:])[:2]
    with open(qfile, encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]
    graph = Graph(os.environ.get("GRAPH_PATH", "causal_graph.json"))
    checker = PlanChecker(graph)
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(lambda q: answer_one(graph, checker, q), questions))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"team": TEAM, "answers": answers}, f, indent=2, ensure_ascii=False)
    u = llm.USAGE
    print(f"Wrote {len(answers)} answers to {out}. LLM: {u['calls']} calls, "
          f"{u['prompt_tokens']} prompt + {u['completion_tokens']} completion tokens "
          f"({(u['prompt_tokens'] + u['completion_tokens']) / max(1, len(answers)):.0f}/question)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
