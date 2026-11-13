"""Answer questions from causal_graph.json.

Usage:  python -m agent.run questions.yaml answers.json

Per question:
  1. LLM call "plan": turn the question into search terms for the cause and the outcome.
  2. Code: find edges whose subject matches the cause and whose object matches the outcome.
  3. LLM call "filter": keep the edges that really bear on the question and flag edges
     whose object measures the opposite quantity (e.g. mortality vs survival).
  4. Code: direction per edge, per paper, then across papers; certainty and confidence.
  5. Template answer + a trace file in outputs/traces/ that shows every step.
"""
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import yaml

from agent import llm
from agent.graph import Graph, aggregate

TEAM = "Team Geo"
MAX_CANDIDATES = 80
TRACE_DIR = "outputs/traces"

PLAN_SYSTEM = """You turn a question about an ecological restoration evidence graph into search terms.
The graph has edges: SUBJECT (an intervention or condition) -> OBJECT (a measured outcome).
Node text looks like: "Increased native forb cover | entity: native forb | attribute: cover".
Return JSON only:
{"cause": {"label": str, "terms": [str], "exclude": [str]},
 "outcome": {"label": str, "terms": [str], "attributes": [str], "exclude": [str]}}
Rules:
- terms: 3-10 lowercase word stems that NAME the thing itself; a node must contain one of
  them. Stems match the start of a word ("herbicid" matches herbicides). Add specific
  examples (e.g. herbicide: "glyphosate", "picloram"; mulch: "straw", "wood chip").
- NEVER put generic words in terms: application, treatment, control, management, effect,
  change, cover, abundance, density, species, plant, vegetation, surface, level.
  Those go in outcome.attributes (used only for context).
- Use the term "native" (exactly) when the outcome is about native species.
- exclude: stems that would wrongly match (e.g. for native plants exclude "invasive",
  "exotic", "non-native")."""

GENERIC = {"application", "applications", "treatment", "treatments", "control", "management",
           "effect", "effects", "change", "cover", "abundance", "density", "species", "plant",
           "plants", "vegetation", "surface", "level", "levels", "organic", "c"}


def core_terms(terms):
    """Drop generic words the model slipped in, unless nothing else is left."""
    terms = [str(t).strip().lower() for t in terms or [] if str(t).strip()]
    specific = [t for t in terms if t not in GENERIC]
    return specific or terms


FILTER_SYSTEM = """You check candidate edges from an evidence graph against a question.
You are given the intervention and outcome being asked about, then lines "index: SUBJECT -> OBJECT".
Be strict: when unsure, leave the edge out.
Keep an edge only if SUBJECT is an instance of the question's intervention/condition AND
OBJECT measures the question's outcome (or its direct opposite).
Watch for traps: native vs non-native/invasive are DIFFERENT outcomes; a subject that is
the absence/control of the intervention does not count.
"flip" lists kept edges whose OBJECT measures the OPPOSITE quantity of the asked outcome
(e.g. mortality when the question asks about survival, bare ground when it asks about
plant cover). Do not flip for any other reason.
Return JSON only: {"keep": [int], "flip": [int]}"""


def plan(question):
    return llm.chat_json(PLAN_SYSTEM, question, max_tokens=300)


def filter_edges(graph, question, p, edges):
    lines = []
    for i, e in enumerate(edges):
        s, o = graph.nodes[e["subject"]], graph.nodes[e["object"]]
        lines.append(f"{i}: {s['name']} -> {o['name']}")
    user = (f"Question: {question}\nIntervention: {p['cause'].get('label')}\n"
            f"Outcome: {p['outcome'].get('label')}\n\nCandidates:\n" + "\n".join(lines))
    out = llm.chat_json(FILTER_SYSTEM, user, max_tokens=60 + 4 * len(edges))
    keep = [edges[i] for i in out.get("keep", []) if isinstance(i, int) and 0 <= i < len(edges)]
    flips = {edges[i]["id"] for i in out.get("flip", [])
             if isinstance(i, int) and 0 <= i < len(edges)}
    return keep, flips & {e["id"] for e in keep}


def searched_for(p):
    c, o = p["cause"], p["outcome"]
    return (f"edges whose subject node matches '{c.get('label')}' "
            f"(terms: {', '.join(c.get('terms', []))}) and whose object node measures "
            f"'{o.get('label')}' (terms: {', '.join(o.get('terms', []))})")


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
    return (f"{c} {WORDS[agg['direction']]} {o}, based on {len(kept)} claims from "
            f"{agg['n_papers']} studies (per-study verdicts: {tally}). "
            f'Example ({cite}): "{quote}"')


def answer_one(graph, q):
    qid, question = q["id"], " ".join(str(q["question"]).split())
    trace = {"id": qid, "question": question}
    try:
        p = plan(question)
        trace["plan"] = p
        p["cause"]["terms"] = core_terms(p["cause"].get("terms"))
        p["outcome"]["terms"] = core_terms(p["outcome"].get("terms"))
        cands = graph.candidates(p["cause"]["terms"], p["cause"].get("exclude"),
                                 p["outcome"]["terms"], p["outcome"].get("exclude"))
        trace["n_candidates"] = len(cands)
        cands = graph.spread_by_paper(cands, MAX_CANDIDATES)
        kept, flips = filter_edges(graph, question, p, cands) if cands else ([], set())
        trace["kept"] = [e["id"] for e in kept]
        trace["flipped"] = sorted(flips)
        trace["rejected"] = [e["id"] for e in cands if e not in kept]

        if not kept:
            entry = {
                "id": qid, "status": "no_evidence",
                "answer": (f"The graph has no claim linking {p['cause']['label']} to "
                           f"{p['outcome']['label']}."),
                "direction": None,
                "confidence": 0.8 if not cands else 0.65,
                "certainty": None, "edge_ids": [], "searched_for": searched_for(p)}
        else:
            agg = aggregate(graph, kept, flips)
            trace["aggregate"] = agg
            entry = {
                "id": qid, "status": "answered",
                "answer": write_answer(graph, p, agg, kept),
                "direction": agg["direction"], "confidence": agg["confidence"],
                "certainty": agg["certainty"], "edge_ids": [e["id"] for e in kept],
                "searched_for": None}
        trace["edges"] = [{
            "id": e["id"], "direction": graph.direction(e),
            "flipped": e["id"] in flips,
            "subject": graph.nodes[e["subject"]]["name"],
            "predicate": e["predicate"],
            "object": graph.nodes[e["object"]]["name"],
            "paper": e["source_document"], "grade": e.get("certainty_grade"),
            "sentence": e["original_sentence"]} for e in kept]
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
    print(f"[{qid}] {entry['status']} {entry.get('direction')} "
          f"conf={entry['confidence']} edges={len(entry['edge_ids'])}", file=sys.stderr)
    return entry


def main():
    qfile, out = (sys.argv[1:3] + ["questions.yaml", "answers.json"][len(sys.argv) - 1:])[:2]
    with open(qfile, encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]
    graph = Graph(os.environ.get("GRAPH_PATH", "causal_graph.json"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(lambda q: answer_one(graph, q), questions))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"team": TEAM, "answers": answers}, f, indent=2, ensure_ascii=False)
    u = llm.USAGE
    print(f"Wrote {len(answers)} answers to {out}. LLM: {u['calls']} calls, "
          f"{u['prompt_tokens']} prompt + {u['completion_tokens']} completion tokens "
          f"({(u['prompt_tokens'] + u['completion_tokens']) / max(1, len(answers)):.0f}/question)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
