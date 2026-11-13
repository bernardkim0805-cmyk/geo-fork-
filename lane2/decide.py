"""Stage 3: edges -> direction, certainty, confidence, status. Deterministic.

How direction is read (STUB rules; verify against schema descriptions!):
  - negated=True / causal_language=no_relationship  -> no_effect
  - else the OBJECT node's state_or_change_qualifier: increased/decreased/unchanged
  - predicates prevents/disrupts flip the sign (TODO: confirm with schema)
  - anything else -> unclear
Aggregate: one direction holds a clear majority -> that direction;
otherwise mixed.  Certainty = median edge certainty_grade (crude).
Confidence: placeholder heuristics. Calibrate later against hand-checked cases.
"""
from __future__ import annotations

from collections import Counter

from .graph import Graph
from .schema import Decision, Evidence, Grounding, Query

GRADES = ["very_low", "low", "moderate", "high"]
FLIP = {"prevents", "disrupts"}
FLIP_DIR = {"increase": "decrease", "decrease": "increase"}


def edge_direction(edge: dict, graph: Graph) -> str:
    if edge.get("negated") or edge.get("causal_language") == "no_relationship":
        return "no_effect"
    q = graph.nodes[edge["object"]].get("state_or_change_qualifier")
    d = {"increased": "increase", "decreased": "decrease", "unchanged": "no_effect"}.get(q, "unclear")
    if edge.get("predicate") in FLIP:
        d = FLIP_DIR.get(d, d)
    return d


def describe_search(query: Query, g: Grounding) -> str:
    return (
        f"edges whose subject node matches {g.intervention_terms or [query.intervention]} "
        f"({len(g.intervention_nodes)} nodes) and whose object node matches "
        f"{g.outcome_terms or [query.outcome]} ({len(g.outcome_nodes)} nodes)"
    )


def decide(query: Query, g: Grounding, ev: Evidence, graph: Graph) -> Decision:
    if query.parse_method == "failed":
        return Decision(status="no_evidence", confidence=0.3,
                        searched_for="could not parse an intervention and outcome from the question")
    if not ev.edge_ids:
        return Decision(status="no_evidence", confidence=0.6, searched_for=describe_search(query, g))

    edges = [graph.edges[e] for e in ev.edge_ids]
    tally = Counter(edge_direction(e, graph) for e in edges)
    informative = {k: v for k, v in tally.items() if k != "unclear"}
    n = sum(informative.values())

    if n == 0:
        direction = "unclear"
    else:
        top, top_n = max(informative.items(), key=lambda kv: kv[1])
        direction = top if top_n / n >= 0.7 else "mixed"

    grades = sorted(GRADES.index(e["certainty_grade"]) for e in edges if e.get("certainty_grade") in GRADES)
    certainty = GRADES[grades[len(grades) // 2]] if grades else "very_low"

    # Placeholder confidence: more agreeing edges -> more confident, capped.
    agree = informative.get(direction, 0) if direction not in ("mixed", "unclear") else 0
    confidence = min(0.9, 0.4 + 0.1 * agree) if agree else 0.4

    return Decision(status="answered", direction=direction, certainty=certainty,
                    confidence=round(confidence, 2), edge_ids=ev.edge_ids, tally=dict(tally))
