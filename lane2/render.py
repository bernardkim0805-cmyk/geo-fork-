"""Stage 4: Decision -> answers.json entry.

STUB: f-string templates, no LLM. Later: Rosetta-style statements, evidence
maps in outputs/ listed under `extras`, optional LLM phrasing constrained to
the retrieved fields.
"""
from __future__ import annotations

from .schema import Decision, Query, Question

LEAD = {
    "increase": "The evidence suggests an increase in",
    "decrease": "The evidence suggests a decrease in",
    "no_effect": "Studies tested this and found no effect on",
    "mixed": "The evidence is mixed on its effect on",
    "unclear": "A relationship is reported, but the direction of change in",
}

FINDINGS = {
    "increase": "an increase",
    "decrease": "a decrease",
    "no_effect": "no effect",
    "mixed": "mixed findings",
    "unclear": "an unclear direction",
}


# Parse found no cause->effect pair, so nothing was searched. Without this the
# effect template printed "no claim linking grassland restoration to None" at
# the decide stage's confidence, a confident claim about evidence nobody looked
# for. For no_evidence, confidence is P(the graph really can't answer); for
# these questions it usually can, we just don't handle the type, so it is low.
UNSUPPORTED = {
    "gap_map": ("This question asks where the evidence is dense or thin, which means "
                "counting claims across the graph rather than reading one cause and "
                "effect. This system only answers cause-and-effect questions, so it "
                "did not map the evidence and cannot say where the gaps are.", 0.1),
    "trajectory": ("This question asks how long recovery takes or in what order, and "
                   "no measured outcome could be identified to look up. This system "
                   "did not search the graph for it.", 0.2),
    "interaction": ("This question asks whether stressors combine or modify each other, "
                    "and no measured outcome could be identified to look up. This "
                    "system did not search the graph for it.", 0.2),
    "effect": ("The cause and the measured outcome could not both be identified in the "
               "question, so the graph was not searched.", 0.3),
}


def _unsupported(q: Question, query: Query) -> dict:
    answer, confidence = UNSUPPORTED.get(query.mode, UNSUPPORTED["effect"])
    found = (f"causes {query.interventions}" if query.interventions else "no cause") + \
            (f", outcomes {query.outcomes}" if query.outcomes else ", no measured outcome")
    return {
        "id": q.id,
        "status": "no_evidence",
        "answer": answer,
        "direction": None,
        "confidence": confidence,
        "certainty": None,
        "edge_ids": [],
        "searched_for": (f"nothing: the question was read as mode '{query.mode}' with "
                         f"{found}, so there was no cause->effect pair to search for"),
    }


def render(q: Question, query: Query, d: Decision) -> dict:
    if not query.pairs:
        return _unsupported(q, query)
    if d.status == "no_evidence":
        answer = (f"The provided evidence contains no claim linking {query.intervention} to {query.outcome}."
                  if query.intervention else "The intervention and outcome could not be identified in the question.")
    else:
        counts = ", ".join(f"{v} reporting {FINDINGS[k]}" for k, v in sorted(d.tally.items(), key=lambda kv: -kv[1]))
        tail = " is unclear" if d.direction == "unclear" else ""
        n_papers = sum(d.tally.values())
        answer = (f"{LEAD[d.direction]} {query.outcome}{tail} "
                  f"({n_papers} {'study' if n_papers == 1 else 'studies'}, "
                  f"{counts}; {d.certainty.replace('_', ' ')} certainty).")
        if d.notes:
            answer += " " + " ".join(d.notes)
    return {
        "id": q.id,
        "status": d.status,
        "answer": answer,
        "direction": d.direction if d.status == "answered" else None,
        "confidence": d.confidence,
        "certainty": d.certainty if d.status == "answered" else None,
        "edge_ids": d.edge_ids if d.status == "answered" else [],
        "searched_for": d.searched_for if d.status == "no_evidence" else None,
    }
