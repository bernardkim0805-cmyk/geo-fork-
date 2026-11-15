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


def render(q: Question, query: Query, d: Decision) -> dict:
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
