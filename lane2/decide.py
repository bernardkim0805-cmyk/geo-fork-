"""Stage 3: edges -> direction, certainty, confidence, status. Deterministic.

Spec: docs/decide.md (rule numbers R1-R4 below refer to it).
Tests: tests/test_decide.py.  Tune numbers in CONFIG only.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .schema import Decision, Evidence, Grounding, Query

# --------------------------------------------------------------------------
# CONFIG: every tunable number/list. Change these, not the logic.
# --------------------------------------------------------------------------
CONFIG = {
    "majority": 0.66,  # top direction share needed to avoid "mixed"
    "excluded_predicates": {"mediates", "moderates", "regulates", "precedes"},
    "predicate_sign": {"causes": 1, "contributes_to": 1, "enables": 1,
                       "associated_with": 1, "correlated_with": 1,
                       "prevents": -1, "disrupts": -1},
    "object_sign": {"increased": 1, "introduced": 1, "initiated": 1,
                    "decreased": -1, "removed": -1, "absent": -1,
                    "terminated": -1, "interrupted": -1, "aborted": -1},
    "negative_subject_qualifiers": {"removed", "absent", "decreased",
                                    "terminated", "interrupted", "aborted"},
    # question phrases that are themselves about a removal -> don't flip subject
    "removal_words": r"\b(remov|exclu|exclos|reduc|absen|thin|clear|control|suppress|"
                     r"eradicat|cull|kill|cut|harvest|release from|elimin|without|no )",
    "comparator_weight": {"untreated_control": 1.0, "pre_treatment_baseline": 1.0,
                          "spatial_or_temporal_control": 1.0, "passive_management_arm": 1.0,
                          "alternative_treatment": 0.5, "reference_undisturbed": 0.5},
    "comparator_default_weight": 0.75,  # not_reported / not_applicable / missing
    # alternative_treatment whose description is really a no-treatment control
    "control_words": r"\b(control|un\w+ed|unburn\w*|without|no[ -]\w+|bare|untouched)",
    "language_weight": {"direct_causal": 1.0, "uncertain_causal": 0.75,
                        "associational": 0.6, "no_relationship": 1.0},
    "associative_predicates": {"associated_with", "correlated_with"},
    "associative_weight": 0.6,
    # confidence placeholders (calibrate against tests/cases.yaml)
    "conf_base": 0.45, "conf_per_paper": 0.08, "conf_cap": 0.9,
    "conf_mixed": 0.45, "conf_unclear": 0.3, "head_to_head_penalty": 0.85,
    "conf_no_evidence_gap": 0.75,    # both sides grounded, nothing joins them
    "conf_no_evidence_miss": 0.55,   # a side grounded to nothing (maybe our miss)
    "conf_parse_failed": 0.3,
}
GRADES = ["very_low", "low", "moderate", "high"]
DIRS = ("increase", "decrease", "no_effect")


@dataclass
class EdgeVote:
    edge_id: str
    doc: str
    direction: Optional[str]      # increase|decrease|no_effect|unclear|None (=excluded)
    weight: float = 0.0
    head_to_head: bool = False    # compared to another treatment / reference, not "nothing"
    associative: bool = False
    flipped_subject: bool = False
    grade: Optional[str] = None
    reason: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 1. Read one edge
# --------------------------------------------------------------------------
def _comparator_weight(edge: dict) -> tuple[float, bool]:
    comp = (edge.get("comparator") or [{}])[0]
    ctype = comp.get("comparator_type")
    desc = comp.get("comparator_description") or ""
    if ctype == "alternative_treatment" and re.search(CONFIG["control_words"], desc, re.I):
        return 1.0, False  # mislabeled: really a no-treatment control
    w = CONFIG["comparator_weight"].get(ctype, CONFIG["comparator_default_weight"])
    return w, ctype in ("alternative_treatment", "reference_undisturbed")


def read_edge(edge: dict, graph, query: Query) -> EdgeVote:
    v = EdgeVote(edge_id=edge["id"], doc=edge.get("source_document") or edge["id"],
                 direction=None, grade=edge.get("certainty_grade"))
    pred = edge.get("predicate")

    # R1: not a direct X -> Y effect
    if pred in CONFIG["excluded_predicates"]:
        v.reason = f"excluded: predicate {pred}"
        return v

    cw, v.head_to_head = _comparator_weight(edge)
    lang = edge.get("causal_language")
    lw = CONFIG["language_weight"].get(lang, 0.75)
    v.associative = pred in CONFIG["associative_predicates"] or lang == "associational"
    if pred in CONFIG["associative_predicates"]:
        lw = min(lw, CONFIG["associative_weight"])
    v.weight = round(cw * lw, 3)

    # R2: null result
    if edge.get("negated") or lang == "no_relationship":
        v.direction, v.reason = "no_effect", "negated / no_relationship"
        return v

    obj_q = graph.nodes[edge["object"]].get("state_or_change_qualifier")
    # R3: unchanged (team decision: treat as no_effect)
    if obj_q == "unchanged":
        v.direction, v.reason = "no_effect", "object unchanged"
        return v

    # R4: subject sign x predicate sign x object sign
    o = CONFIG["object_sign"].get(obj_q)
    p = CONFIG["predicate_sign"].get(pred)
    if o is None or p is None:
        v.direction, v.reason = "unclear", f"no sign (predicate={pred}, object={obj_q})"
        return v
    s = 1
    subj_q = graph.nodes[edge["subject"]].get("state_or_change_qualifier")
    if subj_q in CONFIG["negative_subject_qualifiers"]:
        asks_about_removal = re.search(CONFIG["removal_words"], (query.intervention or "") + " ", re.I)
        if not asks_about_removal:
            s, v.flipped_subject = -1, True
    v.direction = "increase" if s * p * o > 0 else "decrease"
    v.reason = f"sign {s}*{p}*{o}" + (" (subject flipped)" if v.flipped_subject else "")
    return v


# --------------------------------------------------------------------------
# 2-3. Combine per paper, then across papers
# --------------------------------------------------------------------------
def _paper_votes(votes: list[EdgeVote]) -> list[dict]:
    by_doc = defaultdict(list)
    for v in votes:
        by_doc[v.doc].append(v)
    papers = []
    for doc, vs in by_doc.items():
        dirs = defaultdict(float)
        for v in vs:
            if v.direction in DIRS:
                dirs[v.direction] += v.weight
        total = sum(dirs.values())
        grades = [GRADES.index(v.grade) for v in vs if v.grade in GRADES]
        papers.append({
            "doc": doc,
            "share": {d: w / total for d, w in dirs.items()} if total else {},
            "weight": max(v.weight for v in vs),
            "grade": max(grades) if grades else 0,
            "head_to_head": all(v.head_to_head for v in vs),
            "associative": all(v.associative for v in vs),
        })
    return papers


def _paper_label(p: dict) -> str:
    """What one paper says overall: its top direction, or mixed if split."""
    if not p["share"]:
        return "unclear"
    top = max(p["share"], key=p["share"].get)
    return top if p["share"][top] >= CONFIG["majority"] else "mixed"


def describe_search(query: Query, g: Grounding) -> str:
    return (
        f"edges whose subject node matches {g.intervention_terms or [query.intervention]} "
        f"({len(g.intervention_nodes)} nodes) and whose object node matches "
        f"{g.outcome_terms or [query.outcome]} ({len(g.outcome_nodes)} nodes)"
    )


def _no_evidence(query: Query, g: Grounding, extra_note: str = "") -> Decision:
    both_grounded = bool(g.intervention_nodes) and bool(g.outcome_nodes)
    conf = CONFIG["conf_no_evidence_gap"] if both_grounded else CONFIG["conf_no_evidence_miss"]
    return Decision(status="no_evidence", confidence=conf,
                    searched_for=describe_search(query, g) + extra_note)


def decide(query: Query, g: Grounding, ev: Evidence, graph) -> Decision:
    if query.parse_method == "failed":
        return Decision(status="no_evidence", confidence=CONFIG["conf_parse_failed"],
                        searched_for="could not parse an intervention and outcome from the question")

    votes = [read_edge(graph.edges[e], graph, query) for e in ev.edge_ids]
    voting = [v for v in votes if v.direction is not None]
    if not voting:
        note = "; only indirect edges (mediates/moderates/...) were found" if votes else ""
        d = _no_evidence(query, g, note)
        d.votes = votes
        return d

    papers = _paper_votes(voting)
    totals = defaultdict(float)
    for p in papers:
        for d_, s in p["share"].items():
            totals[d_] += s * p["weight"]
    total = sum(totals.values())

    notes = []
    if total == 0:
        direction, top_share, agreeing = "unclear", 0.0, 0
    else:
        top, top_w = max(totals.items(), key=lambda kv: kv[1])
        top_share = top_w / total
        direction = top if top_share >= CONFIG["majority"] else "mixed"
        agreeing = sum(1 for p in papers if _paper_label(p) == top)

    # 4. certainty
    grades = sorted(p["grade"] for p in papers)
    level = grades[len(grades) // 2]
    all_h2h = all(p["head_to_head"] for p in papers)
    if direction == "mixed":
        level -= 1
    if all(p["associative"] for p in papers):
        level -= 1
        notes.append("All evidence is associational, not causal.")
    if all_h2h:
        level -= 1
        notes.append("Every study compares against another treatment or a reference site, "
                     "not against no treatment.")
    certainty = GRADES[max(level, 0)]

    # 5. confidence (placeholder)
    if direction == "mixed":
        conf = CONFIG["conf_mixed"]
    elif direction == "unclear":
        conf = CONFIG["conf_unclear"]
    else:
        conf = min(CONFIG["conf_cap"], CONFIG["conf_base"] + CONFIG["conf_per_paper"] * agreeing)
        conf *= top_share
    if all_h2h:
        conf *= CONFIG["head_to_head_penalty"]

    flipped = sum(1 for v in voting if v.flipped_subject)
    if flipped:
        notes.append(f"{flipped} edge(s) describe the absence/removal of the intervention; "
                     "their direction was inverted.")
    excluded = len(votes) - len(voting)
    if excluded:
        notes.append(f"{excluded} indirect edge(s) (mediates/moderates/...) not counted.")

    tally = defaultdict(int)
    for p in papers:
        tally[_paper_label(p)] += 1

    return Decision(status="answered", direction=direction, certainty=certainty,
                    confidence=round(conf, 2), edge_ids=[v.edge_id for v in voting],
                    tally=dict(tally), notes=notes, votes=votes)
