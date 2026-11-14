"""Stage 3: edges -> direction, certainty, confidence, status.

Deterministic except D2: when an edge's subject is a removal/absence, one small
cached LLM call decides whether that subject IS the question's intervention or
its opposite (falls back to a word list when the LLM is unavailable).

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
    # D2 fallback (no LLM): question phrases that are themselves about a removal -> no flip.
    # Tightened: control/cut/thin/clear/harvest/'no ' removed (e.g. 'erosion control').
    "removal_words": r"\b(remov|exclu|exclos|absen|suppress|eradicat|cull|elimin|"
                     r"release from|without|withdraw|cessation)",
    "removal_llm": True,             # D2: ask the LLM when BRINE_* are set (env or .env)
    "removal_llm_max_tokens": 5,     # thinking is off; the answer is one word
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
    # confidence: fitted by tests/calibrate_decide.py on tests/decide_cases.yaml
    # (2026-11-14, 31 signed-off cases; correct = status + direction + exact certainty)
    "conf_base": 0.55, "conf_per_paper": 0.125, "conf_cap": 0.9,
    "conf_mixed": 0.57,              # 2/3 right, shrunk toward 0.5 (K=4)
    "conf_unclear": 0.3,             # no cases yet: prior kept
    "conf_floor": 0.2, "conf_ceiling": 0.9,  # E3 clamp
    "head_to_head_penalty": 0.85,
    "single_study_mixed_penalty": 0.85,  # D1: one paper that disagrees with itself
    "conf_no_evidence": 0.75,        # D3: 7/8 right, shrunk toward 0.5 (K=4)
    "conf_parse_failed": 0.3,
    "bad_input_penalty": 0.9,
    # C1: study design -> vote weight (evidential_basis.evidence_types; max if several)
    "evidence_weight": {"randomized_experiment": 1.0, "quasi_experiment": 0.9,
                        "natural_experiment": 0.9, "observational_longitudinal": 0.8,
                        "observational_cross_sectional": 0.7, "structural_equation_model": 0.7,
                        "mechanistic_study": 0.7, "modeling_simulation": 0.5, "case_study": 0.5},
    "evidence_default_weight": 0.7,
    # C2: drop certainty a level when every paper is observational. OFF: certainty_grade
    # already encodes design (randomized ~ moderate, cross-sectional ~ low), so this
    # would double-count. Kept switchable; see docs/decide.md section 4.
    "observational_certainty_drop": False,
    "experimental_types": {"randomized_experiment", "quasi_experiment", "natural_experiment"},
    # C3: reported effect strength -> vote weight (independent of certainty_grade)
    "strength_weight": {"strong": 1.0, "moderate": 0.9, "not_specified": 0.9,
                        "weak": 0.75, "negligible": 0.5},
    "strength_default_weight": 0.9,
    # B1: sign words in a node name, used when the object has no up/down qualifier
    "name_plus": r"\b(increas\w*|higher|greater|more|enhanc\w*|elevat\w*|rais\w*|ris(e|ing)|larger|improv\w*|gain\w*)\b",
    "name_minus": r"\b(decreas\w*|lower|reduc\w*|fewer|less|loss|lost|declin\w*|smaller|diminish\w*)\b",
    # B2/B3: edges that don't state X's effect on the LEVEL of Y -> excluded like R1
    "variation_object": r"\b(variation|variability|variance|heterogeneity)\b",
    "interaction_subject": r"\binteraction|\s[x\u00d7]\s|\band (year|time)\b",        # x confidence when some edge_ids were unusable
}
GRADES = ["very_low", "low", "moderate", "high"]
DIRS = ("increase", "decrease", "no_effect")
ABSTAIN = "abstain"   # cited as context, but carries no direction vote (B1, B5)


@dataclass
class EdgeVote:
    edge_id: str
    doc: str
    direction: Optional[str]      # increase|decrease|no_effect|abstain|None (=excluded)
    weight: float = 0.0
    head_to_head: bool = False    # compared to another treatment / reference, not "nothing"
    associative: bool = False
    flipped_subject: bool = False
    observational: bool = False   # no randomized/quasi/natural experiment behind it (C2)
    grade: Optional[str] = None
    reason: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 1. Read one edge
# --------------------------------------------------------------------------
def _comparator_weight(edge: dict) -> tuple[float, bool]:
    comp = (edge.get("comparator") or [{}])[0] or {}
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

    # B2/B3: interaction subjects and "variation in Y" objects say nothing about
    # what X does to the level of Y
    subj_name = graph.nodes[edge["subject"]].get("name") or ""
    obj_name = graph.nodes[edge["object"]].get("name") or ""
    if re.search(CONFIG["interaction_subject"], subj_name, re.I):
        v.reason = "excluded: interaction subject"
        return v
    if re.search(CONFIG["variation_object"], obj_name, re.I):
        v.reason = "excluded: variation-in object"
        return v

    cw, v.head_to_head = _comparator_weight(edge)
    lang = edge.get("causal_language")
    lw = CONFIG["language_weight"].get(lang, 0.75)
    v.associative = pred in CONFIG["associative_predicates"] or lang == "associational"
    if pred in CONFIG["associative_predicates"]:
        lw = min(lw, CONFIG["associative_weight"])
    types = (edge.get("evidential_basis") or {}).get("evidence_types") or []
    ew = max((CONFIG["evidence_weight"].get(t, CONFIG["evidence_default_weight"]) for t in types),
             default=CONFIG["evidence_default_weight"])
    v.observational = not (set(types) & CONFIG["experimental_types"])
    desc = (edge.get("strength") or {}).get("qualitative_descriptor")
    sw = CONFIG["strength_weight"].get(desc, CONFIG["strength_default_weight"])
    v.weight = round(cw * lw * ew * sw, 3)

    # R2: null result
    if edge.get("negated") or lang == "no_relationship":
        v.direction, v.reason = "no_effect", "negated / no_relationship"
        return v

    obj_q = graph.nodes[edge["object"]].get("state_or_change_qualifier")
    # R3: unchanged (team decision: treat as no_effect)
    if obj_q == "unchanged":
        v.direction, v.reason = "no_effect", "object unchanged"
        return v

    subj_q = graph.nodes[edge["subject"]].get("state_or_change_qualifier")
    # B5: an 'unchanged' subject means X itself didn't vary, so it can't say what X does
    if subj_q == "unchanged":
        v.direction, v.reason = ABSTAIN, "subject unchanged"
        return v

    # R4: subject sign x predicate sign x object sign
    o = CONFIG["object_sign"].get(obj_q)
    from_name = False
    if o is None:  # B1: fall back to sign words in the object's name
        o, from_name = _name_sign(obj_name), True
    p = CONFIG["predicate_sign"].get(pred)
    if o is None or p is None:
        v.direction, v.reason = ABSTAIN, f"no sign (predicate={pred}, object={obj_q})"
        return v
    s = 1
    if subj_q in CONFIG["negative_subject_qualifiers"]:
        subj_name = graph.nodes[edge["subject"]].get("name") or ""
        if subject_is_opposite(query.intervention or "", subj_name):
            s, v.flipped_subject = -1, True
    v.direction = "increase" if s * p * o > 0 else "decrease"
    v.reason = (f"sign {s}*{p}*{o}" + (" (object sign from name)" if from_name else "")
                + (" (subject flipped)" if v.flipped_subject else ""))
    return v


# --------------------------------------------------------------------------
# D2: is a removal/absence subject the question's intervention, or its opposite?
# --------------------------------------------------------------------------
LLM_STATS = {"calls": 0, "cache_hits": 0, "fallbacks": 0}
_OPPOSITE_CACHE: dict[tuple[str, str], bool] = {}

_PROMPT = (
    "A question asks about the effect of an intervention: \"{x}\".\n"
    "A study's claim has this subject, which describes something being removed, absent "
    "or reduced: \"{s}\".\n"
    "Does the subject describe the intervention itself happening (SAME), e.g. "
    "'weed removal' for 'weed control', 'release from browsing' for 'browsing exclusion'; "
    "or the intervention NOT happening / its opposite (OPPOSITE), e.g. 'absence of "
    "fertilization' for 'fertilization', 'weed removal' for 'competing vegetation'?\n"
    "Answer with one word: SAME or OPPOSITE."
)


def _regex_opposite(intervention: str) -> bool:
    return not re.search(CONFIG["removal_words"], intervention + " ", re.I)


def _llm_opposite(intervention: str, subject: str) -> Optional[bool]:
    from .llm import chat  # local import: decide must work without network config
    out = chat([{"role": "user", "content": _PROMPT.format(x=intervention, s=subject)}],
               temperature=0, max_tokens=CONFIG["removal_llm_max_tokens"])
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S).strip().upper()
    if "OPPOSITE" in out:
        return True
    if "SAME" in out:
        return False
    return None


def subject_is_opposite(intervention: str, subject: str) -> bool:
    """True -> flip the subject's sign. One LLM call per (intervention, subject), cached."""
    key = (intervention.strip().lower(), subject.strip().lower())
    if key in _OPPOSITE_CACHE:
        LLM_STATS["cache_hits"] += 1
        return _OPPOSITE_CACHE[key]
    ans = None
    from . import llm
    if CONFIG["removal_llm"] and llm.configured():
        try:
            LLM_STATS["calls"] += 1
            ans = _llm_opposite(intervention, subject)
        except Exception:
            ans = None
    if ans is None:
        LLM_STATS["fallbacks"] += 1
        ans = _regex_opposite(intervention)
    _OPPOSITE_CACHE[key] = ans
    return ans


def _name_sign(name: str) -> Optional[int]:
    """B1: +1/-1 if the name has only up-words or only down-words, else None."""
    plus = re.search(CONFIG["name_plus"], name or "", re.I)
    minus = re.search(CONFIG["name_minus"], name or "", re.I)
    if plus and not minus:
        return 1
    if minus and not plus:
        return -1
    return None


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
            "observational": all(v.observational for v in vs),
        })
    return papers


def _paper_label(p: dict) -> str:
    """What one paper says overall: its top direction, or mixed if split."""
    if not p["share"]:
        return "unclear"
    top = max(p["share"], key=p["share"].get)
    return top if p["share"][top] >= CONFIG["majority"] else "mixed"


def describe_search(query: Query, g: Grounding = None) -> str:
    """D3: built from the Query alone (the retrieve -> decide contract is edge ids only)."""
    return (f"edges whose subject (cause) is '{query.intervention}' and whose object "
            f"(effect) is '{query.outcome}', in that direction")


def _clamp(conf: float) -> float:
    """E3: never more certain than conf_ceiling nor less than conf_floor."""
    return round(min(CONFIG["conf_ceiling"], max(CONFIG["conf_floor"], conf)), 2)


def _no_evidence(query: Query, g: Grounding = None, extra_note: str = "") -> Decision:
    """D3: retrieve returned nothing usable; trust it."""
    return Decision(status="no_evidence", confidence=_clamp(CONFIG["conf_no_evidence"]),
                    searched_for=describe_search(query) + extra_note)


def _usable_edges(edge_ids, graph) -> tuple[list[str], list[str]]:
    """F1/F2: dedupe (keep order); drop ids not in the graph or with a missing node."""
    usable, bad, seen = [], [], set()
    for eid in edge_ids or []:
        if eid in seen:
            continue
        seen.add(eid)
        e = graph.edges.get(eid)
        if e is None or e.get("subject") not in graph.nodes or e.get("object") not in graph.nodes:
            bad.append(eid)
        else:
            usable.append(eid)
    return usable, bad


def decide(query: Query, g: Grounding, ev: Evidence, graph) -> Decision:
    if query.parse_method == "failed":
        return Decision(status="no_evidence", confidence=CONFIG["conf_parse_failed"],
                        searched_for="could not parse an intervention and outcome from the question")

    usable, bad = _usable_edges(ev.edge_ids, graph)
    votes = [read_edge(graph.edges[e], graph, query) for e in usable]
    voting = [v for v in votes if v.direction is not None]
    if not voting:
        note = ("; only edges that don't give X's direct effect on Y's level were found "
                "(mediates/moderates, interactions, variation-only)") if votes else ""
        if bad:
            note += f"; {len(bad)} edge id(s) were missing from the graph"
        d = _no_evidence(query, g, note)
        d.votes = votes
        return d

    all_papers = _paper_votes(voting)
    papers = [p for p in all_papers if p["share"]] or all_papers  # abstain-only papers sit out
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
    if CONFIG["observational_certainty_drop"] and all(p["observational"] for p in papers):
        level -= 1
        notes.append("Every study is observational (no experiment).")
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
    if direction == "mixed" and len(papers) == 1:
        # D1: certainty already dropped a level for "mixed"; don't drop it twice
        conf *= CONFIG["single_study_mixed_penalty"]
        split = defaultdict(int)
        for v in voting:
            if v.direction in DIRS:
                split[v.direction] += 1
        parts = ", ".join(f"{n} {k}" for k, n in sorted(split.items(), key=lambda kv: -kv[1]))
        notes.append(f"One study, and its results disagree ({parts} edges): "
                     "the effect likely depends on species, site or time.")
    if bad:
        conf *= CONFIG["bad_input_penalty"]
        notes.append(f"{len(bad)} edge id(s) skipped: not in the graph or missing a node.")

    flipped = sum(1 for v in voting if v.flipped_subject)
    if flipped:
        notes.append(f"{flipped} edge(s) describe the absence/removal of the intervention; "
                     "their direction was inverted.")
    excluded = len(votes) - len(voting)
    if excluded:
        notes.append(f"{excluded} edge(s) not counted: indirect (mediates/moderates/...), "
                     "interaction or variation-only.")
    abstained = sum(1 for v in voting if v.direction == ABSTAIN)
    if abstained:
        notes.append(f"{abstained} edge(s) cited as context only: no up/down direction.")

    tally = defaultdict(int)
    for p in papers:
        tally[_paper_label(p)] += 1

    return Decision(status="answered", direction=direction, certainty=certainty,
                    confidence=_clamp(conf), edge_ids=[v.edge_id for v in voting],
                    tally=dict(tally), notes=notes, votes=votes)
