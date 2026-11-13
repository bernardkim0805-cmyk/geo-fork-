"""Load causal_graph.json and apply the reading rules from HANDOFF.md section 3.

Everything here is plain code, with no LLM, so the same edge always gets the same reading.
"""
import json
import re
from collections import Counter, defaultdict

GRADES = ["very_low", "low", "moderate", "high"]

# "native" must not match "non-native" (HANDOFF gotcha 4).
NATIVE = r"(?<!non-)(?<!non)\bnative"
NOT_NATIVE = r"non-?native|nonnative|exotic|invasive|introduced|weed|alien"

# Subjects that describe the ABSENCE of the intervention (gotcha 7).
NEGATED_SUBJECT = re.compile(
    r"^(absence of|lack of|no |without )|\buntreated\b|\bunburn|\bunharvest|"
    r"\bunmowed|\bungrazed|\bcontrol (plot|site|treatment|area)s?\b|\breference\b")

SIGN = {"increased": 1, "decreased": -1, "absent": -1, "removed": -1, "terminated": -1}


class Graph:
    def __init__(self, path="causal_graph.json"):
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
        self.nodes = {n["id"]: n for n in g["nodes"]}
        self.edges = g["edges"]
        self.edge_by_id = {e["id"]: e for e in self.edges}
        self.docs = {d["document_id"]: d for d in g["source_documents"]}
        # Precompute one lowercase searchable string per node.
        self.text = {nid: self._node_text(n) for nid, n in self.nodes.items()}

    @staticmethod
    def _node_text(n):
        parts = [n.get("name"), n.get("embedding_text"), n.get("entity_term"),
                 n.get("measured_attribute"), n.get("applied_to")]
        return " | ".join(str(p) for p in parts if p).lower()

    # ---------- direction of a single edge ----------
    def direction(self, e):
        """increase / decrease / no_effect / unclear for one edge, read from the object node."""
        if e.get("negated") or e.get("causal_language") == "no_relationship":
            return "no_effect"                      # gotcha 2: null result
        if e["predicate"] in ("prevents", "disrupts"):
            return "decrease"                       # gotcha 3: never double-negate
        q = self.nodes[e["object"]].get("state_or_change_qualifier")
        return {1: "increase", -1: "decrease"}.get(SIGN.get(q, 0), "unclear")

    # ---------- finding candidate edges ----------
    @staticmethod
    def compile_terms(terms):
        pats = []
        for t in terms or []:
            t = str(t).strip().lower()
            if not t:
                continue
            pats.append(NATIVE if t == "native" else r"\b" + re.escape(t))
        return re.compile("|".join(pats)) if pats else None

    def candidates(self, cause_terms, cause_excl, out_terms, out_excl):
        cause, out = self.compile_terms(cause_terms), self.compile_terms(out_terms)
        c_ex, o_ex = self.compile_terms(cause_excl), self.compile_terms(out_excl)
        if cause is None or out is None:
            return []
        # If the outcome is about natives, drop non-native / invasive objects.
        if any(str(t).strip().lower() == "native" for t in out_terms) and not any(
                re.search(NOT_NATIVE, str(t)) for t in out_terms):
            o_ex = re.compile((o_ex.pattern + "|" if o_ex else "") + NOT_NATIVE)
        found = []
        for e in self.edges:
            st, ot = self.text[e["subject"]], self.text[e["object"]]
            if not cause.search(st) or not out.search(ot):
                continue
            if c_ex and c_ex.search(st):
                continue
            if o_ex and o_ex.search(ot):
                continue
            if NEGATED_SUBJECT.search(self.nodes[e["subject"]]["name"].lower()):
                continue
            found.append(e)
        return found

    def spread_by_paper(self, edges, cap):
        """Keep at most `cap` edges, taking them round-robin across papers so one
        big paper cannot crowd out the others."""
        by_doc = defaultdict(list)
        for e in edges:
            by_doc[e["source_document"]].append(e)
        out, i = [], 0
        while len(out) < min(cap, len(edges)):
            for lst in by_doc.values():
                if i < len(lst) and len(out) < cap:
                    out.append(lst[i])
            i += 1
        return out


# ---------- aggregation: per paper first, then across papers (gotcha 6) ----------
def flip(d):
    return {"increase": "decrease", "decrease": "increase"}.get(d, d)


def aggregate(graph, edges, flipped_ids):
    per_paper = defaultdict(list)
    for e in edges:
        d = graph.direction(e)
        if e["id"] in flipped_ids:
            d = flip(d)
        per_paper[e["source_document"]].append(d)

    verdicts = {}
    for doc, ds in per_paper.items():
        c = Counter(x for x in ds if x != "unclear")
        if not c:
            verdicts[doc] = "unclear"
        elif c["increase"] and c["decrease"]:
            verdicts[doc] = "mixed"   # same paper, opposite results (gotcha 8)
        elif len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
            verdicts[doc] = "mixed"   # as many null results as effects
        else:
            verdicts[doc] = c.most_common(1)[0][0]

    vc = Counter(verdicts.values())
    decided = sum(vc[k] for k in ("increase", "decrease", "no_effect", "mixed"))
    if decided == 0:
        direction, agreement = "unclear", 0.0
    else:
        top, top_n = max(((k, vc[k]) for k in ("increase", "decrease", "no_effect")),
                         key=lambda kv: kv[1])
        agreement = top_n / decided
        if top_n == 0:
            direction = "mixed"       # every paper was internally mixed
        elif agreement >= 0.7:
            direction = top
        elif vc["increase"] and vc["decrease"]:
            direction = "mixed"
        elif vc["no_effect"] and (vc["increase"] or vc["decrease"]):
            # effects and nulls but no opposite effects: the evidence disagrees on size
            direction = "mixed"
        else:
            direction = top

    # Certainty: start from the typical edge grade, adjust GRADE-style.
    grades = sorted(GRADES.index(e.get("certainty_grade") or "low") for e in edges)
    level = grades[len(grades) // 2]
    n_papers = len(per_paper)
    rct = sum("randomized_experiment" in (e.get("evidential_basis") or {}).get(
        "evidence_types", []) for e in edges)
    if n_papers >= 3 and agreement >= 0.8 and rct >= len(edges) / 2:
        level += 1
    if n_papers == 1:
        level -= 1
    if direction in ("mixed", "unclear"):
        level -= 1
    certainty = GRADES[max(0, min(3, level))]

    # Confidence: a first guess, to be fitted on a dev set later.
    conf = 0.45 + 0.08 * min(n_papers, 4) + 0.2 * max(0.0, agreement - 0.5)
    if direction == "unclear":
        conf = 0.35
    elif direction == "mixed":
        conf = min(conf, 0.6)
    conf = round(max(0.3, min(0.9, conf)), 2)

    return {"direction": direction, "certainty": certainty, "confidence": conf,
            "papers": dict(verdicts), "paper_counts": dict(vc),
            "agreement": round(agreement, 2), "n_papers": n_papers}
