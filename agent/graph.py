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

# ---- fix 2.5: inverse attribute pairs. Each entry: (pair name, class A regex, class B regex).
# "increase" in class A means "decrease" in class B and the other way round. Depth-type
# patterns are tested first, so "Deeper water table position" counts as depth, not level.
INVERSE_PAIRS = [
    ("water_table",
     ("depth", re.compile(r"water[- ]?table depth|depth to (the )?(water table|groundwater)|distance to (the )?water table|"
                          r"depth of (the )?water table|deeper water[- ]?table|water[- ]?table .*\bdepth\b|\bdepth to water")),
     ("level", re.compile(r"water[- ]?table (level|height|position|rise|elevation)|water level|groundwater level|"
                          r"\bwater[- ]?table\b(?! depth)"))),
    ("survival",
     ("mortality", re.compile(r"\bmortalit|\bdeath|\bdied\b|\bdead\b|\bdie-?off")),
     ("survival", re.compile(r"\bsurviv"))),
]

# ---- fix 2.6: second-order properties of an outcome. An edge about the variability of the
# water table does not say whether the water table went up. Keyed by the word a question
# would have to contain for the property to count as asked about.
SECOND_ORDER = [
    ("variability", re.compile(r"variab|varianc|fluctuat|\b(water[\s-]?table|level|temperature) range\b")),
    ("seasonality", re.compile(r"seasonality|seasonal (variab|variation|pattern|timing|shift|dynamic|cycle)")),
    ("stability", re.compile(r"stabilit")),
    ("heterogeneity", re.compile(r"heterogen")),
    ("timing", re.compile(r"\btiming\b|\bphenolog")),
    ("rate of change", re.compile(r"rate of change|decline rate|\brate of (increase|decrease)")),
    ("similarity", re.compile(r"similar(ity)? to|resembl|comparable to")),
]


class Graph:
    def __init__(self, path="causal_graph.json"):
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
        self.nodes = {n["id"]: n for n in g["nodes"]}
        self.edges = g["edges"]
        self.edge_by_id = {e["id"]: e for e in self.edges}
        self.docs = {d["document_id"]: d for d in g["source_documents"]}
        # Precompute one lowercase searchable string per node, and one context string per edge.
        self.text = {nid: self._node_text(n) for nid, n in self.nodes.items()}
        self.context = {e["id"]: self._edge_context(e) for e in self.edges}

    @staticmethod
    def _node_text(n):
        parts = [n.get("name"), n.get("embedding_text"), n.get("entity_term"),
                 n.get("measured_attribute"), n.get("applied_to")]
        return " | ".join(str(p) for p in parts if p).lower()

    def _edge_context(self, e):
        doc = self.docs.get(e.get("source_document"), {})
        return " | ".join(str(x) for x in (e.get("ecosystem_context"), e.get("original_sentence"),
                                           doc.get("title")) if x).lower().replace("_", " ")

    # ---------- direction of a single edge ----------
    def direction(self, e):
        """increase / decrease / no_effect / unclear for one edge, read from the object node."""
        if e.get("negated") or e.get("causal_language") == "no_relationship":
            return "no_effect"                      # gotcha 2: null result
        if e["predicate"] in ("prevents", "disrupts"):
            return "decrease"                       # gotcha 3: never double-negate
        q = self.nodes[e["object"]].get("state_or_change_qualifier")
        return {1: "increase", -1: "decrease"}.get(SIGN.get(q, 0), "unclear")

    # ---------- finding candidate edges (fixes 2.2 and 2.3) ----------
    @staticmethod
    def compile_terms(terms):
        pats = []
        for t in terms or []:
            t = str(t).strip().lower()
            if not t:
                continue
            # spaces in a term also match hyphens: "water table" finds "water-table depth"
            pats.append(NATIVE if t == "native" else r"\b" + re.escape(t).replace(r"\ ", r"[\s-]+"))
        return re.compile("|".join(pats)) if pats else None

    def side_patterns(self, side):
        """Compiled (head, modifiers, exclude) regexes for one side of a validated plan."""
        head = self.compile_terms(side.get("head_terms") or side.get("terms"))
        mods = self.compile_terms(side.get("modifier_terms"))
        excl = self.compile_terms(side.get("exclude"))
        allterms = [str(t).lower() for t in (side.get("head_terms") or []) + (side.get("modifier_terms") or [])]
        # A side about natives must not match non-native / invasive nodes.
        if "native" in allterms and not any(re.search(NOT_NATIVE, t) for t in allterms):
            excl = re.compile((excl.pattern + "|" if excl else "") + NOT_NATIVE)
        return head, mods, excl

    def node_matches(self, pats, node_id, edge_id):
        """Fix 2.2: (head matched, modifiers matched). The head must match the node text and
        the exclusions must not. Modifiers may match the node or the edge's context (sentence,
        ecosystem, paper title); when they do not, the edge is still a candidate but the
        filter step is told which context word the graph never states."""
        head, mods, excl = pats
        if head is None:
            return False, False
        text = self.text[node_id]
        if not head.search(text) or (excl and excl.search(text)):
            return False, False
        mods_ok = mods is None or bool(mods.search(text) or mods.search(self.context[edge_id]))
        return True, mods_ok

    def candidates(self, plan):
        """Edges whose SUBJECT matches the cause side and whose OBJECT matches the outcome side,
        fully modifier-matched edges first. Returns (forward edges, ids whose modifiers were
        not found, number of edges that matched only in the reverse orientation)."""
        cp, op = self.side_patterns(plan["cause"]), self.side_patterns(plan["outcome"])
        full, partial, reversed_n = [], [], 0
        for e in self.edges:
            s_ok, s_mods = self.node_matches(cp, e["subject"], e["id"])
            o_ok, o_mods = self.node_matches(op, e["object"], e["id"])
            if s_ok and o_ok:
                if NEGATED_SUBJECT.search(self.nodes[e["subject"]]["name"].lower()):
                    continue
                (full if (s_mods and o_mods) else partial).append(e)
            elif self.node_matches(op, e["subject"], e["id"])[0] and self.node_matches(cp, e["object"], e["id"])[0]:
                reversed_n += 1                     # fix 2.3: runs the wrong way for this question
        return full + partial, [e["id"] for e in partial], reversed_n

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

    # ---------- fix 2.5: polarity ----------
    @staticmethod
    def attribute_class(text):
        """(pair name, class) for a node or phrase text, or None if no inverse pair applies."""
        text = str(text).lower()
        for pair, (a_name, a_re), (b_name, b_re) in INVERSE_PAIRS:
            if a_re.search(text):
                return pair, a_name
            if b_re.search(text):
                return pair, b_name
        return None

    def asked_class(self, outcome_side):
        """What the question measures, from its own phrase first, then its head terms."""
        for t in [outcome_side.get("phrase") or ""] + list(outcome_side.get("head_terms") or []):
            c = self.attribute_class(t)
            if c:
                return c
        return None

    def normalise_direction(self, e, asked):
        """(direction_raw, direction, inverted) for one edge against the asked attribute."""
        raw = self.direction(e)
        obj = self.attribute_class(self.nodes[e["object"]]["name"] + " | " + str(self.nodes[e["object"]].get("measured_attribute") or ""))
        if asked and obj and obj[0] == asked[0] and obj[1] != asked[1] and raw in ("increase", "decrease"):
            return raw, flip(raw), True
        return raw, raw, False

    # ---------- fix 2.6: second-order attributes ----------
    def second_order(self, e, question):
        """Reason string if the edge's object measures only second-order properties the
        question did not ask about, else None. An object that carries an asked-about property
        ("seasonal variability" for a variability question) counts."""
        obj = self.nodes[e["object"]]["name"].lower() + " | " + str(self.nodes[e["object"]].get("measured_attribute") or "").lower()
        q = question.lower()
        hits = [name for name, pat in SECOND_ORDER if pat.search(obj)]
        if not hits:
            return None
        if any(pat.search(q) or name.split()[0] in q for name, pat in SECOND_ORDER if name in hits):
            return None
        return f"object measures {hits[0]}, not the asked attribute"

    # ---------- fix 2.4: dedupe ----------
    @staticmethod
    def _norm(s):
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())

    def dedupe(self, edges, directions):
        """One edge per (paper, subject, object, direction). Returns (unique edges, dropped ids)."""
        seen, out, dups = set(), [], []
        for e in edges:
            key = (e["source_document"], self._norm(self.nodes[e["subject"]]["name"]),
                   self._norm(self.nodes[e["object"]]["name"]), directions[e["id"]])
            if key in seen:
                dups.append(e["id"])
                continue
            seen.add(key)
            out.append(e)
        return out, dups


# ---------- aggregation: per paper first, then across papers (gotcha 6) ----------
def flip(d):
    return {"increase": "decrease", "decrease": "increase"}.get(d, d)


def confidence_answered(n_papers, agreement):
    """Fix 2.7: README defines confidence as the probability the answer is correct; this is the
    team's calibration curve for it. Rises with the number of papers, falls with disagreement:
    1 agreeing paper -> 0.57, 3 -> 0.78, 7 -> 0.88."""
    return round(0.35 + 0.55 * agreement * (1 - 0.6 ** n_papers), 2)


def confidence_no_evidence(had_candidates):
    """0.50 when nothing matched either side's head; 0.40 when candidates existed but every one
    was filtered out (evidence may exist that did not qualify)."""
    return 0.40 if had_candidates else 0.50


def aggregate(graph, edges, directions):
    """`directions` maps edge id -> final direction (after polarity and LLM flips)."""
    per_paper = defaultdict(list)
    for e in edges:
        per_paper[e["source_document"]].append(directions[e["id"]])

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
    if direction == "mixed":
        agreement = max(vc.values()) / decided      # share of the largest group

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

    conf = confidence_answered(n_papers, agreement)
    return {"direction": direction, "certainty": certainty, "confidence": conf,
            "papers": dict(verdicts), "paper_counts": dict(vc),
            "agreement": round(agreement, 2), "n_papers": n_papers, "n_edges": len(edges)}
