"""Code checks on the LLM's search plan (fixes 2.1 and 2.2 of fix-agent-trap.md).

The plan step asks the LLM to turn each side of the question into search terms. The LLM
drifts: for "water table position" it added `rewetting` and `drainage` (actions that CHANGE
the water table, in opposite directions) and `hydrology` (an umbrella term). Those terms
turned D-010 into "what does rewetting do to peat?".

Plan shape per side, after validation:
    phrase          the side's phrase as the question says it
    head            the head noun ("peatland restoration" -> "restoration")
    head_terms      the phrase, the head and synonyms of the head  -> an edge node MUST match one
    modifier_terms  the other words of the phrase and synonyms     -> context filter only
    exclude         stems that would wrongly match
    dropped         [{term, reason}] every term the checks refused
    low_grounding   True when the phrase itself matches no node (type check impossible)

Rules, in the order applied to each side's head terms:
    2  opposites never share a set (checked on the raw LLM list, before anything is dropped)
    1  entity type: a head term must name the same CAMO entity_type as the phrase
    4  umbrella: a term with no word in common with the phrase and far more matches than it
       (breadth is measured inside the phrase's modifier context, since that is how head
       terms are used), or a term that matches no node at all in that context
    3  no term on both sides (cross_side, after both sides are checked)
"""
import re
from collections import Counter

# Pairs that can never both describe one side of a question.
OPPOSITES = [
    ("rewet", "drain"), ("restor", "degrad"), ("addition", "removal"), ("increas", "decreas"),
    ("planting", "clearing"), ("planting", "logging"), ("burn", "fire suppression"),
    ("grazing exclusion", "grazing"), ("flood", "dry"), ("fertiliz", "nutrient removal"),
]

GENERIC = {"application", "applications", "treatment", "treatments", "control", "management",
           "effect", "effects", "change", "species", "plant", "plants", "vegetation", "surface",
           "organic", "c", "the", "of", "and", "in", "on", "to", "a", "an", "or", "/", "&"}

UMBRELLA_RATIO = 5          # a term matching > 5x the nodes of the phrase (in context) is too broad

_Q_SHAPE = re.compile(r"what (?:effect|impact|influence)s? (?:does|do|did) (?P<x>.+?) have on (?P<y>.+?)\??$", re.I)


def phrases_from_question(question):
    """(cause phrase, outcome phrase) read from the hidden-set question shape, or (None, None)
    for any other shape (the plan's labels are used instead)."""
    m = _Q_SHAPE.search(" ".join(question.split()))
    return (m.group("x").strip(), m.group("y").strip()) if m else (None, None)


def _clean(terms):
    out = []
    for t in terms or []:
        t = " ".join(str(t).lower().split())
        if t and t not in out:
            out.append(t)
    return out


def _stem_conflict(a, b):
    """Equal, or one starts the other (peat / peatland, herbicide / herbicides)."""
    a, b = a.lower(), b.lower()
    short, long_ = sorted((a, b), key=len)
    return a == b or (len(short) >= 4 and long_.startswith(short))


def _words(s):
    return [w for w in re.findall(r"[a-z][a-z-]*", s.lower()) if w not in GENERIC]


class PlanChecker:
    def __init__(self, graph):
        self.g = graph

    # ---- what the graph knows about a term ---------------------------------------------
    def nodes_for(self, term, within=None):
        """Node ids whose text contains `term` at a word start; with `within` (a compiled
        regex of modifier terms) only nodes that also mention one of the modifiers."""
        pat = re.compile(r"\b" + re.escape(term.lower()).replace(r"\ ", r"[\s-]+"))
        return [nid for nid, t in self.g.text.items()
                if pat.search(t) and (within is None or within.search(t))]

    def named_nodes_for(self, term):
        """Nodes that ARE the thing (entity_term or name starts with the term), as opposed to
        nodes that mention it: "time since rewetting" mentions rewetting but is a duration."""
        t = term.lower()
        out = []
        for nid, n in self.g.nodes.items():
            ent = str(n.get("entity_term") or "").lower()
            name = re.sub(r"^(increased|decreased|reduced|higher|lower|greater|more|less|elevated)\s+",
                          "", str(n.get("name") or "").lower())
            if ent.startswith(t) or name.startswith(t):
                out.append(nid)
        return out

    def dominant_type(self, term):
        """Dominant CAMO entity_type of the nodes a term names, or None when it names none."""
        ids = self.named_nodes_for(term) or self.nodes_for(term)
        c = Counter(self.g.nodes[i].get("entity_type") for i in ids if self.g.nodes[i].get("entity_type"))
        return c.most_common(1)[0][0] if c else None

    # ---- one side -----------------------------------------------------------------------
    def check_side(self, side, phrase):
        phrase = " ".join((phrase or side.get("label") or "").lower().split())
        head = " ".join(str(side.get("head") or (phrase.split()[-1] if phrase else "")).lower().split())
        head_terms = _clean(side.get("head_terms") or side.get("terms"))
        mods = [t for t in _clean(side.get("modifier_terms")) if t not in GENERIC]
        if not mods and len(_words(phrase)) > 1:                   # LLM gave none: derive them
            mods = [w for w in _words(phrase) if w != head]
        if phrase and phrase not in head_terms:
            head_terms.insert(0, phrase)                           # the phrase is always a term
        if head and head not in head_terms:
            head_terms.insert(1 if head_terms else 0, head)
        dropped = []
        phrase_words = set(_words(phrase))
        phrase_type = self.dominant_type(phrase) if phrase else None

        def in_phrase(t):
            return bool(phrase) and (t in phrase or any(_stem_conflict(t, w) for w in phrase_words))

        # Rule 2: opposites, on the raw list. If the two members differ in type, the one that
        # matches the phrase's type stays; otherwise the later one goes.
        raw = head_terms + mods
        banned = set()
        for a, b in OPPOSITES:
            hits_a = [t for t in raw if a in t and not in_phrase(t)]
            hits_b = [t for t in raw if b in t and t not in hits_a and not in_phrase(t)]
            if not (hits_a and hits_b):
                continue
            ta, tb = self.dominant_type(hits_a[0]), self.dominant_type(hits_b[0])
            if phrase_type and ta != tb and phrase_type in (ta, tb):
                losers = hits_b if ta == phrase_type else hits_a
            else:
                losers = hits_b if raw.index(hits_a[0]) < raw.index(hits_b[0]) else hits_a
            for t in losers:
                banned.add(t)
                dropped.append({"term": t, "reason": f"opposites '{a}' / '{b}' cannot share one side"})
        head_terms = [t for t in head_terms if t not in banned]
        mods = [t for t in mods if t not in banned]

        # Context for breadth: a head term is only ever used together with the modifiers.
        within = re.compile("|".join(r"\b" + re.escape(m).replace(r"\ ", r"[\s-]+") for m in mods)) if mods else None
        phrase_nodes = set(self.nodes_for(phrase)) if phrase else set()
        side["low_grounding"] = not phrase_nodes

        kept = []
        for t in head_terms:
            if t == phrase:
                kept.append(t)
                continue
            shares_word = in_phrase(t) or bool(phrase_words & set(_words(t)))
            # Rule 1: same kind of thing as the phrase.
            if phrase_type and not side["low_grounding"]:
                tt = self.dominant_type(t)
                if tt and tt != phrase_type:
                    dropped.append({"term": t, "reason": f"entity type {tt} differs from the phrase's {phrase_type}"})
                    continue
            nodes = self.nodes_for(t, within)
            # Rule 4: umbrella, or nothing at all, in the phrase's context.
            if not shares_word and phrase_nodes and len(nodes) > UMBRELLA_RATIO * len(phrase_nodes):
                dropped.append({"term": t, "reason": f"umbrella: {len(nodes)} nodes in context vs {len(phrase_nodes)} for the phrase"})
                continue
            if not shares_word and not nodes:
                dropped.append({"term": t, "reason": "ungrounded: no node combines it with the phrase's modifiers"
                                + ("" if within else " (no node names it)")})
                continue
            if not shares_word and not within and not self.named_nodes_for(t):
                dropped.append({"term": t, "reason": "ungrounded: no node is named by this term"})
                continue
            kept.append(t)

        side.update({"phrase": phrase, "head": head, "head_terms": kept, "modifier_terms": mods,
                     "exclude": _clean(side.get("exclude")), "dropped": dropped})
        side["terms"] = kept + [m for m in mods if m not in kept]
        return side

    # ---- both sides ---------------------------------------------------------------------
    def cross_side(self, plan):
        """Rule 3: a term (or its stem) may not sit on both sides. It stays only on the side
        whose own phrase contains it; if neither phrase does, it leaves both."""
        c, o = plan["cause"], plan["outcome"]

        def contains(side, t):
            return t in side["phrase"] or any(_stem_conflict(t, w) for w in _words(side["phrase"]))

        for key in ("head_terms", "modifier_terms"):
            for t in list(c[key]):
                clashes = [u for k2 in ("head_terms", "modifier_terms") for u in o[k2] if _stem_conflict(t, u)]
                for u in clashes:
                    in_c, in_o = contains(c, t), contains(o, u)
                    if in_c and in_o:
                        continue
                    if not in_o:
                        for k2 in ("head_terms", "modifier_terms"):
                            if u in o[k2]:
                                o[k2].remove(u)
                                o["dropped"].append({"term": u, "reason": f"also on the cause side as '{t}'"})
                    if not in_c and t in c[key]:
                        c[key].remove(t)
                        c["dropped"].append({"term": t, "reason": f"also on the outcome side as '{u}'"})
        for s in (c, o):
            s["terms"] = s["head_terms"] + [m for m in s["modifier_terms"] if m not in s["head_terms"]]
        return plan

    def validate(self, plan, question):
        cx, ox = phrases_from_question(question)
        self.check_side(plan["cause"], cx)
        self.check_side(plan["outcome"], ox)
        return self.cross_side(plan)
