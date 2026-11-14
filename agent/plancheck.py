"""Code checks on the LLM's search plan (fixes 2.1 and 2.2 of fix-agent-trap.md).

The plan step asks the LLM to turn each side of the question into search terms. The LLM
drifts: for "water table position" it added `rewetting` and `drainage` (actions that CHANGE
the water table, in opposite directions) and `hydrology` (an umbrella term). Those terms
turned D-010 into "what does rewetting do to peat?".

Plan shape per side, after validation:
    phrase          the side's phrase as the question says it
    head            the head noun ("peatland restoration" -> "restoration")
    head_terms      the phrase, the head and synonyms of the head  -> an edge node MUST match one
    modifier_terms  the other words of the phrase and synonyms     -> context, judged by the filter
    exclude         stems that would wrongly match
    dropped         [{term, reason}] every term the checks refused
    low_grounding   True when the phrase itself matches no node (type check impossible)

Rules, in the order applied to each side's head terms:
    2  opposites never share a set (checked on the raw LLM list, before anything is dropped)
    1  entity type: a head term must name the same CAMO entity_type as the phrase
    4  umbrella: a term with no word in common with the phrase and far more matches than it
       (breadth measured inside the phrase's modifier context), or a term nothing in the
       graph is named by
    3  no term on both sides (cross_side, after both sides are checked)

Two things the graph forced on the design:
  * Attribute words (cover, abundance, density, level, depth, flux, survival ...) are exempt
    from rules 1 and 4. CAMO types a node by its ENTITY, so "Cover of Sphagnum" is a taxon
    node; the attribute word itself has no entity type, and attribute synonyms are legitimately
    broad on their own.
  * A phrase whose head is a frame noun ("coarse woody debris RETENTION", "herbicide
    APPLICATION") has its meaning in the modifier. The modifier terms are promoted to head
    terms, since the graph names the thing ("Woody debris presence"), not the frame.
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

# Heads that carry no meaning of their own: the modifier is the concept. Negating frames
# (exclusion, removal, cessation) are NOT here: "grazing exclusion" must not search for grazing.
FRAME_NOUNS = {"application", "applications", "treatment", "treatments", "retention", "presence",
               "use", "addition", "placement", "installation", "management", "practice", "practices",
               "program", "programs", "programme", "intervention", "action", "actions", "deployment",
               "provision", "supply"}

# Distinct intervention categories (the hidden set asks about one at a time). A synonym offered
# for one category may not be the name of another: grazing is not a synonym of mowing.
# Order matters: "grazing exclusion" must land in the exclusion/fencing category, not grazing.
CATEGORIES = [("exclusion", "exclos", "fenc"), ("mow", "cutting"), ("graz", "livestock"), ("burn", "fire"),
              ("herbicid", "glyphosate"), ("seeding", "sowing", "seed addition"), ("planting",),
              ("fertiliz", "fertilis"), ("thinning",), ("harvest", "logging", "clearcut"), ("mulch",),
              ("irrigat",), ("inoculat",), ("scarif",)]

UMBRELLA_RATIO = 5          # a term matching > 5x the nodes of the phrase (in context) is too broad
MIN_NAMED = 2               # nodes needed before a term's entity type is trusted

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
    """Equal, one starts the other (peat / peatland), or a shared stem of 6+ letters
    (fertilizer / fertilization)."""
    a, b = a.lower(), b.lower()
    short, long_ = sorted((a, b), key=len)
    if a == b or (len(short) >= 4 and long_.startswith(short)):
        return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= 6


def _words(s):
    return [w for w in re.findall(r"[a-z][a-z-]*", s.lower()) if w not in GENERIC]


def _category(term):
    for i, stems in enumerate(CATEGORIES):
        if any(st in term for st in stems):
            return i
    return None


def _pat(term):
    return r"\b" + re.escape(term.lower()).replace(r"\ ", r"[\s-]+")


class PlanChecker:
    def __init__(self, graph):
        self.g = graph
        # Attribute words: words that END a measured_attribute value at least 3 times and do so
        # more often than they START an entity_term (cover, abundance, depth, survival: yes;
        # peat, restoration, recovery, drainage: no, those are entities).
        attr_last, ent_first = Counter(), Counter()
        for n in graph.nodes.values():
            a = str(n.get("measured_attribute") or "").lower()
            if a and ":" not in a and _words(a):
                attr_last[_words(a)[-1]] += 1
            e = str(n.get("entity_term") or "").lower()
            if e and ":" not in e and _words(e):
                ent_first[_words(e)[0]] += 1
        self.attribute_words = {w for w, c in attr_last.items() if c >= 3 and c > ent_first[w]}

    # ---- what the graph knows about a term ---------------------------------------------
    def nodes_for(self, term, within=None):
        """Node ids whose text contains `term` at a word start; with `within` (a compiled
        regex of modifier terms) only nodes that also mention one of the modifiers."""
        pat = re.compile(_pat(term))
        return [nid for nid, t in self.g.text.items()
                if pat.search(t) and (within is None or within.search(t))]

    def named_nodes_for(self, term):
        """Nodes that ARE the thing (entity_term or name starts with the term), as opposed to
        nodes that mention it: "time since rewetting" mentions rewetting but is a duration."""
        pat = re.compile("^" + _pat(term).replace(r"\b", "", 1))
        out = []
        for nid, n in self.g.nodes.items():
            ent = str(n.get("entity_term") or "").lower()
            name = re.sub(r"^(increased|decreased|reduced|higher|lower|greater|more|less|elevated)\s+",
                          "", str(n.get("name") or "").lower())
            if pat.match(ent) or pat.match(name):
                out.append(nid)
        return out

    def dominant_type(self, term):
        """Dominant CAMO entity_type of the nodes a term names, or None when too few name it."""
        ids = self.named_nodes_for(term)
        if len(ids) < MIN_NAMED:
            ids = [] if ids else self.nodes_for(term)
            if len(ids) < MIN_NAMED:
                return None
        c = Counter(self.g.nodes[i].get("entity_type") for i in ids if self.g.nodes[i].get("entity_type"))
        return c.most_common(1)[0][0] if c else None

    def is_attribute(self, term):
        """True when the term's last word is an attribute word ("species richness" -> richness)."""
        ws = _words(term)
        return bool(ws) and ws[-1] in self.attribute_words

    # ---- one side -----------------------------------------------------------------------
    def check_side(self, side, phrase):
        phrase = " ".join((phrase or side.get("label") or "").lower().split())
        head = " ".join(str(side.get("head") or (phrase.split()[-1] if phrase else "")).lower().split())
        head_terms = _clean(side.get("head_terms") or side.get("terms"))
        mods = [t for t in _clean(side.get("modifier_terms")) if t not in GENERIC]
        if not mods and len(_words(phrase)) > 1:                   # LLM gave none: derive them
            mods = [w for w in _words(phrase) if w != head]
        dropped = []
        promoted = False
        if head in FRAME_NOUNS and mods:                           # the modifier is the concept
            promoted = True
            head_terms = [t for t in head_terms if t != head and t not in FRAME_NOUNS]
            for m in mods:
                if m not in head_terms:
                    head_terms.append(m)
            mods = []
        if phrase and phrase not in head_terms:
            head_terms.insert(0, phrase)                           # the phrase is always a term
        if head and head not in head_terms and not promoted:
            head_terms.insert(1 if head_terms else 0, head)
        phrase_words = set(_words(phrase))
        phrase_type = self.dominant_type(phrase) if phrase else None

        def in_phrase(t):
            return bool(phrase) and (t in phrase or any(_stem_conflict(t, w) for w in phrase_words))

        # Rule 2: opposites, on the raw list. The member whose entity type is the phrase's
        # stays when exactly one does; otherwise both go (neither is a synonym of the phrase).
        raw = head_terms + mods
        banned = set()
        for a, b in OPPOSITES:
            hits_a = [t for t in raw if a in t and not in_phrase(t)]
            hits_b = [t for t in raw if b in t and t not in hits_a and not in_phrase(t)]
            if not (hits_a and hits_b):
                continue
            ta, tb = self.dominant_type(hits_a[0]), self.dominant_type(hits_b[0])
            if phrase_type and (ta == phrase_type) != (tb == phrase_type):
                losers = hits_b if ta == phrase_type else hits_a
            else:
                losers = hits_a + hits_b
            for t in losers:
                banned.add(t)
                dropped.append({"term": t, "reason": f"opposites '{a}' / '{b}' cannot share one side"})
        head_terms = [t for t in head_terms if t not in banned]
        mods = [t for t in mods if t not in banned]

        # Context for breadth: a head term is used together with the modifiers.
        within = re.compile("|".join(_pat(m) for m in mods)) if mods else None
        phrase_nodes = set(self.nodes_for(phrase)) if phrase else set()
        side["low_grounding"] = not phrase_nodes

        kept = []
        for t in head_terms:
            if t == phrase or in_phrase(t):
                kept.append(t)
                continue
            shares_word = bool(phrase_words & set(_words(t)))
            attribute = self.is_attribute(t)                        # attributes: no entity-type reasoning
            pc, tc = _category(phrase), _category(t)
            if tc is not None and tc != pc:
                dropped.append({"term": t, "reason": "names a different intervention category"})
                continue
            # Rule 1: same kind of thing as the phrase.
            if phrase_type and not side["low_grounding"] and not attribute:
                tt = self.dominant_type(t)
                if tt and tt != phrase_type:
                    dropped.append({"term": t, "reason": f"entity type {tt} differs from the phrase's {phrase_type}"})
                    continue
            nodes = self.nodes_for(t, within)
            # Rule 4: umbrella, or nothing at all.
            if not shares_word and not attribute and phrase_nodes and len(nodes) > UMBRELLA_RATIO * len(phrase_nodes):
                dropped.append({"term": t, "reason": f"umbrella: {len(nodes)} nodes in context vs {len(phrase_nodes)} for the phrase"})
                continue
            if not shares_word and not nodes and not self.named_nodes_for(t):
                dropped.append({"term": t, "reason": "ungrounded: no node is named by it and none combines it with the phrase's modifiers"})
                continue
            kept.append(t)

        side.update({"phrase": phrase, "head": head, "head_terms": kept, "modifier_terms": mods,
                     "exclude": _clean(side.get("exclude")), "dropped": dropped,
                     "head_promoted_from_modifiers": promoted})
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
