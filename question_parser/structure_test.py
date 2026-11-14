"""Tests for structured sides (groups) and sub-questions.

"herbicides such as glyphosate" is a class plus a member; "population growth and regulation" is
one topic. The parse must keep that distinction, templates must carry it, and the number of
sub-questions must fall out of the parse."""
import json
import llm
from pipeline import Pipeline, expand_subquestions

RESULTS = []
def case(name):
    def deco(fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"CRASH {type(e).__name__}: {str(e)[:90]}"
        RESULTS.append(ok)
        print(f"{'ok    ' if ok else 'BROKEN'} {name:60} {detail}")
        return fn
    return deco

Q_HERB = "Do herbicides such as glyphosate increase the population of Oryza sativa?"
GOOD = {"kind": "intervention_outcomes", "topic": "population of Oryza sativa",
        "triples": [{"s": {"class": "herbicides", "members": ["glyphosate"], "relation": "exemplar"},
                     "p": "any", "o": "population of Oryza sativa"}]}
FLAT = {"kind": "intervention_outcomes", "topic": "the population of Oryza sativa",
        "triples": [{"s": "herbicides such as glyphosate", "p": "any", "o": "the population of Oryza sativa"}]}

# ---------------- reply validation ----------------
@case("validate accepts a correct exemplar group")
def _():
    llm.validate(json.loads(json.dumps(GOOD)), Q_HERB); return True, "accepted"

@case("validate rejects a flat parse when 'such as' needs a group")
def _():
    try: llm.validate(json.loads(json.dumps(FLAT)), Q_HERB); return False, "accepted a flat parse"
    except ValueError as e: return "exemplar" in str(e), str(e)[:70]

@case("validate rejects invented members / bad relation / empty members")
def _():
    bad = [{"class": "herbicides", "members": ["atrazine"], "relation": "exemplar"},
           {"class": "herbicides", "members": ["glyphosate"], "relation": "kinda"},
           {"class": "herbicides", "members": [], "relation": "exemplar"},
           {"class": None, "members": ["glyphosate"], "relation": "exemplar"}]
    rej = 0
    for g in bad:
        p = json.loads(json.dumps(GOOD)); p["triples"][0]["s"] = g
        try: llm.validate(p, Q_HERB)
        except ValueError: rej += 1
    return rej == len(bad), f"{rej}/{len(bad)} rejected"

@case("compound topic stays one phrase (no group required)")
def _():
    q = "How does grazing affect population growth and regulation?"
    p = {"kind": "intervention_outcomes", "topic": "population growth and regulation",
         "triples": [{"s": "grazing", "p": "any", "o": "population growth and regulation"}]}
    llm.validate(p, q); return True, "accepted flat compound topic"

@case("detector ignores loose uses (as such, like, rather than, compared with)")
def _():
    qs = ["As such, does grazing reduce soil carbon?", "Does it work in such a way that X?",
          "Would plots like this recover?", "If the rarest, rather than the typical, drove it?",
          "How does X compare, compared with natural wetlands?"]
    hits = [q for q in qs if llm.detect_markers(q)]
    return not hits, f"false alarms: {hits}"

@case("detector catches such as / including / especially / e.g. / versus")
def _():
    qs = ["Do herbicides such as glyphosate help?", "Do pests including aphids matter?",
          "Do grasses, especially fescue, spread?", "Do shrubs (e.g. alder) fix nitrogen?",
          "Partial harvesting versus clearcutting?"]
    return all(llm.detect_markers(q) for q in qs), "all caught"

@case("corrective retry: flat first reply, grouped second reply")
def _():
    replies = [json.dumps(FLAT), json.dumps(GOOD)]
    orig = llm.call_model
    llm.call_model = lambda system, user, max_tokens=800: (replies.pop(0), 100)
    try:
        parse, tokens = llm.llm_parse(Q_HERB)
    finally:
        llm.call_model = orig
    return parse["triples"][0]["s"]["relation"] == "exemplar" and tokens == 200, f"tokens={tokens}"

@case("two bad replies -> raises (pipeline reports llm_failed)")
def _():
    orig = llm.call_model
    llm.call_model = lambda system, user, max_tokens=800: (json.dumps(FLAT), 100)
    try:
        P = Pipeline("data/graph.json", llm.llm_parse, confirm=False)
        r = P.parse(Q_HERB)
    finally:
        llm.call_model = orig
    return r["route"] == "llm_failed", f"route={r['route']} tokens={r['tokens']}"

@case("leading articles are dropped from every phrase")
def _():
    raw = {"kind": "x", "topic": "the population of Oryza sativa",
           "triples": [{"s": {"class": "the herbicides", "members": ["a glyphosate"], "relation": "exemplar"},
                        "p": "any", "o": "An outcome"}]}
    p = llm.strip_articles(raw)
    t0 = p["triples"][0]
    return (p["topic"], t0["s"]["class"], t0["s"]["members"], t0["o"]) == \
        ("population of Oryza sativa", "herbicides", ["glyphosate"], "outcome"), json.dumps(t0)[:70]

@case("end to end: LLM reply with 'the' becomes a reusable template")
def _():
    orig = llm.call_model
    reply = {"kind": "intervention_outcomes", "topic": "the population of Oryza sativa",
             "triples": [{"s": {"class": "herbicides", "members": ["glyphosate"], "relation": "exemplar"},
                          "p": "any", "o": "the population of Oryza sativa"}]}
    llm.call_model = lambda system, user, max_tokens=800: (json.dumps(reply), 100)
    try:
        P = Pipeline("data/graph.json", llm.llm_parse, confirm=False)
        P.parse(Q_HERB)
        llm.call_model = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not be called"))
        r = P.parse("Do insecticides such as imidacloprid increase the population of Apis mellifera?")
    finally:
        llm.call_model = orig
    p = r["parse"] or {}
    return r["route"] == "frame_cache" and p["triples"][0]["s"]["members"] == ["imidacloprid"], \
        f"route={r['route']} o={p.get('triples', [{}])[0].get('o')!r}"

# ---------------- templates carry groups ----------------
@case("template learned from a group question generalises")
def _():
    P = Pipeline("data/graph.json", lambda q: GOOD if q == Q_HERB else None, confirm=False)
    P.parse(Q_HERB)
    r = P.parse("Do insecticides such as imidacloprid increase the population of Apis mellifera?")
    s = (r["parse"] or {}).get("triples", [{}])[0].get("s")
    return s == {"class": "insecticides", "members": ["imidacloprid"], "relation": "exemplar"}, \
        f"route={r['route']} s={s}"

@case("'such as' never hides inside a single-topic blank")
def _():
    q1 = "What happens to seed dispersal after grazing?"
    P = Pipeline("data/graph.json", lambda q: {"kind": "x", "topic": "seed dispersal"} if q == q1 else None,
                 confirm=False)
    P.parse(q1)
    r = P.parse("What happens to herbivores such as deer after grazing?")
    return r["route"] != "frame_cache", f"route={r['route']}"

@case("two KNOWN terms joined by 'and' are not one blank")
def _():
    q1 = "What happens to seed dispersal after grazing?"
    P = Pipeline("data/graph.json", lambda q: {"kind": "x", "topic": "seed dispersal"} if q == q1 else None,
                 confirm=False)
    P.parse(q1)
    r = P.parse("What happens to mowing and logging after grazing?")
    return r["route"] != "frame_cache", f"route={r['route']}"

@case("compound topic with one known word still fills a blank")
def _():
    q1 = "What happens to seed dispersal after grazing?"
    P = Pipeline("data/graph.json", lambda q: {"kind": "x", "topic": "seed dispersal"} if q == q1 else None,
                 confirm=False)
    P.parse(q1)
    r = P.parse("What happens to population growth and regulation after grazing?")
    return r["route"] == "frame_cache", f"route={r['route']} topic={(r['parse'] or {}).get('topic')!r}"

# ---------------- sub-questions ----------------
@case("herbicides such as glyphosate -> 2 sub-questions (class + member)")
def _():
    subs, tr = expand_subquestions(GOOD)
    roles = [(x["s"], x["s_role"]) for x in subs]
    return roles == [("herbicides", "class"), ("glyphosate", "member")] and not tr, f"{roles}"

@case("2 causes x 2 outcomes -> 4 sub-questions")
def _():
    p = {"triples": [{"s": {"class": None, "members": ["fire", "grazing"], "relation": "list"}, "p": "any",
                      "o": {"class": None, "members": ["survival", "growth"], "relation": "list"}}]}
    subs, tr = expand_subquestions(p); return len(subs) == 4 and not tr, f"n={len(subs)}"

@case("expansion is capped at 6 and says so")
def _():
    p = {"triples": [{"s": {"class": "herbicides", "members": ["a", "b", "c", "d"], "relation": "exemplar"},
                      "p": "any", "o": {"class": None, "members": ["x", "y"], "relation": "list"}}]}
    subs, tr = expand_subquestions(p); return len(subs) == 6 and tr, f"n={len(subs)} truncated={tr}"

@case("plain question -> 1 sub-question; no triples -> 0")
def _():
    a, _ = expand_subquestions({"triples": [{"s": "grazing", "p": "any", "o": "soil carbon"}]})
    b, _ = expand_subquestions({"topic": "x", "triples": []})
    return len(a) == 1 and len(b) == 0, f"{len(a)}, {len(b)}"

@case("pipeline output carries subquestions / n_subquestions")
def _():
    P = Pipeline("data/graph.json", lambda q: GOOD, confirm=False)
    r = P.parse(Q_HERB)
    return r.get("n_subquestions") == 2 and "subquestions" in r, f"n={r.get('n_subquestions')}"

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
