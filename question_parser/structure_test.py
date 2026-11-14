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

# =====================================================================================
# Context with roles, combination, comparison tags, spelling hints, split warnings
# =====================================================================================
import os, re, sys, tempfile
from pipeline import split_warnings

Q_BERNARD = ("When herbicides such as glyphosate are used against invasive plants in grassland resotration, "
             "do they also reduce native plants cover and richness?")
_SHAPE = re.compile(r"When (.+?) such as (.+?) are used against (.+?) in (.+?), do they also reduce (.+?)\?$")


def shaped_reply(q):
    """What a correct LLM returns for the 'When X such as Y are used against Z in W' shape."""
    m = _SHAPE.match(q)
    cls, mem, target, setting, outcome = m.groups()
    return {"kind": "intervention_outcomes", "topic": outcome,
            "triples": [{"s": {"class": cls, "members": [mem], "relation": "exemplar"},
                         "p": "any", "o": outcome, "o_qualifier": ["decreased"]}],
            "context": [{"text": target, "role": "purpose"}, {"text": setting, "role": "setting"}]}


def with_model(reply_fn, fn):
    """Run fn() with llm.call_model answering reply_fn(question) as JSON (100 tokens a call)."""
    orig = llm.call_model
    def fake(system, user, max_tokens=800):
        q = user.split("Q: ", 1)[1].split("\n", 1)[0]
        return json.dumps(reply_fn(q)), 100
    llm.call_model = fake
    try:
        return fn()
    finally:
        llm.call_model = orig


@case("validate accepts context items {text, role}")
def _():
    llm.validate(shaped_reply(Q_BERNARD), Q_BERNARD); return True, "accepted"

@case("validate rejects old string list / unknown role / invented text / extra keys")
def _():
    bad = [["invasive plants", "grassland resotration"],
           [{"text": "invasive plants", "role": "target"}],
           [{"text": "tallgrass prairie", "role": "setting"}],
           [{"text": "invasive plants", "role": "purpose", "why": "x"}],
           [{"text": "", "role": "setting"}],
           "grassland resotration"]
    rej = []
    for ctx in bad:
        p = shaped_reply(Q_BERNARD); p["context"] = ctx
        try: llm.validate(p, Q_BERNARD)
        except ValueError: rej.append(True); continue
        rej.append(False)
    # the same rules apply to a per-triple context
    p = shaped_reply(Q_BERNARD); p["triples"][0]["context"] = ["grassland resotration"]
    try: llm.validate(p, Q_BERNARD); rej.append(False)
    except ValueError: rej.append(True)
    return all(rej), f"{sum(rej)}/{len(rej)} rejected"

@case("strip_articles also cleans context text")
def _():
    p = llm.strip_articles({"kind": "x", "triples": [{"s": "a", "o": "b", "context": [{"text": "the forests", "role": "setting"}]}],
                            "context": [{"text": "A tallgrass prairie", "role": "setting"}]})
    return (p["context"][0]["text"], p["triples"][0]["context"][0]["text"]) == ("tallgrass prairie", "forests"), "ok"

@case("Bernard's question: 2 sub-questions, both carry setting + purpose")
def _():
    def run():
        P = Pipeline("data/graph.json", llm.llm_parse, confirm=False)
        return P.parse(Q_BERNARD)
    r = with_model(shaped_reply, run)
    subs = r["subquestions"]
    want_ctx = [{"text": "invasive plants", "role": "purpose"}, {"text": "grassland resotration", "role": "setting"}]
    ok = (r["route"] == "llm" and len(subs) == 2
          and [(x["s"], x["s_role"]) for x in subs] == [("herbicides", "class"), ("glyphosate", "member")]
          and all(x["context"] == want_ctx for x in subs)
          and all(x["s_group"] == "exemplar" and x["o_group"] is None for x in subs)
          and subs[0]["group_id"] == subs[1]["group_id"] == "t0:s"
          and r["spelling"].get("resotration") == "restoration")
    return ok, f"route={r['route']} n={len(subs)} spelling={r['spelling']} group_id={subs[0]['group_id'] if subs else None}"

@case("per-triple context replaces the question-level context")
def _():
    p = {"kind": "intervention_outcomes", "topic": "bird abundance",
         "context": [{"text": "Ontario", "role": "setting"}],
         "triples": [{"s": "burning", "p": "any", "o": "bird abundance", "context": [{"text": "grasslands", "role": "setting"}]},
                     {"s": "thinning", "p": "any", "o": "bird abundance"}]}
    subs, _ = expand_subquestions(p)
    return ([x["context"][0]["text"] for x in subs] == ["grasslands", "Ontario"]), f"{[x['context'] for x in subs]}"

@case("context never changes the number of sub-questions")
def _():
    base = {"triples": [{"s": {"class": "herbicides", "members": ["glyphosate", "picloram"], "relation": "exemplar"},
                         "p": "any", "o": {"class": None, "members": ["cover", "richness"], "relation": "list"}}]}
    n0 = len(expand_subquestions(base)[0])
    many = json.loads(json.dumps(base))
    many["context"] = [{"text": t, "role": r} for t, r in
                       [("a", "setting"), ("b", "purpose"), ("c", "time"), ("d", "population"), ("e", "setting")]]
    many["triples"][0]["context"] = [{"text": "f", "role": "setting"}, {"text": "g", "role": "time"}]
    n1 = len(expand_subquestions(many)[0])
    return n0 == n1 == 6, f"without context {n0}, with 7 context items {n1}"

@case("combination: ONE sub-question whose s is the member list")
def _():
    q = "Does deer browsing plus competing vegetation reduce seedling survival?"
    reply = {"kind": "interaction", "topic": "seedling survival",
             "triples": [{"s": {"class": None, "members": ["deer browsing", "competing vegetation"],
                                "relation": "combination"}, "p": "any", "o": "seedling survival"}]}
    llm.validate(json.loads(json.dumps(reply)), q)
    P = Pipeline("data/graph.json", lambda x: reply, confirm=False)
    r = P.parse(q)
    subs = r["subquestions"]
    ok = (len(subs) == 1 and subs[0]["s"] == ["deer browsing", "competing vegetation"]
          and subs[0]["s_role"] == "combination" and subs[0]["s_group"] == "combination")
    return ok, f"n={len(subs)} s={subs[0]['s'] if subs else None}"

@case("combination needs 2 members; 'combined with' requires a combination group")
def _():
    q = "Does burning combined with glyphosate reduce invasive grass cover?"
    one = {"kind": "x", "triples": [{"s": {"class": None, "members": ["burning"], "relation": "combination"},
                                     "p": "any", "o": "invasive grass cover"}]}
    flat = {"kind": "x", "triples": [{"s": "burning", "p": "any", "o": "invasive grass cover"}]}
    rej = 0
    for p in (one, flat):
        try: llm.validate(p, q)
        except ValueError: rej += 1
    return rej == 2 and "combination" in llm.detect_markers(q) and not llm.detect_markers(
        "Plus, does burning reduce cover?"), f"{rej}/2 rejected"

@case("contrast sub-questions share a group_id")
def _():
    p = {"triples": [{"s": {"class": None, "members": ["partial harvesting", "clearcutting"], "relation": "contrast"},
                      "p": "any", "o": "bird richness"}]}
    subs, _ = expand_subquestions(p)
    return (len(subs) == 2 and len({x["group_id"] for x in subs}) == 1 and subs[0]["group_id"] == "t0:s"
            and all(x["s_group"] == "contrast" for x in subs)), f"{[x['group_id'] for x in subs]}"

@case("cap keeps the class question and reports how many were dropped")
def _():
    p = {"triples": [{"s": {"class": "herbicides", "members": ["a", "b", "c", "d", "e", "f", "g"], "relation": "exemplar"},
                      "p": "any", "o": "cover"}]}
    subs, dropped = expand_subquestions(p, count_dropped=True)
    P = Pipeline("data/graph.json", lambda q: None, confirm=False)
    return subs[0]["s_role"] == "class" and dropped == 2 and len(subs) == 6, f"first={subs[0]['s']} dropped={dropped}"

def _probation_run(q1, q2, q3):
    def run():
        P = Pipeline("data/graph.json", llm.llm_parse, confirm=True)
        r1, r2 = P.parse(q1), P.parse(q2)
        llm.call_model = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("LLM must not be called"))
        return P, r1, r2, P.parse(q3)
    return with_model(shaped_reply, run)

@case("template with context blanks: probation, then reused for a new setting")
def _():
    # Same sentence shape as Bernard's question. The outcome is "native plant richness" because
    # spaCy tags "cover" in "native plants cover and richness" as a VERB, and the noun-phrase rule
    # (not loosened here) refuses that blank; see the next case.
    tail = "do they also reduce native plant richness?"
    q1 = f"When herbicides such as glyphosate are used against invasive plants in grassland resotration, {tail}"
    q2 = f"When fungicides such as captan are used against invasive plants in wetland restoration, {tail}"
    q3 = f"When insecticides such as imidacloprid are used against invasive plants in boreal peatlands, {tail}"
    P, r1, r2, r3 = _probation_run(q1, q2, q3)
    ctx = (r3["parse"] or {}).get("context") or []
    setting = [c["text"] for c in ctx if c["role"] == "setting"]
    ok = (r1["route"] == "llm" and r2["route"] == "llm" and P.counts["confirmed"] == 1
          and r3["route"] == "frame_cache" and setting == ["boreal peatlands"]
          and r3["subquestions"][0]["context"] == ctx)
    return ok, f"routes={r1['route']},{r2['route']},{r3['route']} confirmed={P.counts['confirmed']} setting={setting}"

@case("Bernard's exact wording: outcome blank refused ('cover' tagged as a verb), safe fallback to LLM")
def _():
    q2 = ("When fungicides such as captan are used against invasive plants in wetland restoration, "
          "do they also reduce native plants cover and richness?")
    P = Pipeline("data/graph.json", lambda q: shaped_reply(q), confirm=False)
    P.parse(Q_BERNARD)
    m = P.frames.match(q2)
    refused = bool(m) and "rejected" in m and "native plants cover and richness" in m["bad_fillers"]
    ctx_ok = all(P.frames._is_noun_phrase(x) for x in ("invasive plants", "wetland restoration"))
    return refused and ctx_ok, ("outcome blank refused, context blanks accepted" if refused and ctx_ok else str(m)[:90])

@case("frames.json from version 1 is discarded on load; version 2 round-trips")
def _():
    d = tempfile.mkdtemp()
    old, new = os.path.join(d, "v1.json"), os.path.join(d, "v2.json")
    json.dump([{"frame": "Does {A} reduce {B}?", "template": {"s": "x"}, "recompute": [], "status": "active"}], open(old, "w"))
    P = Pipeline("data/graph.json", lambda q: {"kind": "x", "topic": "soil carbon"}, confirm=False)
    P.frames.load(old)
    n_old = len(P.frames.frames)
    P.parse("Does fire reduce soil carbon?"); P.frames.save(new)
    P2 = Pipeline("data/graph.json", lambda q: None, confirm=False); P2.frames.load(new)
    return n_old == 0 and len(P2.frames.frames) == 1 and json.load(open(new))["version"] == 2, \
        f"v1 kept {n_old}, v2 reloaded {len(P2.frames.frames)}"

@case("split warnings: 'mowing and logging' flat yes, compound topic no")
def _():
    P = Pipeline("data/graph.json", lambda q: None, confirm=False)
    q1 = "Does mowing and logging reduce bird abundance?"
    w1 = split_warnings({"triples": [{"s": "mowing and logging", "p": "any", "o": "bird abundance"}]}, q1, P.terms)
    q2 = "How does grazing affect population growth and regulation?"
    w2 = split_warnings({"triples": [{"s": "grazing", "p": "any", "o": "population growth and regulation"}]}, q2, P.terms)
    return len(w1) == 1 and "unsplit" in w1[0] and not w2, f"{w1} / {w2}"

@case("spelling hints are output only (not in the cached parse)")
def _():
    P = Pipeline("data/graph.json", lambda q: shaped_reply(q), confirm=False)
    r = P.parse(Q_BERNARD)
    stored = json.dumps(P.exact) + json.dumps([f["template"] for f in P.frames.frames], default=str)
    return r["spelling"] and "restoration" not in stored.replace("resotration", ""), f"spelling={r['spelling']}"

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
