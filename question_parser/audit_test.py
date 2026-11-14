"""Audit: attack the assumptions in pipeline.py that the earlier tests never touched."""
import time, json, traceback
from pipeline import Pipeline

def fresh(llm, confirm=False):
    return Pipeline("data/graph.json", llm, confirm=confirm)

def check(name, fn):
    try:
        ok, detail = fn()
        print(f"{'ok    ' if ok else 'BROKEN'} {name:58} {detail}")
    except Exception as e:
        print(f"{'CRASH '} {name:58} {type(e).__name__}: {str(e)[:70]}")

# 1. Learning blanks a phrase by plain text replacement: does it also hit the phrase INSIDE other words?
def substring_blanking():
    q1 = "Does mulching increase plant cover and seedling recovery?"
    P = fresh(lambda q: {"topic": "cover", "kind": "x"} if q == q1 else None)
    P.parse(q1)
    frame = P.frames.frames[0]["frame"] if P.frames.frames else None
    return ("recovery" in (frame or "")), f"learned frame: {frame}"

# 2. Phrase replacement inside the stored ANSWER (JSON text): does it corrupt other fields?
def json_corruption():
    q1 = "What happens to cover after grazing?"
    parse = {"kind": "recovery_question", "topic": "cover", "note": "discovery"}
    P = fresh(lambda q: parse if q == q1 else None)
    P.parse(q1)
    r = P.parse("What happens to seed dispersal after grazing?")
    p = r["parse"] or {}
    return p.get("kind") == "recovery_question" and p.get("note") == "discovery" and p.get("topic") == "seed dispersal", \
        f"kind={p.get('kind')!r} note={p.get('note')!r} topic={p.get('topic')!r}"

# 3. One phrase contained in another ("seed dispersal" and "dispersal"): unfilled blanks left behind?
def nested_phrases():
    q1 = "Does seed dispersal limit dispersal distance?"
    parse = {"s": "seed dispersal", "o": "dispersal distance", "topic": "dispersal"}
    P = fresh(lambda q: parse if q == q1 else None)
    P.parse(q1)
    r = P.parse("Does seed predation limit dispersal distance?")
    out = json.dumps(r["parse"]) if r["parse"] else r["route"]
    return "{" not in out.replace('{"', "").replace('{ "', ""), out[:110]

# 4. Fields the LLM DERIVED from the topic (entity type, ontology id) get copied to a new topic.
def derived_fields():
    q1 = "Does prescribed burning reduce soil carbon?"
    parse = {"s": "prescribed burning", "s_entity_type": "management_intervention",
             "o": "soil carbon", "o_entity_term": "ENVO:soil_carbon"}
    P = fresh(lambda q: parse if q == q1 else None)
    P.parse(q1)
    r = P.parse("Does wolf predation reduce elk abundance?")
    p = r["parse"] or {}
    return p.get("s_entity_type") != "management_intervention", \
        f"s={p.get('s')!r} s_entity_type={p.get('s_entity_type')!r} o_entity_term={p.get('o_entity_term')!r}"

# 5. Template poisoning: ONE bad LLM answer (subject/object swapped) is copied to every later question.
def poisoning():
    q1 = "Does fire reduce seed dispersal?"
    bad = {"s": "seed dispersal", "o": "fire"}          # the LLM got it backwards once
    import re as _re
    def llm(q):
        if q == q1: return bad
        m = _re.match(r"Does (.+) reduce (.+)\?", q)
        return {"s": m.group(1), "o": m.group(2)}       # correct every other time
    P = fresh(llm, confirm=True)
    P.parse(q1)
    outs = [P.parse(f"Does {a} reduce {b}?")["parse"] for a, b in
            [("grazing", "soil carbon"), ("drought", "tree growth"), ("logging", "bird abundance")]]
    swapped = sum(o and o["s"] == b for o, (a, b) in zip(outs, [("grazing", "soil carbon"), ("drought", "tree growth"), ("logging", "bird abundance")]))
    return swapped == 0, f"{swapped}/3 later questions inherited the swapped answer (dropped={P.counts['dropped']}, confirmed={P.counts['confirmed']})"

# 6. More than 8 phrases in one LLM answer (blank names only go A..H).
def many_blanks():
    words = ["fire", "grazing", "drought", "logging", "mowing", "liming", "flooding", "thinning", "seeding"]
    q1 = "Do " + ", ".join(words) + " affect soil carbon?"
    parse = {"triples": [{"s": w, "o": "soil carbon"} for w in words]}
    P = fresh(lambda q: parse if q == q1 else None)
    r = P.parse(q1)
    return len(P.frames.frames) == 0, f"route={r['route']}, templates learned={len(P.frames.frames)} (too many phrases: answer kept, no template)"

# 7. Junk input.
def junk_inputs():
    P = fresh(lambda q: None)
    res = []
    for q in ["", "   ", "?", "????", "\n\t", "a", "🌲🔥?", "x" * 5000]:
        r = P.parse(q); res.append(r["route"])
    return True, f"routes={set(res)}"

# 8. Speed: long question vs a template with several blanks (blank search tries every split).
def slow_matching():
    q1 = "How do fire and grazing and drought and logging affect soil carbon?"
    parse = {"triples": [{"s": "fire", "o": "soil carbon"}, {"s": "grazing", "o": "soil carbon"},
                         {"s": "drought", "o": "soil carbon"}, {"s": "logging", "o": "soil carbon"}]}
    P = fresh(lambda q: parse if q == q1 else None)
    P.parse(q1)
    long_q = "How do " + " and ".join(["fire"] * 60) + " affect soil carbon?"
    t = time.time(); P.parse(long_q); dt = time.time() - t
    return dt < 2, f"{dt:.2f}s for a 60-'and' question"

# 9. A matching-but-refused template stops the search: a later, safe template is never tried.
def first_refusal_blocks():
    qa = "Which interventions can restore cover?"
    qb = "Which interventions can restore the opposite of cover?"
    answers = {qa: {"topic": "cover", "polarity": "same"}, qb: {"topic": "cover", "polarity": "opposite"}}
    P = fresh(lambda q: answers.get(q))
    P.parse(qa); P.parse(qb)                      # second one goes to the LLM and is learned
    P.llm = lambda q: None
    r = P.parse("Which interventions can restore the opposite of seed dispersal?")
    p = r["parse"] or {}
    return r["route"] == "frame_cache" and p.get("polarity") == "opposite", f"route={r['route']} polarity={p.get('polarity')!r}"

# 10. LLM copies a phrase with different capitals/punctuation than the question.
def llm_case_mismatch():
    q1 = "Does fire reduce Seed Dispersal?"
    P = fresh(lambda q: {"topic": "seed dispersal"} if q == q1 else None)
    P.parse(q1)
    return len(P.frames.frames) == 1, f"templates learned: {len(P.frames.frames)}"

# 11. Templates survive a restart?
def persistence():
    q1 = "Does fire reduce seed dispersal?"
    P = fresh(lambda q: {"topic": "seed dispersal"})
    P.parse(q1); P.frames.save("/tmp/frames_test.json")
    P2 = fresh(lambda q: None)
    loader = getattr(P2.frames, "load", None)
    if loader: loader("/tmp/frames_test.json")
    r = P2.parse("Does fire reduce soil carbon?")
    return r["route"] == "frame_cache", f"after restart: {r['route']} (load method exists: {bool(loader)})"

for name, fn in [("1 phrase blanked inside other words", substring_blanking),
                 ("2 stored answer corrupted by phrase replace", json_corruption),
                 ("3 nested phrases leave unfilled blanks", nested_phrases),
                 ("4 derived fields copied to a different topic", derived_fields),
                 ("5 one bad LLM answer poisons later questions", poisoning),
                 ("6 more than 8 phrases in one answer", many_blanks),
                 ("7 junk input (empty, emoji, 5000 chars)", junk_inputs),
                 ("8 slow matching on long questions", slow_matching),
                 ("9 refused template blocks a safe one", first_refusal_blocks),
                 ("10 LLM phrase differs in case from question", llm_case_mismatch),
                 ("11 templates survive a restart", persistence)]:
    check(name, fn)
