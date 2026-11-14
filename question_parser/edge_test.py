"""Pre-push edge cases: anything that crashes the pipeline or silently corrupts its output."""
import json, copy
from pipeline import Pipeline

RESULTS = []
def case(name):
    def deco(fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"CRASH {type(e).__name__}: {str(e)[:80]}"
        RESULTS.append(ok)
        print(f"{'ok    ' if ok else 'BROKEN'} {name:55} {detail}")
        return fn
    return deco

def P_(llm, confirm=False):
    return Pipeline("data/graph.json", llm, confirm=confirm)

Q = "What happens to seed dispersal after grazing?"
Q2 = "What happens to soil carbon after grazing?"

# ---------------- real API behaviour ----------------
@case("API raises (timeout / 500 / rate limit)")
def _():
    def boom(q): raise TimeoutError("api timed out")
    r = P_(boom).parse(Q); return r["route"] == "llm_failed", f"route={r['route']}"

@case("API returns a non-dict (string / list / number / None)")
def _():
    out = []
    for bad in ["{not json", ["a"], 42, None, True]:
        r = P_(lambda q, b=bad: b).parse(Q); out.append(r["route"])
    return all(x in ("llm_failed", "llm_needed") for x in out), f"routes={out}"

@case("API slot values of odd types (int, None, list, '')")
def _():
    p = {"s": 5, "o": None, "topic": ["seed dispersal"], "x": "", "triples": [{"s": "", "o": "?"}]}
    P = P_(lambda q: p); r = P.parse(Q); return r["route"] == "llm", f"route={r['route']} templates={len(P.frames.frames)}"

@case("API answer contains the internal blank marker key")
def _():
    p = {"topic": "seed dispersal", "weird": {"__blank__": "A"}}
    P = P_(lambda q: p if q == Q else None); P.parse(Q)
    r = P.parse(Q2); pp = r["parse"] or {}
    return pp.get("weird") == {"__blank__": "A"}, f"weird={pp.get('weird')!r} route={r['route']}"

@case("free-text field mentioning the topic is not reused stale")
def _():
    p = {"topic": "seed dispersal", "summary": "Asks how seed dispersal responds to grazing"}
    P = P_(lambda q: p if q == Q else None); P.parse(Q)
    r = P.parse(Q2); s = (r["parse"] or {}).get("summary")
    return s is None or "seed dispersal" not in s, f"summary={s!r}"

@case("dict KEY containing the topic is not reused stale")
def _():
    p = {"topic": "seed dispersal", "by_topic": {"seed dispersal": 1}}
    P = P_(lambda q: p if q == Q else None); P.parse(Q)
    r = P.parse(Q2); bt = (r["parse"] or {}).get("by_topic")
    return bt is None or "seed dispersal" not in bt, f"by_topic={bt!r} route={r['route']}"

# ---------------- caller behaviour ----------------
@case("caller mutating a returned answer doesn't corrupt the cache")
def _():
    P = P_(lambda q: {"topic": "seed dispersal", "kind": "x"})
    a = P.parse(Q)["parse"]; a["kind"] = "MUTATED"; a["topic"] = "garbage"
    b = P.parse(Q)["parse"]
    return b["kind"] == "x" and b["topic"] == "seed dispersal", f"second answer={b}"

@case("non-string question (None, int, bytes)")
def _():
    P = P_(lambda q: None); routes = []
    for q in [None, 42, b"bytes?"]:
        routes.append(P.parse(q)["route"])
    return True, f"routes={routes}"

@case("huge question (2 million characters)")
def _():
    r = P_(lambda q: None).parse("seed " * 400000 + "?"); return True, f"route={r['route']}"

@case("empty question answered by the LLM is not cached for all empties")
def _():
    P = P_(lambda q: {"topic": "x"}); P.parse("?"); r = P.parse("!!")
    return r["route"] != "exact_cache", f"route of a different empty input: {r['route']}"

# ---------------- question text that looks like internal syntax ----------------
@case("user types '{A}' or '⟦A⟧' in a question")
def _():
    q = "What happens to seed dispersal after {A} grazing ⟦B⟧?"
    P = P_(lambda q: {"topic": "seed dispersal"} if "seed" in q else None)
    P.parse(q); r = P.parse("What happens to soil carbon after {A} grazing ⟦B⟧?")
    tp = (r["parse"] or {}).get("topic")
    return tp in (None, "soil carbon"), f"route={r['route']} topic={tp!r}"

@case("regex characters in topic (C++, (n), *, $, \\)")
def _():
    out = []
    for t in ["C++ models", "plots (n=5)", "a*b index", "$5 fees", "back\\slash terms"]:
        q = f"What happens to {t} after grazing?"
        P = P_(lambda q, t=t: {"topic": t}); P.parse(q); out.append(len(P.frames.frames))
    return True, f"templates learned per case={out}"

@case("blank swallowing clause punctuation ('fire; and grazing')")
def _():
    P = P_(lambda q: {"topic": "seed dispersal"} if q == Q else None); P.parse(Q)
    r = P.parse("What happens to soil carbon; and the creek after grazing?")
    return r["route"] != "frame_cache", f"route={r['route']} topic={(r['parse'] or {}).get('topic')!r}"

# ---------------- probation edge ----------------
@case("probation: API fails while confirming")
def _():
    calls = {"n": 0}
    def flaky(q):
        calls["n"] += 1
        if calls["n"] == 2: raise ConnectionError("dropped")
        return {"topic": "seed dispersal" if "seed" in q else "soil carbon"}
    P = P_(flaky, confirm=True); P.parse(Q); r1 = P.parse(Q2); r2 = P.parse("What happens to tree growth after grazing?")
    return r1["route"] == "llm_failed", f"routes={r1['route']},{r2['route']} templates={[f['status'] for f in P.frames.frames]}"

@case("save/load: missing or corrupt file")
def _():
    P = P_(lambda q: None)
    open("/tmp/bad.json", "w").write("{corrupt")
    msgs = []
    for path in ["/tmp/does_not_exist.json", "/tmp/bad.json"]:
        try: P.frames.load(path); msgs.append("loaded?")
        except Exception as e: msgs.append(type(e).__name__)
    return True, f"{msgs} (raising here is acceptable: caller decides)"

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
