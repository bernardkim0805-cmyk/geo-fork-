"""Run the pipeline on the run-on bank. The LLM stand-in returns the main topic and, if the
tail has one, the tail's topic, which is what a real LLM copying phrases would produce."""
import collections, json, time
from pipeline import Pipeline
from runon_truth import truth

cur = {}
def llm(q):
    p = {"kind": "standin", "topic": cur["main"], "context": []}
    if cur["tail_topic"]: p["context"].append({"topic": cur["tail_topic"]})
    return p

P = Pipeline("data/graph.json", llm)
routes, wrong, t0 = collections.Counter(), [], time.time()
for t in truth:
    cur.update(t)
    r = P.parse(t["q"])
    routes[r["route"]] += 1
    if r["route"] in ("frame_cache", "exact_cache"):
        got_main = r["parse"]["topic"]
        got_tail = r["parse"]["context"][0]["topic"] if r["parse"]["context"] else None
        if got_main != t["main"] or got_tail != t["tail_topic"]:
            wrong.append((t["q"], got_main, got_tail))
print("routes:", dict(routes), f"| {time.time()-t0:.0f}s")
print("template answers with a wrong topic:", len(wrong))
for w in wrong[:8]: print("  ", w)
