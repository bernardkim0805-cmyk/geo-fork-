import time, resource, io, contextlib, statistics as st
t0 = time.time()
from pipeline import Pipeline
from runon_truth import truth
import re
t_import = time.time() - t0

t0 = time.time(); P = Pipeline("data/graph.json", lambda q: None, confirm=False); t_start = time.time() - t0

# learn many templates: run-on bank (1000 shapes) + varied bank (133)
qs_var = [m.group(2) for m in re.finditer(r"^(\d+)\. (.+)$", open("data/questions_varied.md").read(), re.M)]
old = [m.group(2) for m in re.finditer(r"^(\d+)\. (.+)$", open("data/questions.md").read(), re.M)]
TOP = [re.match(r"Which mechanisms most strongly determine (.+), and how", old[i*10]).group(1) for i in range(100)]
cur = {}
P.llm = lambda q: dict(cur)
def timed(q):
    t = time.perf_counter(); r = P.parse(q); return (time.perf_counter() - t) * 1000, r["route"]

def measure(label, items):
    res = {}
    for q, ans in items:
        cur.clear(); cur.update(ans)
        ms, route = timed(q); res.setdefault(route, []).append(ms)
    print(f"{label:46}", {k: f"n={len(v)} median={st.median(v):.1f}ms p95={sorted(v)[int(.95*len(v))-1]:.1f}ms" for k, v in res.items()})

measure("varied bank, learning (0 -> 133 templates)", [(q, {"topic": TOP[i//10]}) for i, q in enumerate(qs_var)][:200])
measure("run-on bank, learning (133 -> 1133 templates)", [(t["q"], {"topic": t["main"]}) for t in truth])
P.exact = {}
measure("varied bank again, 1133 templates stored", [(q, {"topic": TOP[i//10]}) for i, q in enumerate(qs_var)])
P.llm = lambda q: None
measure("unseen questions, 1133 templates (full scan)", [(f"Does the {w} thing really matter for {w} ecology here?", {}) for w in ["fox","owl","elk","bee","ant","oak","fir","yew"]*25])
mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"\nimport {t_import:.1f}s | startup (load graph + spaCy + vocab) {t_start:.1f}s | peak memory {mem:.0f} MB | templates {len(P.frames.frames)}")
