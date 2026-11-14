"""Run the real pipeline on the VARIED 1,000-question bank and check every cached answer.

LLM stand-in for this run: whenever the pipeline reaches layer 3, we count one LLM call
and return a minimal parse that contains the question's true topic (taken from the bank's
block structure: questions 1-10 share topic 1, 11-20 topic 2, ...). A real LLM would return a
richer parse, but for counting calls and testing template reuse, the topic is what matters:
it is the phrase the template learner turns into a blank.
"""
import re, json, collections
from pipeline import Pipeline

qs = [m.group(2) for m in re.finditer(r"^(\d+)\. (.+)$", open("data/questions_varied.md").read(), re.M)]
old = [m.group(2) for m in re.finditer(r"^(\d+)\. (.+)$", open("data/questions.md").read(), re.M)]
TOPICS = [re.match(r"Which mechanisms most strongly determine (.+), and how", old[i * 10]).group(1)
          for i in range(100)]

current_topic = None
def counting_llm(question):
    return {"kind": "standin", "topic": current_topic, "triples": []}

P = Pipeline("data/graph.json", counting_llm)
routes, wrong, out = collections.Counter(), [], []
for i, q in enumerate(qs):
    current_topic = TOPICS[i // 10]
    r = P.parse(q)
    routes[r["route"]] += 1
    if r["route"] == "frame_cache" and r["parse"].get("topic") != current_topic:
        wrong.append((i + 1, q, r["parse"].get("topic"), current_topic))
    out.append({"n": i + 1, "question": q, "true_topic": current_topic, **r})

print("routes:", dict(routes))
print("LLM calls:", P.counts["llm"], "| templates learned:", len(P.frames.frames))
print("cached answers with the WRONG topic:", len(wrong))
for w in wrong[:10]: print("  WRONG", w)
refused = collections.Counter(s.split("rejected ")[1] for x in out for s in x["trace"] if "blank rejected" in s)
print("templates matched but blank refused:", sum(refused.values()))
for k, v in refused.most_common(8): print("  ", v, k)
json.dump(out, open("varied_results.json", "w"), indent=1)
