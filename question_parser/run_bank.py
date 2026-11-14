"""Run the real pipeline on the 1,000-question bank against the real graph."""
import re, json, sys, collections
from pipeline import Pipeline
from llm_standin import standin_llm

questions = [m.group(2) for m in re.finditer(r"^(\d+)\. (.+)$", open("data/questions.md").read(), re.M)]
print(f"{len(questions)} questions loaded")

P = Pipeline("data/graph.json", standin_llm)
print(f"vocabulary from graph: {P.vocab_size[0]} terms, {P.vocab_size[1]} ecosystem contexts\n")

show = {1, 2, 3, 11, 12, 461, 462, 1000}          # question numbers to print a full trace for
routes = collections.Counter()
results = []
for i, q in enumerate(questions, 1):
    r = P.parse(q)
    routes[r["route"]] += 1
    results.append({"n": i, "question": q, **r})
    if i in show or r["route"] in ("llm_needed",) and routes["llm_needed"] <= 5:
        print(f"Q{i}: {q}")
        for step in r["trace"]: print("   " + step)
        if r["parse"]: print("   OUTPUT:", json.dumps(r["parse"]))
        print()

print("routes:", dict(routes))
print("LLM calls made:", P.counts["llm"], "| LLM calls still needed:", P.counts["llm_failed"])
P.frames.save("frames_test_bank.json")
json.dump(results, open("bank_results.json", "w"), indent=1)
