"""Ground truth for the run-on bank: split each question into (construction, tail),
and find its main topic and (if any) the second topic inside the tail."""
import re, collections
qs = [m.group(2) for m in re.finditer(r"^(\d+)\. (.+)$", open("data/questions_runon.md").read(), re.M)]
old = [m.group(2) for m in re.finditer(r"^(\d+)\. (.+)$", open("data/questions.md").read(), re.M)]
TOPICS = [re.match(r"Which mechanisms most strongly determine (.+), and how", old[i * 10]).group(1) for i in range(100)]
TAILS = [  # (name, regex for the tail; group 'x' = second topic if any)
 ("and_logger", r" and the logger continued sampling after the researchers had left\?$"),
 ("which_team", r",? which the second team said were missing from every map\?$"),
 ("in_contrast", r", in contrast, the canopy surveys registered a decline in (?P<x>.+)\?$"),
 ("although", r"; although the soil cores had been collected only after the flood\?$"),
 ("however", r", however the unburned plots showed no comparable change in (?P<x>.+)\?$"),
 ("fused_crew", r" the next field crew found fewer indicator species beside the creek\?$"),
 ("and_model", r"; and the model output predicted a larger response in (?P<x>.+)\?$"),
 ("comma_splice", r", the follow-up transect recorded a sharper shift in (?P<x>.+)\?$"),
]
truth = []
for q in qs:
    for name, pat in TAILS:
        m = re.search(pat, q)
        if m:
            head, tail_topic = q[:m.start()], m.groupdict().get("x")
            break
    else:
        raise SystemExit("no tail: " + q)
    cands = [t for t in TOPICS if t in head]
    truth.append({"q": q, "tail": name, "head": head, "cands": cands, "tail_topic": tail_topic})
# pick the main topic: the candidate whose blanked construction recurs across the bank
cons = collections.Counter()
for t in truth:
    for c in t["cands"]: cons[t["head"].replace(c, "{T}")] += 1
for t in truth:
    t["main"] = max(t["cands"], key=lambda c: (cons[t["head"].replace(c, "{T}")], len(c)))
    t["construction"] = t["head"].replace(t["main"], "{T}")
if __name__ == "__main__":
    print(len(truth), "questions |", len({t['construction'] for t in truth}), "constructions |",
          collections.Counter(t["tail"] for t in truth))
    print("questions with >1 topic in the main part:", sum(len(t["cands"]) > 1 for t in truth))
    print("tail topic == main topic:", sum(t["tail_topic"] == t["main"] for t in truth))
    print("distinct (construction, tail) frames:", len({(t['construction'], t['tail']) for t in truth}))
