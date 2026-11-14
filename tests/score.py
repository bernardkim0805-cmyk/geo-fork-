"""Score an answers.json against a gold file.

Usage: python tests/score.py ANSWERS.json GOLD.yaml [GOLD2.yaml ...]

Only ids present in a gold file are scored. A question counts as correct when
the status matches and, for answered questions, the direction is the gold
direction or one of its `also_ok` alternatives.
Brier = mean (confidence - correct)^2: 0 is perfect, 0.25 is "always 0.5".
"""
import json, sys
import yaml

answers = {a["id"]: a for a in json.load(open(sys.argv[1]))["answers"]}
gold = {}
for path in sys.argv[2:]:
    gold.update(yaml.safe_load(open(path))["gold"])

rows, brier, n_status, n_dir, n_dir_total, n_ok = [], 0.0, 0, 0, 0, 0
for qid, g in gold.items():
    a = answers.get(qid)
    if a is None:
        rows.append((qid, "MISSING", "", "", "", "")); brier += 1; continue
    status_ok = a.get("status") == g["status"]
    okdirs = [g.get("direction")] + list(g.get("also_ok") or [])
    if g["status"] == "answered":
        n_dir_total += 1
        dir_ok = a.get("direction") in okdirs
        n_dir += dir_ok
        correct = status_ok and dir_ok
    else:
        # also_ok on a no_evidence gold means "answering with this direction is defensible too"
        correct = status_ok or (a.get("status") == "answered" and a.get("direction") in okdirs[1:])
    n_status += status_ok; n_ok += correct
    conf = float(a.get("confidence") or 0)
    brier += (conf - correct) ** 2
    trap = "TRAP" in (g.get("note") or "")
    got = a.get("direction") or a.get("status")
    want = g.get("direction") or g["status"]
    rows.append((qid, "ok" if correct else "WRONG", got, want, f"{conf:.2f}", "trap" if trap else ""))

w = max(len(r[0]) for r in rows)
print(f"{'id':<{w}}  result  got              gold             conf")
for r in rows:
    print(f"{r[0]:<{w}}  {r[1]:<6}  {str(r[2]):<15}  {str(r[3]):<15}  {r[4]:<5} {r[5]}")
n = len(gold)
traps = [r for r in rows if r[5] == "trap"]
print(f"\ncorrect {n_ok}/{n} | status {n_status}/{n} | direction {n_dir}/{n_dir_total} (answerable)"
      f" | traps {sum(r[1]=='ok' for r in traps)}/{len(traps)} | Brier {brier/n:.3f}")
