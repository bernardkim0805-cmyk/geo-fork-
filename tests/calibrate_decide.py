"""Fit decide's confidence constants to tests/decide_cases.yaml (docs/decide_tasks.md E).

    .venv/bin/python -m tests.calibrate_decide

Prints suggested CONFIG values; copy them into lane2/decide.py by hand (the
numbers stay readable and reviewable in one place).

E1  correct = status + direction + exact certainty (the "strict" score).
E2  keep the formula. Grid-search conf_base / conf_per_paper / conf_cap on the
    cases decide answers with a direction (increase/decrease/no_effect).
    Flat groups (mixed, unclear, no_evidence) get their observed accuracy.
E3  guard against overfitting:
    - flat groups: shrink toward 0.5 with K pseudo-cases,
        conf = (right + K*0.5) / (n + K)
    - formula: minimise Brier + (K/n) * mean((conf - 0.5)^2), a ridge toward 0.5
    - everything clamped to [CONF_FLOOR, CONF_CEILING] inside decide.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import lane2.decide as dec
from lane2.graph import Graph

from .eval_decide import load_cases, run_case, score

K = 4  # pseudo-cases at 0.5


def frange(a, b, step):
    n = int(round((b - a) / step))
    return [round(a + i * step, 3) for i in range(n + 1)]


def run_all(cases, graph):
    out = []
    for c in cases:
        d = run_case(c, graph)
        out.append((c, d, score(c, d)["strict"]))
    return out


def main() -> int:
    graph = Graph()
    cases = load_cases()
    base = run_all(cases, graph)

    groups = defaultdict(list)
    for c, d, ok in base:
        key = "no_evidence" if d.status == "no_evidence" else (
            d.direction if d.direction in ("mixed", "unclear") else "signed")
        groups[key].append(ok)

    print("E1 strict accuracy by group (current decide):")
    for k, v in groups.items():
        print(f"  {k:12} n={len(v):2d} right={sum(v):2d} acc={sum(v) / len(v):.2f}")

    print("\nE3 shrunk flat-group values (K=%d):" % K)
    flat = {"mixed": "conf_mixed", "unclear": "conf_unclear", "no_evidence": "conf_no_evidence"}
    suggestions = {}
    for g, key in flat.items():
        v = groups.get(g, [])
        if not v:  # no data: keep the current (prior) value rather than defaulting to 0.5
            suggestions[key] = dec.CONFIG[key]
            print(f"  {key:18} = {dec.CONFIG[key]}   (n=0, keep current)")
            continue
        val = round((sum(v) + K * 0.5) / (len(v) + K), 2)
        suggestions[key] = val
        print(f"  {key:18} = {val}   (n={len(v)})")

    # E2: grid over the formula on 'signed' cases, with a ridge toward 0.5
    signed_cases = [c for c, d, _ in base if d.status == "answered"
                    and d.direction not in ("mixed", "unclear")]
    best = None
    saved = {k: dec.CONFIG[k] for k in ("conf_base", "conf_per_paper", "conf_cap")}
    for b, pp, cap in itertools.product(frange(0.3, 0.9, 0.05), frange(0.0, 0.15, 0.025),
                                        (0.8, 0.85, 0.9)):
        dec.CONFIG.update(conf_base=b, conf_per_paper=pp, conf_cap=cap)
        rows = run_all(signed_cases, graph)
        n = len(rows)
        brier = sum((d.confidence - ok) ** 2 for _, d, ok in rows) / n
        ridge = (K / n) * sum((d.confidence - 0.5) ** 2 for _, d, _ in rows) / n
        obj = brier + ridge
        if best is None or obj < best[0]:
            best = (obj, brier, b, pp, cap)
    dec.CONFIG.update(saved)
    obj, brier, b, pp, cap = best
    suggestions.update(conf_base=b, conf_per_paper=pp, conf_cap=cap)
    print(f"\nE2 formula fit on {len(signed_cases)} signed cases: conf_base={b} conf_per_paper={pp} "
          f"conf_cap={cap}  (Brier {brier:.3f}, objective {obj:.3f})")

    print("\nSuggested CONFIG:")
    for k, v in suggestions.items():
        print(f'    "{k}": {v},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
