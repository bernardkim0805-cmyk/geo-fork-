"""Score lane2.decide against tests/decide_cases.yaml (docs/decide_tasks.md A5).

    .venv/bin/python -m tests.eval_decide              # all cases
    .venv/bin/python -m tests.eval_decide -v           # + vote trace for every case
    .venv/bin/python -m tests.eval_decide --case DC-15 # one case, with trace
    .venv/bin/python -m tests.eval_decide --signed-off # only user-approved cases

Feeds each case's edge_ids straight into decide() (no parse/ground/retrieve).

Pass rule (per case): status exact, direction exact, certainty within one
level, cited edges a subset of the expected cited edges.
Strict (the calibration target, E1): status + direction + exact certainty.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from lane2.decide import GRADES, decide
from lane2.graph import Graph
from lane2.schema import Evidence, Grounding, Query

CASES = Path(__file__).resolve().parent / "decide_cases.yaml"


def load_cases(path: Path = CASES) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    return [{**defaults, **c} for c in data["cases"]]


def run_case(c: dict, graph: Graph):
    query = Query(intervention=c["intervention"], outcome=c["outcome"],
                  raw=c["question"], parse_method="regex")
    g = Grounding(intervention_terms=[c["intervention"]], outcome_terms=[c["outcome"]])
    return decide(query, g, Evidence(edge_ids=list(c["edge_ids"])), graph)


def _cert_dist(a, b):
    if a is None or b is None:
        return 0 if a == b else 9
    return abs(GRADES.index(a) - GRADES.index(b))


def score(c: dict, d) -> dict:
    exp = c["expected"]
    allowed = c["edge_ids"] if exp.get("cited_edges") == "all" else (exp.get("cited_edges") or [])
    direction = d.direction if d.status == "answered" else None
    certainty = d.certainty if d.status == "answered" else None
    r = {
        "status": d.status == exp["status"],
        "direction": direction == exp.get("direction"),
        "certainty": _cert_dist(certainty, exp.get("certainty")) <= 1,
        "cited": set(d.edge_ids) <= set(allowed),
    }
    r["pass"] = all(r.values())
    r["strict"] = r["status"] and r["direction"] and _cert_dist(certainty, exp.get("certainty")) == 0
    r["got"] = (d.status, direction, certainty, d.confidence)
    r["exp"] = (exp["status"], exp.get("direction"), exp.get("certainty"))
    return r


def trace(c, d, graph):
    for v in d.votes:
        e = graph.edges.get(v.edge_id)
        if e is None:
            print(f"      {v.edge_id}: {v.reason}")
            continue
        s, o = graph.nodes.get(e["subject"], {}), graph.nodes.get(e["object"], {})
        print(f"      {v.edge_id[10:]} {str(v.direction):9} w={v.weight:<5} "
              f"[{s.get('name', '?')[:35]}|{s.get('state_or_change_qualifier')}] {e['predicate']}"
              f"{'/NEG' if e.get('negated') else ''} [{o.get('name', '?')[:40]}|"
              f"{o.get('state_or_change_qualifier')}]  ({v.reason})")
    for n in d.notes:
        print(f"      note: {n}")


def calibration(rows):
    """Mean confidence vs strict accuracy per band, plus Brier score (E4)."""
    bands = defaultdict(list)
    for conf, ok in rows:
        bands[min(int(conf * 10), 9)].append((conf, ok))
    print("\n  confidence band   n   mean conf   strict acc")
    for b in sorted(bands):
        xs = bands[b]
        print(f"  {b / 10:.1f}-{(b + 1) / 10:.1f}        {len(xs):3d}   {sum(c for c, _ in xs) / len(xs):9.2f}"
              f"   {sum(ok for _, ok in xs) / len(xs):10.2f}")
    brier = sum((conf - ok) ** 2 for conf, ok in rows) / len(rows)
    print(f"  Brier score (strict): {brier:.3f}   (lower is better; always-0.5 = 0.250)")
    return brier


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--signed-off", action="store_true")
    args = ap.parse_args(argv)

    graph = Graph()
    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    if args.signed_off:
        cases = [c for c in cases if c.get("status_review") == "signed_off"]

    totals = defaultdict(int)
    calib = []
    for c in cases:
        d = run_case(c, graph)
        r = score(c, d)
        for k in ("pass", "strict", "status", "direction", "certainty", "cited"):
            totals[k] += r[k]
        calib.append((d.confidence, r["strict"]))
        known = bool(c.get("known_limit"))
        totals["known"] += known
        totals["known_pass"] += known and r["pass"]
        flag = "PASS" if r["pass"] else ("MISS*" if known else "FAIL")
        misses = ",".join(k for k in ("status", "direction", "certainty", "cited") if not r[k])
        print(f"{flag} {c['id']:6} {str(c.get('trap') or ''):20} got={r['got']} exp={r['exp']}"
              + (f"  [{misses}]" if misses else ""))
        if args.verbose or args.case or (not r["pass"] and not known):
            trace(c, d, graph)

    n = len(cases) or 1
    print(f"\n{totals['pass']}/{len(cases)} pass   strict {totals['strict']}/{len(cases)}   "
          f"status {totals['status']}  direction {totals['direction']}  "
          f"certainty±1 {totals['certainty']}  cited {totals['cited']}")
    if totals["known"]:
        print(f"(MISS* = known_limit case, {totals['known_pass']}/{totals['known']} pass; "
              f"excluding them: {totals['pass'] - totals['known_pass']}/{len(cases) - totals['known']} pass)")
    if calib:
        calibration(calib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
