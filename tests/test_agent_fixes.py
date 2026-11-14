"""Tests for the agent fixes in fix-agent-trap.md (D-010 reversed-direction trap and friends).

Run:  .venv/bin/python -m unittest tests/test_agent_fixes.py -v

Tests 1-9 are unit tests on the code checks: no LLM calls, the plans are written by hand in
the shape the plan step returns. Test 10 is a regression test over the LAST PIPELINE RUN
(outputs/dev_answers.json + outputs/traces); it is skipped when no run exists.
"""
import copy
import json
import re
import unittest
from pathlib import Path

import yaml

from agent.graph import Graph, aggregate, confidence_answered, confidence_no_evidence
from agent.plancheck import PlanChecker

ROOT = Path(__file__).resolve().parent.parent
G = None
PC = None

Q004 = "What effect does peatland restoration have on water table position?"
Q010 = "What effect does water table position have on peatland restoration?"

# The plan the LLM produced for D-010 in the failing run, in the current plan shape.
PLAN_010 = {
    "cause": {"label": "Water table position", "head": "position",
              "head_terms": ["water table position", "position", "water table", "water level",
                             "hydrology", "drainage", "rewetting", "saturation"],
              "modifier_terms": ["water table", "groundwater"], "exclude": []},
    "outcome": {"label": "Peatland restoration", "head": "restoration",
                "head_terms": ["peatland restoration", "restoration", "rehabilitation", "recovery"],
                "modifier_terms": ["peatland", "peat", "bog", "fen", "wetland"], "exclude": []},
}
PLAN_004 = {
    "cause": {"label": "Peatland restoration", "head": "restoration",
              "head_terms": ["peatland restoration", "restoration", "rewetting", "drainage", "reclamation"],
              "modifier_terms": ["peatland", "peat", "bog", "fen"], "exclude": []},
    "outcome": {"label": "Water table position", "head": "position",
                "head_terms": ["water table position", "position", "level", "height", "depth", "peat"],
                "modifier_terms": ["water table", "groundwater"], "exclude": []},
}

# Real edges from causal_graph.json used as fixtures.
E_RESTORATION_TO_HEIGHT = "camo:edge_1457861d5179"   # Restoration of the peatland -> Increased water table height
E_REWET_TO_PEAT_DEPTH = "camo:edge_dc2ee5a13644"     # rewetting (ditch blocking) -> increased depth of fibric peat
E_RECLAMATION_TO_WT_DEPTH = "camo:edge_2d03e06fe56b"  # peat-harvesting and reclamation -> water-table depth (increased)
E_SIMILAR_A = "camo:edge_44c436b0855d"               # Rewetting -> Water table level similar to intact bog
E_SIMILAR_B = "camo:edge_1891adf190bf"               # same paper, same subject, same object
E_SEASONAL_VARIABILITY = "camo:edge_7b75a85395d2"    # Restoration of the peatland -> increased seasonal variability of the water table


def setUpModule():
    global G, PC
    G = Graph(ROOT / "causal_graph.json")
    PC = PlanChecker(G)


def dropped_terms(side):
    return {d["term"] for d in side["dropped"]}


class PlanCheckTests(unittest.TestCase):
    def test_1_actions_and_umbrella_terms_are_dropped(self):
        p = PC.validate(copy.deepcopy(PLAN_010), Q010)
        c = p["cause"]
        for bad in ("rewetting", "drainage", "hydrology"):
            self.assertIn(bad, dropped_terms(c), f"{bad} should have been dropped")
            self.assertNotIn(bad, c["head_terms"] + c["modifier_terms"])
        for good in ("water table", "water level"):
            self.assertIn(good, c["head_terms"] + c["modifier_terms"])
        for d in c["dropped"]:
            self.assertTrue(d["reason"], f"drop of {d['term']} has no reason logged")

    def test_2_opposites_never_share_a_set(self):
        p = PC.validate(copy.deepcopy(PLAN_004), Q004)
        terms = p["cause"]["head_terms"] + p["cause"]["modifier_terms"]
        has_rewet = any("rewet" in t for t in terms)
        has_drain = any("drain" in t for t in terms)
        self.assertTrue(has_rewet != has_drain or not (has_rewet or has_drain))
        self.assertTrue(has_rewet, "rewetting IS peatland restoration and should survive")
        self.assertIn("drainage", dropped_terms(p["cause"]))

    def test_3_term_on_both_sides_stays_on_one(self):
        p = PC.validate(copy.deepcopy(PLAN_004), Q004)
        self.assertIn("peat", p["cause"]["modifier_terms"])
        self.assertNotIn("peat", p["outcome"]["head_terms"] + p["outcome"]["modifier_terms"])
        self.assertIn("peat", dropped_terms(p["outcome"]))


class MatchingTests(unittest.TestCase):
    def test_4_head_word_is_required(self):
        p = PC.validate(copy.deepcopy(PLAN_010), Q010)
        e = G.edge_by_id[E_REWET_TO_PEAT_DEPTH]
        self.assertEqual(G.nodes[e["object"]]["name"], "increased depth of fibric peat")
        self.assertFalse(G.node_matches(G.side_patterns(p["outcome"]), e["object"], e["id"]),
                         "an object that only mentions peat must not match 'peatland restoration'")

    def test_5_orientation(self):
        p004 = PC.validate(copy.deepcopy(PLAN_004), Q004)
        p010 = PC.validate(copy.deepcopy(PLAN_010), Q010)
        fwd004, _ = G.candidates(p004)
        fwd010, reversed010 = G.candidates(p010)
        self.assertIn(E_RESTORATION_TO_HEIGHT, {e["id"] for e in fwd004})
        self.assertNotIn(E_RESTORATION_TO_HEIGHT, {e["id"] for e in fwd010})
        self.assertGreaterEqual(reversed010, 1)
        self.assertEqual(fwd010, [], "D-010 has no edge with a water-table node as the subject and restoration as the outcome")

    def test_6_dedupe_counts_papers(self):
        a, b = G.edge_by_id[E_SIMILAR_A], G.edge_by_id[E_SIMILAR_B]
        self.assertEqual(a["source_document"], b["source_document"])
        dirs = {a["id"]: G.direction(a), b["id"]: G.direction(b)}
        unique, dups = G.dedupe([a, b], dirs)
        self.assertEqual(len(unique), 1)
        self.assertEqual(dups, [b["id"]])
        agg = aggregate(G, unique, dirs)
        self.assertEqual((agg["n_edges"], agg["n_papers"]), (1, 1))

    def test_7_polarity_depth_vs_level(self):
        asked = G.asked_class({"phrase": "water table position", "head_terms": ["water table position"]})
        self.assertEqual(asked, ("water_table", "level"))
        raw, final, inverted = G.normalise_direction(G.edge_by_id[E_RECLAMATION_TO_WT_DEPTH], asked)
        self.assertEqual((raw, final, inverted), ("increase", "decrease", True))
        raw, final, inverted = G.normalise_direction(G.edge_by_id[E_RESTORATION_TO_HEIGHT], asked)
        self.assertEqual((raw, final, inverted), ("increase", "increase", False))
        # the other way round: a depth question sees the height edge inverted
        asked_depth = G.asked_class({"phrase": "water table depth", "head_terms": []})
        self.assertEqual(asked_depth, ("water_table", "depth"))
        self.assertEqual(G.normalise_direction(G.edge_by_id[E_RESTORATION_TO_HEIGHT], asked_depth)[1:], ("decrease", True))

    def test_8_second_order_attributes_are_excluded(self):
        e = G.edge_by_id[E_SEASONAL_VARIABILITY]
        self.assertIn("variability", G.nodes[e["object"]]["name"])
        self.assertIsNotNone(G.second_order(e, Q004))
        self.assertIsNone(G.second_order(e, "What effect does peatland restoration have on water table variability?"))
        self.assertIsNotNone(G.second_order(G.edge_by_id[E_SIMILAR_A], Q004), "similarity to a reference is a comparison, not a direction")
        self.assertIsNone(G.second_order(G.edge_by_id[E_RESTORATION_TO_HEIGHT], Q004))


class ConfidenceTests(unittest.TestCase):
    def test_9_confidence_depends_on_evidence(self):
        full = [confidence_answered(n, 1.0) for n in range(1, 8)]
        self.assertEqual(full, sorted(full))
        self.assertEqual(len(set(full)), 7, "strictly increasing with n")
        self.assertEqual((full[0], full[2], full[6]), (0.57, 0.78, 0.88))
        self.assertGreater(confidence_answered(3, 1.0), confidence_answered(3, 0.6))
        self.assertGreater(confidence_answered(3, 0.6), confidence_answered(3, 0.34))
        self.assertNotEqual(confidence_answered(1, 1.0), confidence_answered(7, 1.0))
        self.assertEqual(confidence_no_evidence(False), 0.50)
        self.assertEqual(confidence_no_evidence(True), 0.40)


class MirrorPairRegression(unittest.TestCase):
    """Over the last pipeline run: mirror questions must not share evidence with swapped roles."""
    SHAPE = re.compile(r"what effect does (.+?) have on (.+?)\?$", re.I)

    def _load_run(self):
        ans = ROOT / "outputs" / "dev_answers.json"
        if not ans.exists():
            self.skipTest("no pipeline run in outputs/")
        answers = {a["id"]: a for a in json.loads(ans.read_text())["answers"]}
        traces = {}
        for qid in answers:
            tp = ROOT / "outputs" / "traces" / f"{qid}.json"
            if tp.exists():
                traces[qid] = json.loads(tp.read_text())
        return answers, traces

    def _pairs(self):
        qs = yaml.safe_load((ROOT / "dev_questions.yaml").read_text())["questions"]
        parsed = {}
        for q in qs:
            m = self.SHAPE.match(" ".join(str(q["question"]).split()))
            if m:
                parsed[str(q["id"])] = (m.group(1).lower(), m.group(2).lower())
        pairs = []
        for a, (x, y) in parsed.items():
            for b, (x2, y2) in parsed.items():
                if a < b and (x, y) == (y2, x2):
                    pairs.append((a, b))
        return pairs

    def test_10_mirror_pairs(self):
        pairs = self._pairs()
        self.assertIn(("D-004", "D-010"), pairs)
        answers, traces = self._load_run()
        for a, b in pairs:
            if a not in traces or b not in traces:
                self.skipTest(f"no traces for {a}/{b} in the last run")
            ta, tb = traces[a], traces[b]
            ea = {(e["subject"], e["object"]) for e in ta.get("edges", [])}
            eb = {(e["subject"], e["object"]) for e in tb.get("edges", [])}
            swapped = {(s, o) for (s, o) in ea if (o, s) in eb}
            self.assertFalse(swapped, f"{a}/{b} share an edge with swapped roles: {swapped}")
            self.assertFalse(set(ta.get("kept", [])) & set(tb.get("kept", [])), f"{a}/{b} cite the same edge id")
            # a kept subject may not match ONLY the other question's cause terms
            for me, other in ((ta, tb), (tb, ta)):
                mine = G.compile_terms(me["plan"]["cause"]["head_terms"])
                theirs = G.compile_terms(other["plan"]["cause"]["head_terms"])
                for e in me.get("edges", []):
                    s = e["subject"].lower()
                    if theirs and theirs.search(s):
                        self.assertTrue(mine and mine.search(s),
                                        f"{me['id']} kept '{e['subject']}', which matches only {other['id']}'s cause")
        # D-010 specifically
        a010 = answers.get("D-010")
        t010 = traces.get("D-010", {})
        if a010 is None:
            self.skipTest("last run did not include D-010")
        if a010["status"] == "answered":
            for e in t010.get("edges", []):
                self.assertRegex(e["subject"].lower(), r"water[\s-]?table|water level",
                                 "D-010 may only be answered from edges whose subject is a water-table node")
        else:
            self.assertEqual(a010["status"], "no_evidence")


if __name__ == "__main__":
    unittest.main()
