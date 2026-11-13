"""Tests for lane2/decide.py. One test per rule in docs/decide.md.

    .venv/bin/python -m unittest tests.test_decide -v
"""
from __future__ import annotations

import unittest

from lane2.decide import decide, read_edge
from lane2.schema import Evidence, Grounding, Query


class FakeGraph:
    """Just the bits decide() touches: .nodes and .edges dicts."""

    def __init__(self):
        self.nodes, self.edges = {}, {}
        self._n = 0

    def node(self, name, qualifier=None, etype="environmental_variable"):
        self._n += 1
        nid = f"n{self._n}"
        self.nodes[nid] = {"id": nid, "name": name, "entity_type": etype,
                           "state_or_change_qualifier": qualifier}
        return nid

    def edge(self, subj, pred, obj, doc="doc1", negated=False, lang="direct_causal",
             grade="moderate", comparator="untreated_control", comp_desc="control"):
        eid = f"e{len(self.edges) + 1}"
        self.edges[eid] = {
            "id": eid, "subject": subj, "predicate": pred, "object": obj,
            "source_document": doc, "negated": negated,
            "causal_language": "no_relationship" if negated else lang,
            "certainty_grade": grade,
            "comparator": [{"comparator_type": comparator, "comparator_description": comp_desc}],
        }
        return eid


def q(x="fertilization", y="seedling survival"):
    return Query(intervention=x, outcome=y, raw="", parse_method="regex")


def run(g, eids, query=None):
    query = query or q()
    gr = Grounding(intervention_nodes=["x"], outcome_nodes=["y"])
    return decide(query, gr, Evidence(edge_ids=eids), g)


class ReadEdge(unittest.TestCase):
    def setUp(self):
        self.g = FakeGraph()
        self.treat = self.g.node("fertilization", "introduced", "management_intervention")

    def test_R1_moderates_is_excluded(self):
        e = self.g.edge(self.treat, "moderates", self.g.node("survival", "increased"))
        self.assertIsNone(read_edge(self.g.edges[e], self.g, q()).direction)

    def test_R2_negated_is_no_effect_even_with_increased_object(self):
        e = self.g.edge(self.treat, "causes", self.g.node("survival", "increased"), negated=True)
        self.assertEqual(read_edge(self.g.edges[e], self.g, q()).direction, "no_effect")

    def test_R3_unchanged_object_is_no_effect(self):
        e = self.g.edge(self.treat, "causes", self.g.node("water table", "unchanged"))
        self.assertEqual(read_edge(self.g.edges[e], self.g, q()).direction, "no_effect")

    def test_R4_plain_causes_increased(self):
        e = self.g.edge(self.treat, "causes", self.g.node("survival", "increased"))
        self.assertEqual(read_edge(self.g.edges[e], self.g, q()).direction, "increase")

    def test_R4_prevents_decreased_is_increase(self):
        # real edge: "caging of small trees" prevents "decreased breakage"
        cage = self.g.node("caging of small trees", "introduced", "management_intervention")
        e = self.g.edge(cage, "prevents", self.g.node("breakage", "decreased"))
        self.assertEqual(read_edge(self.g.edges[e], self.g, q("caging", "breakage")).direction, "increase")

    def test_R4_absent_subject_flips(self):
        # real edge: "absence of fertilization" [absent] causes "increased seedling survival"
        absent = self.g.node("absence of fertilization", "absent", "management_intervention")
        e = self.g.edge(absent, "causes", self.g.node("survival", "increased"))
        v = read_edge(self.g.edges[e], self.g, q("fertilization", "seedling survival"))
        self.assertEqual(v.direction, "decrease")
        self.assertTrue(v.flipped_subject)

    def test_R4_removed_subject_not_flipped_when_question_asks_about_removal(self):
        # real edge: "Release from moose browsing" [removed] -> "Increased sapling height"
        rel = self.g.node("Release from moose browsing", "removed", "management_intervention")
        e = self.g.edge(rel, "associated_with", self.g.node("sapling height", "increased"))
        v = read_edge(self.g.edges[e], self.g, q("browsing exclusion", "sapling height"))
        self.assertEqual(v.direction, "increase")

    def test_R4_present_object_is_unclear(self):
        e = self.g.edge(self.treat, "causes", self.g.node("composition", "present"))
        self.assertEqual(read_edge(self.g.edges[e], self.g, q()).direction, "unclear")

    def test_head_to_head_halves_weight_but_control_label_is_rescued(self):
        y = self.g.node("survival", "increased")
        h2h = self.g.edge(self.treat, "causes", y, comparator="alternative_treatment",
                          comp_desc="peat donor material")
        fake = self.g.edge(self.treat, "causes", y, comparator="alternative_treatment",
                           comp_desc="unfenced plots")
        uns = self.g.edge(self.treat, "causes", y, comparator="alternative_treatment",
                          comp_desc="unscarified plots after 15 years")
        self.assertEqual(read_edge(self.g.edges[h2h], self.g, q()).weight, 0.5)
        self.assertEqual(read_edge(self.g.edges[fake], self.g, q()).weight, 1.0)
        self.assertEqual(read_edge(self.g.edges[uns], self.g, q()).weight, 1.0)


class Aggregate(unittest.TestCase):
    def setUp(self):
        self.g = FakeGraph()
        self.x = self.g.node("fertilization", "introduced", "management_intervention")
        self.up = self.g.node("survival", "increased")
        self.down = self.g.node("survival", "decreased")

    def test_four_down_one_null_is_decrease(self):
        # README's example: "four studies report lower ..., one reports no effect"
        eids = [self.g.edge(self.x, "causes", self.down, doc=f"d{i}") for i in range(4)]
        eids.append(self.g.edge(self.x, "causes", self.up, doc="d9", negated=True))
        d = run(self.g, eids)
        self.assertEqual((d.status, d.direction), ("answered", "decrease"))

    def test_one_paper_many_edges_does_not_outvote_many_papers(self):
        eids = [self.g.edge(self.x, "causes", self.up, doc="big") for _ in range(10)]
        eids += [self.g.edge(self.x, "causes", self.down, doc=f"d{i}") for i in range(3)]
        self.assertEqual(run(self.g, eids).direction, "decrease")

    def test_split_papers_is_mixed_and_certainty_drops(self):
        eids = [self.g.edge(self.x, "causes", self.up, doc="a"),
                self.g.edge(self.x, "causes", self.down, doc="b")]
        d = run(self.g, eids)
        self.assertEqual(d.direction, "mixed")
        self.assertEqual(d.certainty, "low")  # moderate, minus one level for mixed

    def test_one_split_paper_is_mixed_in_tally(self):
        eids = [self.g.edge(self.x, "causes", self.up, doc="a"),
                self.g.edge(self.x, "causes", self.down, doc="a")]
        d = run(self.g, eids)
        self.assertEqual((d.direction, d.tally), ("mixed", {"mixed": 1}))

    def test_excluded_edges_are_not_cited(self):
        keep = self.g.edge(self.x, "causes", self.up, doc="a")
        drop = self.g.edge(self.x, "moderates", self.up, doc="a")
        self.assertEqual(run(self.g, [keep, drop]).edge_ids, [keep])

    def test_head_to_head_only_lowers_confidence_and_adds_note(self):
        normal = run(self.g, [self.g.edge(self.x, "causes", self.up, doc="a")])
        h2h = run(self.g, [self.g.edge(self.x, "causes", self.up, doc="b",
                                        comparator="alternative_treatment", comp_desc="peat")])
        self.assertLess(h2h.confidence, normal.confidence)
        self.assertTrue(any("another treatment" in n for n in h2h.notes))

    def test_no_edges_is_no_evidence(self):
        d = run(self.g, [])
        self.assertEqual((d.status, d.direction), ("no_evidence", None))
        self.assertTrue(d.searched_for)

    def test_only_excluded_edges_is_no_evidence(self):
        d = run(self.g, [self.g.edge(self.x, "moderates", self.up)])
        self.assertEqual(d.status, "no_evidence")

    def test_confidence_in_range(self):
        eids = [self.g.edge(self.x, "causes", self.up, doc=f"d{i}") for i in range(20)]
        self.assertLessEqual(run(self.g, eids).confidence, 0.95)


if __name__ == "__main__":
    unittest.main()
