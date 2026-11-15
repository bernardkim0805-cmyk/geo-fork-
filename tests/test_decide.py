"""Tests for lane2/decide.py. One test per rule in docs/decide.md.

    .venv/bin/python -m unittest tests.test_decide -v
"""
from __future__ import annotations

import unittest

import lane2.decide as _dec
from lane2 import roles
from lane2.decide import decide, read_edge
from lane2.schema import Evidence, Grounding, Query

# Unit tests never call the real endpoint: the D5 role check is switched on only
# inside RoleCheck, with a fake LLM.
_dec.CONFIG["role_check"] = False


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
             grade="moderate", comparator="untreated_control", comp_desc="control",
             evidence="randomized_experiment", strength="strong"):
        eid = f"e{len(self.edges) + 1}"
        self.edges[eid] = {
            "id": eid, "subject": subj, "predicate": pred, "object": obj,
            "source_document": doc, "negated": negated,
            "causal_language": "no_relationship" if negated else lang,
            "certainty_grade": grade,
            "comparator": [{"comparator_type": comparator, "comparator_description": comp_desc}],
            "evidential_basis": {"evidence_types": [evidence]},
            "strength": {"qualitative_descriptor": strength},
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

    def test_C1_study_design_weights_the_vote(self):
        up = self.g.node("survival", "increased")
        rct = self.g.edge(self.treat, "causes", up)
        obs = self.g.edge(self.treat, "causes", up, evidence="observational_cross_sectional")
        model = self.g.edge(self.treat, "causes", up, evidence="modeling_simulation")
        w = [read_edge(self.g.edges[e], self.g, q()).weight for e in (rct, obs, model)]
        self.assertEqual(w, sorted(w, reverse=True))
        self.assertEqual(w[0], 1.0)

    def test_C3_weak_or_negligible_strength_weighs_less(self):
        up = self.g.node("survival", "increased")
        strong = self.g.edge(self.treat, "causes", up)
        weak = self.g.edge(self.treat, "causes", up, strength="weak")
        negl = self.g.edge(self.treat, "causes", up, strength="negligible")
        w = [read_edge(self.g.edges[e], self.g, q()).weight for e in (strong, weak, negl)]
        self.assertTrue(w[0] > w[1] > w[2])

    def test_C1_missing_evidential_basis_does_not_throw(self):
        e = self.g.edge(self.treat, "causes", self.g.node("survival", "increased"))
        del self.g.edges[e]["evidential_basis"], self.g.edges[e]["strength"]
        self.assertEqual(read_edge(self.g.edges[e], self.g, q()).direction, "increase")

    def test_B1_present_object_without_sign_words_abstains(self):
        e = self.g.edge(self.treat, "causes", self.g.node("distinct composition", "present"))
        self.assertEqual(read_edge(self.g.edges[e], self.g, q()).direction, "abstain")

    def test_B1_unsigned_object_takes_sign_from_name(self):
        up = self.g.edge(self.treat, "causes", self.g.node("higher native cover", None))
        down = self.g.edge(self.treat, "causes", self.g.node("reduced seedling survival", "present"))
        self.assertEqual(read_edge(self.g.edges[up], self.g, q()).direction, "increase")
        self.assertEqual(read_edge(self.g.edges[down], self.g, q()).direction, "decrease")

    def test_B1_name_with_both_signs_abstains(self):
        e = self.g.edge(self.treat, "causes", self.g.node("Increased CEC and decreased H", None))
        self.assertEqual(read_edge(self.g.edges[e], self.g, q()).direction, "abstain")

    def test_B2_variation_object_is_excluded(self):
        e = self.g.edge(self.treat, "associated_with", self.g.node("variation in basal area", "increased"))
        self.assertIsNone(read_edge(self.g.edges[e], self.g, q()).direction)

    def test_B3_interaction_subject_is_excluded(self):
        for name in ("interaction of scarification and year", "scarification x cutting",
                     "scarification treatment and time"):
            subj = self.g.node(name, "occurred", "management_intervention")
            e = self.g.edge(subj, "causes", self.g.node("seedling density", "increased"))
            self.assertIsNone(read_edge(self.g.edges[e], self.g, q()).direction, name)

    def test_B5_unchanged_subject_abstains(self):
        subj = self.g.node("water table position", "unchanged")
        e = self.g.edge(subj, "causes", self.g.node("moss cover", "increased"))
        self.assertEqual(read_edge(self.g.edges[e], self.g, q()).direction, "abstain")

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

    def test_D1_single_split_paper_gets_penalty_and_note(self):
        one = run(self.g, [self.g.edge(self.x, "causes", self.up, doc="a"),
                           self.g.edge(self.x, "causes", self.down, doc="a")])
        two = run(self.g, [self.g.edge(self.x, "causes", self.up, doc="b"),
                           self.g.edge(self.x, "causes", self.down, doc="c")])
        self.assertLess(one.confidence, two.confidence)
        self.assertEqual(one.certainty, "low")  # moderate - 1 for mixed, not dropped twice
        self.assertTrue(any(n.startswith("One study") for n in one.notes))

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

    def test_D3_no_evidence_ignores_grounding_and_names_the_query(self):
        rich = decide(q("mowing", "soil carbon"),
                      Grounding(intervention_nodes=["a"], outcome_nodes=["b"]), Evidence([]), self.g)
        bare = decide(q("mowing", "soil carbon"), Grounding(), Evidence([]), self.g)
        self.assertEqual(rich.confidence, bare.confidence)
        self.assertIn("'mowing'", bare.searched_for)
        self.assertIn("'soil carbon'", bare.searched_for)

    def test_only_excluded_edges_is_no_evidence(self):
        d = run(self.g, [self.g.edge(self.x, "moderates", self.up)])
        self.assertEqual(d.status, "no_evidence")

    def test_B1_abstaining_paper_does_not_dilute_the_vote(self):
        eids = [self.g.edge(self.x, "causes", self.up, doc="a"),
                self.g.edge(self.x, "causes", self.up, doc="b"),
                self.g.edge(self.x, "causes", self.g.node("distinct community", "present"), doc="c")]
        d = run(self.g, eids)
        self.assertEqual(d.direction, "increase")
        self.assertEqual(d.tally, {"increase": 2})
        self.assertEqual(len(d.edge_ids), 3)  # the abstaining edge is still cited as context

    def test_B1_only_abstaining_edges_is_unclear(self):
        d = run(self.g, [self.g.edge(self.x, "causes", self.g.node("distinct community", "present"))])
        self.assertEqual((d.status, d.direction), ("answered", "unclear"))

    def test_C2_observational_drop_is_off_by_default(self):
        from lane2.decide import CONFIG
        eids = [self.g.edge(self.x, "causes", self.up, doc="a", evidence="observational_cross_sectional")]
        self.assertFalse(CONFIG["observational_certainty_drop"])
        self.assertEqual(run(self.g, eids).certainty, "moderate")
        CONFIG["observational_certainty_drop"] = True
        try:
            self.assertEqual(run(self.g, eids).certainty, "low")
        finally:
            CONFIG["observational_certainty_drop"] = False

    def test_confidence_in_range(self):
        eids = [self.g.edge(self.x, "causes", self.up, doc=f"d{i}") for i in range(20)]
        self.assertLessEqual(run(self.g, eids).confidence, 0.95)

    def test_E3_confidence_is_clamped(self):
        from lane2.decide import CONFIG
        eids = [self.g.edge(self.x, "causes", self.up, doc=f"d{i}") for i in range(20)]
        self.assertLessEqual(run(self.g, eids).confidence, CONFIG["conf_ceiling"])
        saved = CONFIG["conf_mixed"]
        CONFIG["conf_mixed"] = 0.01
        try:
            d = run(self.g, [self.g.edge(self.x, "causes", self.up, doc="a"),
                             self.g.edge(self.x, "causes", self.down, doc="b")])
            self.assertEqual(d.confidence, CONFIG["conf_floor"])
        finally:
            CONFIG["conf_mixed"] = saved


class RemovalCheck(unittest.TestCase):
    """D2: LLM decides whether a removal subject is the intervention or its opposite."""

    def setUp(self):
        import os
        import lane2.decide as dec
        import lane2.llm as llm
        self.dec, self.llm, self.os = dec, llm, os
        dec._OPPOSITE_CACHE.clear()
        for k in dec.LLM_STATS:
            dec.LLM_STATS[k] = 0
        self._chat, self._configured = llm.chat, llm.configured
        self.g = FakeGraph()
        self.removal = self.g.node("Weed removal", "removed", "management_intervention")
        self.e = self.g.edge(self.removal, "causes", self.g.node("seedling growth", "increased"))

    def tearDown(self):
        self.llm.chat, self.llm.configured = self._chat, self._configured
        self.dec._OPPOSITE_CACHE.clear()

    def _with_llm(self, reply):
        self.llm.configured = lambda: True
        self.calls = []

        def fake_chat(messages, **kw):
            self.calls.append(messages)
            if isinstance(reply, Exception):
                raise reply
            return reply
        self.llm.chat = fake_chat

    def test_llm_same_means_no_flip(self):
        self._with_llm("<think>weed control removes weeds</think>SAME")
        v = read_edge(self.g.edges[self.e], self.g, q("weed control", "seedling growth"))
        self.assertEqual(v.direction, "increase")

    def test_llm_opposite_means_flip(self):
        self._with_llm("OPPOSITE")
        v = read_edge(self.g.edges[self.e], self.g, q("competing vegetation", "seedling growth"))
        self.assertEqual(v.direction, "decrease")

    def test_llm_answer_is_cached(self):
        self._with_llm("OPPOSITE")
        for _ in range(3):
            read_edge(self.g.edges[self.e], self.g, q("competing vegetation", "seedling growth"))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.dec.LLM_STATS["cache_hits"], 2)

    def test_llm_error_falls_back_to_word_list(self):
        self._with_llm(RuntimeError("endpoint down"))
        v = read_edge(self.g.edges[self.e], self.g, q("weed removal", "seedling growth"))
        self.assertEqual(v.direction, "increase")  # 'remov' in the question -> no flip
        self.assertEqual(self.dec.LLM_STATS["fallbacks"], 1)

    def test_no_llm_call_without_a_removal_subject(self):
        self._with_llm("OPPOSITE")
        plain = self.g.node("fertilization", "introduced", "management_intervention")
        e = self.g.edge(plain, "causes", self.g.node("growth", "increased"))
        read_edge(self.g.edges[e], self.g, q())
        self.assertEqual(self.calls, [])

    def test_tightened_fallback_flips_erosion_control(self):
        self.llm.configured = lambda: False
        subj = self.g.node("absence of erosion control", "absent", "management_intervention")
        e = self.g.edge(subj, "causes", self.g.node("sediment loss", "increased"))
        v = read_edge(self.g.edges[e], self.g, q("erosion control", "sediment loss"))
        self.assertEqual(v.direction, "decrease")

    # D4: inverse subject by NAME ("distance from X"), not by qualifier
    def _distance_edge(self, qualifier="increased"):
        subj = self.g.node("Distance from contamination source", qualifier)
        return self.g.edge(subj, "associated_with", self.g.node("taxonomic richness", "increased"))

    def test_D4_distance_from_x_flips_when_llm_says_opposite(self):
        self._with_llm("OPPOSITE")
        v = read_edge(self.g.edges[self._distance_edge()], self.g,
                      q("contaminants", "aquatic communities"))
        self.assertEqual(v.direction, "decrease")
        self.assertTrue(v.flipped_subject)
        self.assertEqual(len(self.calls), 1)

    def test_D4_distance_from_x_not_flipped_when_llm_says_same(self):
        self._with_llm("SAME")
        v = read_edge(self.g.edges[self._distance_edge()], self.g,
                      q("remediation actions", "aquatic communities"))
        self.assertEqual(v.direction, "increase")
        self.assertFalse(v.flipped_subject)

    def test_D4_decreased_distance_is_a_double_inverse_and_never_flips(self):
        self._with_llm("OPPOSITE")  # closer to the source = more contaminant
        v = read_edge(self.g.edges[self._distance_edge("decreased")], self.g,
                      q("contaminants", "aquatic communities"))
        self.assertEqual(v.direction, "increase")
        self.assertEqual(self.calls, [])

    def test_D4_fallback_without_llm_flips_distance(self):
        self.llm.configured = lambda: False
        v = read_edge(self.g.edges[self._distance_edge()], self.g,
                      q("contaminants", "aquatic communities"))
        self.assertEqual(v.direction, "decrease")


class RoleCheck(unittest.TestCase):
    """D5: one batched call classifies subjects vs X and objects vs Y."""

    def setUp(self):
        import lane2.llm as llm
        self.llm = llm
        self._chat, self._configured = llm.chat, llm.configured
        roles.clear()
        _dec._OPPOSITE_CACHE.clear()
        _dec.CONFIG["role_check"] = True
        self.g = FakeGraph()
        self.calls = []

    def tearDown(self):
        self.llm.chat, self.llm.configured = self._chat, self._configured
        _dec.CONFIG["role_check"] = False
        _dec.CONFIG["role_disagree"] = "abstain"
        roles.clear()

    def _answers(self, mapping):
        """Fake LLM: answer each numbered item by the node name it carries."""
        self.llm.configured = lambda: True

        def fake_chat(messages, **kw):
            self.calls.append(messages)
            out = []
            for line in messages[-1]["content"].splitlines():
                parts = line.split(" ", 1)
                if parts[0].isdigit():
                    name = parts[1].split(" [")[0]
                    out.append(f"{parts[0]} {mapping.get(name, 'UNSURE')}")
            return "\n".join(out)
        self.llm.chat = fake_chat

    def _edge(self, subj, subj_q, obj, obj_q, doc="doc1"):
        return self.g.edge(self.g.node(subj, subj_q), "causes", self.g.node(obj, obj_q), doc=doc)

    def test_unrelated_object_is_excluded_and_noted(self):
        self._answers({"Glyphosate": "SAME", "non-native forb cover": "UNRELATED",
                       "native forb cover": "DOWN"})
        e1 = self._edge("Glyphosate", "occurred", "non-native forb cover", "increased")
        e2 = self._edge("Glyphosate", "occurred", "native forb cover", "decreased", doc="doc2")
        d = run(self.g, [e1, e2], q("glyphosate", "native plant cover"))
        self.assertEqual(d.direction, "decrease")
        self.assertEqual(d.edge_ids, [e2])
        self.assertTrue(any("different thing" in n for n in d.notes))

    def test_unrelated_subject_is_excluded(self):
        self._answers({"soil amendment": "UNRELATED", "richness": "UP"})
        e = self._edge("soil amendment", "occurred", "richness", "increased")
        d = run(self.g, [e], q("contaminants", "aquatic communities"))
        self.assertEqual(d.status, "no_evidence")

    def test_opposite_subject_flips(self):
        self._answers({"Time since harvest": "OPPOSITE", "regeneration density": "UP"})
        e = self._edge("Time since harvest", "increased", "regeneration density", "increased")
        d = run(self.g, [e], q("harvesting", "regeneration"))
        self.assertEqual(d.direction, "decrease")

    def test_inverse_object_disagreeing_with_qualifier_abstains_by_default(self):
        self._answers({"logging": "SAME", "failure of regeneration": "UP"})
        e = self._edge("logging", "occurred", "failure of regeneration", "decreased")
        v = run(self.g, [e], q("logging", "recovery")).votes[0]
        self.assertEqual(v.direction, "abstain")

    def test_inverse_object_trusted_when_configured(self):
        _dec.CONFIG["role_disagree"] = "llm"
        self._answers({"logging": "SAME", "failure of regeneration": "UP"})
        e = self._edge("logging", "occurred", "failure of regeneration", "decreased")
        self.assertEqual(run(self.g, [e], q("logging", "recovery")).direction, "increase")

    def test_role_check_fills_a_missing_object_sign(self):
        self._answers({"fertilization": "SAME", "seedling vigour": "UP"})
        e = self._edge("fertilization", "introduced", "seedling vigour", "present")
        self.assertEqual(run(self.g, [e]).direction, "increase")

    def test_one_call_per_decide_and_cached_after(self):
        self._answers({"fertilization": "SAME", "survival": "UP"})
        eids = [self._edge("fertilization", "introduced", "survival", "increased", doc=f"d{i}")
                for i in range(5)]
        run(self.g, eids)
        run(self.g, eids)
        self.assertEqual(len(self.calls), 1)

    def test_endpoint_down_falls_back_to_rules(self):
        self.llm.configured = lambda: True

        def boom(messages, **kw):
            raise RuntimeError("down")
        self.llm.chat = boom
        e = self._edge("fertilization", "introduced", "survival", "increased")
        self.assertEqual(run(self.g, [e]).direction, "increase")

    def test_combined_pass_reads_each_edge_against_its_own_x(self):
        self._answers({"mat removal": "OPPOSITE", "germination": "DOWN"})
        e = self._edge("mat removal", "removed", "germination", "decreased")
        sub = q("competing vegetation", "recovery")
        d = decide(q("deer browsing", "recovery"), Grounding(), Evidence(edge_ids=[e]),
                   self.g, {e: sub})
        # less vegetation -> less germination, so vegetation -> more germination
        self.assertEqual(d.direction, "increase")
        self.assertEqual(self.calls[0][-1]["content"].splitlines()[0],
                         "CAUSE X = 'competing vegetation':")
        self.assertTrue(d.votes[0].flipped_subject)


    def test_combined_pass_picks_the_pair_the_edge_is_about(self):
        # found by both pairs; richness ranked it higher, but it is a cover edge
        self._answers({"Glyphosate": "SAME", "native forb cover": "DOWN"})
        mapping = {("native plant richness", "native forb cover"): "UNRELATED"}
        inner = self.llm.chat

        def chat(messages, **kw):
            out, y = [], None
            for line in messages[-1]["content"].splitlines():
                if line.startswith("EFFECT Y = "):
                    y = line.split("'")[1]
                parts = line.split(" ", 1)
                if parts[0].isdigit():
                    name = parts[1].split(" [")[0]
                    code = mapping.get((y, name)) or inner(
                        [{"content": f"{parts[0]} {parts[1]}"}]).split(" ", 1)[1]
                    out.append(f"{parts[0]} {code}")
            return "\n".join(out)
        self.llm.chat = chat
        e = self._edge("Glyphosate", "occurred", "native forb cover", "decreased")
        rich, cover = q("glyphosate", "native plant richness"), q("glyphosate", "native plant cover")
        d = decide(q("glyphosate", "native plant richness"), Grounding(),
                   Evidence(edge_ids=[e]), self.g, {e: [rich, cover]})
        self.assertEqual(d.direction, "decrease")
        self.assertEqual(d.edge_ids, [e])

class BadInput(unittest.TestCase):
    """F1-F3: decide never throws on bad edge input."""

    def setUp(self):
        self.g = FakeGraph()
        self.x = self.g.node("fertilization", "introduced", "management_intervention")
        self.up = self.g.node("survival", "increased")
        self.good = self.g.edge(self.x, "causes", self.up, doc="a")

    def test_unknown_edge_id_is_skipped_with_note_and_penalty(self):
        clean = run(self.g, [self.good])
        d = run(self.g, [self.good, "camo:edge_doesnotexist"])
        self.assertEqual((d.status, d.direction, d.edge_ids), ("answered", "increase", [self.good]))
        self.assertLess(d.confidence, clean.confidence)
        self.assertTrue(any("skipped" in n for n in d.notes))

    def test_edge_with_missing_node_is_skipped(self):
        broken = self.g.edge(self.x, "causes", "n_missing", doc="b")
        d = run(self.g, [self.good, broken])
        self.assertEqual(d.edge_ids, [self.good])

    def test_only_bad_ids_is_no_evidence(self):
        d = run(self.g, ["camo:edge_nope", "camo:edge_nada"])
        self.assertEqual(d.status, "no_evidence")
        self.assertIn("missing from the graph", d.searched_for)

    def test_duplicate_ids_are_deduped(self):
        d = run(self.g, [self.good, self.good, self.good])
        self.assertEqual(d.edge_ids, [self.good])

    def test_empty_or_null_comparator_does_not_throw(self):
        e1 = self.g.edge(self.x, "causes", self.up, doc="c")
        self.g.edges[e1]["comparator"] = []
        e2 = self.g.edge(self.x, "causes", self.up, doc="d")
        self.g.edges[e2]["comparator"] = [None]
        e3 = self.g.edge(self.x, "causes", self.up, doc="e")
        del self.g.edges[e3]["comparator"]
        d = run(self.g, [e1, e2, e3])
        self.assertEqual(d.direction, "increase")

    def test_none_edge_list_is_no_evidence(self):
        from lane2.decide import decide as _decide
        d = _decide(q(), Grounding(), Evidence(edge_ids=None), self.g)
        self.assertEqual(d.status, "no_evidence")


if __name__ == "__main__":
    unittest.main()
