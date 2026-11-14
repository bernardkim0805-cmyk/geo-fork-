"""Stage 2b: grounded node sets -> edges linking them.

Every pass runs, and their results are unioned. An earlier design stopped at the
first pass that found anything, which meant a question answered by three direct
edges never saw the forty the text-match passes would have found.

  direct              subject in intervention_nodes AND object in outcome_nodes.
                      The strongest evidence: both sides grounded to real nodes.
  outcome_text        subject grounded, object only keyword-matches outcome_terms.
                      Catches outcomes whose node text the grounding stage missed.
  intervention_text   the mirror image: object grounded, subject keyword-matched.
  reverse             object in intervention_nodes AND subject in outcome_nodes,
                      i.e. the claim points Y->X. Reported separately and never
                      merged into edge_ids, since asserting X->Y from it would
                      invert the causal direction.

Edges are not filtered by ecosystem. Measuring it on b3 showed a hard filter
removing 21 of 27 edges, including every `decreased` one, which turned a correct
"mixed" into a confident wrong "increase". The ecosystem match is reported per
edge instead, so the decide stage can down-weight rather than discard.

Multi-hop (X -> mediator -> Y) stays out: the graph holds ~112 such paths against
2,816 direct ones.
"""
from __future__ import annotations

import re

from .graph import Graph
from .ground import _expand_terms, _term_pattern, ground
from .schema import Evidence, Grounding, Query


def _build_pats(terms: list[str]) -> list[re.Pattern[str]]:
    """Patterns for the terms as they stand, stemmed but not word-split.

    These terms already include ground's expansion of the question's phrases
    plus its synonyms. Splitting them again here searched bare words that no
    stage ever chose -- "planting stock" became "planting" and matched
    "planting season".
    """
    return [_term_pattern(t) for t in dict.fromkeys(terms) if len(t) > 2]


def _text_matches(text: str, pats: list[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in pats)


def _direct(g: Grounding, graph: Graph, outcomes: set[str]) -> list[str]:
    hits = []
    for nid in g.intervention_nodes:
        for eid in graph.edges_by_subject.get(nid, []):
            if graph.edges[eid]["object"] in outcomes:
                hits.append(eid)
    return hits


def _outcome_text(g: Grounding, graph: Graph) -> list[str]:
    if not (g.intervention_nodes and g.outcome_terms):
        return []
    pats = _build_pats(g.outcome_terms)
    if not pats:
        return []
    hits = []
    for nid in g.intervention_nodes:
        for eid in graph.edges_by_subject.get(nid, []):
            if _text_matches(graph.node_text(graph.edges[eid]["object"]), pats):
                hits.append(eid)
    return hits


def _intervention_text(g: Grounding, graph: Graph) -> list[str]:
    if not (g.outcome_nodes and g.intervention_terms):
        return []
    pats = _build_pats(g.intervention_terms)
    if not pats:
        return []
    hits = []
    for nid in g.outcome_nodes:
        for eid in graph.edges_by_object.get(nid, []):
            if _text_matches(graph.node_text(graph.edges[eid]["subject"]), pats):
                hits.append(eid)
    return hits


def _reverse(g: Grounding, graph: Graph, interventions: set[str]) -> list[str]:
    """Claims pointing Y->X. Kept out of edge_ids: citing one as evidence for
    X->Y would reverse the causal direction. Surfaced so it is visible."""
    hits = []
    for nid in g.outcome_nodes:
        for eid in graph.edges_by_subject.get(nid, []):
            if graph.edges[eid]["object"] in interventions:
                hits.append(eid)
    return hits


# Weights are log-lift, measured by comparing the confirmed edges in
# tests/cases.yaml against everything else retrieval returned. Positive means
# the value was over-represented among confirmed edges.
#
# Do not hand-tune these. An earlier intuition-based version scored `causes`
# highest and `associated_with` lowest; the data says the opposite, and that
# version ranked WORSE than shuffling the list. Re-measure before changing.
#
# Caveat: fitted on 39 positives, so treat small weights as noise. `in_context`
# is absent deliberately -- it showed a lift of exactly 1.00, no signal at all.
_W_PASS = {"direct": 0.05, "outcome_text": -0.67, "intervention_text": -0.67}
_W_PREDICATE = {}           # re-fitting showed no stable signal here; left empty
_W_CERTAINTY = {"very_low": -2.0}   # 0 confirmed edges across two fits
_W_STRENGTH = {"strong": 0.3, "moderate": -0.3, "not_specified": -1.14}
_W_EVIDENCE = {"observational_cross_sectional": 0.41,
               "observational_longitudinal": -0.78, "mechanistic_study": -2.0}
_W_NEGATED = -0.8           # a "no relationship" claim is usually not the answer
_W_NO_QUALIFIER = -0.89     # decide reads the object's qualifier; without one it is mute
_W_TAXON = 0.8             # unmeasured: b4 has no confirmed edge to fit against


def _score(eid: str, graph: Graph, found_by: dict[str, list[str]],
           in_context: bool, taxon_hint: bool) -> float:
    """Rank an edge by how well it answers the question.

    Summed log-lift, i.e. naive Bayes over the measured features. Nothing here
    filters -- a low score sinks an edge rather than removing it.
    """
    e = graph.edges[eid]
    first = (found_by.get(eid) or ["direct"])[0]
    score = _W_PASS.get(first, 0.0)
    score += _W_PREDICATE.get(e.get("predicate"), 0.0)
    score += _W_CERTAINTY.get(e.get("certainty_grade"), 0.0)
    score += _W_STRENGTH.get(
        (e.get("strength") or {}).get("qualitative_descriptor"), 0.0)
    types = (e.get("evidential_basis") or {}).get("evidence_types") or []
    score += _W_EVIDENCE.get(types[0] if types else None, 0.0)
    if e.get("negated"):
        score += _W_NEGATED
    if not graph.nodes.get(e["object"], {}).get("state_or_change_qualifier"):
        score += _W_NO_QUALIFIER
    # questions name abstract categories ("wildlife") while the graph stores
    # specific measures ("clutch size"); applied_to is the structural link
    if taxon_hint and _applies_to_taxon(graph, e["object"]):
        score += _W_TAXON
    return score


def _applies_to_taxon(graph: Graph, node_id: str) -> bool:
    for a in graph.nodes.get(node_id, {}).get("applied_to") or []:
        if isinstance(a, dict) and a.get("entity_type") == "taxon":
            return True
    return False


# Words that name living things in general rather than a measurable variable.
# When a question uses one, an outcome applied to a taxon is likely relevant
# even when no wording matches -- "wildlife" never appears near "clutch size".
_TAXON_WORDS = {
    "wildlife", "animal", "animals", "fauna", "bird", "birds", "fish",
    "invertebrate", "invertebrates", "insect", "insects", "amphibian",
    "amphibians", "mammal", "mammals", "species", "community", "communities",
    "biodiversity", "taxa", "organism", "organisms",
}


def retrieve(query: Query, graph: Graph, debug: bool = False,
             on_grounded=None) -> Evidence:
    """Phrases to edges, in one stage.

    Grounding used to be a pipeline stage of its own. It was only ever retrieve's
    first step -- expand the phrases, match them to nodes -- and keeping it
    separate meant the text-match passes below re-did the same matching against
    the same node text with the same patterns. It is now a helper this function
    calls, so there is one place where a phrase becomes a node.
    """
    g = ground(query, graph, debug=debug)
    if on_grounded:  # progress report only (live viewer)
        on_grounded(g)
    ev = _edges_for(g, graph)
    ev.grounding = g
    return ev


def _edges_for(g: Grounding, graph: Graph) -> Evidence:
    outcomes, interventions = set(g.outcome_nodes), set(g.intervention_nodes)
    allowed = set(g.contexts)
    taxon_hint = any(w in _TAXON_WORDS
                     for t in g.outcome_terms for w in t.lower().split())

    passes = {
        "direct": _direct(g, graph, outcomes),
        "outcome_text": _outcome_text(g, graph),
        "intervention_text": _intervention_text(g, graph),
    }
    by_pass = {name: sorted(set(ids)) for name, ids in passes.items()}
    reverse = sorted(set(_reverse(g, graph, interventions)))

    # union, strongest pass first so `found_by` reads in priority order
    edge_ids, found_by = [], {}
    for name in ("direct", "outcome_text", "intervention_text"):
        for eid in by_pass[name]:
            if eid not in found_by:
                found_by[eid] = []
                edge_ids.append(eid)
            found_by[eid].append(name)

    def ctx_ok(eid: str) -> bool:
        """Edges carrying no ecosystem_context at all count as in-context:
        5% of the graph has none, and treating those as foreign loses evidence."""
        if not allowed:
            return True
        ctx = graph.edges[eid].get("ecosystem_context")
        return not ctx or ctx in allowed

    out_of_context = sorted(e for e in edge_ids if not ctx_ok(e))
    oo_set = set(out_of_context)

    # Best evidence first. Nothing is dropped -- the caller decides where to cut,
    # and the ordering is what makes a cut meaningful.
    scores = {e: _score(e, graph, found_by, e not in oo_set, taxon_hint)
              for e in edge_ids}
    ranked = sorted(edge_ids, key=lambda e: (-scores[e], e))

    stats = {
        "contexts": sorted(allowed),
        "passes": [
            {"name": name, "found": len(by_pass[name]),
             "new": sum(1 for e in by_pass[name] if found_by.get(e, [None])[0] == name)}
            for name in ("direct", "outcome_text", "intervention_text")
        ],
        "reverse_found": len(reverse),
        "out_of_context": len(out_of_context),
        "total": len(edge_ids),
        "taxon_hint": taxon_hint,
        "score_range": [round(min(scores.values()), 2),
                        round(max(scores.values()), 2)] if scores else [0, 0],
    }

    return Evidence(
        edge_ids=ranked,
        stats=stats,
        by_pass=by_pass,
        found_by=found_by,
        out_of_context=out_of_context,
        reverse_edge_ids=reverse,
        scores={e: round(s, 2) for e, s in scores.items()},
    )
