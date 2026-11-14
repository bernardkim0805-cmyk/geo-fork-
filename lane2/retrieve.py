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
from .schema import Evidence, Grounding

_STOPWORDS = {"the", "a", "an", "of", "on", "in", "and", "or", "to", "for", "with", "by"}


def _build_pats(terms: list[str]) -> list[re.Pattern[str]]:
    meaningful = [t for t in terms if t.lower() not in _STOPWORDS and len(t) > 2]
    return [
        re.compile(r"\b" + re.escape(t).replace(r"\ ", r"[\s_-]+") + r"\w*", re.I)
        for t in meaningful
    ]


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


def retrieve(g: Grounding, graph: Graph) -> Evidence:
    outcomes, interventions = set(g.outcome_nodes), set(g.intervention_nodes)
    allowed = set(g.contexts)

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
    }

    return Evidence(
        edge_ids=sorted(edge_ids),
        stats=stats,
        by_pass=by_pass,
        found_by=found_by,
        out_of_context=out_of_context,
        reverse_edge_ids=reverse,
    )
