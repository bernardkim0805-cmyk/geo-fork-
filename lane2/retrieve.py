"""Stage 2b: grounded node sets -> edges linking them.

Three passes, tried in order, stopping as soon as edges are found:
  1. Direct match: subject ∈ intervention_nodes AND object ∈ outcome_nodes.
     Direction is enforced — an object→subject edge is a trap, excluded here.
  2. Outcome fallback: intervention nodes are grounded but outcome grounding was
     too sparse.  Scan all edges from intervention_nodes; include any edge whose
     object node's embedding_text keyword-matches the outcome_terms (which
     include LLM synonyms when the LLM ran).
  3. Intervention fallback (symmetric): outcome nodes are grounded but
     intervention grounding failed.  Walk edges_by_object for outcome_nodes;
     include any edge whose subject matches intervention_terms.

Multi-hop (X -> mediator -> Y) was considered but the graph has only ~112 such
paths vs 2 816 direct ones, so the signal/cost ratio doesn't justify it.
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


def retrieve(g: Grounding, graph: Graph) -> Evidence:
    outcomes = set(g.outcome_nodes)

    # Pass 1: direct edges (subject=intervention, object=outcome)
    edge_ids: list[str] = []
    for nid in g.intervention_nodes:
        for eid in graph.edges_by_subject.get(nid, []):
            if graph.edges[eid]["object"] in outcomes:
                edge_ids.append(eid)

    if edge_ids:
        return Evidence(edge_ids=sorted(set(edge_ids)))

    # Pass 2: outcome grounding too sparse — text-match the object node
    if g.intervention_nodes and g.outcome_terms:
        pats = _build_pats(g.outcome_terms)
        if pats:
            for nid in g.intervention_nodes:
                for eid in graph.edges_by_subject.get(nid, []):
                    obj_text = graph.node_text(graph.edges[eid]["object"])
                    if _text_matches(obj_text, pats):
                        edge_ids.append(eid)

    if edge_ids:
        return Evidence(edge_ids=sorted(set(edge_ids)))

    # Pass 3: intervention grounding too sparse — text-match the subject node
    if g.outcome_nodes and g.intervention_terms:
        pats = _build_pats(g.intervention_terms)
        if pats:
            for nid in g.outcome_nodes:
                for eid in graph.edges_by_object.get(nid, []):
                    sub_text = graph.node_text(graph.edges[eid]["subject"])
                    if _text_matches(sub_text, pats):
                        edge_ids.append(eid)

    return Evidence(edge_ids=sorted(set(edge_ids)))
