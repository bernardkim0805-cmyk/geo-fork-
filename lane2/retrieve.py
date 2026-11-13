"""Stage 2b: grounded node sets -> edges linking them.

STUB: edges whose subject is an intervention node and whose object is an
outcome node. Direction matters: an edge outcome -> intervention does NOT
answer "what does X do to Y" (likely trap territory).
Later: multi-hop paths (X -> mediator -> Y), embedding fallback over edges.
"""
from __future__ import annotations

from .graph import Graph
from .schema import Evidence, Grounding


def retrieve(g: Grounding, graph: Graph) -> Evidence:
    outcomes = set(g.outcome_nodes)
    edge_ids = []
    for nid in g.intervention_nodes:
        for eid in graph.edges_by_subject.get(nid, []):
            if graph.edges[eid]["object"] in outcomes:
                edge_ids.append(eid)
    return Evidence(edge_ids=sorted(set(edge_ids)))
