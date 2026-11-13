"""Stage 2a: Query phrases -> graph node ids.  <-- THE HARD PART. Iterate here.

STUB: split the phrase into keyword terms ("mulch & surface cover" ->
["mulch", "surface cover"]) and keep nodes whose text contains any term.

Known weaknesses, roughly in priority order:
  - misses synonyms (straw, hay, erosion blanket, wood chips... for "mulch")
  - over-matches (compound treatments, "native and non-native", mulch decomposition)
  - no entity_type filtering beyond a loose intervention/outcome split
Ideas: synonym table per category, embeddings over `embedding_text`,
an offline LLM pass that labels every node with a category once.
"""
from __future__ import annotations

import re

from .graph import Graph
from .schema import Grounding, Query

# Which entity types can sit on each side. Loose on purpose for now.
CAUSE_TYPES = {"management_intervention", "environmental_variable", "environmental_process", "taxon"}
OUTCOME_TYPES = {"environmental_variable", "environmental_process", "taxon"}

STOPWORDS = {"the", "a", "an", "of", "on", "in", "and", "or", "to", "for", "with", "by"}


def terms_for(phrase: str) -> list[str]:
    """'mulch & surface cover' -> ['mulch', 'surface cover']"""
    parts = re.split(r"\s*(?:&|/|,|\band\b|\bor\b)\s*", phrase.lower())
    out = []
    for p in parts:
        p = re.sub(r"[^\w\s-]", " ", p)
        words = [w for w in p.split() if w not in STOPWORDS]
        if words:
            out.append(" ".join(words))
    return out


def _match(graph: Graph, terms: list[str], allowed_types: set[str]) -> list[str]:
    pats = [re.compile(r"\b" + re.escape(t).replace(r"\ ", r"[\s_-]+") + r"\w*", re.I) for t in terms]
    hits = []
    for nid, n in graph.nodes.items():
        if n.get("entity_type") not in allowed_types:
            continue
        text = graph.node_text(nid)
        if any(p.search(text) for p in pats):
            hits.append(nid)
    return hits


def ground(query: Query, graph: Graph) -> Grounding:
    if not query.intervention or not query.outcome:
        return Grounding()
    it, ot = terms_for(query.intervention), terms_for(query.outcome)
    return Grounding(
        intervention_nodes=_match(graph, it, CAUSE_TYPES),
        outcome_nodes=_match(graph, ot, OUTCOME_TYPES),
        intervention_terms=it,
        outcome_terms=ot,
    )
