"""Stage 2a: Query phrases -> graph node ids.  <-- THE HARD PART. Iterate here.

Strategy A: LLM synonym expansion -> keyword match.
  One batched LLM call per question (both phrases together).
  Set GROUND_STRATEGY=A (default).

Strategy B: Sentence-embedding cosine similarity over node embedding_text.
  Tokens per ground() call: 0 (local model, no LLM).
  Set GROUND_STRATEGY=B.
  Requires: sentence-transformers installed.
"""
from __future__ import annotations

import json
import os
import re
import sys

from .llm import chat_counted
from .graph import Graph
from .schema import Grounding, Query

# Which entity types can sit on each side. Loose on purpose for now.
CAUSE_TYPES = {"management_intervention", "environmental_variable", "environmental_process", "taxon"}
OUTCOME_TYPES = {"environmental_variable", "environmental_process", "taxon"}

STOPWORDS = {"the", "a", "an", "of", "on", "in", "and", "or", "to", "for", "with", "by"}

_BATCH_SYSTEM = """\
You are a synonym expander for ecological restoration science.
Given phrases X (intervention/cause) and Y (outcome/effect), return SHORT
synonyms (1-2 words, 3-word max) that appear in ecology or restoration papers
as node names or measured attributes. Include common English AND scientific terms.
Return ONLY: {"x": [...8-12 strings...], "y": [...8-12 strings...]}. No explanation.
Example:
X: "mulch application"  Y: "invasive species abundance"
Output: {"x": ["mulching", "mulch", "straw mulch", "organic mulch", "surface cover"],
         "y": ["exotic", "non-native", "invasive", "alien species", "weed cover", "exotic grass"]}
"""

_synonym_cache: dict[str, list[str]] = {}


def _llm_synonyms_batch(x_phrase: str, y_phrase: str) -> tuple[list[str], list[str], int]:
    """Expand both phrases in one LLM call. Returns (x_terms, y_terms, tokens)."""
    xk, yk = x_phrase.lower().strip(), y_phrase.lower().strip()
    x_cached = _synonym_cache.get(xk)
    y_cached = _synonym_cache.get(yk)
    if x_cached is not None and y_cached is not None:
        return x_cached, y_cached, 0

    try:
        raw, tokens = chat_counted(
            [{"role": "system", "content": _BATCH_SYSTEM},
             {"role": "user", "content": f'X: "{x_phrase}"  Y: "{y_phrase}"'}],
            max_tokens=1200,
            temperature=0,
        )
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        obj = json.loads(raw)
        if isinstance(obj, dict):
            xs = [str(s).strip() for s in obj.get("x", []) if s]
            ys = [str(s).strip() for s in obj.get("y", []) if s]
            _synonym_cache[xk] = xs
            _synonym_cache[yk] = ys
            return xs, ys, tokens
    except Exception:
        pass
    return [], [], 0


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


# Very common ecology / environment words that are too broad to use alone as
# keyword terms (would match hundreds of unrelated nodes).
_BROAD_WORDS = {
    "soil", "land", "fire", "water", "plant", "species", "cover", "surface",
    "ground", "site", "area", "carbon", "organic", "native", "total", "layer",
    "type", "depth", "level", "rate", "change", "effect", "increase", "decrease",
}


def _expand_terms(terms: list[str]) -> list[str]:
    """For multi-word terms like 'exotic abundance', also try matching just the
    first word ('exotic') so it hits nodes named 'exotic species densities'.
    Skips first words that are too generic to use alone (e.g. 'soil', 'cover')."""
    expanded = list(terms)
    for t in terms:
        words = [w for w in t.split() if w.lower() not in STOPWORDS and len(w) > 3]
        if len(words) > 1:
            first = words[0]
            if first.lower() not in _BROAD_WORDS and first not in expanded:
                expanded.append(first)
    return list(dict.fromkeys(expanded))


def _match(graph: Graph, terms: list[str], allowed_types: set[str],
           stats: dict | None = None) -> list[str]:
    """Nodes whose text matches any term. `stats` records what the type filter
    removed, so the UI can show the funnel instead of only the survivors."""
    effective = _expand_terms(terms)
    pats = [re.compile(r"\b" + re.escape(t).replace(r"\ ", r"[\s_-]+") + r"\w*", re.I) for t in effective]
    hits, wrong_type = [], 0
    for nid, n in graph.nodes.items():
        text = graph.node_text(nid)
        if not any(p.search(text) for p in pats):
            continue
        if n.get("entity_type") not in allowed_types:
            wrong_type += 1
            continue
        hits.append(nid)
    if stats is not None:
        stats["expanded_terms"] = effective
        stats["text_matched"] = len(hits) + wrong_type
        stats["dropped_wrong_type"] = wrong_type
        stats["kept"] = len(hits)
        # which single term is responsible for the most hits — the usual culprit
        # when grounding explodes to hundreds of nodes
        per_term = {}
        for t, p in zip(effective, pats):
            per_term[t] = sum(1 for nid in hits if p.search(graph.node_text(nid)))
        stats["per_term"] = dict(sorted(per_term.items(), key=lambda kv: -kv[1]))
    return hits


_embed_cache: dict = {}  # node_id -> np.ndarray, populated once per process

def _embed_match(graph: Graph, phrase: str, allowed_types: set[str], top_k: int = 15) -> tuple[list[str], int]:
    """Option B: cosine similarity over node embedding_text. Returns (node_ids, 0 tokens)."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model_name = "all-MiniLM-L6-v2"
        if "model" not in _embed_cache:
            _embed_cache["model"] = SentenceTransformer(model_name)
        model = _embed_cache["model"]

        # build node matrix once per process
        if "matrix" not in _embed_cache:
            ids, texts = [], []
            for nid, n in graph.nodes.items():
                ids.append(nid)
                texts.append(graph.node_text(nid))
            mat = model.encode(texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
            _embed_cache["matrix"] = mat
            _embed_cache["ids"] = ids
            _embed_cache["types"] = [graph.nodes[nid].get("entity_type") for nid in ids]

        q_vec = model.encode([phrase], normalize_embeddings=True)[0]
        scores = _embed_cache["matrix"] @ q_vec
        order = np.argsort(scores)[::-1]

        hits = []
        for idx in order:
            if _embed_cache["types"][idx] in allowed_types:
                hits.append(_embed_cache["ids"][idx])
            if len(hits) >= top_k:
                break
        return hits, 0
    except Exception as e:
        print(f"  [ground B] embedding failed: {e}", file=sys.stderr)
        return [], 0


# Words a question uses mapped to the word that appears in ecosystem_context
# labels. Only needed where the question's word is not already in the label.
_CONTEXT_SYNONYMS = {
    "peatland": "peat", "peatlands": "peat", "bog": "peat", "bogs": "peat",
    "prairie": "grassland", "prairies": "grassland", "meadow": "grassland",
    "meadows": "grassland", "steppe": "grassland",
    "woodland": "woodland", "forested": "forest", "forestry": "forest",
    "saltmarsh": "saltmarsh", "salt": "saltmarsh",
    "riparian": "river", "riverine": "river", "streams": "stream",
    "agricultural": "cropland", "agriculture": "cropland", "farmland": "cropland",
    "crop": "cropland", "crops": "cropland",
    "alpine": "alpine", "montane": "montane", "boreal": "boreal",
    "coastal": "coastal", "shoreline": "shoreline", "estuary": "estuarie",
    "reef": "reef", "seagrass": "seagrass", "tundra": "tundra",
    "pasture": "pasture", "oldfield": "oldfield", "plantation": "plantation",
}

# Words that name a habitat only when qualified: "meadow" alone must not pull in
# seagrass_meadows, nor "forest" subtropical_temperate_forested_wetlands.
_CONTEXT_QUALIFIED_ONLY = {"meadow", "reedbed", "shrubland", "woodland", "wetland"}

_CONTEXT_MIN_WORD_LEN = 4


def _singular(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    for suffix in ("es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)]
    return w


def _match_contexts(text: str, graph: Graph) -> list[str]:
    """Which ecosystem_context labels does this question restrict itself to?

    Keyword match against the label's own words, so it costs no tokens. An empty
    result means 'no restriction' — retrieve then accepts any context.
    """
    raw = set(re.findall(r"[a-z]+", text.lower()))
    raw |= {_CONTEXT_SYNONYMS[w] for w in raw if w in _CONTEXT_SYNONYMS}
    words = {_singular(w) for w in raw}

    hits = []
    for label in graph.contexts:
        parts = [p for p in label.split("_") if len(p) >= _CONTEXT_MIN_WORD_LEN]
        singles = {_singular(p) for p in parts}
        # a generic habitat word only counts when the question also names one of
        # the label's qualifiers, so "meadow" does not match seagrass_meadows
        distinctive = singles - _CONTEXT_QUALIFIED_ONLY
        if distinctive & words or (singles & words and not distinctive):
            hits.append(label)
    return sorted(hits)


def ground(query: Query, graph: Graph, debug: bool = False) -> Grounding:
    if not query.intervention or not query.outcome:
        return Grounding()

    strategy = os.environ.get("GROUND_STRATEGY", "A").upper()
    total_tokens = 0
    contexts = _match_contexts(query.raw, graph)
    if debug:
        print(f"  [ground] contexts={contexts or '(unrestricted)'}", file=sys.stderr)

    if strategy == "B":
        x_nodes = list(dict.fromkeys(
            n for p in query.interventions for n in _embed_match(graph, p, CAUSE_TYPES)[0]))
        y_nodes = list(dict.fromkeys(
            n for p in query.outcomes for n in _embed_match(graph, p, OUTCOME_TYPES)[0]))
        total_tokens = 0
        it = [t for p in query.interventions for t in terms_for(p)]
        ot = [t for p in query.outcomes for t in terms_for(p)]
        if debug:
            x_names = [graph.nodes[n]["name"] for n in x_nodes[:5]]
            y_names = [graph.nodes[n]["name"] for n in y_nodes[:5]]
            print(f"  [ground B] X top5={x_names}", file=sys.stderr)
            print(f"  [ground B] Y top5={y_names}", file=sys.stderr)
            print(f"  [ground B] tokens=0 (local embeddings)", file=sys.stderr)
        return Grounding(
            intervention_nodes=x_nodes,
            outcome_nodes=y_nodes,
            intervention_terms=it,
            outcome_terms=ot,
            tokens=0,
            contexts=contexts,
        )

    # Strategy A: one batched LLM call for both synonym expansions.
    # A question may name several causes or effects, so expand every phrase and
    # take the union -- b3 asks about cover AND richness.
    it = [t for phrase in query.interventions for t in terms_for(phrase)]
    ot = [t for phrase in query.outcomes for t in terms_for(phrase)]
    x_synonyms, y_synonyms, total_tokens = _llm_synonyms_batch(
        " ; ".join(query.interventions), " ; ".join(query.outcomes))
    all_it = list(dict.fromkeys(query.interventions + it + x_synonyms))
    all_ot = list(dict.fromkeys(query.outcomes + ot + y_synonyms))

    if debug:
        print(f"  [ground A] X terms={all_it}", file=sys.stderr)
        print(f"  [ground A] Y terms={all_ot}", file=sys.stderr)
        print(f"  [ground A] total_tokens={total_tokens}", file=sys.stderr)

    x_stats, y_stats = {}, {}
    x_nodes = _match(graph, all_it, CAUSE_TYPES, x_stats)
    y_nodes = _match(graph, all_ot, OUTCOME_TYPES, y_stats)

    return Grounding(
        intervention_nodes=x_nodes,
        outcome_nodes=y_nodes,
        intervention_terms=all_it,
        outcome_terms=all_ot,
        tokens=total_tokens,
        contexts=contexts,
        stats={"intervention": x_stats, "outcome": y_stats},
    )
