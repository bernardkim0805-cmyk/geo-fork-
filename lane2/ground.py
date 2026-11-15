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

from . import synonyms
from .llm import chat_counted
from .graph import Graph
from .schema import Grounding, Query

# Which entity types can sit on each side. Loose on purpose for now.
# Below this many dictionary synonyms, the hybrid strategy pays for an LLM call.
_HYBRID_MIN = 3

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
    # "native" stays. Removing it measured +4 points of recall under strategy B,
    # whose top_k caps the node sets -- but strategy A has no such cap, and the
    # same change took b3 from 27 edges to 370. Only trust an ablation run on
    # the strategy that is actually the default.
    # Measurement words that name no subject on their own. "seedling size" was
    # being widened to bare "size", which matches seed size, patch size and
    # clutch size alike.
    "size", "amount", "number", "quantity", "value", "index", "score",
    "content", "status", "condition", "quality", "presence", "absence",
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
        if len(words) < 2:
            continue
        # Both ends, not just the first. Every outcome synonym here begins with
        # a broad word ("native cover", "native richness"), so taking only the
        # first word added nothing at all while the informative word -- richness,
        # abundance, diversity -- sat unused at the end.
        for w in (words[0], words[-1]):
            if w.lower() not in _BROAD_WORDS and w not in expanded:
                expanded.append(w)
    return list(dict.fromkeys(expanded))


def _term_pattern(term: str) -> re.Pattern[str]:
    """Match a term regardless of which side carries the plural.

    `\\w*` alone only lets a term grow a suffix, never lose one, so the question's
    "herbicides" could not reach the node "herbicide application" and matched
    nothing at all. Each word is stemmed and then allowed its own ending back.
    """
    words = [w for w in re.split(r"[\s_-]+", term.strip()) if w]
    parts = [re.escape(_singular(w)) + r"\w*" for w in words]
    if not parts:
        return re.compile(r"(?!x)x")  # matches nothing
    # Allow a qualifier between the words. Node names routinely insert one --
    # "native FORB cover", "native SPECIES richness" -- and requiring the words
    # to be adjacent rejected every such name.
    gap = r"(?:[\s_-]+\w+){0,2}[\s_-]+"
    return re.compile(r"\b" + gap.join(parts), re.I)


def _match(graph: Graph, terms: list[str], allowed_types: set[str],
           stats: dict | None = None, expand: bool = True) -> list[str]:
    """Nodes whose text matches any term. `stats` records what the type filter
    removed, so the UI can show the funnel instead of only the survivors."""
    effective = _expand_terms(terms) if expand else list(dict.fromkeys(terms))
    pats = [_term_pattern(t) for t in effective]
    # A node whose NAME matches is about the phrase. One that matches only in
    # the rest of embedding_text was merely measured on it -- "soil inoculum
    # quantity ... applied to: tree seedlings" is a soil study, not a seedling
    # one. Both are kept so nothing is lost, but named matches come first and
    # the weaker ones are recorded so the caller can cut them.
    named, applied, wrong_type = [], [], 0
    for nid, n in graph.nodes.items():
        name_text = graph.node_name_text(nid)
        on_name = any(p.search(name_text) for p in pats)
        if not on_name and not any(p.search(graph.node_text(nid)) for p in pats):
            continue
        if n.get("entity_type") not in allowed_types:
            wrong_type += 1
            continue
        (named if on_name else applied).append(nid)
    hits = named + applied
    if stats is not None:
        stats["expanded_terms"] = effective
        stats["text_matched"] = len(hits) + wrong_type
        stats["dropped_wrong_type"] = wrong_type
        stats["kept"] = len(hits)
        stats["named"] = len(named)
        stats["applied_only"] = len(applied)
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


# Words that already end in a plural-looking suffix while being singular, or
# whose singular is the same word. Stemming these breaks the match entirely:
# "species" became "specy", so every "* species" term matched nothing.
_INVARIANT = {
    "species", "series", "analysis", "diagnosis", "biomass", "grass", "moss",
    "process", "loss", "access", "stress", "class", "mass", "success", "cross",
    "richness", "dryness", "wetness", "roughness", "openness", "status",
}


def _singular(w: str) -> str:
    if w.lower() in _INVARIANT:
        return w
    if w.endswith("ies") and len(w) > 5:
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


def _attribute_outcomes(graph: Graph, phrases: list[str],
                        nodes: list[str]) -> dict[str, list[str]]:
    """Assign each grounded outcome node to the phrase it best answers.

    A question naming two outcomes is two questions sharing one search, and the
    merged result hides which evidence answers which half. Matching each node
    against a phrase's literal pattern leaves almost everything unassigned --
    the nodes arrived via synonyms belonging to neither phrase -- so this scores
    stemmed word overlap and takes the best, with ties going to both.
    """
    if len(phrases) < 2:
        return {p: list(nodes) for p in phrases}

    # What separates the phrases is what they do NOT share: "native plant cover"
    # against "native plant richness" differs only on cover/richness. Broad
    # words are kept here -- "cover" is uninformative when searching the graph
    # but is the whole distinction when telling two outcomes apart.
    word_sets = [{_singular(w) for w in re.findall(r"[a-z]+", p.lower())} - STOPWORDS
                 for p in phrases]
    shared = set.intersection(*word_sets) if word_sets else set()
    keys = [ws - shared for ws in word_sets]
    # if the phrases differ by nothing usable, fall back to the full word sets
    if not any(keys):
        keys = word_sets

    groups: dict[str, list[str]] = {p: [] for p in phrases}
    for nid in nodes:
        text = {_singular(w) for w in re.findall(r"[a-z]+", graph.node_text(nid).lower())}
        scores = [len(k & text) for k in keys]
        best = max(scores)
        if best == 0:
            continue
        for p, s in zip(phrases, scores):
            if s == best:
                groups[p].append(nid)
    return groups


_SELECT_SYSTEM = """\
You match a researcher's wording to entries that exist in an ecology database.

You get a PHRASE and a numbered list of real entries. Return the numbers of
every entry that could be studying the same thing.

The entries are written by annotators reading papers, so they are long, messy
and specific, and will almost never repeat the phrase word for word. Match on
MEANING, not wording. Be generous: a near match is useful, and the researcher
can discard it later.

MATCH:
- the same thing said differently
  "created marshes" -> "construction or enhancement of ponds"
  "created marshes" -> "restoration of wetlands by dredging of sediments"
  "created marshes" -> "Presence of human-created early seral vegetation and marshes"
- a specific instance of it
  "herbicides" -> "Glyphosate treatment", "Cutting and glyphosate treatments"
- the same thing with extra detail attached (site, year, species, method)
  "fertilization" -> "Forest Pak fertilization of black spruce in 1998"
- singular, plural and word forms
  "marsh" matches "marshes", "marshland", "high marsh species"

DO NOT MATCH:
- something merely measured in or near it
  "created marshes" does NOT match "marsh bird abundance": that is an effect
  observed in a marsh, not the making of one
- a category so broad the phrase is only one small part of it
  "glyphosate" does NOT match "any chemical input"
- a different thing that shares a word
  "seed mix" does NOT match "seeds per fruit"

Return ONLY a JSON array of integers, e.g. [0, 3, 7]. Return [] only when the
list truly holds nothing related.
"""


def _select_from_graph(graph: Graph, phrase: str, allowed_types: set[str],
                       pool: int = 40) -> tuple[list[str], int]:
    """Candidates come from the graph, the model only filters them.

    Asking the model for synonyms outright produced "rice yield" and "grain
    yield" for Oryza sativa: good English, zero nodes, while the generic
    "biomass" it offered alongside matched 370 unrelated ones. Here every
    option is a node that exists, so an invented term is not expressible and an
    empty answer is a real finding.
    """
    cands, _ = _embed_match(graph, phrase, allowed_types, top_k=pool)
    if not cands:
        return [], 0
    listing = "\n".join(f"{i}. {graph.nodes[n].get('name')}"
                        for i, n in enumerate(cands))
    try:
        raw, tokens = chat_counted(
            [{"role": "system", "content": _SELECT_SYSTEM},
             {"role": "user", "content": f'PHRASE: "{phrase}"\n\nENTRIES:\n{listing}'}],
            max_tokens=200, temperature=0)
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        picked = json.loads(raw)
        chosen = [cands[i] for i in picked
                  if isinstance(i, int) and 0 <= i < len(cands)]
        return chosen, tokens
    except Exception:
        return [], 0


# "How long does X take to reach Y?" is not a question about X. Nothing in the
# graph is a created marsh -- five nodes mention marshes at all -- and asking
# for one returns nothing. The cause of a trajectory is elapsed time, and the
# comparison it asks for is restored site against reference site, so those are
# what the cause side should search for. Both halves are needed: of the causes
# lane 3 used for the created-marshes question, time words reach six and site
# words reach seven.
_TRAJECTORY_TERMS = [
    # elapsed time
    "age", "site age", "wetland age", "stand age", "restoration age",
    "time since restoration", "time since reclamation", "time elapsed",
    "years since", "year since", "chronosequence", "succession",
    "post-restoration", "development",
    # which site you are in, i.e. the reference comparison
    "reference site", "reference status", "natural site", "natural wetland",
    "natural reserve", "restored site", "reclaimed site", "created site",
    "site location", "site status", "unrestored", "undisturbed",
]


def ground(query: Query, graph: Graph, debug: bool = False) -> Grounding:
    if not query.intervention or not query.outcome:
        return Grounding()

    strategy = os.environ.get("GROUND_STRATEGY", "A").upper()
    total_tokens = 0
    contexts = _match_contexts(query.raw, graph)
    if debug:
        print(f"  [ground] contexts={contexts or '(unrestricted)'}", file=sys.stderr)

    if strategy == "D":
        # The graph proposes, the model disposes. No anchor gate is needed:
        # an empty selection already means the graph holds nothing for it.
        x_nodes, y_nodes, total_tokens = [], [], 0
        for p in query.interventions:
            picked, t = _select_from_graph(graph, p, CAUSE_TYPES)
            x_nodes += picked
            total_tokens += t
        for p in query.outcomes:
            picked, t = _select_from_graph(graph, p, OUTCOME_TYPES)
            y_nodes += picked
            total_tokens += t
        x_nodes = list(dict.fromkeys(x_nodes))
        y_nodes = list(dict.fromkeys(y_nodes))
        # terms are the names the model actually chose, so retrieve's text
        # passes search real vocabulary rather than invented phrases
        it = [graph.nodes[n].get("name") or "" for n in x_nodes]
        ot = [graph.nodes[n].get("name") or "" for n in y_nodes]
        if debug:
            print(f"  [ground D] X picked {len(x_nodes)}: "
                  f"{[graph.nodes[n].get('name') for n in x_nodes[:4]]}", file=sys.stderr)
            print(f"  [ground D] Y picked {len(y_nodes)}: "
                  f"{[graph.nodes[n].get('name') for n in y_nodes[:4]]}", file=sys.stderr)
            print(f"  [ground D] tokens={total_tokens}", file=sys.stderr)
        return Grounding(
            intervention_nodes=x_nodes, outcome_nodes=y_nodes,
            intervention_terms=query.interventions + it[:10],
            outcome_terms=query.outcomes + ot[:10],
            tokens=total_tokens, contexts=contexts,
            stats={"intervention": {"kept": len(x_nodes), "anchor": len(x_nodes)},
                   "outcome": {"kept": len(y_nodes), "anchor": len(y_nodes)}},
            outcome_groups=_attribute_outcomes(graph, query.outcomes, y_nodes),
        )

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

    # Strategies A and C differ only in where the synonyms come from.
    # A question may name several causes or effects, so expand every phrase and
    # take the union -- b3 asks about cover AND richness.
    it = [t for phrase in query.interventions for t in terms_for(phrase)]
    ot = [t for phrase in query.outcomes for t in terms_for(phrase)]

    # A trajectory question's cause is time and site status, not the thing the
    # question names. The named thing stays in the list so a real intervention
    # ("restoration of wetlands by dredging") is still reachable.
    trajectory = query.mode == "trajectory"
    if trajectory:
        it = list(dict.fromkeys(it + _TRAJECTORY_TERMS))

    if strategy == "C":
        # dictionary lookup: 0 tokens, deterministic, works offline
        x_synonyms = synonyms.expand_all(query.interventions + it)
        y_synonyms = synonyms.expand_all(query.outcomes + ot)
        total_tokens = 0
    elif strategy == "H":
        # Hybrid: the dictionary answers what it knows for free, and the LLM is
        # asked only about a side it left thin. The dictionary is written from
        # vocabulary we have already seen, so an unfamiliar question is exactly
        # where it returns nothing and exactly where the LLM earns its tokens.
        x_synonyms = synonyms.expand_all(query.interventions + it)
        y_synonyms = synonyms.expand_all(query.outcomes + ot)
        thin_x = len(x_synonyms) < _HYBRID_MIN
        thin_y = len(y_synonyms) < _HYBRID_MIN
        total_tokens = 0
        if thin_x or thin_y:
            llm_x, llm_y, total_tokens = _llm_synonyms_batch(
                " ; ".join(query.interventions), " ; ".join(query.outcomes))
            if thin_x:
                x_synonyms = list(dict.fromkeys(x_synonyms + llm_x))
            if thin_y:
                y_synonyms = list(dict.fromkeys(y_synonyms + llm_y))
    else:
        x_synonyms, y_synonyms, total_tokens = _llm_synonyms_batch(
            " ; ".join(query.interventions), " ; ".join(query.outcomes))
    all_it = list(dict.fromkeys(query.interventions + it + x_synonyms))
    all_ot = list(dict.fromkeys(query.outcomes + ot + y_synonyms))

    if debug:
        tag = {"C": "C dict", "H": "H hybrid"}.get(strategy, "A llm")
        print(f"  [ground {tag}] X terms={all_it}", file=sys.stderr)
        print(f"  [ground {tag}] Y terms={all_ot}", file=sys.stderr)
        print(f"  [ground {tag}] total_tokens={total_tokens}", file=sys.stderr)

    # The anchor is the question's own wording. If the graph holds nothing for
    # it, no synonym may stand in: asked about Oryza sativa the model offered
    # "rice yield" (0 nodes) beside "yield" and "biomass" (370 nodes of ant
    # communities and jack pine), and the answer came back "mixed" with
    # moderate certainty. Nothing about rice was ever found. An empty anchor is
    # the finding, not a gap to fill.
    # expand=False on purpose: the anchor is the phrase as asked. Expanding it
    # defeats the check, since "oryza sativa production" would be widened to the
    # bare word "production" and pass on 361 nodes about ant communities.
    x_anchor = _match(graph, query.interventions, CAUSE_TYPES, expand=False)
    y_anchor = _match(graph, query.outcomes, OUTCOME_TYPES, expand=False)

    # A failed anchor also drops the synonyms, not just the nodes: retrieve's
    # text-match passes search these terms directly, so leaving "yield" and
    # "biomass" in the list let them find 22 edges about ant communities for a
    # question about rice, after the node gate had already found nothing.
    # A trajectory question names something the graph does not hold ("created
    # marshes"), so its anchor is the time and site vocabulary instead.
    if trajectory:
        x_anchor = x_anchor or _match(graph, _TRAJECTORY_TERMS, CAUSE_TYPES, expand=False)

    if not x_anchor:
        all_it = list(query.interventions)
    if not y_anchor:
        all_ot = list(query.outcomes)

    # Word-split the question's own phrases, but never the synonyms. A synonym
    # is already an expansion; splitting it expands twice. "seedling size" drew
    # the synonym "planting stock", which split to bare "planting" and matched
    # "planting season" and "planting position" -- neither about seedling size.
    x_split = _expand_terms(query.interventions)
    y_split = _expand_terms(query.outcomes)
    all_it = list(dict.fromkeys(x_split + all_it))
    all_ot = list(dict.fromkeys(y_split + all_ot))

    x_stats, y_stats = {}, {}
    x_nodes = _match(graph, all_it, CAUSE_TYPES, x_stats, expand=False) if x_anchor else []
    y_nodes = _match(graph, all_ot, OUTCOME_TYPES, y_stats, expand=False) if y_anchor else []
    x_stats["anchor"] = len(x_anchor)
    y_stats["anchor"] = len(y_anchor)

    if debug and not (x_anchor and y_anchor):
        missing = []
        if not x_anchor:
            missing.append(f"X={query.interventions}")
        if not y_anchor:
            missing.append(f"Y={query.outcomes}")
        print(f"  [ground] anchor not in graph: {', '.join(missing)}"
              f" -- no evidence, synonyms not used", file=sys.stderr)

    outcome_groups = _attribute_outcomes(graph, query.outcomes, y_nodes)

    return Grounding(
        intervention_nodes=x_nodes,
        outcome_nodes=y_nodes,
        intervention_terms=all_it,
        outcome_terms=all_ot,
        tokens=total_tokens,
        contexts=contexts,
        stats={"intervention": x_stats, "outcome": y_stats},
        outcome_groups=outcome_groups,
    )
