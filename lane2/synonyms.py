"""Strategy C: synonym expansion from a fixed dictionary, no LLM.

Strategy A asks the model to expand each phrase, which costs ~300 tokens per
question and is not reproducible -- the same phrase came back with a different
list between runs at temperature 0. This module answers the same need from a
static table, so it costs nothing, always returns the same thing, and runs with
no network.

The entries are the vocabulary gaps actually observed between the practice
questions and the graph: questions say "plant", the graph says "forb" and
"graminoid"; questions say "wildlife", the graph says "clutch size". Each group
expands in every direction, so any member retrieves all of them.
"""
from __future__ import annotations

# Groups of interchangeable terms. Order inside a group carries no meaning.
_GROUPS: list[list[str]] = [
    # --- life form -------------------------------------------------------
    # Group only terms that can stand in for each other in the same slot.
    # An earlier version put "seedling" beside "plant" and "vegetation"
    # because they are related, and "seedling size" then searched for
    # "vegetation": 50 edges about weed removal and scarification where the
    # answer was 6 about planting tall seedlings. A hypernym matches
    # everything, so relatedness is not enough -- substitutability is.
    ["vegetation", "plant cover", "flora", "sward", "ground cover"],
    ["forb", "graminoid", "herbaceous"],
    ["seedling", "sapling", "transplant", "planting stock", "nursery stock"],
    ["tree", "conifer", "woody plant"],
    ["wildlife", "fauna", "bird", "waterfowl", "nestling", "fish", "mammal",
     "amphibian", "invertebrate", "insect", "songbird"],
    ["microbe", "microbial", "bacteria", "fungi", "mycorrhiza", "mycorrhizal"],

    # --- how abundance is measured ---------------------------------------
    ["richness", "diversity", "species richness", "species diversity",
     "number of species", "taxonomic richness"],
    ["abundance", "cover", "density", "biomass", "frequency", "percent cover",
     "standing crop"],
    ["survival", "mortality", "establishment", "persistence", "recruitment",
     "hatching success", "clutch size", "brood size"],
    ["growth", "height", "diameter", "productivity", "yield", "biomass",
     "basal area", "weight"],

    # --- origin ----------------------------------------------------------
    ["native", "indigenous", "endemic"],
    ["invasive", "exotic", "non-native", "alien", "weed", "introduced"],

    # --- interventions ---------------------------------------------------
    ["herbicide", "glyphosate", "picloram", "sethoxydim", "chemical control",
     "weed control", "spraying"],
    ["fire", "burning", "prescribed burn", "wildfire", "pyric"],
    ["grazing", "browsing", "herbivory", "deer browsing", "trampling"],
    ["seeding", "seed mix", "sowing", "broadcast seed", "direct seeding",
     "revegetation"],
    ["fertilization", "fertilizer", "nutrient addition", "nitrogen addition",
     "amendment"],
    ["mowing", "cutting", "clipping", "haying"],
    ["harvesting", "logging", "clearcutting", "thinning", "partial harvest",
     "silviculture", "forestry"],
    ["tillage", "ploughing", "scarification", "site preparation",
     "soil disturbance"],
    ["rewetting", "restoration", "rehabilitation", "reclamation",
     "moss layer transfer", "realignment"],
    ["mulch", "straw", "surface cover", "litter"],

    # --- contaminants ----------------------------------------------------
    ["contaminant", "pollutant", "tailings", "oil sands", "PCB",
     "acid deposition", "heavy metal", "process water", "effluent"],

    # --- habitat ---------------------------------------------------------
    ["wetland", "marsh", "bog", "fen", "peatland", "swamp", "mire"],
    ["grassland", "prairie", "meadow", "pasture", "steppe", "rangeland"],
    ["forest", "woodland", "boreal", "plantation"],
    ["stream", "river", "riparian", "creek", "watercourse"],

    # --- ecosystem function ----------------------------------------------
    ["carbon", "carbon sink", "carbon storage", "sequestration",
     "net ecosystem exchange", "carbon flux"],
    ["methane", "CH4", "greenhouse gas", "emission"],
    ["nutrient", "nitrogen", "phosphorus", "nutrient cycling"],
    ["water table", "hydrology", "moisture", "water level", "drainage"],
    ["soil", "substrate", "seedbed", "peat"],
]

# term -> every term that shares a group with it, built once at import
_INDEX: dict[str, set[str]] = {}
for _group in _GROUPS:
    for _term in _group:
        _INDEX.setdefault(_term.lower(), set()).update(
            t for t in _group if t.lower() != _term.lower())


_INVARIANT = {"species", "grass", "moss", "biomass", "richness", "status",
              "loss", "process", "stress", "success"}


def _singular(w: str) -> str:
    """The table is keyed on singulars, so "herbicides" found nothing at all
    until lookups were stemmed."""
    if w in _INVARIANT:
        return w
    if w.endswith("ies") and len(w) > 5:
        return w[:-3] + "y"
    for suffix in ("es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)]
    return w


def expand(phrase: str, limit: int = 12) -> list[str]:
    """Dictionary synonyms for a phrase. Returns [] when nothing is known.

    Matches the whole phrase first, then each word, both stemmed, so
    "herbicides" reaches the entry filed under "herbicide".
    """
    out: list[str] = []
    seen: set[str] = set()
    p = phrase.lower().strip()
    keys = {p, _singular(p)}

    def add(terms) -> None:
        for t in terms:
            k = t.lower()
            if k not in seen and k not in keys:
                seen.add(k)
                out.append(t)

    for k in keys:
        if k in _INDEX:
            add(_INDEX[k])
    for word in p.replace("-", " ").split():
        for w in (word, _singular(word)):
            if w in _INDEX:
                add(_INDEX[w])
    return out[:limit]


def expand_all(phrases: list[str], limit: int = 12) -> list[str]:
    """Union of the dictionary expansions of several phrases."""
    out: list[str] = []
    seen: set[str] = set()
    for p in phrases:
        for t in expand(p, limit):
            if t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    return out


def coverage() -> dict:
    """How big the table is, for reporting next to the LLM's token cost."""
    return {"groups": len(_GROUPS), "terms": len(_INDEX)}
