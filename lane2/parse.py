"""Stage 1: question text -> Query, carrying one X->Y pair per sub-question.

The LLM does all of it. Regex patterns used to run first and were removed: they
match on sentence shape and understand nothing, so "Do herbicides such as
glyphosate increase production of Oryza sativa?" matched the pattern
"(do) X (increase) Y" and handed back "herbicides such as glyphosate" as a
single cause. A pattern cannot know that "such as" introduces an example, and
every rule added to the prompt was dead code while regex answered first.

The question is read as a list of LINKS. Inside one link every cause applies to
every outcome ("herbicides such as glyphosate ... native plant cover and
richness" is one link: 2 causes x 2 outcomes). Separate clauses are separate
links and are never crossed: "burning on bird abundance, and grazing on soil
carbon" is two pairs, not four. Crossing everything against everything used to
invent questions nobody asked ("burning on soil carbon").
"""
from __future__ import annotations

import json
import re
import sys

from .llm import chat_counted
from .schema import Query, Question

_PARSE_SYSTEM = """\
You are parsing ecological research questions for a causal graph search.

FIRST decide what kind of question it is, THEN extract. Not every question asks
what one thing does to another, and forcing one that does not into that shape
produces nonsense: asked "which combinations have little evidence, and where is
the graph densest?" a careless reader answers "evidence" and "graph density",
which are properties of the database, not things measured in a field.

"mode" is one of:
  "effect"      X changes Y. The normal case.
  "interaction" do several stressors combine or modify each other.
  "gap_map"     a question ABOUT the evidence: where is it dense, thin, missing.
                Counting studies, not reading one.
  "trajectory"  how long something takes, or in what order things recover.

Then "links": a list of {"interventions": [...], "outcomes": [...]}.
  "interventions": the causes, treatments or actions studied
  "outcomes": the ecological responses or effects MEASURED IN THE FIELD
Inside ONE link, every intervention is asked about every outcome. A separate
clause about another cause or effect ("..., and what does Z do to W?", "or does
Z reduce W?") is a NEW link; "respectively" makes one link per pair. Never link
a cause to an outcome the question does not ask it about.

Rules for phrases:
- Each phrase is short (2-4 words). Drop articles, conjunctions, filler.
- List EVERY distinct cause and EVERY distinct effect. "native plant cover and
  richness" is two outcomes, and each gets the shared words:
  "native plant cover", "native plant richness". Five listed treatments are five
  interventions.
- "X such as A, B and C" means BOTH: list the general term X *and* each of
  A, B, C separately. The examples are usually the things a database actually
  records, so dropping them loses the evidence.
- An outcome is something a researcher measured in an ecosystem: cover,
  richness, biomass, survival, methane flux. It is NEVER a word about the study
  or the database: "evidence", "graph density", "studies", "data", "research
  gaps", "attributes that lag", "time to recovery". If the only candidates are
  words of that sort, the outcomes list is EMPTY. That is the right answer, and
  the mode says why.
- Do not invent phrases the question does not ask about.
- Return ONLY valid JSON: {"mode": "...", "links": [{"interventions": [...], "outcomes": [...]}]}

Examples:
Q: "When herbicides such as glyphosate are used against invasive plants, do they also reduce native plant cover and richness?"
A: {"mode": "effect", "links": [{"interventions": ["herbicides", "glyphosate"], "outcomes": ["native plant cover", "native plant richness"]}]}

Q: "What effect does prescribed burning have on bird abundance, and how does grazing affect soil carbon?"
A: {"mode": "effect", "links": [{"interventions": ["prescribed burning"], "outcomes": ["bird abundance"]}, {"interventions": ["grazing"], "outcomes": ["soil carbon"]}]}

Q: "What effects do contaminants such as oil sands tailings, PCBs and acid deposition have on wildlife?"
A: {"mode": "effect", "links": [{"interventions": ["contaminants", "oil sands tailings", "PCBs", "acid deposition"], "outcomes": ["wildlife"]}]}

Q: "For grassland restoration in Canada, which intervention-outcome combinations have little or no evidence, and where is the graph densest?"
A: {"mode": "gap_map", "links": [{"interventions": ["grassland restoration"], "outcomes": []}]}

Q: "How long do created marshes take to reach biodiversity and function comparable to natural reference wetlands, and which attributes lag?"
A: {"mode": "trajectory", "links": [{"interventions": ["created marshes"], "outcomes": ["biodiversity", "wetland function"]}]}

Q: "Where sites face several stressors at once, for example deer browsing plus competing vegetation, do the stressors interact?"
A: {"mode": "interaction", "links": [{"interventions": ["deer browsing", "competing vegetation"], "outcomes": ["restoration recovery"]}]}
"""


def _as_phrases(value) -> list[str]:
    """Accept a list or a lone string, and drop blanks and duplicates."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out = []
    for v in items:
        s = str(v).strip()
        if s and s.lower() not in {p.lower() for p in out}:
            out.append(s)
    return out


def _dedupe(seq) -> list:
    out, seen = [], set()
    for v in seq:
        k = v.lower() if isinstance(v, str) else tuple(x.lower() for x in v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


_MODES = {"effect", "interaction", "gap_map", "trajectory"}


def _links_of(obj: dict) -> list[dict]:
    """The reply's links. Tolerates the older flat shape (one link) and singular keys."""
    raw_links = obj.get("links")
    if not isinstance(raw_links, list):
        raw_links = [obj]
    links = []
    for ln in raw_links:
        if not isinstance(ln, dict):
            continue
        xs = _as_phrases(ln.get("interventions") or ln.get("intervention"))
        ys = _as_phrases(ln.get("outcomes") or ln.get("outcome"))
        if xs or ys:
            links.append({"interventions": xs, "outcomes": ys})
    return links


def _llm_parse(text: str) -> tuple[list[dict], str, int]:
    try:
        raw, tokens = chat_counted(
            [{"role": "system", "content": _PARSE_SYSTEM},
             {"role": "user", "content": text}],
            max_tokens=400,
            temperature=0,
        )
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        obj = json.loads(raw)
        mode = str(obj.get("mode") or "effect").strip().lower()
        return _links_of(obj), (mode if mode in _MODES else "effect"), tokens
    except Exception:
        return [], "effect", 0


def parse(q: Question, debug: bool = False) -> Query:
    text = " ".join(q.text.split())
    links, mode, tokens = _llm_parse(text)
    xs = _dedupe(x for ln in links for x in ln["interventions"])
    ys = _dedupe(y for ln in links for y in ln["outcomes"])
    # Cross only inside a link: a link's causes were all asked about its outcomes,
    # while a cause from another clause was not. No cap: every pair is searched.
    pairs = _dedupe((x, y) for ln in links
                    for x in ln["interventions"] for y in ln["outcomes"])
    if debug:
        print(f"  [parse] mode={mode} tokens={tokens} links={links}", file=sys.stderr)
    if not pairs:
        # No X->Y to search. For a gap_map question that is the honest result,
        # not a failure: it asks where evidence is thin, which is a counting job
        # over the graph rather than one cause and one effect.
        return Query(None, None, raw=text, tokens=tokens, mode=mode,
                     interventions=xs, outcomes=ys, links=links,
                     parse_method="not_causal" if mode != "effect" else "failed")

    if debug:
        print(f"  [parse] {len(pairs)} pair(s): "
              + "; ".join(f"{x} -> {y}" for x, y in pairs[:6]), file=sys.stderr)
    return Query(None, None, raw=text, parse_method="llm", tokens=tokens,
                 interventions=xs, outcomes=ys, pairs=pairs, mode=mode, links=links)
