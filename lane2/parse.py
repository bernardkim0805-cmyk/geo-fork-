"""Stage 1: question text -> Query, carrying one X->Y pair per sub-question.

The LLM does all of it. Regex patterns used to run first and were removed: they
match on sentence shape and understand nothing, so "Do herbicides such as
glyphosate increase production of Oryza sativa?" matched the pattern
"(do) X (increase) Y" and handed back "herbicides such as glyphosate" as a
single cause. A pattern cannot know that "such as" introduces an example, and
every rule added to the prompt was dead code while regex answered first.
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

Then:
  "interventions": the causes, treatments or actions studied
  "outcomes": the ecological responses or effects MEASURED IN THE FIELD

Rules:
- Each phrase is short (2-4 words). Drop articles, conjunctions, filler.
- List EVERY distinct cause and EVERY distinct effect. "cover and richness" is
  two outcomes, not one. Five listed treatments are five interventions.
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
- Return ONLY valid JSON: {"mode": "...", "interventions": [...], "outcomes": [...]}

Examples:
Q: "When herbicides such as glyphosate are used against invasive plants, do they also reduce native plant cover and richness?"
A: {"mode": "effect", "interventions": ["herbicides", "glyphosate"], "outcomes": ["native plant cover", "native plant richness"]}

Q: "What effects do contaminants such as oil sands tailings, PCBs and acid deposition have on wildlife?"
A: {"mode": "effect", "interventions": ["contaminants", "oil sands tailings", "PCBs", "acid deposition"], "outcomes": ["wildlife"]}

Q: "For grassland restoration in Canada, which intervention-outcome combinations have little or no evidence, and where is the graph densest?"
A: {"mode": "gap_map", "interventions": ["grassland restoration"], "outcomes": []}

Q: "How long do created marshes take to reach biodiversity and function comparable to natural reference wetlands, and which attributes lag?"
A: {"mode": "trajectory", "interventions": ["created marshes"], "outcomes": ["biodiversity", "wetland function"]}

Q: "Where sites face several stressors at once, for example deer browsing plus competing vegetation, do the stressors interact?"
A: {"mode": "interaction", "interventions": ["deer browsing", "competing vegetation"], "outcomes": ["restoration recovery"]}

Q: "Which forestry practices maintain biodiversity in Canadian forests?"
A: {"mode": "effect", "interventions": ["forestry practices"], "outcomes": ["biodiversity"]}
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


_MODES = {"effect", "interaction", "gap_map", "trajectory"}


def _llm_parse(text: str) -> tuple[list[str], list[str], str, int]:
    try:
        raw, tokens = chat_counted(
            [{"role": "system", "content": _PARSE_SYSTEM},
             {"role": "user", "content": text}],
            max_tokens=400,
            temperature=0,
        )
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        obj = json.loads(raw)
        # tolerate the older singular keys in case the model falls back to them
        xs = _as_phrases(obj.get("interventions") or obj.get("intervention"))
        ys = _as_phrases(obj.get("outcomes") or obj.get("outcome"))
        mode = str(obj.get("mode") or "effect").strip().lower()
        return xs, ys, (mode if mode in _MODES else "effect"), tokens
    except Exception:
        return [], [], "effect", 0


def parse(q: Question, debug: bool = False) -> Query:
    text = " ".join(q.text.split())
    xs, ys, mode, tokens = _llm_parse(text)
    if debug:
        print(f"  [parse] mode={mode} tokens={tokens} X={xs} Y={ys}", file=sys.stderr)
    if not (xs and ys):
        # No X->Y to search. For a gap_map question that is the honest result,
        # not a failure: it asks where evidence is thin, which is a counting job
        # over the graph rather than one cause and one effect.
        return Query(None, None, raw=text, tokens=tokens, mode=mode,
                     interventions=xs, outcomes=ys,
                     parse_method="not_causal" if mode != "effect" else "failed")

    # Every cause against every outcome. "herbicides such as glyphosate ... rice
    # production" is two questions, not one: something can hold for glyphosate
    # specifically and not for herbicides in general, and merging them into one
    # search loses exactly that distinction.
    pairs = [(x, y) for x in xs for y in ys]
    if debug:
        print(f"  [parse] {len(pairs)} pair(s): "
              + "; ".join(f"{x} -> {y}" for x, y in pairs[:6]), file=sys.stderr)
    return Query(None, None, raw=text, parse_method="llm", tokens=tokens,
                 interventions=xs, outcomes=ys, pairs=pairs, mode=mode)
