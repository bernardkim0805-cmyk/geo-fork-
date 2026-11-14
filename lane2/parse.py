"""Stage 1: question text -> Query(intervention, outcome).

Regex for simple "What effect does X have on Y?" phrasing.
LLM fallback for everything else.
"""
from __future__ import annotations

import json
import re
import sys

from .llm import chat_counted
from .schema import Query, Question

PATTERNS = [
    re.compile(r"what (?:effect|impact|influence)s? (?:does|do|did) (?P<x>.+?) have on (?P<y>.+?)\??$", re.I),
    re.compile(r"how (?:does|do) (?P<x>.+?) (?:affect|influence|change|impact) (?P<y>.+?)\??$", re.I),
    re.compile(r"(?:does|do) (?P<x>.+?) (?:affect|increase|decrease|reduce|change) (?P<y>.+?)\??$", re.I),
]

_PARSE_SYSTEM = """\
You are parsing ecological research questions for a causal graph search.
Extract every cause and every effect the question asks about:
  "interventions": the causes, treatments or actions studied (what is done or changes)
  "outcomes": the ecological responses or effects measured (what is affected)

Rules:
- Each phrase is short (2-4 words). Drop articles, conjunctions, filler.
- List EVERY distinct cause and EVERY distinct effect. "cover and richness" is
  two outcomes, not one. Five listed treatments are five interventions.
- "X such as Y" means the subject is X in general, not just Y. Use the general term.
- Do not invent phrases the question does not ask about.
- Return ONLY valid JSON: {"interventions": [...], "outcomes": [...]}
- If there is no cause/effect structure at all, return {"interventions": [], "outcomes": []}

Examples:
Q: "When herbicides such as glyphosate are used against invasive plants, do they also reduce native plant cover and richness?"
A: {"interventions": ["herbicides"], "outcomes": ["native plant cover", "native plant richness"]}

Q: "How do seedling size, nursery fertilization and mycorrhizal inoculation affect survival and growth?"
A: {"interventions": ["seedling size", "nursery fertilization", "mycorrhizal inoculation"], "outcomes": ["seedling survival", "seedling growth"]}

Q: "Which forestry practices maintain biodiversity in Canadian forests?"
A: {"interventions": ["forestry practices"], "outcomes": ["biodiversity"]}
"""


PRONOUNS = {"they", "it", "this", "that", "these", "those", "them", "he", "she", "we", "you", "i"}


def _looks_bad(phrase: str) -> bool:
    """Regex sometimes grabs a pronoun or filler instead of the real phrase."""
    words = [w for w in re.sub(r"[^\w\s]", " ", phrase.lower()).split() if w]
    if not words:
        return True
    if words[0] in PRONOUNS:
        return True
    if len(" ".join(words)) < 4:
        return True
    return False


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


def _llm_parse(text: str) -> tuple[list[str], list[str], int]:
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
        return xs, ys, tokens
    except Exception:
        return [], [], 0


def parse(q: Question, debug: bool = False) -> Query:
    text = " ".join(q.text.split())
    for pat in PATTERNS:
        m = pat.search(text)
        if m:
            x, y = m.group("x").strip(), m.group("y").strip()
            if not _looks_bad(x) and not _looks_bad(y):
                return Query(x, y, raw=text, parse_method="regex")
            break
    xs, ys, tokens = _llm_parse(text)
    if debug:
        print(f"  [parse llm] tokens={tokens} X={xs} Y={ys}", file=sys.stderr)
    if xs and ys:
        return Query(None, None, raw=text, parse_method="llm", tokens=tokens,
                     interventions=xs, outcomes=ys)
    return Query(None, None, raw=text, parse_method="failed", tokens=tokens)
