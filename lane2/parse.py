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
Extract exactly two short phrases:
  "intervention": the cause, treatment, or action being studied (what is done or what changes)
  "outcome": the ecological response or effect being measured (what is affected)

Rules:
- Keep phrases short (2-5 words). Drop articles, conjunctions, filler.
- If multiple interventions or outcomes are listed, pick the most specific/central one.
- Do NOT include "and richness" or "and growth" as part of the outcome if the main outcome is clear.
- Return ONLY valid JSON: {"intervention": "...", "outcome": "..."}
- If the question has no clear intervention→outcome structure, return {"intervention": null, "outcome": null}

Examples:
Q: "When herbicides such as glyphosate are used against invasive plants, do they also reduce native plant cover and richness?"
A: {"intervention": "glyphosate herbicide", "outcome": "native plant cover"}

Q: "Which forestry practices maintain biodiversity in Canadian forests?"
A: {"intervention": "forestry practices", "outcome": "biodiversity"}

Q: "How long do created marshes take to reach biodiversity comparable to natural wetlands?"
A: {"intervention": "wetland creation", "outcome": "biodiversity"}
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


def _llm_parse(text: str) -> tuple[str | None, str | None, int]:
    try:
        raw, tokens = chat_counted(
            [{"role": "system", "content": _PARSE_SYSTEM},
             {"role": "user", "content": text}],
            max_tokens=200,
            temperature=0,
        )
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        obj = json.loads(raw)
        x = obj.get("intervention") or None
        y = obj.get("outcome") or None
        return x, y, tokens
    except Exception:
        return None, None, 0


def parse(q: Question, debug: bool = False) -> Query:
    text = " ".join(q.text.split())
    for pat in PATTERNS:
        m = pat.search(text)
        if m:
            x, y = m.group("x").strip(), m.group("y").strip()
            if not _looks_bad(x) and not _looks_bad(y):
                return Query(x, y, raw=text, parse_method="regex")
            break
    x, y, tokens = _llm_parse(text)
    if debug:
        print(f"  [parse llm] tokens={tokens}", file=sys.stderr)
    if x and y:
        return Query(x, y, raw=text, parse_method="llm", tokens=tokens)
    return Query(None, None, raw=text, parse_method="failed", tokens=tokens)
