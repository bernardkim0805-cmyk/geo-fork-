"""Stage 1: question text -> Query(intervention, outcome).

STUB: regex for the hidden-set shape "What effect does X have on Y?".
Later: LLM fallback for phrasings the regex misses.
"""
from __future__ import annotations

import re

from .schema import Query, Question

PATTERNS = [
    re.compile(r"what (?:effect|impact|influence)s? (?:does|do|did) (?P<x>.+?) have on (?P<y>.+?)\??$", re.I),
    re.compile(r"how (?:does|do) (?P<x>.+?) (?:affect|influence|change|impact) (?P<y>.+?)\??$", re.I),
    re.compile(r"(?:does|do) (?P<x>.+?) (?:affect|increase|decrease|reduce|change) (?P<y>.+?)\??$", re.I),
]


def parse(q: Question) -> Query:
    text = " ".join(q.text.split())
    for pat in PATTERNS:
        m = pat.search(text)
        if m:
            return Query(m.group("x").strip(), m.group("y").strip(), raw=text, parse_method="regex")
    return Query(None, None, raw=text, parse_method="failed")
