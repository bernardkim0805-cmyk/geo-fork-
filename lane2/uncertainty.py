"""DEV ONLY (delete before the final run): ask the model how unsure it is that
the parse captured the question.

Ported from Shady's `question_parser/llm.py` on branch `actual-parse`
(commit 04ba815). The prompt and the 0-100 contract are his, unchanged. What
differs is the plumbing: that branch has its own LLM client and its own parse
format, so this version calls our `lane2.llm.chat_counted` and serialises a
`Query` instead of his `{"kind", "triples"}` dict.

The scorer is advisory. It never raises, never filters, and nothing downstream
reads it -- decide.py is untouched. It exists so a wrong parse is visible in the
UI instead of silently producing a confident wrong answer.

    score_uncertainty(question, query) -> (score 0-100 | None, reason, tokens)
"""
from __future__ import annotations

import json
import re

from .llm import chat_counted
from .schema import Query

# Shady's prompt, verbatim.
UNCERTAINTY_PROMPT = """You check whether a structured parse correctly captures an ecology question.
You get the question and the parse (JSON). Judge ONLY whether the parse captures what the question
asks: the right cause(s) as "s", the right effect(s) as "o", nothing missing, nothing invented,
nothing reversed. Do not judge whether the science is true.

Reply with ONE JSON object and nothing else:
  {"uncertainty": <integer 0-100>, "reason": "<one short sentence>"}

0 = certain the parse is right. 100 = certain it is wrong or useless. Use the full range."""


def _extract_json(text: str) -> dict:
    """The reply should be one JSON object; tolerate code fences or prose around
    it. Same tolerance as his `extract_json`."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def _as_parse(query: Query) -> dict:
    """Our Query in the shape the prompt describes: causes under "s", effects
    under "o". Sends every phrase, not just the first, or a question naming two
    outcomes would look correctly parsed while half of it was dropped."""
    return {"s": query.interventions, "o": query.outcomes}


# (question, parse json) -> (score, reason). Module-level like ground.py's
# caches: never pay twice for the same pair inside one run.
_cache: dict[tuple[str, str], tuple[int, str]] = {}


def score_uncertainty(question: str, query: Query) -> tuple[int | None, str, int, bool]:
    """Returns (score 0-100 or None, reason, tokens, cached). Never raises.

    None means no judgement is available -- the call failed or the reply was
    unusable. That is reported as-is rather than as a confident 0 or 100.
    """
    if query is None or query.parse_method == "failed" or not (
            query.interventions and query.outcomes):
        return 100, "no usable parse was produced", 0, False

    q = " ".join(str(question).split())
    parse = _as_parse(query)
    key = (q.lower(), json.dumps(parse, sort_keys=True, default=str))
    if key in _cache:
        score, reason = _cache[key]
        return score, reason, 0, True

    try:
        text, tokens = chat_counted(
            [{"role": "system", "content": UNCERTAINTY_PROMPT},
             {"role": "user", "content":
                 f"Question: {q}\nParse: {json.dumps(parse, ensure_ascii=False)}"}],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        # the call never landed, so it cost nothing
        return None, f"uncertainty check failed: {type(e).__name__}: {str(e)[:80]}", 0, False

    try:
        obj = _extract_json(text)
        score = max(0, min(100, int(round(float(obj["uncertainty"])))))
        reason = str(obj.get("reason", ""))[:300]
    except Exception:
        # the reply was paid for even though it is unusable; don't cache it
        return None, f"unusable reply: {text[:80]!r}", tokens, False

    _cache[key] = (score, reason)
    return score, reason, tokens, False


def band(score: int | None) -> str:
    """Coarse label for the UI. Cut points are arbitrary -- they are there to
    colour a badge, not to gate anything."""
    if score is None:
        return "unknown"
    if score <= 20:
        return "low"
    if score <= 50:
        return "moderate"
    if score <= 75:
        return "high"
    return "very_high"
