"""Stage 3 helper: is an edge about the question's X and Y, their inverse, or something else?

The sign rules in decide.py read qualifiers mechanically, so they are fooled by
nodes that mean the inverse of what the question names ("Time since harvest"
for harvesting, "failure of regeneration" for recovery) and by nodes about a
different thing that share a word ("non-native forb cover" for native cover).

One batched LLM call per decide() classifies every not-yet-seen node:
  cause  (subject vs X): SAME | OPPOSITE | UNRELATED | UNSURE
  effect (object  vs Y): UP | DOWN | NONE | UNRELATED | UNSURE
Answers are cached per (side, phrase, node name, qualifier) for the process, so
the combined pass after the per-pair passes costs nothing.

Never raises: no endpoint, an error or an unparseable line just leaves the item
unknown, and decide falls back to its rule-based reading.
"""
from __future__ import annotations

import re

CAUSE, EFFECT = "cause", "effect"
CAUSE_CODES = {"SAME", "OPPOSITE", "UNRELATED", "UNSURE"}
EFFECT_CODES = {"UP", "DOWN", "NONE", "UNRELATED", "UNSURE"}
CHUNK = 60                 # items per call; keeps a small model on task
TOKENS_PER_ITEM = 6        # "12 OPPOSITE\n" with slack

_CACHE: dict[tuple, str] = {}
STATS = {"calls": 0, "items": 0, "cache_hits": 0, "failures": 0}

SYSTEM = """\
Match nodes from ecology papers to a question. [brackets] = change the paper reports.

CAUSE items, vs the question's cause X:
SAME: more or presence of X. Narrower, broader or unspecified forms count
 ('glyphosate application' for 'herbicides'; 'harvest disturbance' for 'partial harvesting').
OPPOSITE: only less or absence of X ('Time since harvest [increased]' for
 'harvesting'; 'Reduced deer densities' for 'deer browsing'; 'distance from smelter
 [increased]' for 'contaminants').
UNRELATED: a different cause, including a different treatment ('clearcutting' for
 'partial harvesting'; 'replanting' for 'competing vegetation').

EFFECT items, vs the question's outcome Y. Say what happened to Y. Change words
in a name (Increased, Decreased, Higher, Lower, Impaired) repeat the bracket:
count the change once ('Decreased cover of Agrostis [decreased]' is DOWN for
'non-native plant cover'). Flip only when the node measures the inverse of Y:
UP or DOWN: 'failure of regeneration [decreased]' is UP for 'recovery'; 'seedling
 mortality [increased]' is DOWN for 'survival'; 'Impaired copepod composition
 [decreased]' is DOWN for 'aquatic communities'.
NONE: about Y, but no direction stated, or unchanged.
UNRELATED: a different quantity or group ('non-native forb cover' for 'native
 plant cover'; 'vole survival' for 'seedling survival'). Unspecified forms count
 as Y ('Richness' for 'native plant richness').

UNSURE: either kind, when you cannot tell.
Reply one line per item: <number> <ANSWER>. Nothing else."""


def _key(side: str, phrase: str, name: str, qual) -> tuple:
    return (side, (phrase or "").strip().lower(), (name or "").strip().lower(), qual or "")


def get(side: str, phrase: str, name: str, qual) -> str | None:
    """Cached verdict, or None if never classified (or the call failed)."""
    v = _CACHE.get(_key(side, phrase, name, qual))
    if v is not None:
        STATS["cache_hits"] += 1
    return v


def peek(side: str, phrase: str, name: str, qual) -> str | None:
    """Like get(), without counting a cache hit (used to choose between sub-questions)."""
    return _CACHE.get(_key(side, phrase, name, qual))


def _format(items: list[tuple]) -> str:
    """Group items under their question phrase, numbered 1..n."""
    lines, last = [], None
    for i, (side, phrase, name, qual) in enumerate(items, 1):
        if (side, phrase) != last:
            label = "CAUSE X" if side == CAUSE else "EFFECT Y"
            lines.append(f"{label} = '{phrase}':")
            last = (side, phrase)
        lines.append(f"{i} {name}" + (f" [{qual}]" if qual else ""))
    return "\n".join(lines)


def _parse(text: str, items: list[tuple]) -> dict[int, str]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    out = {}
    for m in re.finditer(r"^\s*(\d+)[.):]?\s+([A-Z]+)", text, re.M):
        i, code = int(m.group(1)), m.group(2)
        if 1 <= i <= len(items):
            ok = CAUSE_CODES if items[i - 1][0] == CAUSE else EFFECT_CODES
            if code in ok:
                out[i] = code
    return out


def prefetch(items) -> None:
    """Classify every uncached (side, phrase, name, qualifier) in as few calls as possible."""
    from . import llm
    todo, seen = [], set()
    for side, phrase, name, qual in items:
        k = _key(side, phrase, name, qual)
        if not phrase or not name or k in _CACHE or k in seen:
            continue
        seen.add(k)
        todo.append((side, phrase.strip(), name.strip(), qual or ""))
    if not todo or not llm.configured():
        return
    # causes first, then effects, each grouped by phrase: fewer header lines
    todo.sort(key=lambda t: (t[0] != CAUSE, t[1]))
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        try:
            STATS["calls"] += 1
            reply = llm.chat([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": _format(chunk)}],
                             temperature=0, max_tokens=TOKENS_PER_ITEM * len(chunk) + 20)
        except Exception:
            STATS["failures"] += 1
            continue
        got = _parse(reply, chunk)
        STATS["items"] += len(got)
        for i, code in got.items():
            _CACHE[_key(*chunk[i - 1])] = code


def clear() -> None:
    _CACHE.clear()
    for k in STATS:
        STATS[k] = 0
