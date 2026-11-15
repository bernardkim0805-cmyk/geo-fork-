"""LLM retrieval by meaning, top-down: papers first, then their edges.

The keyword passes in retrieve.py judge one node name at a time, so they are
blind to everything around it. "PCB concentrations in deer mice" never matches
"wildlife", and a paper titled "Effects of oil sands tailings compounds ... in
nestling tree swallows" contributes nothing when its nodes are named "Decreased
fledging success". The model can see that context; a regex cannot.

Reading the whole graph is not affordable (~108k tokens as bare subject->object
lines), so this narrows in two calls:

  1. pick_papers   question + every paper title      -> relevant papers
  2. pick_edges    those papers' edges, one per line -> edges per cause/effect

Nothing here raises. A failed or unparseable call returns an empty result and
the caller keeps whatever the keyword passes found.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from .graph import Graph
from .llm import chat_counted

# Edges per pick_edges call. Above this the list is split, so one over-generous
# paper selection cannot overflow the context window.
_EDGE_BATCH = 120
_NAME_MAX = 90

_PAPER_SYSTEM = """\
You screen papers for an evidence review in ecology and restoration.

You get a QUESTION, the CAUSES and EFFECTS it asks about, and a numbered list of
paper titles. Return the numbers of every paper that plausibly reports evidence
on any of the causes (or a specific instance of one) affecting any of the
effects (or a specific instance of one).

Match on meaning, not wording: a named species is an instance of "wildlife",
liming or source removal is a remediation action, low pH is acidification.
Include a paper when unsure; a later step reads its findings and discards it.
Exclude papers that share only a place or a word with the question.

Return ONLY a JSON array of integers, e.g. [3, 17, 42]. Return [] if none.
"""

_EDGE_SYSTEM = """\
You select findings for an evidence review in ecology and restoration.

You get a QUESTION, numbered CAUSES, numbered EFFECTS, and numbered findings
grouped by paper. Each finding reads "cause -> effect"; "(NOT)" marks a tested
relationship that was not found, which is still evidence.

For each cause/effect combination, list the findings whose left side is that
cause (or a specific instance, level or proxy of it) and whose right side is
that effect (or a specific instance or measure of it). Match on meaning. Use
the paper title to interpret short or vague finding names.

Be strict about the left side. A relevant paper also reports other drivers --
date, year, age, weather, predators, fish, model settings -- and those findings
are NOT evidence for the cause; skip them. Skip a finding whose right side is
not an effect on the things asked about (for example water or soil chemistry
when the question asks about animals).

Return ONLY lines of the form "cause#/effect#: numbers", using ranges for runs,
with no other text. Omit combinations with no findings. Example:
1/2: 4, 9-14, 20
3/1: 12
"""


def _json(raw: str):
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    m = re.search(r"[\[{].*[\]}]", raw, re.S)
    return json.loads(m.group(0) if m else raw)


def _parse_picks(raw: str) -> list[tuple[tuple[int, int], list[int]]]:
    """'1/2: 4, 9-14' lines -> ((cause index, effect index), finding numbers).

    Lines rather than JSON: the model pretty-printed a JSON list one number per
    line, which tripled the output tokens and truncated mid-array.
    """
    out = []
    for line in raw.splitlines():
        m = re.match(r"\W*(\d+)\s*/\s*(\d+)\W*:(.*)$", line.strip())
        if not m:
            continue
        nums: list[int] = []
        for a, b in re.findall(r"(\d+)(?:\s*-\s*(\d+))?", m.group(3)):
            lo, hi = int(a), int(b or a)
            if hi - lo <= 500:
                nums.extend(range(lo, hi + 1))
        out.append(((int(m.group(1)) - 1, int(m.group(2)) - 1), nums))
    return out


def _edges_by_doc(graph: Graph) -> dict[str, list[str]]:
    cache = getattr(graph, "_edges_by_doc", None)
    if cache is None:
        cache = defaultdict(list)
        for eid, e in graph.edges.items():
            cache[e.get("source_document")].append(eid)
        graph._edges_by_doc = cache
    return cache


def _header(question: str, causes: list[str], effects: list[str]) -> str:
    c = "\n".join(f"{i}. {x}" for i, x in enumerate(causes, 1))
    e = "\n".join(f"{i}. {y}" for i, y in enumerate(effects, 1))
    return f"QUESTION: {question}\n\nCAUSES:\n{c}\n\nEFFECTS:\n{e}"


def pick_papers(question: str, causes: list[str], effects: list[str],
                graph: Graph) -> tuple[list[str], int]:
    """Document ids the model thinks could hold evidence, and tokens spent."""
    by_doc = _edges_by_doc(graph)
    docs = [d for d in graph.docs if by_doc.get(d)]
    listing = "\n".join(f"{i}. {(graph.docs[d].get('title') or d)[:160]}"
                        for i, d in enumerate(docs))
    try:
        raw, tokens = chat_counted(
            [{"role": "system", "content": _PAPER_SYSTEM},
             {"role": "user", "content":
                 f"{_header(question, causes, effects)}\n\nPAPERS:\n{listing}"}],
            max_tokens=600, temperature=0)
        picked = _json(raw)
        return [docs[i] for i in dict.fromkeys(picked)
                if isinstance(i, int) and 0 <= i < len(docs)], tokens
    except Exception:
        return [], 0


def _edge_line(graph: Graph, eid: str) -> str:
    e = graph.edges[eid]
    s = (graph.nodes.get(e["subject"], {}).get("name") or "")[:_NAME_MAX]
    o = (graph.nodes.get(e["object"], {}).get("name") or "")[:_NAME_MAX]
    return f"{s} -> {o}" + (" (NOT)" if e.get("negated") else "")


def pick_edges(question: str, causes: list[str], effects: list[str],
               doc_ids: list[str], graph: Graph
               ) -> tuple[dict[tuple[str, str], list[str]], int]:
    """(cause, effect) -> edge ids the model says bear on it, and tokens spent."""
    by_doc = _edges_by_doc(graph)
    # batches of whole papers, so a paper's title always sits above its edges
    batches, cur, n = [], [], 0
    for d in doc_ids:
        k = len(by_doc[d])
        if cur and n + k > _EDGE_BATCH:
            batches.append(cur)
            cur, n = [], 0
        cur.append(d)
        n += k
    if cur:
        batches.append(cur)

    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    total = 0
    for batch in batches:
        ids, lines = [], []
        for d in batch:
            lines.append(f"\nPAPER: {(graph.docs[d].get('title') or d)[:160]}")
            for eid in by_doc[d]:
                lines.append(f"{len(ids)}. {_edge_line(graph, eid)}")
                ids.append(eid)
        try:
            raw, tokens = chat_counted(
                [{"role": "system", "content": _EDGE_SYSTEM},
                 {"role": "user", "content":
                     f"{_header(question, causes, effects)}\n\nFINDINGS:"
                     + "\n".join(lines)}],
                max_tokens=500, temperature=0)
            total += tokens
        except Exception:
            continue
        for key, nums in _parse_picks(raw):
            ci, ei = key
            if not (0 <= ci < len(causes) and 0 <= ei < len(effects)):
                continue
            pair = (causes[ci], effects[ei])
            for i in nums:
                if 0 <= i < len(ids) and ids[i] not in out[pair]:
                    out[pair].append(ids[i])
    return dict(out), total


def llm_retrieve(question: str, causes: list[str], effects: list[str],
                 graph: Graph) -> dict:
    """Both steps. Returns papers, per-pair edges, the flat edge list and tokens."""
    papers, t1 = pick_papers(question, causes, effects, graph)
    by_pair, t2 = pick_edges(question, causes, effects, papers, graph) if papers else ({}, 0)
    edge_ids = list(dict.fromkeys(e for ids in by_pair.values() for e in ids))
    return {"papers": papers, "by_pair": by_pair, "edge_ids": edge_ids,
            "tokens": {"papers": t1, "edges": t2, "total": t1 + t2}}
