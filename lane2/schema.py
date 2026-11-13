"""Hand-off contracts between pipeline stages.

Each stage takes the previous stage's dataclass and returns its own. Agree on
these and teammates can swap out one stage without touching the others.

    Question --parse--> Query --ground--> Grounding --retrieve--> Evidence
             --decide--> Decision --render--> Answer (dict for answers.json)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Question:
    id: str
    text: str


@dataclass
class Query:
    """What the question asks about, as plain phrases (not yet node ids)."""
    intervention: Optional[str]  # cause side, e.g. "mulch & surface cover"
    outcome: Optional[str]       # effect side, e.g. "invasive / non-native species abundance"
    raw: str = ""
    parse_method: str = ""       # "regex" | "llm" | "failed": useful for debugging


@dataclass
class Grounding:
    """Which graph nodes each side of the query maps to."""
    intervention_nodes: list[str] = field(default_factory=list)
    outcome_nodes: list[str] = field(default_factory=list)
    intervention_terms: list[str] = field(default_factory=list)  # what we searched for
    outcome_terms: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    """Edges that connect the grounded intervention nodes to outcome nodes."""
    edge_ids: list[str] = field(default_factory=list)


@dataclass
class Decision:
    status: str                          # "answered" | "no_evidence"
    direction: Optional[str] = None      # increase|decrease|no_effect|mixed|unclear
    certainty: Optional[str] = None      # high|moderate|low|very_low
    confidence: float = 0.5
    edge_ids: list[str] = field(default_factory=list)
    searched_for: Optional[str] = None
    tally: dict = field(default_factory=dict)  # per-direction paper counts, for rendering
    notes: list[str] = field(default_factory=list)  # plain-English caveats for render
    votes: list = field(default_factory=list)       # per-edge EdgeVote trace (debug / extras)
