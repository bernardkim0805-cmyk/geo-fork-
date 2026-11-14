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
    kind: str = ""   # intervention_outcomes | species_selection | outcome_trajectories
                     # | prioritization | graph  -- only the first is plain cause->effect


@dataclass
class Query:
    """What the question asks about, as plain phrases (not yet node ids).

    A question may name several causes or several effects -- b3 asks about
    native plant cover AND richness, c4 about five interventions. `interventions`
    and `outcomes` hold all of them; `intervention` and `outcome` stay as the
    first of each so the render and decide stages keep working unchanged.
    """
    intervention: Optional[str]  # cause side, e.g. "mulch & surface cover"
    outcome: Optional[str]       # effect side, e.g. "invasive / non-native species abundance"
    raw: str = ""
    parse_method: str = ""       # "regex" | "llm" | "failed": useful for debugging
    tokens: int = 0              # LLM tokens spent parsing (0 if regex hit)
    interventions: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    # one (cause, effect) per sub-question. Each is searched and judged on its
    # own, because a finding can hold for glyphosate and not for herbicides
    # generally, and one merged search cannot tell the two apart.
    pairs: list[tuple[str, str]] = field(default_factory=list)
    # effect | interaction | gap_map | trajectory. Decided before extraction,
    # so a question about the evidence itself is not forced into cause/effect.
    mode: str = "effect"
    # outcome phrase -> the direction the question takes for granted, e.g. b3
    # asks whether herbicides "also reduce" native cover. Recording it lets the
    # answer disagree with the premise instead of quietly answering a different
    # question. Only question_parser fills this in.
    presumed: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # keep the single and plural forms consistent whichever one was set
        if not self.interventions and self.intervention:
            self.interventions = [self.intervention]
        elif self.interventions and not self.intervention:
            self.intervention = self.interventions[0]
        if not self.outcomes and self.outcome:
            self.outcomes = [self.outcome]
        elif self.outcomes and not self.outcome:
            self.outcome = self.outcomes[0]
        if not self.pairs and self.intervention and self.outcome:
            self.pairs = [(self.intervention, self.outcome)]


@dataclass
class Grounding:
    """Which graph nodes each side of the query maps to."""
    intervention_nodes: list[str] = field(default_factory=list)
    outcome_nodes: list[str] = field(default_factory=list)
    intervention_terms: list[str] = field(default_factory=list)  # what we searched for
    outcome_terms: list[str] = field(default_factory=list)
    tokens: int = 0              # LLM tokens spent on synonym expansion (0 for strategy B)
    contexts: list[str] = field(default_factory=list)  # ecosystem_context values the
                                 # question is restricted to; empty means no restriction
    stats: dict = field(default_factory=dict)  # how many nodes each filter dropped
    # outcome phrase -> the node ids it grounded to. A question asking about
    # cover AND richness is two separate X->Y questions sharing one search, and
    # merging them hides which evidence answers which half.
    outcome_groups: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Evidence:
    """Edges that connect the grounded intervention nodes to outcome nodes.

    `edge_ids` is the answer: every X->Y edge any pass found, deduplicated. The
    rest is provenance the decide stage can weigh — a direct edge is stronger
    evidence than one found only by keyword-matching an endpoint.
    """
    edge_ids: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    # pass name -> edge ids it found; an edge can appear under several
    by_pass: dict[str, list[str]] = field(default_factory=dict)
    # edge id -> pass names that found it, strongest first
    found_by: dict[str, list[str]] = field(default_factory=dict)
    # subset of edge_ids whose ecosystem_context is outside the question's.
    # Still in edge_ids: filtering them out loses real counter-evidence.
    out_of_context: list[str] = field(default_factory=list)
    # claims pointing Y->X. NOT in edge_ids — citing one would invert direction.
    reverse_edge_ids: list[str] = field(default_factory=list)
    # edge id -> rank score. edge_ids is sorted by this, best first, so taking
    # the first N is a principled cut rather than an arbitrary one.
    scores: dict[str, float] = field(default_factory=dict)
    # What retrieve matched the phrases to. Grounding stopped being a pipeline
    # stage and became retrieve's first step, but decide still needs the terms
    # and node counts to explain what was searched.
    grounding: "Grounding" = field(default_factory=lambda: Grounding())


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
    # how direction/certainty/confidence were reached, step by step (trace / UI only)
    explain: dict = field(default_factory=dict)
