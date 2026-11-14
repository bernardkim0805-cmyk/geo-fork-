# Decide stage: spec

`lane2/decide.py` turns the retrieved edges into `direction`, `certainty`,
`confidence` and `status`. It is deterministic: the graph decides, not the LLM.
Every rule below has a test in `tests/test_decide.py`. Every number lives in
the `CONFIG` block at the top of `decide.py`.

Sources for the rules: the `description`s in `schema/causalmosaic.yaml`
(CausalEdge.negated, ClaimStrengthEnum, StateOrChangeQualifierEnum,
CausalPredicateEnum "variable level" sign composition).

## 1. Read each edge (one vote per edge)

| Rule | Why |
| --- | --- |
| R1. `predicate` is `mediates`, `moderates`, `regulates` or `precedes` → excluded, not cited | These don't state a direct X → Y effect |
| R2. `negated: true` or `causal_language: no_relationship` → `no_effect` | Canonical null result. The object's qualifier is the *tested* direction; ignore it |
| B2. Object name says "variation / variability / variance / heterogeneity in Y" → excluded, not cited | Says Y differed among treatments, not whether X raised or lowered it |
| B3. Subject is an interaction ("interaction of…", "X × Y", "X and year/time") → excluded, not cited | An "it depends" result, not X's own effect |
| R3. Object qualifier `unchanged` → `no_effect` | Team decision (2026-11-13), confirmed in review 2026-11-14 (DC-19, DC-20 pass). Labels are noisy (DC-21: grizzly movement), which aggregation absorbs |
| B5. Subject qualifier `unchanged` → abstain (cited, no vote) | X itself didn't vary, so the edge can't say what X does |
| R4. Otherwise **direction = subject sign × predicate sign × object sign** | Schema's variable-level convention. Using only the object's sign flips edges |

Signs:

- **Predicate:** `causes` / `contributes_to` / `enables` = +1. `prevents` / `disrupts` = −1. `associated_with` / `correlated_with` = +1, but they get a lower weight (unsigned association).
- **Object:** `increased` / `introduced` / `initiated` = +1. `decreased` / `removed` / `absent` / `terminated` / `interrupted` / `aborted` = −1. `present` / `occurred` / `ongoing` / none = no sign → **B1:** take the sign from words in the object's name (higher/greater/more/… = +1, lower/reduced/fewer/loss/… = −1). If the name has none, or both kinds, the edge **abstains**: it is cited as context but doesn't vote.
- **Subject:** −1 if the qualifier is `removed` / `absent` / `decreased` / `terminated` / `interrupted` / `aborted`, **unless the question itself asks about a removal** (its intervention phrase contains a word like removal, exclusion, reduction, thinning, control, suppression, absence). Otherwise +1.
  - "What effect does **fertilization** have on survival?" + `absence of fertilization` [absent] causes `increased survival` → **decrease**
  - "What effect does **browsing exclusion** have on height?" + `release from moose browsing` [removed] → `increased height` → **increase** (no flip: the removal is the question's own subject)

## 2. Weight each vote

`weight = comparator weight × language weight × study-design weight (C1) × strength weight (C3)`

| Comparator | Weight |
| --- | --- |
| `untreated_control`, `pre_treatment_baseline`, `spatial_or_temporal_control`, `passive_management_arm` | 1.0 |
| `alternative_treatment` whose description mentions a control (control, untreated, unfenced, unprepared, without, no …) | 1.0 (label fixed) |
| `alternative_treatment` (true head-to-head) | 0.5 |
| `reference_undisturbed` (restored vs pristine, not vs unrestored) | 0.5 |
| `not_reported`, `not_applicable`, missing | 0.75 |

| causal_language / predicate | Weight |
| --- | --- |
| `direct_causal` | 1.0 |
| `uncertain_causal` | 0.75 |
| `associational`, or predicate `associated_with` / `correlated_with` | 0.6 |
| `no_relationship` (null) | 1.0 |

| Study design (`evidential_basis.evidence_types`, C1) | Weight |
| --- | --- |
| `randomized_experiment` | 1.0 |
| `quasi_experiment`, `natural_experiment` | 0.9 |
| `observational_longitudinal` | 0.8 |
| `observational_cross_sectional`, `structural_equation_model`, `mechanistic_study`, missing | 0.7 |
| `modeling_simulation`, `case_study` | 0.5 |

| `strength.qualitative_descriptor` (C3) | Weight |
| --- | --- |
| `strong` | 1.0 |
| `moderate`, `not_specified`, missing | 0.9 |
| `weak` | 0.75 |
| `negligible` | 0.5 |

`strength` is used because it does not track `certainty_grade` (strong edges are
spread across moderate/low/very_low), so it adds information.

## 3. Combine: count papers, not edges

- Inside each `source_document`, the paper's weighted votes become a share over {increase, decrease, no_effect}. Each paper counts once, at the weight of its strongest edge, so one paper with 10 edges can't outvote four papers.
- Papers whose edges all abstain sit out (they don't dilute the vote, tally or certainty).
- Across papers: the top direction holding ≥ `MAJORITY` (0.66) of the total → that direction. Otherwise `mixed`. If no paper has a readable direction → `unclear`.
- Cite only the edges that voted (R1-excluded edges are never cited).
- If every retrieved edge was excluded (R1, B2, B3), the status is `no_evidence`, and `searched_for` says so.

## 4. Certainty (strength of the evidence)

Take the best `certainty_grade` in each paper, then the median across papers.
Drop one level for each of these (never below `very_low`):
the direction is `mixed`; every voting edge is associational; every paper is head-to-head / reference-only.
The graph contains no `high` grades.

C2 (drop a level when every paper is observational) is implemented but **off**
(`observational_certainty_drop`). Measured 2026-11-14: `certainty_grade` already
encodes design (randomized: 2176 moderate / 165 low; cross-sectional: 1365 low /
190 very_low; modelling and case studies: all very_low), so the extra drop
double-counts. Turning it on changed no eval_decide result.

## 5. Confidence (probability we are right): PLACEHOLDER until tests/cases.yaml exists

- answered: `base + per_paper × agreeing papers` (capped), scaled by the top share; `mixed` and `unclear` get flat low values; × `head_to_head_penalty` when every paper is head-to-head.
- no_evidence (D3): zero usable edges from retrieve → trust it. One value, `conf_no_evidence`. The Grounding split was dropped: ground is being folded into retrieve, and the retrieve → decide contract is edge ids only. `searched_for` is built from the Query ("edges whose subject (cause) is 'X' and whose object (effect) is 'Y', in that direction"), plus a note when edges existed but were all excluded or missing.

## 5b. One study that disagrees with itself (D1)

If only one paper votes and the result is `mixed`, confidence × `single_study_mixed_penalty`
(0.85) and a note gives the split ("One study, and its results disagree (2 no_effect,
2 increase edges) …"). Certainty is not dropped again: the `mixed` rule already did.

## 6. Notes for render

`Decision.notes` carries plain-English caveats (head-to-head comparisons, excluded edges, flipped subjects) so render can be honest without re-deriving anything.

## 7. Bad input (never throws)

- `edge_ids` are deduplicated (first occurrence kept).
- An id that is not in the graph, or whose subject/object node is missing, is skipped. Confidence × `bad_input_penalty` (0.9) and a note is added.
- If nothing usable is left → `no_evidence`, and `searched_for` mentions the missing ids.
- An empty, `None` or `[None]` comparator falls back to the default comparator weight.
