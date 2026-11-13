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
| R3. Object qualifier `unchanged` → `no_effect` | Team decision (2026-11-13). The schema treats it as "maintained", not as a null result. Revisit later |
| R4. Otherwise **direction = subject sign × predicate sign × object sign** | Schema's variable-level convention. Using only the object's sign flips edges |

Signs:

- **Predicate:** `causes` / `contributes_to` / `enables` = +1. `prevents` / `disrupts` = −1. `associated_with` / `correlated_with` = +1, but they get a lower weight (unsigned association).
- **Object:** `increased` / `introduced` / `initiated` = +1. `decreased` / `removed` / `absent` / `terminated` / `interrupted` / `aborted` = −1. `present` / `occurred` / `ongoing` / none = **unclear** (no up or down).
- **Subject:** −1 if the qualifier is `removed` / `absent` / `decreased` / `terminated` / `interrupted` / `aborted`, **unless the question itself asks about a removal** (its intervention phrase contains a word like removal, exclusion, reduction, thinning, control, suppression, absence). Otherwise +1.
  - "What effect does **fertilization** have on survival?" + `absence of fertilization` [absent] causes `increased survival` → **decrease**
  - "What effect does **browsing exclusion** have on height?" + `release from moose browsing` [removed] → `increased height` → **increase** (no flip: the removal is the question's own subject)

## 2. Weight each vote

`weight = comparator weight × language weight`

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

## 3. Combine: count papers, not edges

- Inside each `source_document`, the paper's weighted votes become a share over {increase, decrease, no_effect}. Each paper counts once, at the weight of its strongest edge, so one paper with 10 edges can't outvote four papers.
- Across papers: the top direction holding ≥ `MAJORITY` (0.66) of the total → that direction. Otherwise `mixed`. If no paper has a readable direction → `unclear`.
- Cite only the edges that voted (R1-excluded edges are never cited).
- If every retrieved edge was excluded by R1, the status is `no_evidence`, and `searched_for` says that only indirect edges were found.

## 4. Certainty (strength of the evidence)

Take the best `certainty_grade` in each paper, then the median across papers.
Drop one level for each of these (never below `very_low`):
the direction is `mixed`; every voting edge is associational; every paper is head-to-head / reference-only.
The graph contains no `high` grades.

## 5. Confidence (probability we are right): PLACEHOLDER until tests/cases.yaml exists

- answered: `base + per_paper × agreeing papers` (capped), scaled by the top share; `mixed` and `unclear` get flat low values; × `head_to_head_penalty` when every paper is head-to-head.
- no_evidence: higher when both sides grounded to nodes but no edge joins them (a real gap), lower when one side grounded to nothing (could be a grounding miss).

## 6. Notes for render

`Decision.notes` carries plain-English caveats (head-to-head comparisons, excluded edges, flipped subjects) so render can be honest without re-deriving anything.
