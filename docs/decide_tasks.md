# Decide stage: task list

Scope: `lane2/decide.py` only. Input = `Query` + `Evidence(edge_ids)` from
retrieve (contract **locked**: edge IDs only). Output = `Decision`. Ground,
retrieve and render are out of scope.

Working rules (agreed 2026-11-14):
- One commit per task on the `decide` branch. Each commit = code + test +
  matching update to `docs/decide.md`. **No PRs yet.**
- Calibrate confidence **last**, after the rules stop changing.
- Every tunable number stays in `CONFIG`.

---

## A. Test harness (do first)

- [ ] **A1. Case file format.** Create `tests/decide_cases.yaml`, separate from
  `tests/cases.yaml` (which stays as the broad end-to-end practice set). Fields per case:
  `id`, `question`, `intervention`, `outcome`, `edge_ids` (the input),
  `expected: {status, direction, certainty, cited_edges}`, `trap` (type or null),
  `labeled_by`, `checked_by`, `notes`.
- [ ] **A2. Draft ~30 single-pair cases** in hidden-set shape
  ("What effect does X have on Y?"):
  ~10 `no_evidence` (empty or only-excluded edge lists), ~8 traps, ~12 normal.
  Traps must cover: negated/null results, "absence of X" subjects, a question that
  is itself about a removal, head-to-head comparators, reversed cause/effect,
  `unchanged` objects (2–3, to verify R3), "variation in Y", interaction
  subjects, unsigned objects, a single paper that disagrees with itself.
  Pull some pairs from the edge lists of the existing practice cases.
  Claude labels from `original_sentence` + schema descriptions.
- [ ] **A3. Independent check by 3 subagents.** Each reads the edges and sentences
  cold (no access to Claude's labels or reasoning) and labels
  status/direction/certainty. Record agreement per case.
- [ ] **A4. Final review by the user**, starting with the cases where the
  subagents disagreed. Only cases the user signs off on count.
- [ ] **A5. `tests/eval_decide.py`.** Feed each case's `edge_ids` straight into
  `decide()`. Pass rules: status exact; direction exact; certainty within ±1
  level; cited edges ⊆ expected edges. Print a per-case diff plus each edge's
  vote trace (`EdgeVote.reason`). Also report strict accuracy
  (status + direction + exact certainty), which is the calibration target (see E).

## B. Edge-pattern handling (`read_edge`)

- [ ] **B1. Unsigned objects** (508 non-null edges: qualifier None / `present` /
  `occurred` / `ongoing`). Fall back to sign words in the node name
  (higher/greater/more/increase → +; lower/reduced/fewer/loss/decline → −).
  If none: the edge **sits out** (cited as context, no vote). `unclear` only when
  no edge has a usable sign.
- [ ] **B2. Exclude "variation in / variability of Y" objects** (~115). Same handling
  as R1: no vote, not cited, counted in notes.
- [ ] **B3. Exclude "interaction of X and …" subjects** (~46). Same handling as R1.
- [ ] **B4. R3 stays: `unchanged` object → `no_effect`.** Verify against the 2–3
  A2 cases. Note: none of the 121 are `negated`, and some labels are noisy
  (e.g. node "Changed Shannon diversity" with qualifier `unchanged`).
- [ ] **B5. Subject qualifier `unchanged`** (166 edges, mostly "water table
  position"): currently treated as +1 without any decision. Pick a rule
  (likely: sit out) and add a test.

## C. Fields decide ignores

- [ ] **C1. `evidential_basis.evidence_types` → vote weight.** Randomized /
  quasi-experiment weigh more than observational / cross-sectional / model.
  Numbers go in `CONFIG`.
- [ ] **C2. `evidential_basis` → certainty.** If every voting paper is
  observational, certainty drops one level (like the all-associational rule).
  Watch for double-counting with `certainty_grade`.
- [ ] **C3. Look at `strength`.** Present on every edge; check whether its
  qualifier carries information beyond `certainty_grade`. Use it or write down
  why not.

## D. Rule decisions

- [ ] **D1. One paper that disagrees with itself → `mixed`** with a single-study
  penalty: lower confidence and certainty, plus a note
  ("one study; results differ by …").
- [ ] **D2. "Does the question ask about a removal?" → LLM call at runtime.**
  Replace the `removal_words` regex with a small Qwen call (`lane2/llm.py`).
  Run it only when at least one voting edge has a negative subject qualifier.
  Cache it per intervention phrase. Fall back to a **tightened** regex when the
  LLM is unavailable (no `control`/`cut`/`thin`/`no `). Trade-offs to track:
  energy per question is scored (12 pts), and decide stops being fully
  deterministic. Log the call count in the eval output.
- [ ] **D3. Zero edges from retrieve → trust it.** `no_evidence` with a single
  calibrated confidence. `searched_for` is built from `Query` only (contract
  is locked). Edges that exist but are all excluded keep their own note.

## E. Confidence calibration (last)

- [ ] **E1. Definition of correct:** status + direction + certainty
  (reference = an ecologist's answer, which our checked cases stand in for).
- [ ] **E2. Keep the formula; tune its constants** (`conf_base`,
  `conf_per_paper`, `conf_mixed`, `conf_unclear`, `head_to_head_penalty`,
  single-study penalty, no_evidence value) against A5's strict accuracy.
- [ ] **E3. Guard against overfitting:** pull values toward 0.5 when few
  cases support them; clamp to about [0.2, 0.9].
- [ ] **E4. Report calibration** in `eval_decide.py`: mean confidence vs. actual
  accuracy per band, plus a Brier score.

## F. Stability

- [ ] **F1. Bad input never throws.** If an edge_id isn't in the graph, a node is
  missing, or the comparator is empty/None: skip that edge, add a note, lower
  confidence slightly. If nothing usable is left → `no_evidence`.
- [ ] **F2. Dedupe `edge_ids`** before voting.
- [ ] **F3. Tests** for each F1/F2 case in `tests/test_decide.py`.

---

## Suggested order

A1 → A2 → A3 → A4 → A5 → F1–F3 → B1–B5 → C1–C3 → D1 → D3 → D2 → E1–E4

(Stability comes early because it's cheap and keeps the eval from crashing.
D2 comes after the other rules so its energy cost can be measured against a
working baseline.)
