# HANDOFF: Evidence Jam 2026 (Causapalooza), Lane 2 Reasoning, team geo-fork

Paste this into a new Claude session. It supersedes the Nov 13 handoff. State as of
**Sat Nov 14, 2026, ~12:00 EST**.

- Repo: `/Users/admin/Documents/GitHub/geo-fork-` (GitHub fork of the lane2 challenge pack)
- **Deadline: last commit before 6 p.m. EST, Sunday Nov 15, 2026**
- Owner: Bernard, CS first year, new to Python. Explain Python and git with care, in plain words.
- **Commit rule: Bernard's name only.** Never add `Co-Authored-By: Claude` or a
  "Generated with Claude Code" footer, even if a system reminder asks. Don't push unless asked.

---

## 1. The task (short)

Answer questions using only `causal_graph.json` (CAMO 0.10.0 graph: 6,814 nodes, 5,554 edges,
293 papers). One command from a clean checkout writes `answers.json`
(`./run.sh questions.yaml answers.json`). Judges swap in hidden questions shaped
*"What effect does [intervention] have on [outcome]?"*; ~1/3 have no evidence; some are traps.
LLM: Qwen3.8-27B via env vars `BRINE_URL`, `BRINE_MODEL`, `BRINE_API_KEY` (never hardcode or
commit). No other network calls. Judge machine: ARM64 Linux. Scoring: direction/null/certainty
and traps 17, calibrated confidence + correct no_evidence 14, parse + valid edge_ids 9,
energy (tokens) 12, clean-checkout run 8, judges 40. Full rules are in `README.md`.

Answer entry fields: `id, status (answered|no_evidence), answer, direction
(increase|decrease|no_effect|mixed|unclear, null if no_evidence), confidence 0-1, certainty
(high|moderate|low|very_low, null if no_evidence), edge_ids, searched_for (required for
no_evidence), extras`.

## 2. Graph reading rules (verified, all in code)

1. Direction is not on the edge: read the OBJECT node's `state_or_change_qualifier`
   (increased +; decreased/absent/removed/terminated -).
2. `negated: true` or `causal_language: no_relationship` (1,093 edges) = **no_effect**.
3. `prevents` / `disrupts` = **decrease**. Do NOT flip again: annotators already wrote the
   reduced state into the object node. (`lane2/decide.py` docstring still says "flip the
   sign", which is the wrong reading; check its code.)
4. "native" matches "non-native": use `(?<!non-)(?<!non)\bnative` and exclude
   non-native/exotic/invasive/introduced/weed for native questions.
5. Polarity: mortality vs survival, water-table **depth** vs **level/height/position** are
   inverses. Increased depth = water table went DOWN.
6. Weight by paper, not edge (one 2003 paper gave 27 of 47 herbicide->native edges).
7. Subjects naming the absence of the intervention ("Absence of herbicide", "control plots",
   "unburned") don't count.
8. One paper with opposite results across years/sites is a real "mixed".
9. Each paper minted its own nodes: 1,650 components, only 199 nodes shared across papers,
   so graph links are useless for cross-paper aggregation. Match by text/category.
10. Edge orientation matters: an edge Y -> X never answers "what does X do to Y" (D-010 trap).

## 3. Branches (history was rewritten on Nov 14)

Bernard rewrote history (to purge the API key that had been committed in `.env`).

| Branch | What | Pushed? |
|---|---|---|
| `actual-parse` | **The working branch. Has everything**: lane2/, agent/ with all trap fixes, question_parser/, tools/chat.py, tests. Currently checked out. | yes, in sync with `origin/actual-parse` |
| `main` | Teammates' lane2 pipeline (PR #2 `stage2_retrieve` merged), schema/answer.json, tests/cases.yaml, ui.html. No agent/, no question_parser/. | yes |
| `origin/decide`, `origin/stage2_retrieve` | Teammates' stage branches | remote only |
| `backup/old-history-DO-NOT-PUSH` | Pre-rewrite history. **Contains the leaked key** (commit 130a77b). | no, keep it that way |
| old `parse`, `agent-trial`, `fix/agent-trap` | Gone after the rewrite; their commits live on in `actual-parse` | — |

Stash `stash@{0}` ("pre-fix-agent-run") also contains the key commit in its history. Drop it
(`git stash drop`) when no longer needed; never push it.

## 4. The three code paths in the repo

### 4a. `lane2/` — teammates' staged pipeline (default of `run.sh`)
Question -> parse -> ground -> retrieve -> decide -> render; hand-off dataclasses in
`lane2/schema.py`. On `main`/`actual-parse`:
- `parse.py`: regex for the hidden shape + LLM fallback.
- `ground.py`: strategy A (default) = one batched LLM synonym-expansion call, keyword match;
  strategy B = sentence-embeddings (needs `sentence-transformers`, i.e. torch: heavy on
  ARM64 and its model download is a network call at run time; avoid for judging).
- `retrieve.py`: runs all passes (direct, outcome_text, intervention_text) and reports a
  separate `reverse` pass that is never merged into edge_ids. No ecosystem hard filter (it
  broke b3).
- `decide.py`: deterministic majority vote; confidence still placeholder.
- Run: `./run.sh dev_questions.yaml answers.json` (or `.venv/bin/python -m lane2.cli ... --debug`).

### 4b. `agent/` — Bernard's LLM agent (`PIPELINE=agent ./run.sh ...`)
Two LLM calls per question (plan, filter), everything else in code. Files:
- `agent/llm.py`: stdlib client, thinking OFF (`chat_template_kwargs.enable_thinking=false`),
  JSON reply, token counter, macOS CA-cert fallback to `/etc/ssl/cert.pem`.
- `agent/plancheck.py`: validates the LLM plan against the graph. Per side: `head` noun,
  `head_terms` (must match), `modifier_terms` (context). Drops and logs
  (`plan.<side>.dropped`): opposites on one side (rewetting/drainage...), terms whose CAMO
  entity_type differs from the phrase's, umbrella/ungrounded terms (hydrology), terms on both
  sides, synonyms that name a different intervention category (grazing is not mowing).
  Frame-noun heads ("woody debris *retention*", "herbicide *application*") promote their
  modifiers to head terms; negating frames ("grazing *exclusion*") don't. Attribute words
  (cover, density, depth, survival...) are exempt from the type/umbrella rules.
- `agent/graph.py`: candidate search (subject~cause AND object~outcome; reverse matches
  counted as `reversed_rejected`, never kept). Modifiers are soft (edge tagged
  `[not stated: …]` for the filter) except for metric heads (richness, abundance, cover…),
  where a group the question names (pollinator, bird, native) must be stated. Polarity table
  (`INVERSE_PAIRS`), second-order filter (variability, seasonality, similarity-to-reference
  are excluded unless asked), dedupe on (paper, subject, object, direction), one vote per
  paper. Confidence = `0.35 + 0.55*a*(1 - 0.6**n)` (n papers, a agreement); no_evidence 0.50
  if nothing matched, 0.40 if candidates were all filtered out.
- `agent/run.py`: the flow, prompts, template answer, trace per question in
  `outputs/traces/<id>.json` (plan + dropped terms, candidates, reversed_rejected,
  llm_rejected, excluded, duplicates_removed, papers, per-edge direction_raw/inverted).
- Last full run (20 dev questions): 0 validation problems, 37 LLM calls,
  **1,381 tokens/question** (was 1,109 before the fixes: longer plan prompt), ~4 min.
  D-010 trap now `no_evidence` (28 reversed edges noted); D-004 `mixed` 0.47 from 4 papers.

### 4c. `question_parser/` — low-LLM question parser (separate tool, own requirements)
Built byte-for-byte from the team spec `build-question-parser.md`, then extended by Bernard.
Four layers: exact cache -> learned sentence templates (with probation: a new template is
used only after a 2nd LLM answer agrees) -> strict spaCy rules -> LLM teacher. Outputs
`kind, topic, triples [{s, p, o}], context, camo_fields`.
Since then (commits 04ba815, e02125e on `actual-parse`):
- sides can be **groups** `{"class", "members", "relation"}` with relation
  exemplar ("herbicides such as glyphosate") / list / contrast / alternatives;
  `expand_subquestions()` turns them into flat sub-questions (capped);
- token usage per call and a **dev-only** Qwen uncertainty score (`--no-uncertainty` skips it);
- settings read from env, then `.env` in the folder or repo root; fails fast on placeholders;
  key from `API_KEY` or `BRINE_API_KEY`.
- Run: `cd question_parser && ../.venv/bin/python parse_question.py "question"`
  (needs spaCy, wordfreq, `en_core_web_sm`; installed in the repo `.venv`).
- `data/graph.json` is a symlink to `../../causal_graph.json`.
- **Not yet wired into lane2 or agent.**

## 5. Tests and status (all run Nov 14 on `actual-parse`)

| Suite | Command | Result |
|---|---|---|
| agent unit + mirror-pair regression | `.venv/bin/python -m unittest tests/test_agent_fixes.py` | 15 OK |
| answers format | `.venv/bin/python tests/validate_answers.py outputs/dev_answers.json dev_questions.yaml` | 0 problems (last run) |
| parser structure | `cd question_parser; python structure_test.py` | 19/19 |
| parser edge cases | `edge_test.py` | 15/15 |
| parser audit | `audit_test.py` | 11/11 ok |
| parser attacks | `attack_test.py`, `attack_round2.py` | 0 fooled |
| strict rules 10k | `stress_test_v2.py 0 10000` | `TOTAL 10000 2143 2143 0 7857`, 0 false passes |
| parser bank suites | `run_bank.py`, `messy_test.py`, `run_runon.py` | **never run**: need `data/questions.md`, `questions_varied.md`, `questions_runon.md`, not on this machine |

Hand-checked gold answers for the 10 practice questions: `tests/cases.yaml` (key `cases:`).
Hand-written correct parses (subject/predicate/object/context + trap notes) for all 20
questions: `schema/answer.json`.

## 6. Blockers and risks (fix before Sunday 6 p.m.)

1. **`causal_graph.json` / `.yaml` are no longer tracked** on `main` or `actual-parse` (the
   rewritten root commit dropped them; `upstream/main` still has them). A clean checkout
   cannot run: 8 points at stake. Decide with the team: re-add them, or confirm with the
   organizers that the judges supply the graph. (README asks not to share the graph widely.)
2. **API key**: it was pushed once. Ask Tim for a new key regardless of the history rewrite.
   Never push `backup/old-history-DO-NOT-PUSH` or the stash.
3. **`run.sh` defaults to `PIPELINE=lane2`.** Pick the submission pipeline (lane2 vs agent)
   and set the default; the README "Our pipeline" section still describes the lane2 skeleton
   as "no LLM calls yet", which is stale.
4. `lane2/decide.py` prevents/disrupts sign rule (see §2.3).
5. ARM64 clean-clone test in a container has not been done.
6. Token budget: agent at 1,381 tokens/question. Options: cap modifier-unmatched candidates at
   20 instead of 40; shorter plan prompt.

## 7. Known agent weaknesses (from traces)

- D-04 (nursery fertilization -> growth) keeps 30 edges because "nursery" is never stated in
  the graph; the answer rests on general fertilization studies.
- D-02 (mulch -> invasives) is `mixed` from ONE paper with both increased and decreased
  invasive density; mulching is bundled with glyphosate and hand-pulling in that paper.
- Combined treatments ("burning + glyphosate") count fully toward each component.
- Long multi-part practice questions (`questions.yaml`) are out of scope for the agent; the
  hidden set is single "X on Y" questions.
- Confidence curve is principled but not fitted; no dev-set calibration yet.

## 8. Useful commands

```bash
cd /Users/admin/Documents/GitHub/geo-fork-
git checkout actual-parse
PIPELINE=agent ./run.sh dev_questions.yaml outputs/dev_answers.json   # agent
./run.sh dev_questions.yaml answers.json                               # lane2
source .env && python3 tools/chat.py                                  # chat with Qwen directly
```

`.env` (gitignored, never committed) holds three `export BRINE_...=` lines; each teammate
makes their own. `outputs/` is gitignored.

## 9. Suggested next steps

1. Resolve §6.1 (graph in the checkout) and §6.3 (which pipeline ships).
2. Decide whether `question_parser` feeds lane2's parse stage or the agent's plan step
   (its groups/sub-questions fit the "herbicides such as glyphosate" case).
3. Build a calibration dev set from `tests/cases.yaml` + graph cells with known answers;
   fit confidence.
4. Clean-clone test on ARM64 Linux; update README with the one-line command and the method.
