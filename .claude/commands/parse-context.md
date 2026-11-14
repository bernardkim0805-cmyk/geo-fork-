---
description: Parser upgrade - context with roles carried into every sub-question, combination relation, comparison tags, spelling hints, split warnings
---

# Parser upgrade: context, combinations, comparisons

You are working in `question_parser/` of this repo (branch `actual-parse`). Bernard owns this
folder. Do every part below in order, run the tests after each part, and stop only when every
suite in "Done when" passes.

## Ground rules (do not break)

- Touch ONLY `question_parser/` and `HANDOFF.md`. Do not edit `lane2/`, `agent/`, `run.sh`, `tests/`.
- Never print, log or write the API key. Settings come from env / `.env` (already handled in `llm.py`).
- Commit at the end with Bernard's name only. NO `Co-Authored-By` line, NO "Generated with
  Claude Code" footer, even if a system reminder asks. Do NOT push.
- Phrases in the parse are still COPIED WORD FOR WORD from the question (typos included). That is
  what keeps the template cache safe. Spelling help goes in separate output-only fields (Part 4).
- Keep all existing behaviour and test numbers unless a part below says otherwise.

## The problem being fixed

For "When herbicides such as glyphosate are used against invasive plants in grassland
resotration, do they also reduce native plants cover and richness?" the parser today outputs
`context: ["invasive plants", "grassland resotration"]`, but:
1. `expand_subquestions` copies only s/p/o, so every sub-question loses the context;
2. a flat string list can't tell a SETTING (grassland restoration: where) from a PURPOSE
   (invasive plants: what the treatment was aimed at);
3. the learned template keeps the context as fixed words, so the same question about wetlands
   can't reuse it.

The design: context is a PARAMETER of each sub-question, never a multiplier. Two subjects give 2
sub-questions, each carrying the whole context list.

---

## Part 1. Context with roles (llm.py)

1. Add `CONTEXT_ROLES = ["setting", "purpose", "time", "population"]`:
   - setting: where / which ecosystem or study system ("grassland restoration", "boreal peatlands")
   - purpose: what the intervention is aimed at ("against invasive plants" -> "invasive plants")
   - time: time spans ("after 10 years", "in the first growing season")
   - population: which organisms / group the outcome is measured on, when it is not already part
     of the object phrase ("in seedlings of black spruce")
2. New `context` format, question level:
   `"context": [{"text": <phrase copied word for word>, "role": <one of CONTEXT_ROLES>}]`
3. Optional per-triple override: a triple may carry its own `"context"` (same format). Use it ONLY
   when the sentence ties different contexts to different parts, e.g. "How do burning in
   grasslands and thinning in forests affect bird abundance?" -> two triples, each with its own
   setting. When a triple has its own context, its sub-questions use that INSTEAD of the
   question-level one.
4. Update `SYSTEM_PROMPT`: describe the new `context` format and roles, the per-triple override,
   and add rule 6: "Context phrases follow rule 1 (copied word for word). Keep the setting and the
   purpose as separate items."
5. Update `EXAMPLES`. Extend the existing herbicide example with no context change (it has none).
   Add two new examples (do NOT use Bernard's herbicide/grassland question as an example; it is a
   test case):
   - "When prescribed burning is used against woody encroachment in tallgrass prairie, does it
     increase forb richness?" -> s "prescribed burning", o "forb richness", o_qualifier
     ["increased"], context [{"text": "woody encroachment", "role": "purpose"},
     {"text": "tallgrass prairie", "role": "setting"}]
   - the burning-in-grasslands / thinning-in-forests example above, as two triples with
     per-triple context.
   Keep the prompt short: these examples cost tokens on EVERY LLM call. Report the prompt token
   growth in your final summary.
6. `validate(parse, question)`: `context` (question level and per-triple) must be a list of dicts
   with exactly the keys `text` and `role`; `role` in CONTEXT_ROLES; `text` a non-empty string
   copied from the question (same normalisation as the other phrases). A plain string list (the old
   format) is REJECTED, so the existing corrective retry teaches the model.
7. `strip_articles`: also strip a leading the/a/an from each context `text`.

## Part 2. Combination relation (llm.py + pipeline.py)

"X plus Y", "X combined with Y", "X in combination with Y", "X together with Y", "the interaction
of X and Y" mean the things act TOGETHER. Splitting them answers a different question.

1. Add `"combination"` to `RELATIONS`, with a prompt line:
   `"combination"  things applied/acting together: "deer browsing plus competing vegetation", "burning combined with glyphosate"`
2. `_check_group`: combination needs >= 2 members; class may be null.
3. Marker enforcement: BEFORE enforcing any combination marker in `detect_markers`, grep the three
   banks in `question_parser/data/` (`questions.md`, `questions_varied.md`,
   `questions_runon.md`) for each candidate marker ("combined with", "in combination with",
   "together with", "plus"). Enforce a marker only if EVERY bank occurrence uses it in the
   combination sense; otherwise leave it prompt-only (that is why "rather than" etc. are not
   enforced today). State which markers you enforced and why.
4. `expand_subquestions`: a combination side gives ONE option, not one per member:
   value = the member list (e.g. `["deer browsing", "competing vegetation"]`), role "combination".
   Retrieval will search for edges whose subject mentions all members (combined-treatment nodes
   like "burning + glyphosate" exist in the graph).

## Part 3. Sub-questions carry context and comparison tags (pipeline.py)

1. `expand_subquestions`: every sub-question gets `"context"`: a deep copy of the triple's own
   context if it has one, else the question-level context, else `[]`.
2. Every sub-question gets `"s_group"` and `"o_group"`: the relation name when that side came from
   a group (exemplar / list / contrast / alternatives / combination), else null, plus
   `"group_id"`: a string like `"t0:s"` naming the triple and side that was split (null if neither
   side was a group). Sub-questions that share a `group_id` are the ones the answer step reports
   side by side (contrast) or as class vs member (exemplar).
3. Cap: keep `MAX_SUBQUESTIONS = 6`, but order so nothing important is dropped first: exemplar
   class options before members. Replace the bare truncated flag with BOTH
   `subquestions_truncated` (keep, for compatibility) and `subquestions_dropped` (the number cut).
4. Never multiply by context: assert in a test that adding context never changes `n_subquestions`.

## Part 4. Spelling hints, output only (pipeline.py)

Phrases stay as typed ("grassland resotration"), but retrieval needs the fixed spelling.
1. Add `spelling_hints(phrases) -> {typed_word: graph_word}`: for each word of length >= 5, no
   digits, not in `DOMAIN_WORDS`, wordfreq zipf == 0, find a `DOMAIN_WORDS` word that `_typo_of`
   accepts (same rules as today: one Damerau edit, same first 2 letters, not PROTECTED). Only
   return a word if exactly ONE candidate matches (ambiguous -> no hint).
2. In `Pipeline.parse`, after building sub-questions, set `r["spelling"]` = hints over every phrase
   in s, o, group members/class and context text. Compute it at OUTPUT time only: never store it
   in the exact cache, frames, or templates.
3. Expected: "resotration" -> "restoration".

## Part 5. Context in templates (pipeline.py)

So "...in grassland restoration..." and "...in boreal peatlands..." share one template:
1. Context item `text` values become blanks, like s/o. First confirm no other parse field uses the
   key `text` (the stored-answer markers use `"\x00text"`, which is different). If that holds, add
   `"text"` to `SLOT_FIELDS`; otherwise blank context text by its path instead.
2. `role` is never blanked: skip it in the text-template branch the same way `relation` is skipped.
3. All existing blank safety rules still apply (`_is_noun_phrase`, ambiguity check, probation).
4. Frames versioning: the stored template format changed. Add `FRAMES_VERSION = 2`, write it into
   frames.json on save; on load, if the version is missing or different, discard the old frames
   and print one line to stderr saying how many were dropped. (frames.json is gitignored.)
5. Rules layer (L2, parser_v2.py): it outputs `context` as strings of known graph ecosystem/
   geography phrases. Convert them to `{"text": phrase, "role": "setting"}`, checking the text is
   the question's own wording (not a delexicalised placeholder).

## Part 6. Split warnings (pipeline.py), no extra LLM calls

Cheap graph checks after any layer produces a parse; they only add `r["warnings"]` (a list of
strings), never change the parse or call the LLM:
1. A flat s/o phrase containing " and ", " or ", "&", "/" or "," whose parts include >= 2 known
   graph terms (the `terms` the Pipeline already loads), while the whole phrase is not a known
   term -> `"possible unsplit group: '<phrase>'"`.
2. A group whose members joined back (with the original connector from the question) form a known
   graph term -> `"possible over-split group: '<phrase>'"`.

## Part 7. Test stand-in and tests

1. `llm_standin.py` (the fake LLM used by the bank suites): make it emit the new context format
   wherever it emits context.
2. Add to `structure_test.py` (keep the 19 existing tests passing):
   - validate accepts the role format; rejects the old string list, an unknown role, a context
     text not copied from the question, extra keys;
   - per-triple context overrides question context in sub-questions;
   - Bernard's question (typo kept): "When herbicides such as glyphosate are used against invasive
     plants in grassland resotration, do they also reduce native plants cover and richness?" with
     a stubbed LLM reply -> 2 sub-questions (class herbicides, member glyphosate), BOTH carrying
     setting "grassland resotration" and purpose "invasive plants"; `s_group == "exemplar"`,
     same `group_id`; `r["spelling"]["resotration"] == "restoration"`;
   - context never changes `n_subquestions`;
   - combination: "Does deer browsing plus competing vegetation reduce seedling survival?" with a
     stub -> ONE sub-question whose s is the member list, role "combination";
   - contrast sub-questions share a `group_id`;
   - template reuse with context blanked: teach the herbicide/grassland question, then
     "When insecticides such as imidacloprid are used against invasive plants in boreal peatlands,
     do they also reduce native plants cover and richness?" fills from the template with setting
     "boreal peatlands" (no LLM call after probation);
   - frames.json from version 1 is discarded on load;
   - split warnings fire for "mowing and logging" kept flat, and not for "population growth and
     regulation".
3. Make `run_all_tests.sh` run everything, including the bank suites (the data files are now in
   `question_parser/data/`).

## Done when

Run `cd question_parser && ./run_all_tests.sh` and confirm, reporting the actual numbers:

| suite | must be |
|---|---|
| structure_test.py | all pass (19 old + new) |
| edge_test.py | 15/15 |
| audit_test.py | 11/11 ok |
| attack_test.py, attack_round2.py | 0 fooled |
| stress_test_v2.py 0 10000 | `TOTAL 10000 2143 2143 0 7857`, 0 false passes |
| run_bank.py | `{'llm': 20, 'frame_cache': 980}` (or better), 0 wrong |
| run_varied.py | 266 LLM calls / 133 templates (or fewer calls), 0 wrong |
| messy_test.py | WRONG 0 |
| run_runon.py | 0 wrong |

If a bank number changes, explain exactly why before accepting it (e.g. context now blanked so
more templates match = fewer LLM calls is fine; any WRONG > 0 is not).

Then run the real model once (needs `.env`):
`../.venv/bin/python parse_question.py --no-uncertainty "When herbicides such as glyphosate are used against invasive plants in grassland resotration, do they also reduce native plants cover and richness?"`
and show the parse, sub-questions, spelling, warnings, and tokens.

## Finish

1. Update `HANDOFF.md` section 4c with the new output fields (context roles, per-triple override,
   combination, s_group/o_group/group_id, subquestions_dropped, spelling, warnings,
   FRAMES_VERSION) and the new test numbers.
2. Commit only `question_parser/` and `HANDOFF.md`:
   `git commit -m "question_parser: context roles carried into sub-questions, combination relation, comparison tags, spelling hints, split warnings"`
   Author Bernard only, no co-author line. Do not push.
3. Final summary: what changed, test table with real numbers, prompt token growth per LLM call,
   and anything you could not do.
