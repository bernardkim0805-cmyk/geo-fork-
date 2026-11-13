# Lane 2 — Reasoning 

## The Challenge

> **Create a reasoning agent that can answer questions**
> **based on the included graph.**

You will be scored on a hidden set of questions. Every hidden question is
written from the graph itself, so the answer (or the fact that there is no
answer) is always in `causal_graph.json`. Your approach should be able to
answer any question the graph has evidence for, and say so when it has no evidence.

`questions.yaml` is a practice set, built from questions experts and Canadian
biodiversity priorities raise. It shows the kind of question a real user asks.
The hidden set is narrower and more uniform: each question asks what one kind
of intervention or condition does to one kind of outcome, at a broad level
(e.g. "What effect does mulch & surface cover have on invasive / non-native
species abundance?"). About a third of the hidden questions have no evidence in
the graph, and some are written to catch common misreadings of the graph.

You should use the Qwen3.8-27B endpoint that was provided, but if your team decides
a different open weight model is preferred, we can discuss accommodating that. But
the judging must be based on Qwen3.8-27B.

```
causal_graph.yaml / .json   the corpus-level CAMO graph
schema/causalmosaic.yaml    the schema it validates against
questions.yaml              a practice question set, without answers
```

## Start here

```python
import yaml
graph = yaml.safe_load(open("causal_graph.yaml", encoding="utf-8"))
nodes = {n["id"]: n for n in graph["nodes"]}
for edge in graph["edges"]:
    print(nodes[edge["subject"]]["name"], edge["predicate"],
          nodes[edge["object"]]["name"], edge["causal_language"])
```

How to read a node, an edge, a null result or a certainty grade is set out in
the `description` of each class and slot in `schema/causalmosaic.yaml`. Read
those before you trust a direction.

## What you are being scored on

Scripts score your `answers.json` against the hidden set (60 points), and
judges assess the rest (40 points).

| Area | Points | How |
| --- | --- | --- |
| Correct direction, null-vs-effect and certainty; not fooled by trap questions | 17 | script |
| `confidence` matches how often you're right; `no_evidence` (saying what you searched for) when the graph has nothing | 14 | script |
| Answers parse; cited `edge_ids` exist and are the edges that bear on the question | 9 | script |
| Energy per question | 12 | script |
| Runs from a clean checkout | 8 | script |
| Method, faithfulness to the cited edges, honesty about uncertainty, value to a real user, and how easily it extends | 40 | judges |

Judges read your `answers.json` and any additional outputs it points to (see
below), open the edges you cite, and ask your system one real question of
their own.

## Provenance

Every edge carries `source_document` (a `document_id` into
`source_documents`) and `original_sentence`. Most edges (88%) also carry
`source_spans` with character offsets into the source article. Answers should
be traceable to a sentence.

## Answer format

Your command must write `answers.json` in the format below, with one entry per
question. This is what the scripts score.

```json
{
  "team": "Team A",
  "answers": [
    {
      "id": "H-017",
      "status": "answered",
      "answer": "Probably yes: four studies report lower non-native grass cover after burning, one reports no effect.",
      "direction": "decrease",
      "confidence": 0.75,
      "certainty": "moderate",
      "edge_ids": ["camo:edge_3050bbf82ed8", "camo:edge_31392cea762e"],
      "searched_for": null,
      "extras": ["outputs/H-017_evidence_map.html"]
    },
    {
      "id": "H-031",
      "status": "no_evidence",
      "answer": "The graph has no claim linking mowing to soil organic carbon.",
      "direction": null,
      "confidence": 0.9,
      "certainty": null,
      "edge_ids": [],
      "searched_for": "edges whose subject node is mowing or cutting and whose object node's measured_attribute is soil carbon or organic matter"
    }
  ]
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | The question's `id`. A missing entry is scored wrong at full confidence. |
| `status` | yes | `answered`, or `no_evidence` when the graph can't support an answer. |
| `answer` | yes | A short plain-language answer. For `no_evidence`, say what is missing. |
| `confidence` | yes | Your probability, 0 to 1, that this answer is correct. For `no_evidence`, the probability that the graph really can't answer it. |
| `direction` | answered | What happens to the outcome: `increase`, `decrease`, `no_effect` (studies tested it and found nothing), `mixed` (studies disagree) or `unclear`. Null for `no_evidence`. |
| `certainty` | answered | Strength of the evidence: `high`, `moderate`, `low` or `very_low`, however you grade it. Not the same thing as `confidence`. |
| `edge_ids` | answered | The edges in `causal_graph.json` your answer rests on. Every ID must exist. |
| `searched_for` | `no_evidence` | What you looked for and didn't find: which nodes, attributes or relations. |
| `extras` | no | Paths, relative to your repo, to any other outputs for this question. |

**Beyond the required format.** We're very interested in your interpretation of
what a good answer is, and it need not be written language. Your system
doesn't (and maybe shouldn't) answer with a paragraph like "Well, Tim, your
question has the following answer:". Models, meta-models, diagrams, evidence
maps and more are all welcome. Put them alongside `answers.json` and list them
in `extras`; judges will open them. They don't replace the required fields,
which the scripts need.

## Submitting

The last commit before **6 p.m. EST on Sunday, November 15** will be used for judging.

Judges run every submission on one machine, one at a time: an NVIDIA DGX Spark
(GB10, **ARM64 / aarch64** Linux, CUDA). The LLM is Qwen3.8-27B on the shared
LiteLLM endpoint, the same endpoint and model you use during development, so
what works for you will work for the judges.

1. **One command, from a clean checkout.** Your README should give a single
   command that runs the whole pipeline, for example
   `./run.sh questions.yaml answers.json`, reading the graph in this pack.
   Judges will pass the hidden question file in place of `questions.yaml`; it
   has the same shape, with only `id` and `question` on each entry. Pin your
   dependencies (a lock file or `==` versions). The only network call your
   pipeline may make is to the LLM endpoint: no other downloads or APIs at run
   time.
2. **Read the LLM endpoint, model and key from the environment.** Use exactly
   these variables: `BRINE_URL` (the endpoint URL), `BRINE_MODEL` (the model
   name) and `BRINE_API_KEY` (your key). Judges set the same three to run you
   with their own key on the same endpoint. Don't hardcode any of them, and
   don't commit keys.
3. **Output.** `answers.json` in the format above, plus any extra outputs it
   lists in `extras`.

Footprint is energy (Wh) per question answered: what your code uses on the
judging machine plus the model's share, estimated from the tokens your run uses
on the LiteLLM endpoint. It's ranked against the other teams, and it only counts once your submission clears the
lane's accuracy floor (half of answers correct). If more than 5% of entries in
`answers.json` don't parse in the format above, your automatic scores are
halved until it is fixed.

## The graph you are using

The graph is partially the answer to Lane 1, so please avoid sharing this file
widely. It was built automatically by an LLM and may not be accurate or
complete. You will be judged on how effectively your pipeline answers questions
based on **this graph** and will not be penalized for any errors that are in
the graph itself.

Ontology terms (`entity_term`, `measured_attribute`, `applied_to`) are CURIEs
where an exact Wikidata (`wd:`), ENVO, PATO, CHEBI or GO match was found, and
free text otherwise. Each node's `embedding_text` puts the readable label next
to each CURIE, which is the field to embed for retrieval.

---

# Our pipeline (team geo-fork)

**Run it** (one command, from a clean checkout):

```bash
./run.sh questions.yaml answers.json
```

First run creates `.venv/` and installs the pinned `requirements.txt`. LLM
settings come only from `BRINE_URL`, `BRINE_MODEL`, `BRINE_API_KEY` (the
skeleton does not call the LLM yet).

**Dev loop**

```bash
.venv/bin/python -m lane2.cli dev_questions.yaml answers.json --debug   # per-question trace
.venv/bin/python tests/validate_answers.py answers.json dev_questions.yaml
```

**Stages** (`lane2/`), each swappable; hand-off types live in `schema.py`:

| Stage | File | Status |
| --- | --- | --- |
| Parse question -> (intervention, outcome) | `parse.py` | regex stub |
| Ground phrases -> node ids | `ground.py` | keyword stub. **Main focus** |
| Retrieve edges X -> Y | `retrieve.py` | direct edges only |
| Decide direction / certainty / confidence | `decide.py` | majority-vote stub; check trap rules |
| Render answers.json entry | `render.py` | f-string templates |
| LLM client (unused) | `llm.py` | stdlib, env vars only |

`dev_questions.yaml` is our own hidden-style test set; `tests/cases.yaml`
collects hand-checked expected answers.
