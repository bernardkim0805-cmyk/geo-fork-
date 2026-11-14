"""Command-line entry point.

    python parse_question.py "Which interventions restore seed dispersal?"
    python parse_question.py --file questions.txt          (one question per line)

Learned templates are loaded from / saved to frames.json so they survive restarts.
Prints one JSON line per question: {"question", "route", "parse", "uncertainty", "tokens", "trace"}
  uncertainty: {"score": 0-100 or null (Qwen's judgement, DEV ONLY), "reason": ..., "cached": bool}
               always present; score is null when the check is off or failed
  tokens:      {"parse": ..., "uncertainty": ..., "total": ...}.
"""
import argparse, json, os, sys
from pipeline import Pipeline
from llm import llm_parse, score_uncertainty

FRAMES = "frames.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--graph", default="data/graph.json")
    ap.add_argument("--no-uncertainty", action="store_true",
                    help="skip the dev-only Qwen uncertainty check (saves one LLM call per question)")
    ap.add_argument("--no-confirm", action="store_true", help="turn template probation off")
    a = ap.parse_args()
    qs = [l.strip() for l in open(a.file) if l.strip()] if a.file else [a.question]
    if not qs or qs == [None]:
        ap.error("give a question or --file")
    P = Pipeline(a.graph, llm_parse, confirm=not a.no_confirm,
                 checker=None if a.no_uncertainty else score_uncertainty)
    if os.path.exists(FRAMES):
        P.frames.load(FRAMES)
    for q in qs:
        r = P.parse(q)
        print(json.dumps({"question": q, **r}, ensure_ascii=False))
    P.frames.save(FRAMES)
    print(json.dumps({"counts": P.counts}), file=sys.stderr)


if __name__ == "__main__":
    main()
