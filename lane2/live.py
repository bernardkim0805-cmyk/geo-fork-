"""Progress file for the live viewer: `python -m lane2.cli Q.yaml answers.json --live`.

Writes answers.live.json after every stage of every question; ui.html polls it
once a second. Local file writes only: never calls the LLM, so it costs no
energy, and every write is best-effort, so it can never break a run.

Each write goes to a temp file and is renamed into place, so the viewer never
reads a half-written file.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import llm

STAGES = ("parse", "ground", "retrieve", "decide")


class Live:
    def __init__(self, path: Path, questions) -> None:
        self.path = Path(path)
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self._index = {q.id: i for i, q in enumerate(questions)}
        self.state = {
            "kind": "live",
            "running": True,
            "started": time.time(),
            "updated": None,
            "finished": None,
            "total": len(questions),
            "done": 0,
            "tokens": 0,
            # one LLM call at a time: is one in flight, and since when
            "llm": {"busy": False, "since": None, "calls": 0},
            "questions": [{
                "id": q.id, "question": q.text, "kind": q.kind,
                "status": "queued",   # queued | running | done | error
                "stage": None,        # parse | ground | retrieve | decide
                "pair": None,         # index of the pair being searched
                "parse": None,        # parse result, once parse finishes
                "pairs": [],          # per-pair summaries as they finish
                "started": None, "finished": None,
            } for q in questions],
            # finished questions' full traces, keyed by id
            "traces": {},
        }
        llm.HOOKS.append(self._on_llm)
        self.write()

    # -- events -------------------------------------------------------------
    def _q(self, qid: str) -> dict:
        return self.state["questions"][self._index[qid]]

    def stage(self, qid: str, stage: str, **info) -> None:
        q = self._q(qid)
        if q["status"] == "queued":
            q["status"], q["started"] = "running", time.time()
        q["stage"] = stage
        q.update(info)
        self.write()

    def pair_done(self, qid: str, summary: dict) -> None:
        self._q(qid)["pairs"].append(summary)
        self.write()

    def finish(self, qid: str, trace: dict, error: bool = False) -> None:
        q = self._q(qid)
        q["status"] = "error" if error else "done"
        q["stage"], q["finished"] = None, time.time()
        self.state["traces"][qid] = trace
        self.state["done"] += 1
        self.write()

    def close(self) -> None:
        self.state["running"] = False
        self.state["finished"] = time.time()
        self.state["llm"]["busy"] = False
        if self._on_llm in llm.HOOKS:
            llm.HOOKS.remove(self._on_llm)
        self.write()

    def _on_llm(self, event: str) -> None:
        L = self.state["llm"]
        if event == "start":
            L["busy"], L["since"] = True, time.time()
        else:
            L["busy"], L["since"] = False, None
            L["calls"] += 1
        self.write()

    # -- output -------------------------------------------------------------
    def write(self) -> None:
        try:
            self.state["updated"] = time.time()
            self.state["tokens"] = llm.USAGE["prompt_tokens"] + llm.USAGE["completion_tokens"]
            self._tmp.write_text(json.dumps(self.state), encoding="utf-8")
            os.replace(self._tmp, self.path)
        except Exception:
            pass  # the viewer is a convenience; the run must go on
