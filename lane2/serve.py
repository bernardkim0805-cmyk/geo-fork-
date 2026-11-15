"""Local dev server: serves ui.html and lets its Run button start the pipeline.

    ./serve.sh                      # or: .venv/bin/python -m lane2.serve [--port 8765]

Then press "Start run" in the viewer. The run is `python -m lane2.cli
<questions> answers.json --live`, exactly what you would type, so the live view
shows its progress. Dev tool only: the judged run is ./run.sh and never uses it.

Safety: binds 127.0.0.1 only; runs only the pipeline, only on *.yaml question
files in the repo root; one run at a time; API calls must carry a custom header
and a localhost Host, so another website open in your browser can't start runs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ANSWERS = REPO / "answers.json"
LOG = REPO / "answers.run.log"
ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def question_sets() -> list[dict]:
    """*.yaml files in the repo root that hold a question list."""
    out = []
    for p in sorted(REPO.glob("*.yaml")):
        if p.name == "causal_graph.yaml":
            continue
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = data.get("questions") if isinstance(data, dict) else data
        if isinstance(items, list) and items and isinstance(items[0], dict) and "question" in items[0]:
            out.append({"file": p.name, "count": len(items),
                        "ids": [str(q.get("id")) for q in items]})
    return out


class Runner:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.cmd: list[str] = []
        self.started = None
        self.finished = None
        self.returncode = None
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, qfile: str, only: list[str], uncertainty: bool) -> list[str]:
        with self.lock:
            if self.running():
                raise RuntimeError("a run is already going")
            sets = {s["file"]: s for s in question_sets()}
            if qfile not in sets:
                raise ValueError(f"unknown question file {qfile!r}")
            bad = [i for i in only if not ID_RE.match(i) or i not in sets[qfile]["ids"]]
            if bad:
                raise ValueError(f"unknown question id(s): {', '.join(bad)}")
            cmd = [sys.executable, "-m", "lane2.cli", qfile, ANSWERS.name, "--live", "--debug"]
            if only:
                cmd += ["--only", ",".join(only)]
            if uncertainty:
                cmd.append("--uncertainty")
            log = open(LOG, "w", encoding="utf-8")
            log.write("$ " + " ".join(cmd[1:]) + "\n")
            log.flush()
            self.proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            log.close()  # the child keeps its own handle
            self.cmd, self.started = cmd, time.time()
            self.finished = self.returncode = None
            threading.Thread(target=self._wait, args=(self.proc,), daemon=True).start()
            return cmd

    def _wait(self, proc: subprocess.Popen) -> None:
        rc = proc.wait()
        if proc is self.proc:
            self.returncode, self.finished = rc, time.time()

    def stop(self) -> bool:
        with self.lock:
            if not self.running():
                return False
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            # a killed run never writes its final state; mark the live file stopped
            live = ANSWERS.with_suffix(".live.json")
            try:
                d = json.loads(live.read_text(encoding="utf-8"))
                d["running"], d["finished"], d["stopped"] = False, time.time(), True
                live.write_text(json.dumps(d), encoding="utf-8")
            except Exception:
                pass
            return True

    def status(self) -> dict:
        tail = []
        try:
            tail = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        except Exception:
            pass
        return {"server": True, "running": self.running(),
                "command": " ".join(["python"] + self.cmd[1:]) if self.cmd else None,
                "started": self.started, "finished": self.finished,
                "returncode": self.returncode, "log": tail,
                "question_sets": [{"file": s["file"], "count": s["count"], "ids": s["ids"]}
                                  for s in question_sets()]}


RUNNER = Runner()


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the terminal for run output
        pass

    def _ok_origin(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost") and self.headers.get("X-Pipeline") == "1"

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Only these are served. Everything else in the repo is off limits: the
    # handler's root is the working directory, so without this GET /.env
    # returned the BRINE key to anything that could reach the port, including
    # a page on another site using the browser as a proxy.
    SERVABLE = {"/ui.html", "/answers.json", "/answers.trace.json",
                "/answers.live.json", "/ablations.json", "/compare.json"}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/status":
            if not self._ok_origin():
                return self._json(403, {"error": "forbidden"})
            return self._json(200, RUNNER.status())
        if path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/ui.html")
            self.end_headers()
            return
        if path not in self.SERVABLE:
            return self._json(404, {"error": "not found"})
        return super().do_GET()

    def do_POST(self):
        if not self._ok_origin():
            return self._json(403, {"error": "forbidden"})
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except Exception:
            return self._json(400, {"error": "bad JSON"})
        if path == "/api/run":
            only = [s.strip() for s in str(body.get("only") or "").split(",") if s.strip()]
            try:
                cmd = RUNNER.start(str(body.get("questions") or ""), only,
                                   bool(body.get("uncertainty")))
            except RuntimeError as e:
                return self._json(409, {"error": str(e)})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            print(f"[serve] started: python {' '.join(cmd[1:])}", file=sys.stderr)
            return self._json(200, RUNNER.status())
        if path == "/api/stop":
            stopped = RUNNER.stop()
            if stopped:
                print("[serve] run stopped", file=sys.stderr)
            return self._json(200, {**RUNNER.status(), "stopped": stopped})
        return self._json(404, {"error": "not found"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't open a browser tab")
    args = ap.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(REPO)))
    url = f"http://127.0.0.1:{args.port}/ui.html"
    print(f"[serve] viewer at {url}  (Ctrl+C to quit; stops any run in progress)", file=sys.stderr)
    if not args.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RUNNER.stop()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
