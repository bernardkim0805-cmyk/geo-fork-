"""Thin client for the BRINE LiteLLM endpoint (Qwen, OpenAI-compatible).

Settings: BRINE_URL / BRINE_MODEL / BRINE_API_KEY from the environment, falling
back to a `.env` file (repo root, then the current directory) for any setting
that is missing or still a template placeholder. Values are never printed.
Stdlib only. Every call costs energy points, so each use has to earn its place.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.request
from pathlib import Path

KEYS = ("BRINE_URL", "BRINE_MODEL", "BRINE_API_KEY")
REPO_ROOT = Path(__file__).resolve().parent.parent
# Template values like "YOUR_URL" / "<key>" mean the setting was never filled in.
PLACEHOLDER = re.compile(r"^\s*(your[_ -]|<|xxx|changeme|replace)", re.I)

# Running totals (energy is scored on tokens).
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def _ssl_context():
    """python.org builds on macOS ship without CA certs: fall back to the system bundle."""
    ctx = ssl.create_default_context()
    paths = ssl.get_default_verify_paths()
    if not (paths.cafile and os.path.exists(paths.cafile)) and os.path.exists("/etc/ssl/cert.pem"):
        ctx.load_verify_locations("/etc/ssl/cert.pem")
    return ctx


def _load_dotenv() -> dict:
    vals = {}
    for path in (REPO_ROOT / ".env", Path.cwd() / ".env"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$", line)
                if m and m.group(1) not in vals:
                    vals[m.group(1)] = m.group(2).strip('"').strip("'")
    return vals


def _settings() -> dict | None:
    dot = None
    out = {}
    for k in KEYS:
        v = os.environ.get(k, "")
        if not v or PLACEHOLDER.match(v):
            dot = _load_dotenv() if dot is None else dot
            v = dot.get(k, v)
        if not v or PLACEHOLDER.match(v):
            return None
        out[k] = v
    if not out["BRINE_URL"].startswith(("http://", "https://")):
        return None
    return out


def configured() -> bool:
    """True when all three BRINE settings are available (env or .env)."""
    return _settings() is not None


def chat(messages: list[dict], **params) -> str:
    s = _settings()
    if s is None:
        raise RuntimeError("BRINE_URL / BRINE_MODEL / BRINE_API_KEY not set (env or .env)")
    url = s["BRINE_URL"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = {"model": s["BRINE_MODEL"], "messages": messages,
            # Qwen3: skip hidden "thinking" tokens unless a caller asks for them.
            "chat_template_kwargs": {"enable_thinking": False}, **params}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {s['BRINE_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as r:
        data = json.load(r)
    USAGE["calls"] += 1
    u = data.get("usage") or {}
    USAGE["prompt_tokens"] += u.get("prompt_tokens", 0)
    USAGE["completion_tokens"] += u.get("completion_tokens", 0)
    return data["choices"][0]["message"]["content"]
