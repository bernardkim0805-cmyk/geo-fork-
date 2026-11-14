"""Tiny client for the BRINE LiteLLM endpoint (OpenAI-compatible).

Uses only the Python standard library, so there is nothing extra to install.
Reads BRINE_URL, BRINE_MODEL and BRINE_API_KEY from the environment.
"""
import json
import os
import re
import ssl
import threading
import time
import urllib.request

# Running total of tokens, shared by all threads (energy is scored on tokens).
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
_lock = threading.Lock()


def _ssl_context():
    """Verified HTTPS. python.org builds on macOS ship without CA certs, so fall back to
    the system bundle when Python's own default is missing."""
    ctx = ssl.create_default_context()
    paths = ssl.get_default_verify_paths()
    if not (paths.cafile and os.path.exists(paths.cafile)) and os.path.exists("/etc/ssl/cert.pem"):
        ctx.load_verify_locations("/etc/ssl/cert.pem")
    return ctx


_SSL = _ssl_context()


# Template values like "YOUR_URL" / "<key>" mean the setting was never filled in.
PLACEHOLDER = re.compile(r"^\s*(your[_ -]|<|xxx|changeme|replace)", re.I)


def _load_dotenv(path=".env"):
    """KEY=value lines from .env (with or without 'export'). Values are never printed."""
    vals = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            m = re.match(r"\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if m:
                vals[m.group(1)] = m.group(2).strip('"').strip("'")
    return vals


def _config():
    """BRINE settings from the environment, falling back to .env for any setting that is
    missing or still a template placeholder (so running `python -m agent.run` directly works
    too). Exits BEFORE any question runs if a setting is missing, a placeholder, or invalid."""
    dot = _load_dotenv()
    vals = []
    for k in ("BRINE_URL", "BRINE_MODEL", "BRINE_API_KEY"):
        v = os.environ.get(k, "")
        if not v or PLACEHOLDER.match(v):
            v = dot.get(k, v)
        if not v or PLACEHOLDER.match(v):
            raise SystemExit(f"{k} is {'a placeholder' if v else 'not set'}: put the real value "
                             "in .env (see run.sh) or export it.")
        vals.append(v)
    url, model, key = vals
    if not url.startswith(("http://", "https://")):
        raise SystemExit("BRINE_URL must start with http:// or https://")
    return url.rstrip("/"), model, key


def chat_json(system, user, max_tokens=400, retries=3):
    """Send one chat request with thinking OFF and return the reply parsed as JSON."""
    url, model, key = _config()
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0,
        # Qwen3: skip the hidden "thinking" tokens. Saves energy and time.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120, context=_SSL) as resp:
                data = json.loads(resp.read())
            usage = data.get("usage", {})
            with _lock:
                USAGE["calls"] += 1
                USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0)
                USAGE["completion_tokens"] += usage.get("completion_tokens", 0)
            return _parse_json(data["choices"][0]["message"]["content"] or "")
        except Exception as err:  # network hiccup or bad JSON: wait, then retry
            last_err = err
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} tries: {last_err}")


def _parse_json(text):
    """Pull the first {...} block out of the reply (models sometimes add ``` fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])
