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


def _config():
    try:
        return (os.environ["BRINE_URL"].rstrip("/"), os.environ["BRINE_MODEL"],
                os.environ["BRINE_API_KEY"])
    except KeyError as missing:
        raise SystemExit(f"Environment variable {missing} is not set "
                         "(need BRINE_URL, BRINE_MODEL, BRINE_API_KEY).")


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
