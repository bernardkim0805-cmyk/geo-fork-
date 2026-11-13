"""Thin LLM client. NOT used by the skeleton yet: every call costs energy
points, so each use has to earn its place.

Reads BRINE_URL / BRINE_MODEL / BRINE_API_KEY from the environment (README
rule: never hardcode them). Uses only the stdlib so no extra deps.
LiteLLM exposes an OpenAI-compatible /chat/completions route.
"""
from __future__ import annotations

import json
import os
import urllib.request


def chat(messages: list[dict], **params) -> str:
    url = os.environ["BRINE_URL"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = {"model": os.environ["BRINE_MODEL"], "messages": messages, **params}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['BRINE_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]
