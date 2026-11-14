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


def _build_body(messages: list[dict], params: dict) -> dict:
    body = {"model": os.environ["BRINE_MODEL"], "messages": messages, **params}
    # North-Mini-Code-1.0 and Qwen3 on vLLM support disabling the reasoning
    # pass for simple structured-output tasks (saves tokens and latency).
    body.setdefault("chat_template_kwargs", {"enable_thinking": False})
    return body


def chat(messages: list[dict], **params) -> str:
    url = os.environ["BRINE_URL"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = _build_body(messages, params)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['BRINE_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    choice = resp["choices"][0]
    msg = choice["message"]
    # content is null for reasoning models mid-think; only use reasoning_content
    # when the generation finished cleanly (not truncated by length limit).
    content = msg.get("content")
    if content is None and choice.get("finish_reason") != "length":
        content = msg.get("reasoning_content") or ""
    return content or ""


def chat_counted(messages: list[dict], **params) -> tuple[str, int]:
    """Like chat() but also returns total tokens used."""
    url = os.environ["BRINE_URL"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = _build_body(messages, params)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['BRINE_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    choice = resp["choices"][0]
    msg = choice["message"]
    content = msg.get("content")
    if content is None and choice.get("finish_reason") != "length":
        content = msg.get("reasoning_content") or ""
    tokens = resp.get("usage", {}).get("total_tokens", 0)
    return content or "", tokens
