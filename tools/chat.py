"""Chat with the BRINE model in the terminal.

Usage:  source .env && python3 tools/chat.py          (thinking off: fast, cheap)
        source .env && python3 tools/chat.py --think  (let Qwen reason first)
Type 'quit' to exit, 'reset' to clear the conversation.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.llm import _SSL, _config  # reuse the agent's settings and HTTPS fix

url, model, key = _config()
think = "--think" in sys.argv
history = []
print(f"Talking to {model} (thinking {'on' if think else 'off'}). 'quit' to exit, 'reset' to clear.")

while True:
    try:
        text = input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if text in ("quit", "exit"):
        break
    if text == "reset":
        history = []
        print("(cleared)")
        continue
    if not text:
        continue
    history.append({"role": "user", "content": text})
    body = {"model": model, "messages": history, "max_tokens": 2000,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(
        url + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300, context=_SSL) as resp:
            data = json.loads(resp.read())
    except Exception as err:
        print(f"(error: {err})")
        history.pop()
        continue
    reply = data["choices"][0]["message"]["content"] or ""
    history.append({"role": "assistant", "content": reply})
    u = data.get("usage", {})
    print(f"\nqwen> {reply}\n   [{u.get('prompt_tokens')} in / {u.get('completion_tokens')} out tokens]")
