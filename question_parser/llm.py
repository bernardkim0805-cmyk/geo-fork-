"""Layer 3: the real LLM call.

The API key is read from the environment variable API_KEY. `call_model` is the ONLY
provider-specific function: implement it with your provider's SDK (temperature 0, JSON output).
Everything else (prompt, reply checking) is provider-independent.

llm_parse(question) -> dict          a parse in the format below
                    raises           on API errors or unusable replies; the pipeline catches
                                     any exception and returns route "llm_failed"
"""
import json, os, re

API_KEY = os.environ.get("API_KEY")          # "the API key"

KINDS = ["drivers", "interactions", "scale_dependence", "trajectories", "methods_causal",
         "measurement", "intervention_outcomes", "mechanism", "evidence_gaps", "interaction", "other"]
PREDICATES = ["any", "causes", "contributes_to", "associated_with", "correlated_with", "prevents",
              "enables", "disrupts", "regulates", "mediates", "moderates", "precedes"]

SYSTEM_PROMPT = f"""You turn an ecology research question into JSON for a causal evidence graph
(Causal Mosaic schema, CAMO 0.10.0). Reply with ONE JSON object and nothing else.

Fields:
  "kind":    one of {KINDS}
  "topic":   the question's main subject phrase, COPIED WORD FOR WORD from the question
  "triples": list of {{"s": ..., "p": ..., "o": ...}} causal links the question asks about
      "s", "o": phrases COPIED WORD FOR WORD from the question, or an open slot written "?x"
               (what the question asks for), optionally with "s_type" / "o_type" describing it
      "p":     one of {PREDICATES}; use "any" unless the question names a specific relation
      optional per triple: "moderators" (list), "candidates" (list), "o_qualifier" (list of
               increased/decreased/unchanged/present/absent), "rank_by", "feature"
  "context":     optional list of context phrases (places, ecosystems, time spans)
  "camo_fields": optional list of CAMO field names useful for answering

Rules:
  1. Every value of "s", "o" and "topic" must appear in the question exactly as written,
     or be an open slot starting with "?". Never paraphrase, never invent.
  2. Anything you work out ABOUT a phrase (its type, ontology id) goes in a field named after
     the slot: "s_type", "o_type", "s_entity_term", "topic_id". Never anywhere else.
  3. Do not repeat the topic in free-text fields.
  4. Same question shape -> same JSON shape. Be deterministic.

Examples:
"""

EXAMPLES = [
 ("Which mechanisms most strongly determine density dependence, and how do their relative effects change across environmental gradients?",
  {"kind": "drivers", "topic": "density dependence",
   "triples": [{"s": "?x", "s_type": "mechanism", "p": "any", "o": "density dependence",
                "moderators": ["environmental gradients"]}],
   "camo_fields": ["moderation", "context_dependence"]}),
 ("How do human activities alter seed dispersal, and which effects persist after the disturbance or pressure is reduced?",
  {"kind": "intervention_outcomes", "topic": "seed dispersal",
   "triples": [{"s": "human activities", "p": "any", "o": "seed dispersal", "feature": "reversibility"}],
   "camo_fields": ["reversibility"]}),
 ("How should researchers measure soil carbon storage, and what are the main sources of uncertainty or sampling bias?",
  {"kind": "measurement", "topic": "soil carbon storage", "triples": [],
   "camo_fields": ["measured_attribute", "sample_size_value", "sampling_effort_protocol"]}),
]


def build_prompt(question):
    shots = "\n".join(f"Q: {q}\nA: {json.dumps(a)}" for q, a in EXAMPLES)
    return SYSTEM_PROMPT + shots, f"Q: {question}\nA:"


def call_model(system, user):
    """Send one request with the API key and return the reply TEXT.
    Implement with your provider's SDK: temperature=0, max output ~800 tokens, JSON mode if offered.

    Team implementation: the BRINE LiteLLM endpoint (OpenAI-compatible, Qwen3.8-27B), stdlib only.
    Reads BRINE_URL and BRINE_MODEL; the key is API_KEY, or BRINE_API_KEY when API_KEY is unset.
    Raises on any HTTP / network error (the pipeline catches it and reports route "llm_failed")."""
    import ssl, urllib.request
    key = API_KEY or os.environ.get("BRINE_API_KEY")
    url, model = os.environ.get("BRINE_URL"), os.environ.get("BRINE_MODEL")
    if not (key and url and model):
        raise RuntimeError("need API_KEY (or BRINE_API_KEY), BRINE_URL and BRINE_MODEL in the environment")
    body = {"model": model, "temperature": 0, "max_tokens": 800,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False}}      # Qwen3: no hidden reasoning tokens
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    ctx = ssl.create_default_context()
    paths = ssl.get_default_verify_paths()           # python.org macOS builds ship without CA certs
    if not (paths.cafile and os.path.exists(paths.cafile)) and os.path.exists("/etc/ssl/cert.pem"):
        ctx.load_verify_locations("/etc/ssl/cert.pem")
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"] or ""


def extract_json(text):
    """The reply should be one JSON object; tolerate code fences or text around it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def validate(parse, question):
    """Reject replies that break the rules, so they can't become templates."""
    if not isinstance(parse, dict):
        raise ValueError("reply is not an object")
    if parse.get("kind") not in KINDS:
        raise ValueError(f"bad kind {parse.get('kind')!r}")
    if not isinstance(parse.get("triples", []), list):
        raise ValueError("triples is not a list")
    low = " ".join(question.lower().split())
    for t in parse.get("triples", []):
        if not isinstance(t, dict) or t.get("p", "any") not in PREDICATES:
            raise ValueError(f"bad triple {t!r}")
        for slot in ("s", "o"):
            v = t.get(slot)
            if isinstance(v, str) and v and not v.startswith("?") and " ".join(v.lower().split()) not in low:
                raise ValueError(f"{slot}={v!r} is not copied from the question")
    tp = parse.get("topic")
    if isinstance(tp, str) and tp and " ".join(tp.lower().split()) not in low:
        raise ValueError(f"topic={tp!r} is not copied from the question")
    return parse


def llm_parse(question):
    system, user = build_prompt(question)
    return validate(extract_json(call_model(system, user)), question)
