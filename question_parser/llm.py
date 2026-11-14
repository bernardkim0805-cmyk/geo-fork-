"""Layer 3: the real LLM call.

The API key is read from the environment variable API_KEY. `call_model` is the ONLY
provider-specific function: implement it with your provider's SDK (temperature 0, JSON output).
Everything else (prompt, reply checking) is provider-independent.

llm_parse(question) -> (dict, int)   a parse in the format below, and the total tokens the call used
                    raises           on API errors or unusable replies; the pipeline catches
                                     any exception and returns route "llm_failed"
"""
import json, os, re

API_KEY = os.environ.get("API_KEY")          # "the API key"

KINDS = ["drivers", "interactions", "scale_dependence", "trajectories", "methods_causal",
         "measurement", "intervention_outcomes", "mechanism", "evidence_gaps", "interaction", "other"]
RELATIONS = ["exemplar", "list", "contrast", "alternatives"]
PREDICATES = ["any", "causes", "contributes_to", "associated_with", "correlated_with", "prevents",
              "enables", "disrupts", "regulates", "mediates", "moderates", "precedes"]

SYSTEM_PROMPT = f"""You turn an ecology research question into JSON for a causal evidence graph
(Causal Mosaic schema, CAMO 0.10.0). Reply with ONE JSON object and nothing else.

Fields:
  "kind":    one of {KINDS}
  "topic":   the question's main subject phrase, COPIED WORD FOR WORD from the question
  "triples": list of {{"s": ..., "p": ..., "o": ...}} causal links the question asks about
      "s", "o": a phrase COPIED WORD FOR WORD from the question, or an open slot written "?x"
               (what the question asks for), optionally with "s_type" / "o_type" describing it,
               OR a group when the side names several things:
               {{"class": <phrase or null>, "members": [<phrases>], "relation": <relation>}}
               relation is one of {RELATIONS}:
                 "exemplar"     a class with examples: "herbicides such as glyphosate",
                                "including", "especially", "particularly", "(e.g. ...)"
                 "list"         separate things joined by and/commas: "fire and grazing"
                 "contrast"     things compared head to head: "X versus Y"
                 "alternatives" "X or Y"
               A single compound topic is NOT a group: "population growth and regulation",
               "dispersal and colonization" stay one phrase.
      "p":     one of {PREDICATES}; use "any" unless the question names a specific relation
      optional per triple: "moderators" (list), "candidates" (list), "o_qualifier" (list of
               increased/decreased/unchanged/present/absent), "rank_by", "feature"
  "context":     optional list of context phrases (places, ecosystems, time spans)
  "camo_fields": optional list of CAMO field names useful for answering

Rules:
  1. Every phrase in "s", "o" (including a group's class and members) and "topic" must appear
     in the question exactly as written, or be an open slot starting with "?". Never paraphrase.
  2. Anything you work out ABOUT a phrase (its type, ontology id) goes in a field named after
     the slot: "s_type", "o_type", "s_entity_term", "topic_id". Never anywhere else.
  3. Do not repeat the topic in free-text fields.
  4. Same question shape -> same JSON shape. Be deterministic.
  5. Leave out a leading "the", "a" or "an" from phrases.

Examples:
"""

EXAMPLES = [
 ("Do herbicides such as glyphosate increase the population of Oryza sativa?",
  {"kind": "intervention_outcomes", "topic": "population of Oryza sativa",
   "triples": [{"s": {"class": "herbicides", "members": ["glyphosate"], "relation": "exemplar"},
                "p": "any", "o": "population of Oryza sativa", "o_qualifier": ["increased"]}]}),
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


PLACEHOLDER = re.compile(r"^\s*(your[_ -]|<|xxx|changeme|replace)", re.I)


def _dotenv():
    """KEY=value pairs from .env in this folder or the repo root (values never printed)."""
    vals = {}
    for path in (".env", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                m = re.match(r"\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$", line)
                if m:
                    vals.setdefault(m.group(1), m.group(2).strip('"').strip("'"))
    return vals


def _settings():
    """(key, url, model): environment first, then .env for anything missing or a placeholder."""
    dot = _dotenv()
    def get(*names):
        for n in names:
            v = os.environ.get(n, "")
            if v and not PLACEHOLDER.match(v):
                return v
        for n in names:
            v = dot.get(n, "")
            if v and not PLACEHOLDER.match(v):
                return v
        raise RuntimeError(f"{names[0]} is missing or a placeholder: put the real value in .env")
    key, url, model = get("API_KEY", "BRINE_API_KEY"), get("BRINE_URL"), get("BRINE_MODEL")
    if not url.startswith(("http://", "https://")):
        raise RuntimeError("BRINE_URL must start with http:// or https://")
    return key, url, model


def call_model(system, user, max_tokens=800):
    """Send one request with the API key and return (reply TEXT, total tokens used).
    Implement with your provider's SDK: temperature=0, max output ~800 tokens, JSON mode if offered.

    Team implementation: the BRINE LiteLLM endpoint (OpenAI-compatible, Qwen3.8-27B), stdlib only.
    Reads BRINE_URL and BRINE_MODEL; the key is API_KEY, or BRINE_API_KEY when API_KEY is unset.
    Raises on any HTTP / network error (the pipeline catches it and reports route "llm_failed")."""
    import ssl, urllib.request
    key, url, model = _settings()
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
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
    tokens = (data.get("usage") or {}).get("total_tokens") or 0
    return data["choices"][0]["message"]["content"] or "", tokens


def extract_json(text):
    """The reply should be one JSON object; tolerate code fences or text around it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


# Markers that ALWAYS mean "a class with examples" or "a head-to-head contrast". Softer words
# ("like", "for example", "rather than", "compared with") are used too loosely to enforce.
EXEMPLAR_MARKER = re.compile(r"\b(such as|including|especially|particularly)\s+\w|\(\s*e\.g\.", re.I)
CONTRAST_MARKER = re.compile(r"\b(versus|vs\.?)\s+\w", re.I)


def detect_markers(question):
    """Cheap, rule-based: which group relations the question's wording REQUIRES."""
    need = set()
    if EXEMPLAR_MARKER.search(question):
        need.add("exemplar")
    if CONTRAST_MARKER.search(question):
        need.add("contrast")
    return need


def _phrases_of(side):
    """All phrases in one side of a triple: a string, or a group's class + members."""
    if isinstance(side, str):
        return [side]
    if isinstance(side, dict):
        return ([side["class"]] if isinstance(side.get("class"), str) else []) + \
               [m for m in side.get("members", []) if isinstance(m, str)]
    return []


def _check_group(g, where):
    if not isinstance(g, dict) or set(g) - {"class", "members", "relation"}:
        raise ValueError(f"{where}: a group needs only class / members / relation")
    if g.get("relation") not in RELATIONS:
        raise ValueError(f"{where}: bad relation {g.get('relation')!r}")
    if not (g.get("class") is None or isinstance(g.get("class"), str)):
        raise ValueError(f"{where}: class must be a phrase or null")
    mem = g.get("members")
    if not (isinstance(mem, list) and mem and all(isinstance(m, str) and m for m in mem)):
        raise ValueError(f"{where}: members must be a non-empty list of phrases")
    if g["relation"] == "exemplar" and not g.get("class"):
        raise ValueError(f"{where}: an exemplar group needs its class")
    if g["relation"] in ("list", "contrast", "alternatives") and len(mem) < 2:
        raise ValueError(f"{where}: a {g['relation']} group needs at least two members")


_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)


def strip_articles(parse):
    """Drop a leading the/a/an from every phrase (s, o, topic, group class and members).
    Articles carry no meaning for retrieval, and phrases without them can be reused as templates."""
    def fix(v):
        return _ARTICLE.sub("", v) if isinstance(v, str) and not v.startswith("?") else v
    if not isinstance(parse, dict):
        return parse
    if "topic" in parse:
        parse["topic"] = fix(parse["topic"])
    for t in parse.get("triples") or []:
        if not isinstance(t, dict):
            continue
        for slot in ("s", "o"):
            v = t.get(slot)
            if isinstance(v, dict):
                if "class" in v: v["class"] = fix(v["class"])
                if isinstance(v.get("members"), list): v["members"] = [fix(m) for m in v["members"]]
            else:
                t[slot] = fix(v)
    return parse


def validate(parse, question):
    """Reject replies that break the rules, so they can't become templates."""
    if not isinstance(parse, dict):
        raise ValueError("reply is not an object")
    if parse.get("kind") not in KINDS:
        raise ValueError(f"bad kind {parse.get('kind')!r}")
    if not isinstance(parse.get("triples", []), list):
        raise ValueError("triples is not a list")
    low = " ".join(question.lower().split())
    found = set()
    for t in parse.get("triples", []):
        if not isinstance(t, dict) or t.get("p", "any") not in PREDICATES:
            raise ValueError(f"bad triple {t!r}")
        for slot in ("s", "o"):
            v = t.get(slot)
            if isinstance(v, dict):
                _check_group(v, slot)
                found.add(v["relation"])
            elif v is not None and not isinstance(v, str):
                raise ValueError(f"{slot} must be a phrase, an open slot or a group")
            for ph in _phrases_of(v):
                if ph and not ph.startswith("?") and " ".join(ph.lower().split()) not in low:
                    raise ValueError(f"{slot} phrase {ph!r} is not copied from the question")
    missing = detect_markers(question) - found
    if missing:
        raise ValueError(f"the question's wording needs a {'/'.join(sorted(missing))} group, "
                         "but the parse has none")
    tp = parse.get("topic")
    if isinstance(tp, str) and tp and " ".join(tp.lower().split()) not in low:
        raise ValueError(f"topic={tp!r} is not copied from the question")
    return parse


def llm_parse(question):
    system, user = build_prompt(question)
    text, tokens = call_model(system, user)
    try:
        return validate(strip_articles(extract_json(text)), question), tokens
    except ValueError as first:
        # one corrective retry: tell the model exactly which rule it broke
        text2, t2 = call_model(system, user.replace("\nA:", f"\n(Your previous answer was rejected: {first}. "
                                                    "Follow the rules exactly.)\nA:"))
        tokens += t2
        try:
            return validate(strip_articles(extract_json(text2)), question), tokens
        except Exception as e:
            e.tokens = tokens    # both calls were paid for even though the reply is unusable
            raise


# ---------------------------------------------------------------------------
# DEV ONLY (delete later): ask Qwen how unsure it is that a parse is right.
# ---------------------------------------------------------------------------
UNCERTAINTY_PROMPT = """You check whether a structured parse correctly captures an ecology question.
You get the question and the parse (JSON). Judge ONLY whether the parse captures what the question
asks: the right cause(s) as "s", the right effect(s) as "o", nothing missing, nothing invented,
nothing reversed. Do not judge whether the science is true.

Reply with ONE JSON object and nothing else:
  {"uncertainty": <integer 0-100>, "reason": "<one short sentence>"}

0 = certain the parse is right. 100 = certain it is wrong or useless. Use the full range."""


def _public(parse):
    """Drop internal bookkeeping keys (e.g. "_recompute") so the checker judges only the parse."""
    return {k: v for k, v in parse.items() if not str(k).startswith("_")} if isinstance(parse, dict) else parse


def score_uncertainty(question, parse):
    """Returns (score 0-100 or None, reason, tokens). Never raises."""
    if parse is None:
        return 100, "no parse was produced", 0
    try:
        text, tokens = call_model(UNCERTAINTY_PROMPT,
                                  f"Question: {question}\nParse: {json.dumps(_public(parse), ensure_ascii=False)}",
                                  max_tokens=150)
    except Exception as e:
        return None, f"uncertainty check failed: {type(e).__name__}: {str(e)[:80]}", 0
    try:
        obj = extract_json(text)
        score = max(0, min(100, int(round(float(obj["uncertainty"])))))
        return score, str(obj.get("reason", ""))[:300], tokens
    except Exception as e:
        return None, f"unusable reply: {text[:80]!r}", tokens
