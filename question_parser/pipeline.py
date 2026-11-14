"""Question -> structured parse, calling the LLM as rarely as possible.

Every question goes down four layers and stops at the first one that can answer:

  Layer 0  EXACT CACHE   seen this exact question before?           -> reuse, 0 LLM calls
  Layer 1  FRAME CACHE   same sentence frame as an earlier question? -> fill in the blanks, 0 LLM calls
  Layer 2  STRICT RULES  simple "does X affect Y in Z?" question?    -> rule parse, 0 LLM calls
  Layer 3  LLM           none of the above                           -> 1 LLM call, then LEARN a frame
                                                                       so the next similar question
                                                                       stops at layer 1

The LLM is the teacher: each call it makes is turned into a reusable template.
"""
import json, re
from parser_v2 import StrictParser


# ---------------------------------------------------------------------------
# Vocabulary from the CAMO graph (what the strict rules are allowed to recognise)
# ---------------------------------------------------------------------------
def vocab_from_graph(path):
    g = json.load(open(path))
    terms = set()
    for n in g["nodes"]:
        for field in ("entity_term", "measured_attribute"):
            v = (n.get(field) or "").strip().lower()
            # keep plain-English phrases; skip ontology IDs like ENVO:00001998 or wd:Q806838
            if v and ":" not in v and re.fullmatch(r"[a-z][a-z \-]{2,60}", v) and len(v.split()) <= 5:
                terms.add(v)
    contexts = {(e.get("ecosystem_context") or "").replace("_", " ")
                for e in g["edges"] if e.get("ecosystem_context")}
    return sorted(terms), sorted(contexts)


# ---------------------------------------------------------------------------
# Layer 1: frames learned from LLM answers
# ---------------------------------------------------------------------------
import copy

SLOT_FIELDS = ("s", "o", "topic")      # parse fields whose phrases (copied from the question) become blanks
# Fields that describe WHAT a blank's phrase is (its type, its ontology id). They were worked out for the
# old phrase, so they must not be copied onto a new one: they are cleared and listed in "_recompute".
# Convention: anything named "<slot>_..." (s_entity_type, o_entity_term, topic_id) plus these names.
DERIVED_KEYS = {"entity_type", "entity_term", "entity_id", "ontology_id", "curie", "grounding"}
MAX_BLANKS = 8
BLANK_KEY = "\x00blank"     # internal markers: contain a NUL byte, so real API output can't collide
TEXT_KEY = "\x00text"
MAX_QUESTION_CHARS = 5000
BLANK_NAMES = "ABCDEFGH"
# Words that reverse or exclude a topic. A blank containing one goes to the LLM instead.
BLANK_STOP_WORDS = {"not", "no", "never", "nor", "neither", "none", "without", "except", "excluding",
                    "but", "besides", "other", "than", "instead", "lack", "lacking", "absence", "absent",
                    "minus", "non", "anti", "un", "less", "fewer", "free", "deficit", "deficient",
                    "devoid", "opposite", "reverse", "inverse", "zero", "nil", "missing", "loss",
                    "lost", "failed", "failure", "reduced", "decreased", "declining", "decline"}


def _find_phrase(question, phrase):
    """Where `phrase` occurs in the question as whole words (capitals and spacing ignored).
    Returns the list of (start, end) character spans, using the QUESTION's own text."""
    words = [re.escape(w) for w in phrase.split()]
    if not words:
        return []
    pat = r"(?<![A-Za-z0-9])" + r"\s+".join(words) + r"(?![A-Za-z0-9])"
    return [(m.start(), m.end()) for m in re.finditer(pat, question, re.I)]


class _Unlearnable(Exception):
    pass


def _blank_template(obj, phrase_to_name, recompute, path=""):
    """Copy of the LLM answer where each slot value equal to a learned phrase becomes a blank marker,
    and fields derived from that phrase are cleared. Nothing else is touched."""
    if isinstance(obj, list):
        return [_blank_template(v, phrase_to_name, recompute, f"{path}[{i}]") for i, v in enumerate(obj)]
    if not isinstance(obj, dict):
        return obj
    out, blanked = {}, set()
    for k, v in obj.items():
        if any(_find_phrase(str(k), ph) for ph in phrase_to_name):
            raise _Unlearnable(f"key {k!r} mentions a blank phrase")
        if isinstance(v, str) and not (k in SLOT_FIELDS and v.lower().strip() in phrase_to_name):
            text, hit = v, False
            for ph, name in sorted(phrase_to_name.items(), key=lambda x: -len(x[0])):
                for a, b in reversed(_find_phrase(text, ph)):
                    text, hit = text[:a] + "\x00" + name + "\x00" + text[b:], True
            if hit:
                out[k] = {TEXT_KEY: text}
                continue
        if k in SLOT_FIELDS and isinstance(v, str) and v.lower().strip() in phrase_to_name:
            out[k] = {BLANK_KEY: phrase_to_name[v.lower().strip()]}
            blanked.add(k)
        else:
            out[k] = _blank_template(v, phrase_to_name, recompute, f"{path}.{k}")
    if blanked:
        for k in list(out):
            if k in DERIVED_KEYS or any(k.startswith(b + "_") for b in blanked):
                out[k] = None
                recompute.append(f"{path}.{k}".lstrip("."))
    return out


def _fill_template(obj, fillers):
    if isinstance(obj, list):
        return [_fill_template(v, fillers) for v in obj]
    if isinstance(obj, dict):
        if set(obj) == {BLANK_KEY}:
            return fillers[obj[BLANK_KEY]]
        if set(obj) == {TEXT_KEY}:                       # free text that mentioned a blank's phrase
            out = obj[TEXT_KEY]
            for name, val in fillers.items():
                out = out.replace("\x00" + name + "\x00", val)
            return out
        return {k: _fill_template(v, fillers) for k, v in obj.items()}
    return obj


def _norm(obj, skip):
    """Comparable form of a parse: lowercase single-spaced strings, skip the derived fields."""
    if isinstance(obj, list):
        return [_norm(v, skip) for v in obj]
    if isinstance(obj, dict):
        return {k: _norm(v, skip) for k, v in obj.items() if k not in skip and not k.startswith("_")}
    if isinstance(obj, str):
        return " ".join(obj.lower().split())
    return obj


def _parts_from_frame(frame):
    parts = []
    for w, _, _ in _words(re.sub(r"\{([A-H])\}", r" ⟦\1⟧ ", frame)):
        if w.startswith("⟦"):
            parts.append(w[1])
        elif parts and isinstance(parts[-1], list):
            parts[-1].append(w)
        else:
            parts.append([w])
    return parts


class FrameCache:
    def __init__(self, nlp, confirm=True):
        """confirm=True: a new template is on probation until a second LLM answer, for a different
        question of the same shape, agrees with what the template would have produced."""
        self.nlp = nlp          # spaCy, used only to sanity-check what goes into a blank
        self.confirm = confirm
        self.frames = []        # {"frame", "parts", "template", "recompute", "status"}

    # ---- learning ---------------------------------------------------------
    def learn(self, question, parse):
        """Turn (question, LLM parse) into a reusable frame. Returns the frame text, or None if the
        parse can't be safely generalised (invented phrase, phrase swallowed by another, too many)."""
        phrases = {}
        for v in _walk_values(parse, SLOT_FIELDS):
            if not v or v.startswith("?"):
                continue
            if not _find_phrase(question, v):        # not in the question as whole words: invented
                return None
            phrases[v.lower().strip()] = v
        if not phrases or len(phrases) > MAX_BLANKS:
            return None
        frame, phrase_to_name = question, {}
        for i, key in enumerate(sorted(phrases, key=len, reverse=True)):     # longest first
            name = BLANK_NAMES[i]
            phrase_to_name[key] = name
            spans = _find_phrase(frame, phrases[key])
            if not spans:                            # swallowed by a longer phrase: can't generalise
                return None
            for a, b in reversed(spans):
                frame = frame[:a] + "{" + name + "}" + frame[b:]
        recompute = []
        try:
            template = _blank_template(copy.deepcopy(parse), phrase_to_name, recompute)
        except _Unlearnable:
            return None
        parts = _parts_from_frame(frame)
        if any(f["parts"] == parts for f in self.frames):     # already have this shape
            return None
        self.frames.append({"frame": frame, "parts": parts, "template": template, "recompute": recompute,
                            "status": "pending" if self.confirm else "active"})
        return frame

    def confirm_or_drop(self, frame_rec, filled, llm_parse):
        """Probation check: activate the template if the LLM agrees with it, otherwise delete it."""
        skip = {p.split(".")[-1] for p in frame_rec["recompute"]}
        if _norm(filled, skip) == _norm(llm_parse, skip):
            frame_rec["status"] = "active"
            return True
        if frame_rec in self.frames:
            self.frames.remove(frame_rec)
        return False

    # ---- using ------------------------------------------------------------
    def match(self, question):
        """Try templates from most to least specific (most fixed words first). Return the first that
        fits in exactly one safe way: {parse, frame, fillers, typos, record}. If templates fit but
        every fit was unsafe, return {rejected, bad_fillers}. If none fit, None."""
        qw = _words(question)
        first_refusal = None
        order = sorted(self.frames, key=lambda f: -sum(len(p) for p in f["parts"] if isinstance(p, list)))
        for f in order:
            fits = _all_fills(qw, f["parts"], limit=10)
            if not fits:
                continue
            texts = [({k: question[qw[a][1]:qw[b - 1][2]] for k, (a, b) in fills.items()}, typos)
                     for fills, typos in fits]
            good = [(t, ty) for t, ty in texts if all(self._is_noun_phrase(v) for v in t.values())]
            if not good:
                first_refusal = first_refusal or {"rejected": f["frame"], "bad_fillers": list(texts[0][0].values())}
                continue
            if len({tuple(sorted(t.items())) for t, _ in good}) > 1:
                first_refusal = first_refusal or {"rejected": f["frame"], "bad_fillers": [f"ambiguous: {len(good)} ways to fill"]}
                continue
            fillers, typos = good[0]
            parse = _fill_template(f["template"], fillers)
            if f["recompute"]:
                parse["_recompute"] = list(f["recompute"])
            return {"parse": parse, "frame": f["frame"], "fillers": fillers, "typos": typos, "record": f}
        return first_refusal

    def _is_noun_phrase(self, text):
        """A blank may only hold a topic phrase: short, no verbs, no negation, no question mark."""
        if len(text.split()) > 10 or any(c in text for c in "?!;:"):
            return False
        for o, c in ("()", "[]", "{}", "“”", "‘’"):
            if text.count(o) != text.count(c):
                return False
        if text.count('"') % 2:
            return False
        if text.isupper():               # SHOUTED TEXT confuses the tagger; judge it in lowercase
            text = text.lower()
        doc = self.nlp(text)
        if any(t.dep_ == "neg" or t.lower_ in BLANK_STOP_WORDS for t in doc):
            return False
        # A clause needs a conjugated verb (reduces / reduce / reduced-as-past / is) or a modal (can, should).
        # Those mean the blank swallowed part of a sentence, not a topic -> reject.
        # "-ing" (VBG) and participle (VBN) words are fine: "foraging behavior", "created marshes".
        if any(t.tag_ in ("VBZ", "VBP", "VBD", "MD") for t in doc):
            return False
        # pronouns, wh-words and question words never belong in a topic
        if any(t.tag_ in ("PRP", "PRP$", "WDT", "WP", "WP$", "WRB") for t in doc):
            return False
        # A topic is a bare noun phrase. Determiners and quantifiers ("the", "a", "any", "zero"),
        # adverbs ("hardly") and particles never belong in one: they set scope or polarity
        # ("the opposite of X", "zero X", "hardly any X"). Send those to the LLM.
        if any(t.pos_ in ("DET", "NUM", "ADV", "PART", "PRON", "SCONJ") for t in doc):
            return False
        # the phrase has to end in a noun (its head): "foraging behavior", not "predation on".
        # A final "-ing" word counts as a noun here: "responses to warming", "stratification and mixing".
        return doc[-1].pos_ in ("NOUN", "PROPN") or doc[-1].tag_ == "VBG"

    # ---- persistence ------------------------------------------------------
    def save(self, path):
        json.dump([{k: f[k] for k in ("frame", "template", "recompute", "status")} for f in self.frames],
                  open(path, "w"), indent=1)

    def load(self, path):
        for f in json.load(open(path)):
            f["parts"] = _parts_from_frame(f["frame"])
            if not any(g["parts"] == f["parts"] for g in self.frames):
                self.frames.append(f)


# ---------------------------------------------------------------------------
# Word-level matching that ignores formatting and tolerates small typos
# ---------------------------------------------------------------------------
from wordfreq import zipf_frequency

# Tokens: signed / decimal numbers ("-5", "1.5", "10⁶"), words ("can't", "m²", "CO₂"), blank markers,
# and symbols that carry meaning (≥ ≤ < > = ± % ° ...). Everything else (commas, quotes, dashes
# between words, spaces, indents) is formatting and is ignored.
_SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺"
_SUB = "₀₁₂₃₄₅₆₇₈₉"
_MINUS = "−–—‒﹣－-"
_WORD = re.compile(
    rf"⟦[A-H]⟧"                                                        # template blank marker
    rf"|(?<![A-Za-z0-9.{_SUP}{_SUB}])[{_MINUS}]?\d+(?:[.,]\d+)*[{_SUP}]*"  # numbers, optional minus
    rf"|[A-Za-z0-9{_SUP}{_SUB}]+(?:['’][A-Za-z]+)?"                     # words
    rf"|[≥≤<>=±%°$€£×÷≈~]")                                             # meaningful symbols


MAX_TYPOS = 2               # at most this many misspelled fixed words per question
# Words whose meaning is too important to guess at: never accept a typo on or into these.
PROTECTED = {"not", "no", "never", "nor", "none", "without", "increase", "increases", "decrease",
             "decreases", "reduce", "reduces", "raise", "lower", "more", "less", "most", "least",
             "before", "after", "cause", "causes", "caused", "prevent", "prevents", "harm",
             "sustain", "restore", "remove", "add", "only", "all", "any", "both", "either",
             "higher", "lower", "stronger", "weaker", "positive", "negative", "affect", "effect"}


def _words(text):
    """[(token, start, end)]. Lowercased, EXCEPT short tokens with capitals such as 'Mg', 'P',
    'CO₂', 'DNA', whose case can carry meaning (magnesium vs milligram). A capitalised common
    word at the start of a sentence ('How', 'Do') is still lowercased, and SHOUTED text is
    lowercased entirely."""
    shouting = text.isupper()
    out, prev_end = [], 0
    for m in _WORD.finditer(text):
        w = m.group().replace("’", "'")
        if w[0] in _MINUS and len(w) > 1:
            w = "-" + w[1:]
        sentence_start = not out or re.search(r"[.?!:;]", text[prev_end:m.start()])
        keep_case = (not shouting and len(w.rstrip(_SUB + _SUP)) <= 3 and any(c.isupper() for c in w)
                     and not (sentence_start and w.istitle() and zipf_frequency(w.lower(), "en") >= 4))
        out.append((w if keep_case else w.lower(), m.start(), m.end()))
        prev_end = m.end()
    return out


def _edit1(a, b):
    """True if b is a at most one typo away (one letter added, dropped, changed, or two swapped)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return a == b
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


DOMAIN_WORDS = set()        # every word used in the graph; filled in by Pipeline()


def _typo_of(expected, got):
    """Accept `got` as a misspelling of `expected` only if it is clearly a typo:
      - one keystroke off, in a word of 5+ letters, with no digits in either word
      - the change is NOT in the first two letters (that is where meaning-flipping prefixes
        live: re-/de-, a-, un-, in-), so 'rewetting'->'dewetting' or 'synchrony'->'asynchrony' fail
      - neither word is protected
      - `got` is not a word at all: absent from the English frequency list (not merely rare,
        since rare scientific words like 'phosphite' ARE in it) and absent from the graph's own words
    """
    if len(expected) < 5 or expected in PROTECTED or got in PROTECTED:
        return False
    if any(c.isdigit() for c in expected + got) or not got.isalpha():
        return False
    if not _edit1(expected, got) or expected[:2] != got[:2]:
        return False
    return zipf_frequency(got, "en") == 0 and got not in DOMAIN_WORDS


def _all_fills(qw, parts, limit=10):
    """Every way the question's words fit a template, up to `limit`.
    Returns [(fills, typos)], fills = {blank: (first word index, end word index)}."""
    found = []
    def go(i, pos, fills, typos):
        if len(found) >= limit:
            return
        if i == len(parts):
            if pos == len(qw):
                found.append((dict(fills), list(typos)))
            return
        part = parts[i]
        if isinstance(part, list):                       # fixed words
            new = []
            for k, w in enumerate(part):
                if pos + k >= len(qw):
                    return
                got = qw[pos + k][0]
                if got != w:
                    if not _typo_of(w, got):
                        return
                    new.append((w, got))
            if len(typos) + len(new) > MAX_TYPOS:
                return
            go(i + 1, pos + len(part), fills, typos + new)
            return
        if part in fills:                                # repeated blank: same words again
            a, b = fills[part]
            n = b - a
            if [w for w, _, _ in qw[a:b]] == [w for w, _, _ in qw[pos:pos + n]]:
                go(i + 1, pos + n, fills, typos)
            return
        for end in range(pos + 1, len(qw) + 1):          # a blank holds one or more words
            fills[part] = (pos, end)
            go(i + 1, end, fills, typos)
            del fills[part]
    go(0, 0, {}, [])
    return found


def _walk_values(obj, keys):
    """Yield every string stored under one of `keys`, anywhere in a nested parse."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, str):
                yield v
            else:
                yield from _walk_values(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_values(v, keys)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, graph_path, llm, confirm=True):
        """llm: function(question) -> parse dict, or None if it can't answer.
        confirm: put each new template on probation until a second LLM answer agrees with it."""
        terms, contexts = vocab_from_graph(graph_path)
        g = json.load(open(graph_path))
        for n in g["nodes"]:
            DOMAIN_WORDS.update(w for w, _, _ in _words(" ".join(str(n.get(k) or "") for k in
                                ("name", "entity_term", "measured_attribute"))))
        self.rules = StrictParser(terms, contexts)
        self.frames = FrameCache(self.rules.nlp, confirm=confirm)
        self.exact = {}
        self.llm = llm
        self.counts = {"exact_cache": 0, "frame_cache": 0, "strict_rules": 0, "llm": 0, "llm_failed": 0,
                       "confirmed": 0, "dropped": 0}
        self.vocab_size = (len(terms), len(contexts))

    def parse(self, question):
        """Never raises for bad input or a failing API: returns a route saying what happened."""
        if isinstance(question, bytes):
            question = question.decode("utf-8", errors="replace")
        if not isinstance(question, str):
            return {"route": "invalid_input", "parse": None, "trace": [f"not text: {type(question).__name__}"]}
        if len(question) > MAX_QUESTION_CHARS:
            return {"route": "invalid_input", "parse": None, "trace": [f"longer than {MAX_QUESTION_CHARS} characters"]}
        q = " ".join(question.split())               # normalise whitespace
        key = " ".join(w for w, _, _ in _words(q))    # formatting-free key for the exact cache
        if not key:
            return {"route": "invalid_input", "parse": None, "trace": ["no words in question"]}
        r = self._parse(q, key)
        r["parse"] = copy.deepcopy(r["parse"])       # callers can't edit what is stored
        return r

    def _parse(self, q, key):
        trace = []

        # Layer 0 ---------------------------------------------------------
        if key in self.exact:
            self.counts["exact_cache"] += 1
            trace.append("L0 exact cache: HIT")
            return {"route": "exact_cache", "parse": self.exact[key], "trace": trace}
        trace.append("L0 exact cache: miss")

        # Layer 1 ---------------------------------------------------------
        hit = self.frames.match(q)
        pending = None
        if hit and "parse" in hit and hit["record"]["status"] == "pending":
            pending, hit = hit, None                  # on probation: still ask the LLM (below)
            trace.append(f"L1 frame cache: template on probation '{pending['frame']}', checking it against the LLM")
        if hit and "parse" in hit:
            self.counts["frame_cache"] += 1
            trace.append(f"L1 frame cache: HIT  frame='{hit['frame']}'  blanks={hit['fillers']}"
                         + (f"  typos fixed={hit['typos']}" if hit["typos"] else ""))
            self.exact[key] = copy.deepcopy(hit["parse"])
            return {"route": "frame_cache", "parse": hit["parse"], "trace": trace}
        if hit:
            trace.append(f"L1 frame cache: frame matched but blank rejected {hit['bad_fillers']}")
        elif pending:
            pass
        else:
            trace.append(f"L1 frame cache: miss ({len(self.frames.frames)} frames stored)")

        # Layer 2 ---------------------------------------------------------
        r = self.rules.parse(q) if not pending else {"route": "skip", "reasons": ["template on probation"]}
        if r["route"] == "rules":
            self.counts["strict_rules"] += 1
            parse = {"kind": "effect", "triples": [{"s": r["s"], "p": r["p"], "o": r["o"]}],
                     "context": r["context"]}
            trace.append(f"L2 strict rules: PASS -> {parse['triples'][0]}")
            self.exact[key] = copy.deepcopy(parse)
            return {"route": "strict_rules", "parse": parse, "trace": trace}
        trace.append(f"L2 strict rules: reject ({r['reasons'][0]})")

        # Layer 3 ---------------------------------------------------------
        try:
            parse = self.llm(q)
        except Exception as e:                       # timeout, 5xx, rate limit, network: don't crash
            self.counts["llm_failed"] += 1
            trace.append(f"L3 LLM: API ERROR {type(e).__name__}: {str(e)[:80]}"
                         + (" (template stays on probation)" if pending else ""))
            return {"route": "llm_failed", "parse": None, "trace": trace}
        if parse is not None and not isinstance(parse, dict):
            self.counts["llm_failed"] += 1
            trace.append(f"L3 LLM: unusable answer of type {type(parse).__name__}")
            return {"route": "llm_failed", "parse": None, "trace": trace}
        if parse is None:
            self.counts["llm_failed"] += 1
            trace.append("L3 LLM: CALL NEEDED")
            return {"route": "llm_needed", "parse": None, "trace": trace}
        self.counts["llm"] += 1
        trace.append("L3 LLM: called (1 call)")
        if pending:
            if self.frames.confirm_or_drop(pending["record"], pending["parse"], parse):
                self.counts["confirmed"] += 1
                trace.append("   template CONFIRMED: LLM agreed, it is now active")
            else:
                self.counts["dropped"] += 1
                trace.append("   template DROPPED: LLM disagreed with it")
        frame = self.frames.learn(q, parse)
        trace.append(f"   learned frame: '{frame}'" if frame else "   no frame learned")
        self.exact[key] = copy.deepcopy(parse)
        return {"route": "llm", "parse": parse, "trace": trace}
