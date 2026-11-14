"""Strict rule parser for causal questions.

Pipeline:
  1. Known multi-word terms (graph node names) and context phrases are swapped
     for one-word placeholders BEFORE the dependency parse, so spaCy can't split
     or mis-tag them; the originals are mapped back afterwards.
  2. Dependency parse -> (subject, predicate, object, context).
  3. Strict gate: the parse is accepted ONLY if every token in the question is
     accounted for by an allowed role. Anything else -> send to the LLM.

parse(question) -> {"route": "rules", "s":..., "p":..., "o":..., "context":..., }
                or {"route": "llm", "reasons": [...]}
"""
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.util import filter_spans

CAUSAL_VERBS = {"affect", "reduce", "increase", "control", "maintain", "limit", "cause",
                "influence", "improve", "decrease", "drive", "promote", "suppress",
                "enhance", "alter", "determine", "prevent"}
WH_SUBJ_NOUNS = {"intervention", "practice", "action", "treatment", "factor", "management"}
AUX_OK = {"do", "can", "will", "would"}


class StrictParser:
    def __init__(self, terms, contexts):
        """terms: known entity/attribute phrases (e.g. graph node names).
        contexts: known geography/ecosystem phrases (e.g. 'Ontario tallgrass prairies')."""
        self.nlp = spacy.load("en_core_web_sm", exclude=["ner"])
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher.add("TERM", [self.nlp.make_doc(t) for t in terms])
        self.matcher.add("CTX", [self.nlp.make_doc(c) for c in contexts])

    def _delex(self, q):
        """Replace known terms with a placeholder noun ('soil') and contexts with
        'Canada' so the tagger/parser sees simple, familiar words. Returns the parsed
        doc and {token_index: (kind, original_text)}."""
        raw = self.nlp.make_doc(q)
        spans = [Span(raw, s, e, label=m) for m, s, e in self.matcher(raw)]
        spans = sorted(filter_spans(spans), key=lambda sp: sp.start_char)
        out, pos, marks = [], 0, []
        for sp in spans:
            out.append(q[pos:sp.start_char])
            ph = "soil" if sp.label_ == "TERM" else "Canada"
            start = sum(len(x) for x in out)
            out.append(ph)
            marks.append((start, start + len(ph), sp.label_, sp.text.lower()))
            pos = sp.end_char
        out.append(q[pos:])
        doc = self.nlp("".join(out))
        kind = {}
        for a, b, label, text in marks:
            span = doc.char_span(a, b)
            if span is None or len(span) != 1:
                return doc, None
            kind[span[0].i] = (label, text)
        return doc, kind

    def parse(self, q):
        doc, kind = self._delex(q.strip())
        if kind is None:
            return {"route": "llm", "reasons": ["term alignment failed"]}
        self._kind = kind
        why = []
        used = set()  # token indices accounted for

        if len(list(doc.sents)) != 1 or doc[-1].text != "?":
            return {"route": "llm", "reasons": ["not a single question"]}

        verbs = [t for t in doc if t.i not in kind and (t.pos_ == "VERB" or (t.pos_ == "AUX" and t.dep_ == "ROOT"))]
        main = [t for t in verbs if t.lemma_.lower() in CAUSAL_VERBS]
        if len(verbs) != 1 or len(main) != 1:
            return {"route": "llm", "reasons": [f"need exactly one verb and it must be causal (verbs={[v.text for v in verbs]})"]}
        v = main[0]
        used.add(v.i)

        subj = [c for c in v.children if c.dep_ == "nsubj"]
        obj = [c for c in v.children if c.dep_ == "dobj"]
        if len(subj) != 1 or len(obj) != 1:
            return {"route": "llm", "reasons": ["need exactly one subject and one object"]}

        # --- subject: one known TERM token, or a wh-phrase ("what" / "which interventions")
        s_tok = subj[0]
        s_sub = list(s_tok.subtree)
        if kind.get(s_tok.i, ("",))[0] == "TERM" and len(s_sub) == 1:
            s_val = kind[s_tok.i][1]
        elif len(s_sub) == 1 and s_tok.lower_ == "what":
            s_val = "?"
        elif (len(s_sub) == 2 and s_sub[0].lower_ in ("which", "what")
              and s_tok.lemma_.lower() in WH_SUBJ_NOUNS):
            s_val = "?"
        else:
            why.append(f"subject '{' '.join(t.text for t in s_sub)}' is not a single known term or wh-phrase")
            s_val = None
        used.update(t.i for t in s_sub)

        # --- object: one known TERM token, optionally with 'in <CTX>'
        o_tok = obj[0]
        context = []
        if kind.get(o_tok.i, ("",))[0] != "TERM":
            why.append(f"object '{o_tok.text}' is not a known term")
        used.add(o_tok.i)
        for c in o_tok.children:
            if not self._take_context(c, used, context):
                why.append(f"unexpected modifier '{c.text}' on object")

        # --- other verb children: aux, wh 'how', context prep, punctuation
        for c in v.children:
            if c.i in used or c in subj or c in obj:
                continue
            if c.dep_ == "aux" and c.lemma_.lower() in AUX_OK:
                used.add(c.i)
            elif c.dep_ == "advmod" and c.lower_ == "how":
                used.add(c.i)
            elif c.dep_ == "punct":
                used.add(c.i)
            elif self._take_context(c, used, context):
                pass
            else:
                why.append(f"unexpected word '{c.text}' ({c.dep_})")

        # --- full accounting: every token must have a role
        leftover = [t.text for t in doc if t.i not in used]
        if leftover:
            why.append(f"unaccounted words {leftover}")

        if why:
            return {"route": "llm", "reasons": why}
        return {"route": "rules", "s": s_val, "p": v.lemma_.lower(), "o": kind.get(o_tok.i, ("", None))[1],
                "context": sorted(context), "negated": False}

    def _take_context(self, tok, used, context):
        """Accept 'in <CTX>' (optionally followed by a comma). Returns True if consumed."""
        if tok.dep_ == "prep" and tok.lower_ == "in":
            kids = list(tok.children)
            if len(kids) == 1 and kids[0].dep_ == "pobj" and self._kind.get(kids[0].i, ("",))[0] == "CTX" \
               and not list(kids[0].children):
                used.update([tok.i, kids[0].i])
                context.append(self._kind[kids[0].i][1])
                return True
        if tok.dep_ == "punct" and tok.text == ",":
            used.add(tok.i)
            return True
        return False
