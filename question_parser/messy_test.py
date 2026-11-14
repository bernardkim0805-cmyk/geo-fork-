"""Mess up every question in the varied bank and check where each version ends up.

SHOULD be answered from a template (with the right topic):
  format      capitals, commas removed, curly quotes, dashes, line breaks and indents
  typo1       one misspelled fixed word            ("intervetions")
  typo2       two misspelled fixed words
  topic_typo  misspelled word inside the topic     (answer should keep the user's spelling)
MUST go to the LLM (a template answer would be wrong):
  typo3       three misspelled fixed words (over the typo budget)
  real_word   a fixed word swapped for a different real word ("evaluated" -> "estimated")
  not         the word "not" inserted
  dropped     one fixed word deleted
"""
import random, re, json, collections
from wordfreq import zipf_frequency
from pipeline import Pipeline, PROTECTED, _words
import run_varied as RV     # learns the 133 templates from the varied bank (prints its summary)

random.seed(7)
P = RV.P
P.llm = lambda q: None      # from now on, any LLM call is just counted
P.exact = {}                # force everything through the template layer

SWAPS = ["estimated", "ignored", "reported", "selected", "removed", "studied", "measured",
         "rarely", "always", "largest", "smallest", "early", "late", "plants", "animals", "soil"]

def typo(w):
    for _ in range(50):
        i = random.randrange(1, len(w) - 1)
        kind = random.choice(["swap", "drop", "double", "replace"])
        if kind == "swap": t = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        elif kind == "drop": t = w[:i] + w[i + 1:]
        elif kind == "double": t = w[:i] + w[i] + w[i:]
        else: t = w[:i] + random.choice("aeioustrnl") + w[i + 1:]
        if t != w and zipf_frequency(t, "en") < 2.0:
            return t
    return None

def fixed_words(q, topic):
    """(start, end) of fixed words outside the topic, long enough to typo, not protected."""
    spans = [(m.start(), m.end()) for m in re.finditer(re.escape(topic), q)]
    out = []
    for w, a, b in _words(q):
        if any(s <= a < e for s, e in spans): continue
        out.append((a, b, w))
    return out

def replace_at(q, a, b, new): return q[:a] + new + q[b:]

def variants(q, topic):
    v = {}
    v["format"] = random.choice([
        lambda s: s.lower(), lambda s: s.replace(",", ""), lambda s: s.replace("“", '"').replace("”", '"'),
        lambda s: "\n    " + s.replace(" and ", "\n\tand ", 1) + "  ", lambda s: s.replace("—", " - "),
        lambda s: s.upper()])(q)
    fw = [x for x in fixed_words(q, topic) if len(x[2]) >= 5 and x[2] not in PROTECTED]
    random.shuffle(fw)
    def typo_n(n):
        s, picks = q, sorted(fw[:n], key=lambda x: -x[0])
        if len(picks) < n: return None
        for a, b, w in picks:
            t = typo(q[a:b].lower())
            if not t: return None
            s = replace_at(s, a, b, t)
        return s
    v["typo1"], v["typo2"], v["typo3"] = typo_n(1), typo_n(2), typo_n(3)
    tw = [m for m in re.finditer(r"[A-Za-z]{5,}", topic)]
    if tw:
        m = random.choice(tw); t = typo(m.group().lower())
        if t:
            new_topic = topic[:m.start()] + t + topic[m.end():]
            v["topic_typo"] = (q.replace(topic, new_topic), new_topic)
    allw = fixed_words(q, topic)
    if allw:
        a, b, w = random.choice(allw)
        v["real_word"] = replace_at(q, a, b, random.choice([s for s in SWAPS if s != w]))
        a, b, w = random.choice(allw)
        v["dropped"] = (q[:a] + q[b:]).replace("  ", " ")
        a, b, w = random.choice(allw)
        v["not"] = q[:a] + "not " + q[a:]
    return v

SHOULD_HIT = {"format", "typo1", "typo2", "topic_typo"}
stats = collections.defaultdict(collections.Counter)
wrong = []
for i, q in enumerate(RV.qs):
    topic = RV.TOPICS[i // 10]
    for kind, val in variants(q, topic).items():
        if val is None: continue
        mq, expect = (val if isinstance(val, tuple) else (val, topic))
        r = P.parse(mq)
        hit = r["route"] in ("frame_cache", "exact_cache")
        s = stats[kind]; s["n"] += 1
        if hit:
            got = r["parse"].get("topic")
            if kind in SHOULD_HIT and [w.lower() for w, _, _ in _words(got or "")] == [w.lower() for w, _, _ in _words(expect)]: s["template, correct"] += 1
            else:
                s["template, WRONG"] += 1; wrong.append((kind, mq, got))
        else:
            s["sent to LLM"] += 1

print(f"\n{'variant':12}{'n':>6}{'template, correct':>19}{'sent to LLM':>13}{'WRONG':>8}")
for k in ["format", "typo1", "typo2", "topic_typo", "typo3", "real_word", "not", "dropped"]:
    s = stats[k]
    print(f"{k:12}{s['n']:>6}{s['template, correct']:>19}{s['sent to LLM']:>13}{s['template, WRONG']:>8}")
print("\nwrong answers:", len(wrong))
for w in wrong[:15]: print("  ", w)
