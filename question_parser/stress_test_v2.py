"""10,000-case stress test for parser_v2.

Each generated question carries a gold parse. A question that needs more than
(s, p, o, context) to be understood correctly is marked extra=True, so ANY
rules-route answer for it counts as a false pass.

Pass criterion: zero false passes (a wrong parse routed to rules)."""
import random, sys, time
from collections import Counter, defaultdict
from parser_v2 import StrictParser

random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

KNOWN_I = ["prescribed burning", "deer exclusion fencing", "herbicide application",
    "peatland rewetting", "moss layer transfer", "tree planting", "invasive shrub removal",
    "partial harvesting", "clearcutting", "riparian buffer retention", "mulching",
    "native seed addition", "grazing exclusion", "soil decompaction", "dam removal",
    "wetland creation", "cattle grazing", "nutrient addition", "topsoil removal",
    "beaver reintroduction", "thinning", "salvage logging", "liming", "dredging",
    "mycorrhizal inoculation"]
KNOWN_O = ["native plant richness", "seedling survival", "bird abundance",
    "soil carbon", "methane emissions", "water table depth", "invasive plant cover",
    "pollinator diversity", "tree growth", "amphibian abundance", "soil nitrogen",
    "sphagnum cover", "fish abundance", "litter decomposition", "conifer regeneration",
    "understory cover", "insect diversity", "sediment load", "shrub density",
    "species richness", "canopy cover", "microbial biomass", "nitrate leaching",
    "forb abundance", "beetle diversity"]
# Terms NOT in the merge vocabulary (simulates things the graph doesn't contain)
UNKNOWN_I = ["goat browsing", "biochar amendment", "snag creation", "road decommissioning",
    "hydroseeding", "log placement", "fire suppression", "seed drilling"]
UNKNOWN_O = ["lichen cover", "bat activity", "root biomass", "soil respiration",
    "moose browsing pressure", "earthworm density", "water clarity", "fungal diversity"]
KNOWN_C = ["ontario tallgrass prairies", "canadian boreal forests", "alberta peatlands",
    "restored wetlands", "quebec forests", "coastal salt marshes",
    "british columbia grasslands", "urban parks"]
UNKNOWN_C = ["manitoba fens", "arctic tundra", "nova scotia barrens"]
VERBS = [("increase", "increases", "increased"), ("reduce", "reduces", "reduced"),
         ("affect", "affects", "affected"), ("improve", "improves", "improved"),
         ("decrease", "decreases", "decreased"), ("influence", "influences", "influenced"),
         ("limit", "limits", "limited"), ("promote", "promotes", "promoted"),
         ("alter", "alters", "altered"), ("enhance", "enhances", "enhanced")]

def cap(s):  # random capitalisation noise on terms
    r = random.random()
    return s.title() if r < 0.15 else (s.capitalize() if r < 0.3 else s)

def G(q, s, p, o, ctx=None, extra=False, neg=False):
    return {"q": q, "s": s, "p": p, "o": o, "ctx": sorted([ctx] if ctx else []),
            "extra": extra, "neg": neg}

def gen():
    I, O, I2, O2 = (random.choice(KNOWN_I), random.choice(KNOWN_O),
                    random.choice(KNOWN_I), random.choice(KNOWN_O))
    C = random.choice(KNOWN_C) if random.random() < 0.6 else None
    vb, v3, vd = random.choice(VERBS)
    ci = f" in {cap(C)}" if C else ""
    T = {
     # ---- plain (rules may take these)
     "does":        lambda: G(f"Does {cap(I)} {vb} {cap(O)}{ci}?", I, vb, O, C),
     "how_does":    lambda: G(f"How does {cap(I)} {vb} {cap(O)}{ci}?", I, vb, O, C),
     "can":         lambda: G(f"Can {cap(I)} {vb} {cap(O)}{ci}?", I, vb, O, C),
     "which":       lambda: G(f"Which {random.choice(['interventions','practices','treatments','actions'])} {vb} {cap(O)}{ci}?", "?", vb, O, C),
     "what":        lambda: G(f"What {v3} {cap(O)}{ci}?", "?", vb, O, C),
     "ctx_first":   lambda: G(f"In {cap(C)}, does {cap(I)} {vb} {cap(O)}?", I, vb, O, C) if C
                            else G(f"Does {cap(I)} {vb} {cap(O)}?", I, vb, O),
     "unknown_term": lambda: (lambda ui, uo: G(f"Does {ui} {vb} {uo}{ci}?", ui, vb, uo, C))(
                            random.choice(UNKNOWN_I + [I]), random.choice(UNKNOWN_O + [O])),
     # ---- tricky (structure differs, or meaning needs more than s/p/o/ctx)
     "passive":     lambda: G(f"Is {cap(O)} {vd} by {cap(I)}{ci}?", I, vb, O, C),
     "effect_of":   lambda: G(f"What is the effect of {cap(I)} on {cap(O)}{ci}?", I, "affect", O, C),
     "pronoun":     lambda: G(f"When {cap(I)} is used{ci}, does it {vb} {cap(O)}?", I, vb, O, C),
     "lead_to":     lambda: G(f"Does {cap(I)} lead to higher {cap(O)}{ci}?", I, "increase", O, C),
     "what_happens": lambda: G(f"What happens to {cap(O)} after {cap(I)}{ci}?", I, "affect", O, C),
     "negated":     lambda: G(f"Does {cap(I)} not {vb} {cap(O)}{ci}?", I, vb, O, C, neg=True),
     "fail_to":     lambda: G(f"Does {cap(I)} fail to {vb} {cap(O)}{ci}?", I, vb, O, C, neg=True),
     "coord_obj":   lambda: G(f"Does {cap(I)} {vb} {cap(O)} and {cap(O2)}{ci}?", I, vb, O, C, extra=True),
     "coord_subj":  lambda: G(f"Does {cap(I)} or {cap(I2)} {vb} {cap(O)}{ci}?", I, vb, O, C, extra=True),
     "adverb":      lambda: G(f"Does {cap(I)} {random.choice(['significantly','strongly','still','really'])} {vb} {cap(O)}{ci}?", I, vb, O, C, extra=True),
     "time":        lambda: G(f"Does {cap(I)} {vb} {cap(O)} over the long term{ci}?", I, vb, O, C, extra=True),
     "compare":     lambda: G(f"Does {cap(I)} {vb} {cap(O)} more than {cap(I2)}{ci}?", I, vb, O, C, extra=True),
     "conditional": lambda: G(f"Does {cap(I)} {vb} {cap(O)} when {cap(I2)} is applied?", I, vb, O, extra=True),
     "two_clauses": lambda: G(f"Does {cap(I)} {vb} {cap(O)}, and does it {vb} {cap(O2)}?", I, vb, O, extra=True),
     "unknown_ctx": lambda: G(f"Does {cap(I)} {vb} {cap(O)} in {random.choice(UNKNOWN_C)}?", I, vb, O, extra=True),
     "two_ctx":     lambda: G(f"Does {cap(I)} {vb} {cap(O)} in {KNOWN_C[0]} and {KNOWN_C[1]}?", I, vb, O, extra=True),
     "modifier":    lambda: G(f"Does {random.choice(['early','repeated','intensive','annual'])} {cap(I)} {vb} {cap(O)}{ci}?", I, vb, O, C, extra=True),
     "determiner":  lambda: G(f"Does {cap(I)} {vb} the {cap(O)}{ci}?", I, vb, O, C),
     "help_recover": lambda: G(f"Does {cap(I)} help {cap(O)} recover{ci}?", I, "increase", O, C),
     "no_qmark":    lambda: G(f"Does {cap(I)} {vb} {cap(O)}{ci}", I, vb, O, C),
     "could":       lambda: G(f"Could {cap(I)} {vb} {cap(O)}{ci}?", I, vb, O, C),
     "lowercase_q": lambda: G(f"does {I} {vb} {O}{ci.lower()}?", I, vb, O, C),
    }
    name = random.choice(list(T))
    return name, T[name]()

def is_correct(r, g):
    return (not g["extra"] and not g["neg"] and r["s"] == g["s"].lower() and r["p"] == g["p"]
            and r["o"] == g["o"].lower() and r["context"] == g["ctx"] and r["negated"] == g["neg"])

if __name__ == '__main__':
    P = StrictParser(KNOWN_I + KNOWN_O, KNOWN_C)
    stats, fps, t0 = defaultdict(Counter), [], time.time()
    for _ in range(N):
        name, g = gen()
        r = P.parse(g["q"])
        st = stats[name]; st["n"] += 1
        if r["route"] == "rules":
            st["rules"] += 1
            if is_correct(r, g): st["ok"] += 1
            else:
                st["WRONG"] += 1; fps.append((name, g, r))
        else:
            st["llm"] += 1

    tot = Counter()
    print(f"{'template':14}{'n':>6}{'rules':>7}{'ok':>7}{'WRONG':>7}{'llm':>7}")
    for k in sorted(stats, key=lambda k: -stats[k]['rules']):
        s = stats[k]; tot.update(s)
        print(f"{k:14}{s['n']:>6}{s['rules']:>7}{s['ok']:>7}{s['WRONG']:>7}{s['llm']:>7}")
    print(f"{'TOTAL':14}{tot['n']:>6}{tot['rules']:>7}{tot['ok']:>7}{tot['WRONG']:>7}{tot['llm']:>7}")
    print(f"\nfalse passes: {len(fps)}   ({time.time()-t0:.1f}s)")
    for name, g, r in fps[:15]:
        print(f"  [{name}] {g['q']}\n     got {r}\n     gold s={g['s']} p={g['p']} o={g['o']} ctx={g['ctx']} extra={g['extra']} neg={g['neg']}")
