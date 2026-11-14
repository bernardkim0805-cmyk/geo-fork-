"""Adversarial tests: every attack below CHANGES THE MEANING of a question but tries to
still match a learned template. A template answer for any of these is a failure.

Each case: (attack name, question that teaches the template, attacking question).
{T} is the topic blank. The template is learned with topic "soil carbon storage" and
attacked with topic "seed dispersal" (so the exact-question memory can't answer).
"""
from pipeline import Pipeline

CASES = [
 # --- one-keystroke prefix flips on rare scientific words (look like typos to a word list)
 ("a- prefix: synchrony -> asynchrony", "Does {T} increase population synchrony across sites?", "Does {T} increase population asynchrony across sites?"),
 ("re-/de-: rewetting -> dewetting", "How did peatland rewetting change {T}?", "How did peatland dewetting change {T}?"),
 ("re-/de-: revegetation -> devegetation", "Did revegetation of slopes change {T}?", "Did devegetation of slopes change {T}?"),
 ("re-/de-: rewilding -> dewilding", "How does rewilding influence {T}?", "How does dewilding influence {T}?"),
 ("re-/de-: recolonized -> decolonized", "Which sites recolonized after fire show higher {T}?", "Which sites decolonized after fire show higher {T}?"),
 ("a- prefix: seasonal -> aseasonal", "Is {T} seasonal in boreal lakes?", "Is {T} aseasonal in boreal lakes?"),
 ("chemistry: phosphate -> phosphite", "How does phosphate addition change {T}?", "How does phosphite addition change {T}?"),
 ("number: 2050s -> 2080s", "How will {T} change by the 2050s under warming?", "How will {T} change by the 2080s under warming?"),
 ("two flips at once (within 2-typo budget)", "How did rewetting and revegetation change {T}?", "How did dewetting and devegetation change {T}?"),
 # --- meaning carried by symbols that the formatting clean-up throws away
 ("symbol: >= vs <=", "How does {T} change at sites with pH ≥ 6?", "How does {T} change at sites with pH ≤ 6?"),
 ("symbol: > vs <", "How does {T} differ in patches > 5 ha?", "How does {T} differ in patches < 5 ha?"),
 ("symbol: minus sign dropped", "How does {T} change when winter temperatures stay above −5 °C?", "How does {T} change when winter temperatures stay above 5 °C?"),
 ("symbol: % dropped", "How does {T} change within 10% of the reference?", "How does {T} change within 10 of the reference?"),
 ("symbol: decimal 1.5 vs 15", "How does {T} change under 1.5 °C of warming?", "How does {T} change under 15 °C of warming?"),
 ("contraction: can vs can't", "Which interventions can restore {T}?", "Which interventions can't restore {T}?"),
 # --- meaning smuggled INTO the blank (the topic) where the typo rules never look
 ("blank: 'anything except'", "Which interventions are most likely to sustain or restore {T}, and how can their outcomes be evaluated?",
  "Which interventions are most likely to sustain or restore anything except seed dispersal, and how can their outcomes be evaluated?"),
 ("blank: 'everything but'", "Which interventions are most likely to sustain or restore {T}, and how can their outcomes be evaluated?",
  "Which interventions are most likely to sustain or restore everything but seed dispersal, and how can their outcomes be evaluated?"),
 ("blank: 'without'", "Which interventions are most likely to sustain or restore {T}, and how can their outcomes be evaluated?",
  "Which interventions are most likely to sustain or restore soil without seed dispersal, and how can their outcomes be evaluated?"),
 ("blank: 'the absence of'", "Which interventions are most likely to sustain or restore {T}, and how can their outcomes be evaluated?",
  "Which interventions are most likely to sustain or restore the absence of seed dispersal, and how can their outcomes be evaluated?"),
 ("blank: 'non-'", "Which interventions are most likely to sustain or restore {T}, and how can their outcomes be evaluated?",
  "Which interventions are most likely to sustain or restore non-seed dispersal, and how can their outcomes be evaluated?"),
 ("blank: 'other than'", "Which interventions are most likely to sustain or restore {T}, and how can their outcomes be evaluated?",
  "Which interventions are most likely to sustain or restore processes other than seed dispersal, and how can their outcomes be evaluated?"),
 # --- protected-word and real-word guards (expected to hold already)
 ("protected: sustain -> harm", "Which interventions are most likely to sustain {T}?", "Which interventions are most likely to harm {T}?"),
 ("real word: biotic -> abiotic", "Do biotic factors explain variation in {T}?", "Do abiotic factors explain variation in {T}?"),
 ("real word: reforestation -> deforestation", "Does reforestation alter {T} in catchments?", "Does deforestation alter {T} in catchments?"),
 ("real word: nitrate -> nitrite", "How does nitrate leaching relate to {T}?", "How does nitrite leaching relate to {T}?"),
 ("plural/tense: predators -> predator", "Do predators reduce {T}?", "Do predator reduce {T}?"),
]

def run(verbose=True):
    failures = 0
    for name, teach, attack in CASES:
        topic = {"t": "soil carbon storage"}
        P = Pipeline("data/graph.json", lambda q: {"kind": "x", "topic": topic["t"]}, confirm=False)
        P.parse(teach.replace("{T}", "soil carbon storage"))           # learn the template
        P.llm = lambda q: None
        r = P.parse(attack.replace("{T}", "seed dispersal"))
        fooled = r["route"] in ("frame_cache", "exact_cache")
        failures += fooled
        if verbose:
            how = r["trace"][1] if fooled else r["trace"][1][:70]
            print(f"{'FOOLED ' if fooled else 'safe   '} {name:42} {how[:110] if fooled else ''}")
    print(f"\n{failures} of {len(CASES)} attacks fooled the template layer")
    return failures

if __name__ == "__main__":
    run()
