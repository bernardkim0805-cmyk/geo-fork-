"""Round 2: new attacks against the fixed pipeline."""
import attack_test as A
A.CASES = [
 # characters the tokenizer silently drops
 ("superscript: m² vs m", "How does {T} change per m² of plot?", "How does {T} change per m of plot?"),
 ("superscript: 10⁶ vs 10", "How does {T} respond to 10⁶ seeds per hectare?", "How does {T} respond to 10 seeds per hectare?"),
 ("subscript: CO₂ vs CO", "How does elevated CO₂ affect {T}?", "How does elevated CO affect {T}?"),
 ("en-dash used as minus", "How does {T} change when temperatures stay above 5 °C?", "How does {T} change when temperatures stay above –5 °C?"),
 ("fullwidth minus", "How does {T} change when temperatures stay above 5 °C?", "How does {T} change when temperatures stay above －5 °C?"),
 # case carries meaning in chemistry / units
 ("case: Mg (magnesium) vs mg (milligram)", "How does Mg supply affect {T}?", "How does mg supply affect {T}?"),
 ("case: P (phosphorus) vs p", "Does P limitation affect {T}?", "Does p limitation affect {T}?"),
 # new blank smuggling words
 ("blank: 'zero'", "Which interventions can restore {T}?", "Which interventions can restore zero seed dispersal?"),
 ("blank: 'hardly any'", "Which interventions can restore {T}?", "Which interventions can restore hardly any seed dispersal?"),
 ("blank: 'the opposite of'", "Which interventions can restore {T}?", "Which interventions can restore the opposite of seed dispersal?"),
 ("blank: 'a deficit of'", "Which interventions can restore {T}?", "Which interventions can restore a deficit of seed dispersal?"),
 ("blank: '-free'", "Which interventions can restore {T}?", "Which interventions can restore seed dispersal-free sites?"),
 # typo rule: change after the 2nd letter that still flips meaning
 ("prefix guard: upregulated -> unregulated", "Is {T} upregulated after burning?", "Is {T} unregulated after burning?"),
 ("suffix: tolerant -> tolerent (harmless)", "Which species are tolerant of {T}?", "Which species are tolerent of {T}?"),
 ("word order swap", "Does fire suppression increase {T} more than grazing?", "Does grazing suppression increase {T} more than fire?"),
 ("british spelling (harmless, costs a call)", "How does {T} affect foraging behavior?", "How does {T} affect foraging behaviour?"),
]
HARMLESS = {"suffix: tolerant -> tolerent (harmless)", "british spelling (harmless, costs a call)"}
fails = 0
for name, teach, attack in A.CASES:
    from pipeline import Pipeline
    P = Pipeline("data/graph.json", lambda q: {"kind": "x", "topic": "soil carbon storage"}, confirm=False)
    P.parse(teach.replace("{T}", "soil carbon storage")); P.llm = lambda q: None
    r = P.parse(attack.replace("{T}", "seed dispersal"))
    hit = r["route"] in ("frame_cache", "exact_cache")
    bad = hit and name not in HARMLESS
    fails += bad
    tag = "FOOLED " if bad else ("free   " if hit else "to LLM ")
    print(f"{tag} {name:42} {r['parse'].get('topic') if hit else ''}")
print(f"\n{fails} meaning-changing attacks fooled it")
