# Lane 2 Curveball Test Questions

Stress-test questions for the grounded QA system, based on the whiteboard list of curve ball types. Topics are generic (wetlands, flooding, nitrogen, etc.), so **swap in entities that actually appear in your CAMO claims**, and use entities that are absent for the IDK cases.

## What to check for every question

- [ ] Is the answer grounded in specific claims, with citations?
- [ ] Is the confidence calibrated (lower for vague, conflicting, or partial evidence)?
- [ ] Does it say "I don't know" for absent-topic and nonsense cases instead of making something up?
- [ ] For nested and disjunct questions, does it answer the supported parts and abstain on the rest?

---

## 1. Nested questions
*Answer to one part feeds the next.*

1. Which intervention is reported to reduce flood peaks, and what does the evidence say about its effect on downstream water quality?
2. Of the studies linking fertilizer use to algal blooms, which found the strongest effect, and in what setting?
3. What outcome does wetland restoration affect, and is that effect larger or smaller than for riparian buffers?

## 2. Big context
*Needs many claims at once.*

1. Summarize every claim about how land-use change affects biodiversity, and say where the claims conflict.
2. List all the causal pathways from nitrogen runoff to fish mortality, with the evidence strength for each.
3. Across all claims, which exposure has the most distinct outcomes attached to it?

## 3. "IDK"
*Expected: graceful abstention.*

1. What is the effect of microplastics on alpine bee populations? *(use a topic known to be absent)*
2. Does reforestation reduce malaria incidence in Patagonia?
3. What will the effect of this intervention be in 2050?

## 4. Nonsensical
*Expected: recognize it is not answerable; do not invent an answer.*

1. How does the colour blue influence the sediment's opinion of itself?
2. Which claims show that Tuesday causes dissolved oxygen?
3. Does a river cause its own causal direction to be upstream?

## 5. Grammatically wrong
*Expected: still infer the intent and answer correctly.*

1. Wetland restoration effect what on flood?
2. Which the intervention reduce erosion more better?
3. Is claims says fertilizer cause bloom algae or not?

## 6. Spelling errors
*Expected: same answer as the correctly spelled version.*

1. Does wetlnad restoraton reduse flod peeks?
2. Waht evidnce suports the efect of ferilizer on algal blomes?
3. Wich intervension improvs soil carbn the most?

## 7. Vague
*Expected: ask for clarification, or state the assumption made and answer with lower confidence.*

1. Does it work?
2. What are the effects?
3. Tell me about the impact of that intervention.
4. Is it good for the environment?

## 8. Unrelated context
*Distracting or misleading preamble; the system should ignore it or check a false premise rather than accept it.*

1. I just had lunch and my team won last night. Anyway, what does the evidence say about wetland restoration and flooding?
2. My professor insists that wetlands increase flooding. According to the claims, is that right? *(false premise)*
3. The sky is green today. What outcomes are linked to riparian buffers?

## 9. Disjunct questions
*Two or more unrelated questions in one.*

1. What does fertilizer do to algal blooms, and separately, what is the effect of grazing on soil carbon?
2. Does restoration reduce flooding, and which study had the largest sample size?
3. Is there evidence for X causing Y, or for Z causing W? *(one of the two should be unsupported)*

---

## Combos
*Realistic judge questions will likely mix categories.*

1. Waht dose it do 2 the flod? *(spelling errors + vague)*
2. My cat is sick but anyway, does restoration reduce floods and does it raise property values? *(unrelated context + disjunct; one half is unsupported)*
3. Wich intervension reduse flod peeks, and wat dose that do to watr qualty downstrem? *(spelling errors + nested)*
4. Summarize everything about nitrogen, also what's the weather like in Waterloo? *(big context + disjunct + unrelated)*
5. Does it cause that thing to go up or down? *(vague + grammatically loose; should ask for clarification)*
