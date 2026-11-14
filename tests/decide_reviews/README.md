# Blind reviews of tests/decide_cases.yaml (task A3)

Three subagents labeled every case independently. They saw only the question and
the edge_ids (`blind_cases.yaml` shape), never the draft labels or decide's rules.

Result vs. the draft labels: 24/31 agreed on status + direction with certainty
within one level. The 7 disagreements (DC-02, 06, 11, 16, 19, 30, 31) were settled
by the user (task A4); see `decided_by` in the case file. Edges that 2 or 3
reviewers said don't bear on the question were removed from the inputs of
DC-03, DC-12 and DC-13.
