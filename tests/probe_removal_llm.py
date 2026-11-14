"""Live check of decide's D2 LLM call (needs BRINE_* in env or .env).

    .venv/bin/python -m tests.probe_removal_llm

Each probe: (question intervention, removal/absence edge subject, expected).
The first 10 shaped the prompt; the last 6 are held out.
Last run 2026-11-14: 14/16 (held-out 4/6); word list alone 8/10 on the first 10.
"""
from lane2 import llm
from lane2.decide import _llm_opposite, _regex_opposite

PROBES = [
    ("weed removal", "Weed removal", "SAME"),
    ("competing vegetation", "Weed removal", "OPPOSITE"),
    ("tree shelters", "absence of shelter treatment", "OPPOSITE"),
    ("moose browsing exclusion", "Release from moose browsing", "SAME"),
    ("weed control", "Weed removal", "SAME"),
    ("erosion control", "absence of erosion control", "OPPOSITE"),
    ("fertilization", "absence of fertilization", "OPPOSITE"),
    ("thinning", "removal of trees", "SAME"),
    ("canopy cover", "removal of trees", "OPPOSITE"),
    ("grazing", "removal of cattle", "OPPOSITE"),
    # held out
    ("fire suppression", "absence of fire", "SAME"),
    ("prescribed burning", "absence of fire", "OPPOSITE"),
    ("deer fencing", "removal of deer", "SAME"),
    ("beaver activity", "beaver removal", "OPPOSITE"),
    ("invasive species control", "eradication of invasive plants", "SAME"),
    ("shrub cover", "shrub removal", "OPPOSITE"),
]


def main() -> int:
    if not llm.configured():
        print("BRINE_* not configured (env or .env)")
        return 1
    ok = rx_ok = 0
    for x, s, exp in PROBES:
        got = "OPPOSITE" if _llm_opposite(x, s) else "SAME"
        rx = "OPPOSITE" if _regex_opposite(x) else "SAME"
        ok += got == exp
        rx_ok += rx == exp
        print(f"{'ok   ' if got == exp else 'WRONG'} {x!r:28} vs {s!r:34} expect {exp:8} llm {got:8} wordlist {rx}")
    print(f"LLM {ok}/{len(PROBES)}   word list {rx_ok}/{len(PROBES)}   tokens {llm.USAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
