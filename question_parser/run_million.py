"""Run the stress test on 1,000,000 generated questions across all CPU cores."""
import random, sys, time, json
from collections import Counter
from multiprocessing import Pool, cpu_count

TOTAL, CHUNK = 1_000_000, 10_000

def work(seed):
    import stress_test_v2 as S
    from parser_v2 import StrictParser
    random.seed(seed)
    P = StrictParser(S.KNOWN_I + S.KNOWN_O, S.KNOWN_C)
    c, fps, uniq = Counter(), [], set()
    for _ in range(CHUNK):
        name, g = S.gen()
        uniq.add(g["q"])
        r = P.parse(g["q"])
        c["n"] += 1
        if r["route"] == "rules":
            c["rules"] += 1
            if S.is_correct(r, g): c["ok"] += 1
            else: c["wrong"] += 1; fps.append((g["q"], r))
            c[f"rules:{name}"] += 1
        else:
            c["llm"] += 1
    return c, fps, uniq

if __name__ == "__main__":
    t0 = time.time(); tot = Counter(); fps = []; uniq = set()
    with Pool(cpu_count()) as pool:
        for i, (c, f, u) in enumerate(pool.imap_unordered(work, range(1000, 1000 + TOTAL // CHUNK))):
            tot.update(c); fps += f; uniq |= u
            if (i + 1) % 10 == 0:
                print(f"{tot['n']:>9,} done  rules={tot['rules']:,} wrong={tot['wrong']}  {time.time()-t0:.0f}s", flush=True)
    print(json.dumps({"n": tot["n"], "unique_questions": len(uniq), "rules": tot["rules"],
                      "rules_correct": tot["ok"], "false_passes": tot["wrong"], "llm": tot["llm"],
                      "seconds": round(time.time() - t0)}, indent=1))
    for q, r in fps[:20]: print("FALSE PASS:", q, r)
