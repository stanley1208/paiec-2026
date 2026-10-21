"""Apply the pre-registered rule to the LLM difficulty ratings (see rate.py)."""
import json
import random
from pathlib import Path

from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sample = {s["item_id"]: s for s in json.loads((HERE / "sample.json").read_text(encoding="utf-8"))}
ratings = [json.loads(line) for line in (HERE / "ratings.jsonl").read_text(encoding="utf-8").splitlines()]
spent = sum(r["usd"] for r in ratings)
missing = [r for r in ratings if r["rating"] is None]
print(f"{len(ratings)} rated, {len(missing)} without a rating, cost ${spent:.2f}")

results = {}
for bench in ("matharena", "multi_swebench", "real_webagents", "researchcodebench"):
    pairs = [(r["rating"], sample[r["item_id"]]["b"]) for r in ratings
             if r["benchmark"] == bench and r["rating"] is not None]
    rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    rng = random.Random(0)
    boot = []
    for _ in range(2000):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        boot.append(spearmanr([p[0] for p in draw], [p[1] for p in draw]).statistic)
    boot.sort()
    spread = {k: sum(p[0] == k for p in pairs) for k in range(1, 11)}
    results[bench] = rho
    print(f"{bench:18s} n={len(pairs):2d} Spearman {rho:+.3f} (95% bootstrap {boot[50]:+.2f} to {boot[1949]:+.2f})"
          f" | ratings used: { {k: v for k, v in spread.items() if v} }")

passed = sum(rho >= 0.30 for rho in results.values())
go = passed >= 3 and all(rho > 0 for rho in results.values())
print(f"benchmarks with Spearman >= 0.30: {passed} of 4; all positive: {all(r > 0 for r in results.values())}"
      f" -> {'GO' if go else 'NO-GO'}")
