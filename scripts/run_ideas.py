"""Do the agent rule and question groups improve version 2 offline?

    python scripts/run_ideas.py

Decision rule, fixed before the first run (2026-10-06). Each idea is compared with version 2
(item_pooling with default settings) on the same pairs:
  agent rule:      agent_scale = 0 (no metadata nudge for subjects with a harness)
  question groups: group_sd = 1.0 (subject-specific effect per item group)
An idea goes into version 3 only if
  (a) on hidden-sized pairs (pool_items 400, min_pair_items 60) with one partner pair,
      pair-mean ALC improves by more than 0.0003 and by more than 2 standard errors
      (cluster-robust, clusters = subject-benchmark pairs);
  (b) on full-size pairs (the default config, own labels only) it also improves;
  (c) no benchmark gets worse by more than 0.001 in setting (a).
The agent rule can only act on Multi-SWE-bench, the one public benchmark with harness
subjects. "both" is reported for information; version 3 takes the ideas that pass.
Writes results/runs/<timestamp>_ideas/ with config.json (provenance, written first),
per_pair.csv and summary.json.
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_harness import provenance, write  # noqa: E402
from run_pooling import _mean, _paired  # noqa: E402

from paiec import data, protocol, scoring  # noqa: E402
from paiec.predictors import PREDICTORS  # noqa: E402

SEEDS = (0, 1, 2)
SETTINGS = {"hidden": {"pool_items": 400, "min_pair_items": 60, "partners": 1}, "full": {}}
VARIANTS = {"version2": {}, "agent": {"agent_scale": 0.0}, "groups": {"group_sd": 1.0},
            "both": {"agent_scale": 0.0, "group_sd": 1.0}}
IDEAS = ("agent", "groups")

_dataset = None


def _load():
    global _dataset
    _dataset = data.load()


def _task(setting, variant, seed):
    settings = protocol.Settings(seed=seed, evidence="partners", **SETTINGS[setting])
    rows = protocol.run(_dataset, lambda: PREDICTORS["item_pooling"](**VARIANTS[variant]), settings)
    return [{"setting": setting, "variant": variant, **row} for row in rows]


def _table(rows):
    table = {}
    for setting in SETTINGS:
        reference = [r for r in rows if (r["setting"], r["variant"]) == (setting, "version2")]
        for variant in VARIANTS:
            group = [r for r in rows if (r["setting"], r["variant"]) == (setting, variant)]
            benchmarks = sorted({r["benchmark"] for r in group})
            table[f"{setting} {variant}"] = {
                "pairs": len(group),
                "alc_pair_mean": _mean([r["alc"] for r in group]),
                "alc_benchmark_mean": _mean([_mean([r["alc"] for r in group if r["benchmark"] == b])
                                             for b in benchmarks]),
                "brier": {b: _mean([r[f"brier_{b}"] for r in group]) for b in scoring.BUDGETS},
                "vs_version2": _paired(group, reference),
                "vs_version2_by_benchmark": {
                    b: _paired([r for r in group if r["benchmark"] == b],
                               [r for r in reference if r["benchmark"] == b]) for b in benchmarks}}
    return table


def _decision(table):
    decision = {}
    for idea in IDEAS:
        hidden, full = table[f"hidden {idea}"], table[f"full {idea}"]
        diff, se = hidden["vs_version2"]["alc_diff"], hidden["vs_version2"]["alc_diff_se"]
        worst = max(v["alc_diff"] for v in hidden["vs_version2_by_benchmark"].values())
        checks = {"a_hidden_gain_over_0.0003_and_2se": diff < -0.0003 and diff < -2 * se,
                  "b_full_size_improves": full["vs_version2"]["alc_diff"] < 0,
                  "c_no_benchmark_worse_than_0.001": worst <= 0.001}
        decision[idea] = {"checks": checks, "go": all(checks.values())}
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=3)  # each loads the data (~1 GB)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()

    tasks = [(s, v, seed) for s in SETTINGS for v in VARIANTS for seed in SEEDS]
    run_dir = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}_ideas"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {"settings": SETTINGS, "variants": VARIANTS, "seeds": SEEDS,
               "provenance": provenance(None)}
    write(run_dir / "config.json", json.dumps(payload, indent=1) + "\n")

    started = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_load) as pool:
        rows = [row for chunk in pool.map(_task, *zip(*tasks)) for row in chunk]
    table = _table(rows)
    decision = _decision(table)

    with (run_dir / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write(run_dir / "summary.json", json.dumps(
        {"table": table, "decision": decision,
         "elapsed_seconds": round(time.time() - started, 1)}, indent=1) + "\n")

    print(f"{'':18s} {'pairs':>5s} {'ALC':>7s} {'bench':>7s} {'vs version2 (SE)':>18s}  Brier at budgets"
          f" {scoring.BUDGETS}")
    for name, entry in table.items():
        versus = entry["vs_version2"]
        print(f"{name:18s} {entry['pairs']:5d} {entry['alc_pair_mean']:.4f} {entry['alc_benchmark_mean']:.4f}"
              f" {versus['alc_diff']:+.5f} ({versus['alc_diff_se']:.5f})  "
              + " ".join(f"{v:.4f}" for v in entry["brier"].values()))
    for idea in IDEAS + ("both",):
        print(f"by benchmark, hidden {idea}:",
              {b: round(v["alc_diff"], 4) for b, v in table[f"hidden {idea}"]["vs_version2_by_benchmark"].items()})
    print("decision:", json.dumps(decision))
    shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    print(f"saved to {shown.as_posix()}")


if __name__ == "__main__":
    main()
