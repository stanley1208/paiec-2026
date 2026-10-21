"""Does pooling item difficulties from other subjects' labels beat Phase 4 offline?

    python scripts/run_pooling.py

Decision rule, fixed before the first run (2026-10-04). Go on to packaging only if, on
hidden-sized pairs (pool_items 400, min_pair_items 60, independent splits) with one partner
pair on the same benchmark (partners = 1):
  (a) with no partners the pooled model reproduces Phase 4 exactly (max |diff| < 1e-12);
  (b) pair-mean ALC improves by at least 0.001 and by more than 2 standard errors
      (clustered by subject-benchmark pair);
  (c) no benchmark gets worse by more than 0.001;
  (d) on full-size pairs (the default config) with one partner it is not worse by more
      than 0.0005.
Otherwise no-go. More partners and a shared split are reported as sensitivity checks.
Writes results/runs/<timestamp>_pooling/ with config.json (provenance, written first),
per_pair.csv and summary.json.
"""

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_harness import provenance, write  # noqa: E402

from paiec import data, protocol, scoring  # noqa: E402
from paiec.predictors import PREDICTORS  # noqa: E402

SEEDS = (0, 1, 2)
HIDDEN = {"pool_items": 400, "min_pair_items": 60}
# setting -> (settings keywords, variants: name -> (predictor, options, partners))
GRID = {
    "hidden": (HIDDEN, {
        "phase4": ("logit_normal", {"scale": 0.5}, 0),
        "pooled_k0": ("item_pooling", {}, 0),
        "pooled_k1": ("item_pooling", {}, 1),
        "pooled_k2": ("item_pooling", {}, 2),
        "pooled_k4": ("item_pooling", {}, 4)}),
    "hidden_shared_split": (dict(HIDDEN, split="shared"), {
        "phase4": ("logit_normal", {"scale": 0.5}, 0),
        "pooled_k1": ("item_pooling", {}, 1)}),
    "full": ({}, {
        "phase4": ("logit_normal", {"scale": 0.5}, 0),
        "pooled_k1": ("item_pooling", {}, 1)}),
}

_dataset = None


def _mean(values):
    return math.fsum(values) / len(values)


def _load():
    global _dataset
    _dataset = data.load()


def _task(setting, variant, seed):
    keywords, variants = GRID[setting]
    name, options, partners = variants[variant]
    settings = protocol.Settings(seed=seed, evidence="partners", partners=partners, **keywords)
    rows = protocol.run(_dataset, lambda: PREDICTORS[name](**options), settings)
    return [{"setting": setting, "variant": variant, **row} for row in rows]


def _paired(group, reference):
    """Pair-mean ALC difference to reference over all pair results, with a cluster-robust
    standard error (clusters: subject-benchmark pairs, which repeat across seeds)."""
    alc = {(r["seed"], r["benchmark"], r["subject_id"]): r["alc"] for r in reference}
    diffs = [(r["benchmark"], r["subject_id"], r["alc"] - alc[(r["seed"], r["benchmark"], r["subject_id"])])
             for r in group]
    centre = _mean([d for _, _, d in diffs])
    clusters = {}
    for benchmark, subject, d in diffs:
        clusters[benchmark, subject] = clusters.get((benchmark, subject), 0.0) + d - centre
    se = math.sqrt(math.fsum(v * v for v in clusters.values())) / len(diffs)
    return {"alc_diff": centre, "alc_diff_se": se}


def _table(rows):
    table = {}
    for setting, (_, variants) in GRID.items():
        reference = [r for r in rows if (r["setting"], r["variant"]) == (setting, "phase4")]
        for variant in variants:
            group = [r for r in rows if (r["setting"], r["variant"]) == (setting, variant)]
            benchmarks = sorted({r["benchmark"] for r in group})
            entry = {"pairs": len(group),
                     "alc_pair_mean": _mean([r["alc"] for r in group]),
                     "alc_benchmark_mean": _mean([_mean([r["alc"] for r in group if r["benchmark"] == b])
                                                  for b in benchmarks]),
                     "brier": {b: _mean([r[f"brier_{b}"] for r in group]) for b in scoring.BUDGETS},
                     "vs_phase4": _paired(group, reference),
                     "vs_phase4_by_benchmark": {
                         b: _paired([r for r in group if r["benchmark"] == b],
                                    [r for r in reference if r["benchmark"] == b]) for b in benchmarks}}
            if variant == "pooled_k0":
                alc = {(r["seed"], r["benchmark"], r["subject_id"]): r for r in reference}
                entry["max_abs_diff_vs_phase4"] = max(
                    abs(r[f"brier_{b}"] - alc[(r["seed"], r["benchmark"], r["subject_id"])][f"brier_{b}"])
                    for r in group for b in scoring.BUDGETS)
            table[f"{setting} {variant}"] = entry
    return table


def _decision(table):
    k0 = table["hidden pooled_k0"]["max_abs_diff_vs_phase4"]
    k1 = table["hidden pooled_k1"]
    full = table["full pooled_k1"]["vs_phase4"]["alc_diff"]
    worst = max(v["alc_diff"] for v in k1["vs_phase4_by_benchmark"].values())
    checks = {"a_exact_without_partners": k0 < 1e-12,
              "b_gain_at_least_0.001_and_2se": (k1["vs_phase4"]["alc_diff"] <= -0.001 and
                                                k1["vs_phase4"]["alc_diff"] < -2 * k1["vs_phase4"]["alc_diff_se"]),
              "c_no_benchmark_worse_than_0.001": worst <= 0.001,
              "d_full_size_not_worse_than_0.0005": full <= 0.0005}
    return {"checks": checks, "go": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=3)  # each loads the data (~1 GB)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()

    tasks = [(s, v, seed) for s, (_, variants) in GRID.items() for v in variants for seed in SEEDS]
    run_dir = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}_pooling"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {"grid": GRID, "seeds": SEEDS, "provenance": provenance(None)}
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

    print(f"{'':32s} {'pairs':>5s} {'ALC':>7s} {'bench':>7s} {'vs phase4 (SE)':>17s}  Brier at budgets"
          f" {scoring.BUDGETS}")
    for name, entry in table.items():
        versus = entry["vs_phase4"]
        print(f"{name:32s} {entry['pairs']:5d} {entry['alc_pair_mean']:.4f} {entry['alc_benchmark_mean']:.4f}"
              f" {versus['alc_diff']:+.4f} ({versus['alc_diff_se']:.4f})  "
              + " ".join(f"{v:.4f}" for v in entry["brier"].values()))
    print("by benchmark, hidden pooled_k1 vs phase4:",
          {b: round(v["alc_diff"], 4) for b, v in table["hidden pooled_k1"]["vs_phase4_by_benchmark"].items()})
    print("decision:", json.dumps(decision))
    shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    print(f"saved to {shown.as_posix()}")


if __name__ == "__main__":
    main()
