"""Stress-test starting guesses offline: shifted benchmarks and uninformative metadata.

    python scripts/run_stress.py

Every held-out benchmark is made easier or harder by delta logits (delta = 0 is the
normal harness), and the Phase 4 model is also run with its metadata shuffled among the
held-out subjects or reversed. Writes results/runs/<timestamp>_stress/ with config.json
(provenance, written before the run), per_pair.csv and summary.json; the summary also
has per-benchmark results and each variant's difference to Phase 3 at the same delta,
with a standard error clustered by subject-benchmark pair.
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

from run_harness import SETTING_KEYS, provenance, write  # noqa: E402

from paiec import data, protocol, scoring, stress  # noqa: E402
from paiec.predictors import PREDICTORS  # noqa: E402

DELTAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
VARIANTS = {"phase3": ("beta_binomial_subject", {}), "phase4": ("logit_normal", {}),
            "phase4_half_metadata": ("logit_normal", {"scale": 0.5}),
            "phase4_no_metadata": ("logit_normal", {"scale": 0.0})}
CONTROLS = {"phase4_reversed_metadata": ("logit_normal", {"scale": -1.0}),
            "phase4_shuffled_metadata": (None, {})}

_dataset = None


def _mean(values):
    return math.fsum(values) / len(values)


def _load():
    global _dataset
    _dataset = data.load()


def _task(variant, delta, settings):
    name, options = {**VARIANTS, **CONTROLS}[variant]
    rows = []
    for benchmark in sorted(set(_dataset.responses["benchmark"])):
        dataset = (stress.shift_benchmark(_dataset, benchmark, delta, settings.seed)
                   if delta else _dataset)
        if name is None:
            subjects = [{f: dataset.subjects[p.subject_id][f] for f in data.SUBJECT_FIELDS}
                        for p in protocol.eligible_pairs(dataset, benchmark, settings)]
            def make(subjects=subjects, seed=f"{settings.seed}|{benchmark}"):
                return stress.ShuffledMetadata(subjects, seed)
        else:
            def make(name=name, options=options):
                return PREDICTORS[name](**options)
        for row in protocol.run_fold(dataset, benchmark, make, settings):
            rows.append({"variant": variant, "delta": delta, **row})
    return rows


def _paired(group, reference):
    """Mean ALC difference to reference, with a standard error clustered by pair."""
    alc = {(r["seed"], r["benchmark"], r["subject_id"]): r["alc"] for r in reference}
    clusters = {}
    for r in group:
        diff = r["alc"] - alc[(r["seed"], r["benchmark"], r["subject_id"])]
        clusters.setdefault((r["benchmark"], r["subject_id"]), []).append(diff)
    means = [_mean(v) for v in clusters.values()]
    centre = _mean(means)
    variance = math.fsum((m - centre) ** 2 for m in means) / (len(means) - 1)
    return {"alc_diff": centre, "alc_diff_se": math.sqrt(variance / len(means))}


def _table(rows):
    table = {}
    for key in sorted({(r["variant"], r["delta"]) for r in rows}):
        group = [r for r in rows if (r["variant"], r["delta"]) == key]
        reference = [r for r in rows if (r["variant"], r["delta"]) == ("phase3", key[1])]
        by_benchmark = {b: {"pairs": len(g), "alc": _mean([r["alc"] for r in g]),
                            "brier_0": _mean([r["brier_0"] for r in g])}
                        for b in sorted({r["benchmark"] for r in group})
                        for g in [[r for r in group if r["benchmark"] == b]]}
        table[f"{key[0]} delta={key[1]:+g}"] = {
            "variant": key[0], "delta": key[1], "pairs": len(group),
            "alc_pair_mean": _mean([r["alc"] for r in group]),
            "alc_benchmark_mean": _mean([v["alc"] for v in by_benchmark.values()]),
            "brier": {b: _mean([r[f"brier_{b}"] for r in group]) for b in scoring.BUDGETS},
            "vs_phase3": _paired(group, reference) if reference else None,
            "by_benchmark": by_benchmark}
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    parser.add_argument("--workers", type=int, default=3)  # each loads the data (~1 GB)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = [protocol.Settings(seed=seed, **{k: config[k] for k in SETTING_KEYS if k in config})
                for seed in config["seeds"]]
    tasks = [(v, d, s) for v in VARIANTS for d in DELTAS for s in settings]
    tasks += [(v, 0.0, s) for v in CONTROLS for s in settings]
    run_dir = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}_stress"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {"deltas": DELTAS, "variants": VARIANTS, "controls": CONTROLS,
               "config_file": str(args.config), "config": config,
               "provenance": provenance(None)}
    write(run_dir / "config.json", json.dumps(payload, indent=1) + "\n")

    started = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_load) as pool:
        rows = [row for chunk in pool.map(_task, *zip(*tasks)) for row in chunk]
    table = _table(rows)

    with (run_dir / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write(run_dir / "summary.json", json.dumps(
        {"table": table, "elapsed_seconds": round(time.time() - started, 1)}, indent=1) + "\n")

    print(f"{'':38s} {'ALC':>7s} {'bench':>7s} {'vs phase3 (SE)':>17s}  Brier at budgets"
          f" {scoring.BUDGETS}")
    for name, entry in table.items():
        versus = entry["vs_phase3"]
        versus = f"{versus['alc_diff']:+.4f} ({versus['alc_diff_se']:.4f})" if versus else ""
        print(f"{name:38s} {entry['alc_pair_mean']:.4f} {entry['alc_benchmark_mean']:.4f}"
              f" {versus:>17s}  " + " ".join(f"{v:.4f}" for v in entry["brier"].values()))
    shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    print(f"saved to {shown.as_posix()}")


if __name__ == "__main__":
    main()
