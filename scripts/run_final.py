"""How does the submitted version 4 compare with version 3 in conditions closer to the hosted runs?

    python scripts/run_final.py

Descriptive, for the technical report; no decision depends on it (fixed before the first run,
2026-10-09). On the pairs of scripts/run_trust.py and scripts/run_gate.py (held-out benchmarks
cut to rated items, pairs of at least 40 items, seeds 0-2, rating slope fitted without the
held-out benchmark), compare version 3 (trust 0) with version 4 (trust 0.5 with the gate) when
the target pair also sees the labels of K = 0, 1 or 2 other subjects' pairs on the same
benchmark (hosted draws put one or two pairs on some benchmarks), and when only a share c =
1, 0.9 or 0.7 of the items have a rating (the hosted simulations rated 91-98% of labels and
not every target in time). Missing ratings are a fixed pseudo-random subset of items per c.
Writes results/runs/<timestamp>_final/ with config.json (provenance, written first),
per_pair.csv and summary.json.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_gate as rg  # noqa: E402
import run_ratings as rr  # noqa: E402
import run_trust as rt  # noqa: E402
from run_harness import provenance, write  # noqa: E402
from run_pooling import _mean, _paired  # noqa: E402

from paiec import data, protocol, scoring  # noqa: E402

PARTNERS = (0, 1, 2)
COVERAGE = (1.0, 0.9, 0.7)
SEEDS = (0, 1, 2)
MODELS = {"version3": (0.0, False), "version4": (0.5, True)}  # (trust, gate)


def _covered(key, coverage):
    """Whether an item keeps its rating at this coverage (a fixed pseudo-random subset)."""
    digest = hashlib.sha256(f"coverage|{coverage}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < coverage


def _load():
    rt._load()
    real = rr._state["tables"]["real"]
    rr._state["coverage"] = {c: {k: v for k, v in real.items() if _covered(k, c)} for c in COVERAGE}


def _task(model, partners, coverage, seed):
    rr._state["by_content"] = rr._state["coverage"][coverage]
    trust, gate = MODELS[model]
    dataset = rr._state["dataset"]
    settings = protocol.Settings(seed=seed, evidence="partners", partners=partners,
                                 min_pair_items=rr.MIN_PAIR_ITEMS)
    rows = []
    for benchmark in rr.RATED:
        responses = dataset.responses
        keep = (responses["benchmark"] != benchmark) | responses["item_id"].isin(rr._state["rated_items"][benchmark])
        cut = data.Dataset(responses[keep].reset_index(drop=True), dataset.subjects, dataset.items)
        factory = lambda: rg.GatedItemPooling(0.0, benchmark, trust, gate)  # noqa: E731
        for row in protocol.run_fold(cut, benchmark, factory, settings):
            rows.append({"model": model, "partners": partners, "coverage": coverage, **row})
    return rows


def _summary(rows):
    table = {}
    for partners in PARTNERS:
        reference = [r for r in rows if (r["model"], r["partners"]) == ("version3", partners)]
        table[f"k{partners} version3"] = {
            "pairs": len(reference), "alc_pair_mean": _mean([r["alc"] for r in reference]),
            "brier": {b: _mean([r[f"brier_{b}"] for r in reference]) for b in scoring.BUDGETS}}
        for coverage in COVERAGE:
            group = [r for r in rows if (r["model"], r["partners"], r["coverage"]) == ("version4", partners, coverage)]
            table[f"k{partners} version4 c{coverage:g}"] = {
                "pairs": len(group), "alc_pair_mean": _mean([r["alc"] for r in group]),
                "vs_version3": _paired(group, reference),
                "by_benchmark": {b: _paired([r for r in group if r["benchmark"] == b],
                                            [r for r in reference if r["benchmark"] == b])
                                 for b in rr.RATED},
                "brier": {b: _mean([r[f"brier_{b}"] for r in group]) for b in scoring.BUDGETS},
                "brier_vs_version3": {b: _mean([r[f"brier_{b}"] for r in group])
                                      - _mean([r[f"brier_{b}"] for r in reference]) for b in scoring.BUDGETS}}
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=3)  # each loads the data (~1 GB)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()

    tasks = [("version3", k, 1.0, seed) for k in PARTNERS for seed in SEEDS]
    tasks += [("version4", k, c, seed) for k in PARTNERS for c in COVERAGE for seed in SEEDS]
    run_dir = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}_final"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {"models": MODELS, "partners": PARTNERS, "coverage": COVERAGE, "seeds": SEEDS,
               "min_pair_items": rr.MIN_PAIR_ITEMS, "provenance": provenance(None)}
    write(run_dir / "config.json", json.dumps(payload, indent=1) + "\n")

    started = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_load) as pool:
        rows = [row for chunk in pool.map(_task, *zip(*tasks)) for row in chunk]
    table = _summary(rows)

    with (run_dir / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write(run_dir / "summary.json", json.dumps(
        {"table": table, "elapsed_seconds": round(time.time() - started, 1)}, indent=1) + "\n")

    print(f"{'':22s} {'pairs':>5s} {'ALC':>7s} {'vs version 3 (SE)':>20s}  by benchmark")
    for name, entry in table.items():
        versus = entry.get("vs_version3")
        shown = f"{versus['alc_diff']:+.5f} ({versus['alc_diff_se']:.5f})" if versus else ""
        benchmarks = " ".join(f"{b} {v['alc_diff']:+.4f}" for b, v in entry.get("by_benchmark", {}).items())
        print(f"{name:22s} {entry['pairs']:5d} {entry['alc_pair_mean']:.4f} {shown:>20s}  {benchmarks}")
    shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    print(f"saved to {shown.as_posix()}")


if __name__ == "__main__":
    main()
