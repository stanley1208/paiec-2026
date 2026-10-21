"""Does the rating gate protect version 4 when the ratings are wrong? (version 4 review, ROB-1)

    python scripts/run_gate.py

Decision rule, fixed before the first run (2026-10-06). On the pairs of scripts/run_trust.py
(own labels only, seeds 0-2), compare three models: version 3 (trust 0), trust 0.5 without the
gate, and trust 0.5 with the gate (params["rating_gate"]: ratings are switched off on a
benchmark while its rated labels show failed items rated easier than solved ones). Each runs
under three rating scenarios: the real ratings, useless ratings (shuffled within each
benchmark, as in run_trust.py) and reversed ratings (11 - rating). Version 4 keeps the gate if
its largest regret over the three scenarios is no larger than without it, regret being
pair-mean ALC minus the best of the three models' ALC in that scenario.
Writes results/runs/<timestamp>_gate/ with config.json (provenance, written first),
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

import run_ratings as rr  # noqa: E402
import run_trust as rt  # noqa: E402
from run_harness import provenance, write  # noqa: E402
from run_pooling import _mean, _paired  # noqa: E402

from paiec import data, protocol, scoring  # noqa: E402

MODELS = {"version3": (0.0, False), "t0.5": (0.5, False), "t0.5+gate": (0.5, True)}
SCENARIOS = ("real", "useless", "reversed")
SEEDS = (0, 1, 2)


class GatedItemPooling(rt.TrustedItemPooling):
    def __init__(self, scale, held_out, trust, gate):
        super().__init__(scale, held_out, trust)
        self.gate = gate

    def fit(self, train):
        super().fit(train)
        self.params = dict(self.params, rating_gate=self.gate)


def _load():
    rt._load()
    tables = rr._state["tables"]
    tables["reversed"] = {key: 11 - value for key, value in tables["real"].items()}


def _task(scenario, model, seed):
    rr._state["by_content"] = rr._state["tables"][scenario]
    trust, gate = MODELS[model]
    dataset = rr._state["dataset"]
    settings = protocol.Settings(seed=seed, evidence="partners", partners=0, min_pair_items=rr.MIN_PAIR_ITEMS)
    rows = []
    for benchmark in rr.RATED:
        responses = dataset.responses
        keep = (responses["benchmark"] != benchmark) | responses["item_id"].isin(rr._state["rated_items"][benchmark])
        cut = data.Dataset(responses[keep].reset_index(drop=True), dataset.subjects, dataset.items)
        factory = lambda: GatedItemPooling(0.0, benchmark, trust, gate)  # noqa: E731
        for row in protocol.run_fold(cut, benchmark, factory, settings):
            rows.append({"scenario": scenario, "model": model, **row})
    return rows


def _summary(rows):
    table, regret = {}, {}
    for scenario in SCENARIOS:
        reference = [r for r in rows if (r["scenario"], r["model"]) == (scenario, "version3")]
        alc = {}
        for model in MODELS:
            group = [r for r in rows if (r["scenario"], r["model"]) == (scenario, model)]
            alc[model] = _mean([r["alc"] for r in group])
            table[f"{scenario} {model}"] = {
                "alc_pair_mean": alc[model], "vs_version3": _paired(group, reference),
                "by_benchmark": {b: _paired([r for r in group if r["benchmark"] == b],
                                            [r for r in reference if r["benchmark"] == b])["alc_diff"]
                                 for b in rr.RATED},
                "brier": {b: _mean([r[f"brier_{b}"] for r in group]) for b in scoring.BUDGETS}}
        best = min(alc.values())
        for model in MODELS:
            regret.setdefault(model, {})[scenario] = alc[model] - best
    worst = {model: max(r.values()) for model, r in regret.items()}
    keep_gate = worst["t0.5+gate"] <= worst["t0.5"]
    return table, {"regret": regret, "worst_regret": worst, "keep_gate": keep_gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=3)  # each loads the data (~1 GB)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()

    tasks = [(s, m, seed) for s in SCENARIOS for m in MODELS for seed in SEEDS]
    run_dir = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}_gate"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {"models": MODELS, "scenarios": SCENARIOS, "seeds": SEEDS, "provenance": provenance(None)}
    write(run_dir / "config.json", json.dumps(payload, indent=1) + "\n")

    started = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_load) as pool:
        rows = [row for chunk in pool.map(_task, *zip(*tasks)) for row in chunk]
    table, decision = _summary(rows)

    with (run_dir / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write(run_dir / "summary.json", json.dumps(
        {"table": table, "decision": decision, "elapsed_seconds": round(time.time() - started, 1)},
        indent=1) + "\n")

    print(f"{'':20s} {'ALC':>7s} {'vs version 3 (SE)':>20s}  by benchmark")
    for name, entry in table.items():
        versus = entry["vs_version3"]
        print(f"{name:20s} {entry['alc_pair_mean']:.4f} {versus['alc_diff']:+.5f} ({versus['alc_diff_se']:.5f})  "
              + " ".join(f"{b} {v:+.4f}" for b, v in entry["by_benchmark"].items()))
    print("worst-case regret:", {m: round(v, 4) for m, v in decision["worst_regret"].items()})
    print("keep the gate:", decision["keep_gate"])
    shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    print(f"saved to {shown.as_posix()}")


if __name__ == "__main__":
    main()
