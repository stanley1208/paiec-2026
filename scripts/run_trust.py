"""How much should the model trust LLM difficulty ratings? (option A, safety tuning)

    python scripts/run_trust.py

Decision rule, fixed before the first run (2026-10-06). Trust level t scales the fitted
rating slope to t * slope and keeps each item's total prior variance at item_sd^2, so t = 0 is
version 3 and t = 1 is the setting of scripts/run_ratings.py. Each t in {0, 0.25, 0.5, 0.75, 1}
is run on the same pairs as scripts/run_ratings.py (own labels only, seeds 0-2) under three
rating scenarios: the real ratings, half-noisy ratings (each item keeps its rating with
probability 1/2 and otherwise gets a randomly drawn other item's rating from the same
benchmark), and useless ratings (shuffled within each benchmark). The chosen t minimises the
largest regret over the three scenarios, regret being pair-mean ALC minus the best t's ALC
in that scenario.
Writes results/runs/<timestamp>_trust/ with config.json (provenance, written first),
per_pair.csv and summary.json.
"""

import argparse
import collections
import csv
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_ratings as rr  # noqa: E402
from run_harness import provenance, write  # noqa: E402
from run_pooling import _mean, _paired  # noqa: E402

from paiec import data, protocol, scoring  # noqa: E402

TRUST = (0.0, 0.25, 0.5, 0.75, 1.0)
SCENARIOS = ("real", "half_noisy", "useless")
SEEDS = (0, 1, 2)


class TrustedItemPooling(rr.RatedItemPooling):
    def __init__(self, scale, held_out, trust):
        super().__init__(scale, held_out)
        self.trust = trust

    def fit(self, train):
        super().fit(train)
        item_sd, slope, spread = self.params["item_sd"], self.params["rating_slope"], self.params["rating_sd"]
        explained = max(item_sd ** 2 - spread ** 2, 0.0)
        self.params = dict(self.params, rating_slope=self.trust * slope,
                           rating_sd=math.sqrt(item_sd ** 2 - self.trust ** 2 * explained))


def _scenario_tables():
    """Rating lookups by (features, content) for each scenario, seeded, per benchmark."""
    dataset, real = rr._state["dataset"], rr._state["by_content"]
    keys_by_benchmark = collections.defaultdict(set)
    for benchmark, ids in rr._state["rated_items"].items():
        for i in ids:
            keys_by_benchmark[benchmark].add((dataset.items[i]["item_features"], dataset.items[i]["item_content"]))
    tables = {"real": real, "half_noisy": {}, "useless": {}}
    for benchmark, keys in keys_by_benchmark.items():
        keys = sorted(keys)
        values = [real[k] for k in keys]
        shuffled = values[:]
        random.Random(f"useless|{benchmark}").shuffle(shuffled)
        tables["useless"].update(zip(keys, shuffled))
        rng = random.Random(f"half_noisy|{benchmark}")
        tables["half_noisy"].update((k, v if rng.random() < 0.5 else rng.choice(values)) for k, v in zip(keys, values))
    return tables


def _load():
    rr._load()
    rr._state["tables"] = _scenario_tables()


def _task(scenario, trust, seed):
    rr._state["by_content"] = rr._state["tables"][scenario]
    dataset = rr._state["dataset"]
    settings = protocol.Settings(seed=seed, evidence="partners", partners=0, min_pair_items=rr.MIN_PAIR_ITEMS)
    rows = []
    for benchmark in rr.RATED:
        responses = dataset.responses
        keep = (responses["benchmark"] != benchmark) | responses["item_id"].isin(rr._state["rated_items"][benchmark])
        cut = data.Dataset(responses[keep].reset_index(drop=True), dataset.subjects, dataset.items)
        for row in protocol.run_fold(cut, benchmark, lambda: TrustedItemPooling(0.0, benchmark, trust), settings):
            rows.append({"scenario": scenario, "trust": trust, **row})
    return rows


def _summary(rows):
    table, regret = {}, {}
    for scenario in SCENARIOS:
        reference = [r for r in rows if (r["scenario"], r["trust"]) == (scenario, 0.0)]
        alc = {}
        for trust in TRUST:
            group = [r for r in rows if (r["scenario"], r["trust"]) == (scenario, trust)]
            alc[trust] = _mean([r["alc"] for r in group])
            table[f"{scenario} t={trust:g}"] = {
                "alc_pair_mean": alc[trust], "vs_trust0": _paired(group, reference),
                "brier": {b: _mean([r[f"brier_{b}"] for r in group]) for b in scoring.BUDGETS}}
        best = min(alc.values())
        for trust in TRUST:
            regret.setdefault(trust, {})[scenario] = alc[trust] - best
    worst = {trust: max(r.values()) for trust, r in regret.items()}
    chosen = min(TRUST, key=lambda t: (worst[t], -t))
    return table, {"regret": regret, "worst_regret": worst, "chosen_trust": chosen}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=3)  # each loads the data (~1 GB)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()

    tasks = [(s, t, seed) for s in SCENARIOS for t in TRUST for seed in SEEDS]
    run_dir = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}_trust"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {"trust": TRUST, "scenarios": SCENARIOS, "seeds": SEEDS, "provenance": provenance(None)}
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

    print(f"{'':22s} {'ALC':>7s} {'vs t=0 (SE)':>18s}  Brier at budgets {scoring.BUDGETS}")
    for name, entry in table.items():
        versus = entry["vs_trust0"]
        print(f"{name:22s} {entry['alc_pair_mean']:.4f} {versus['alc_diff']:+.5f} ({versus['alc_diff_se']:.5f})  "
              + " ".join(f"{v:.4f}" for v in entry["brier"].values()))
    print("worst-case regret by trust:", {t: round(v, 4) for t, v in decision["worst_regret"].items()})
    print("chosen trust:", decision["chosen_trust"])
    shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    print(f"saved to {shown.as_posix()}")


if __name__ == "__main__":
    main()
