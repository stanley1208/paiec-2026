"""Do LLM difficulty ratings improve version 3 offline? (option A, step 2)

    python scripts/run_ratings.py

Decision rule, fixed before the first run (2026-10-06). Each held-out benchmark is cut to the
items rated by Claude Opus 5.5 (results/llm_rating/: every MathArena problem and every REAL
task, 400 Multi-SWE-bench tasks, 50 ResearchCodeBench tasks), pairs need at least 40 items,
seeds 0-2. The rating-to-difficulty slope and spread are fitted on the other rated
benchmarks only. Option A continues only if, with own labels only (no partner pairs),
version 3 with ratings beats version 3 by at least 0.003 pair-mean ALC and by more than 2
standard errors (clustered by subject-benchmark pair), and also improves on Multi-SWE-bench
and on REAL, the two benchmarks whose item difficulties the model is least likely to
remember. Version 2 and one partner pair are reported for information.
Writes results/runs/<timestamp>_ratings/ with config.json (provenance, written first),
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
from run_pooling import _mean, _paired  # noqa: E402

from paiec import data, item_pooling_core, protocol, scoring  # noqa: E402
from paiec.item_pooling import ItemPooling  # noqa: E402

RATINGS = ROOT / "results" / "llm_rating"
RATED = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
SEEDS = (0, 1, 2)
PARTNERS = (0, 1)
VARIANTS = {"version3": (0.0, False), "version3_rated": (0.0, True),
            "version2": (0.5, False), "version2_rated": (0.5, True)}
MIN_PAIR_ITEMS = 40

_state = {}


def problem_key(dataset, item_id):
    """MathArena item ids differ by prompt wrapper; its problems are keyed by competition and index."""
    item = dataset.items[item_id]
    if dataset.responses_benchmark[item_id] != "matharena":
        return item_id
    fields = dict(kv.split("=", 1) for kv in item["item_features"].split(";") if "=" in kv)
    return f"ma:{fields['competition']}:{fields['problem_idx']}"


def load_ratings(dataset):
    """Ratings and Rasch difficulties by problem key, and the rated item ids of each benchmark."""
    keys = {}
    for name in ("sample_ids.json", "sample_more_ids.json"):
        for row in json.loads((RATINGS / name).read_text(encoding="utf-8")):
            keys[row["item_id"]] = (row["benchmark"], row["key"], row["b"])
    by_key = {}
    for name in ("ratings.jsonl", "ratings_more.jsonl"):
        for line in (RATINGS / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["rating"] is not None:
                benchmark, key, b = keys[row["item_id"]]
                by_key[benchmark, key] = (row["rating"], b)
    responses = dataset.responses
    dataset.responses_benchmark = dict(zip(responses["item_id"], responses["benchmark"]))
    # A rating is reused only for items whose text shows the problem: content shared by several
    # problems (MathArena's system-prompt-only items) could not be rated from the text itself.
    problems_by_content = {}
    for item_id, benchmark in dataset.responses_benchmark.items():
        content = dataset.items[item_id]["item_content"]
        problems_by_content.setdefault(content, set()).add(problem_key(dataset, item_id))
    rated_items, by_content = {b: set() for b in RATED}, {}
    for item_id, benchmark in dataset.responses_benchmark.items():
        if len(problems_by_content[dataset.items[item_id]["item_content"]]) > 1:
            continue
        if benchmark in RATED and (benchmark, problem_key(dataset, item_id)) in by_key:
            rating = by_key[benchmark, problem_key(dataset, item_id)][0]
            rated_items[benchmark].add(item_id)
            item = dataset.items[item_id]
            by_content[item["item_features"], item["item_content"]] = rating
    return by_key, rated_items, by_content


def rating_fit(by_key, held_out):
    """Pooled within-benchmark slope of Rasch difficulty on rating, and the residual sd."""
    sxy = sxx = 0.0
    groups = []
    for benchmark in RATED:
        if benchmark == held_out:
            continue
        rows = [v for (b, _), v in by_key.items() if b == benchmark]
        mr, mb = _mean([r for r, _ in rows]), _mean([d for _, d in rows])
        groups.append([(r - mr, d - mb) for r, d in rows])
        sxy += math.fsum(x * y for x, y in groups[-1])
        sxx += math.fsum(x * x for x, _ in groups[-1])
    slope = sxy / sxx
    residuals = [(y - slope * x) ** 2 for group in groups for x, y in group]
    return slope, math.sqrt(_mean(residuals))


class RatedItemPooling(ItemPooling):
    """Version 2/3 with the ratings as prior item difficulties (slope fitted without held_out)."""

    def __init__(self, scale, held_out):
        super().__init__(scale=scale)
        self.held_out = held_out

    def fit(self, train):
        super().fit(train)
        slope, spread = rating_fit(_state["by_key"], self.held_out)
        self.params = dict(self.params, rating_slope=slope, rating_sd=spread)

    def predict(self, input, labeled=None):
        return item_pooling_core.predict(input, labeled or [], self.params, _rating)


def _rating(item):
    return _state["by_content"].get((item.get("item_features", ""), item.get("item_content", "")))


def _load():
    dataset = data.load()
    by_key, rated_items, by_content = load_ratings(dataset)
    _state.update(dataset=dataset, by_key=by_key, rated_items=rated_items, by_content=by_content)


def _task(variant, partners, seed):
    scale, rated = VARIANTS[variant]
    dataset = _state["dataset"]
    settings = protocol.Settings(seed=seed, evidence="partners", partners=partners,
                                 min_pair_items=MIN_PAIR_ITEMS)
    rows = []
    for benchmark in RATED:
        responses = dataset.responses
        keep = (responses["benchmark"] != benchmark) | responses["item_id"].isin(_state["rated_items"][benchmark])
        cut = data.Dataset(responses[keep].reset_index(drop=True), dataset.subjects, dataset.items)
        make = ((lambda: RatedItemPooling(scale, benchmark)) if rated
                else (lambda: ItemPooling(scale=scale)))
        for row in protocol.run_fold(cut, benchmark, make, settings):
            rows.append({"variant": variant, "partners": partners, **row})
    return rows


def _table(rows):
    table = {}
    for partners in PARTNERS:
        for variant in VARIANTS:
            reference_name = "version3" if variant.startswith("version3") else "version2"
            group = [r for r in rows if (r["variant"], r["partners"]) == (variant, partners)]
            reference = [r for r in rows if (r["variant"], r["partners"]) == (reference_name, partners)]
            benchmarks = sorted({r["benchmark"] for r in group})
            table[f"k{partners} {variant}"] = {
                "pairs": len(group),
                "alc_pair_mean": _mean([r["alc"] for r in group]),
                "brier": {b: _mean([r[f"brier_{b}"] for r in group]) for b in scoring.BUDGETS},
                "vs_unrated": _paired(group, reference),
                "vs_unrated_by_benchmark": {
                    b: _paired([r for r in group if r["benchmark"] == b],
                               [r for r in reference if r["benchmark"] == b]) for b in benchmarks}}
    return table


def _decision(table):
    entry = table["k0 version3_rated"]
    diff, se = entry["vs_unrated"]["alc_diff"], entry["vs_unrated"]["alc_diff_se"]
    by_benchmark = entry["vs_unrated_by_benchmark"]
    checks = {"gain_at_least_0.003_and_2se": diff <= -0.003 and diff < -2 * se,
              "improves_multi_swebench": by_benchmark["multi_swebench"]["alc_diff"] < 0,
              "improves_real_webagents": by_benchmark["real_webagents"]["alc_diff"] < 0}
    return {"checks": checks, "go": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=3)  # each loads the data (~1 GB)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()

    tasks = [(v, k, seed) for v in VARIANTS for k in PARTNERS for seed in SEEDS]
    run_dir = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}_ratings"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {"variants": VARIANTS, "partners": PARTNERS, "seeds": SEEDS,
               "min_pair_items": MIN_PAIR_ITEMS, "provenance": provenance(None)}
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

    print(f"{'':20s} {'pairs':>5s} {'ALC':>7s} {'vs unrated (SE)':>18s}  Brier at budgets {scoring.BUDGETS}")
    for name, entry in table.items():
        versus = entry["vs_unrated"]
        print(f"{name:20s} {entry['pairs']:5d} {entry['alc_pair_mean']:.4f}"
              f" {versus['alc_diff']:+.5f} ({versus['alc_diff_se']:.5f})  "
              + " ".join(f"{v:.4f}" for v in entry["brier"].values()))
    for name in ("k0 version3_rated", "k1 version3_rated"):
        print(f"by benchmark, {name}:",
              {b: round(v["alc_diff"], 4) for b, v in table[name]["vs_unrated_by_benchmark"].items()})
    print("decision:", json.dumps(decision))
    shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    print(f"saved to {shown.as_posix()}")


if __name__ == "__main__":
    main()
