"""Control for scripts/run_ratings.py: the same rated model with the ratings shuffled within each benchmark.

    python scripts/run_shuffle_control.py

Descriptive control, recovered from the session transcript on 2026-10-11. It first ran on
2026-10-06 as an inline script against commit 815c838, and its output (shuffled ratings +0.02702
worse than version 3) was quoted in the AI log but not saved; this script reruns it and saves it.
Setting as in run_ratings.py: variant version3_rated (fitted slope at full trust, no gate), own
labels only, seeds 0-2, held-out benchmarks cut to rated items, pairs of at least 40 items. Each
benchmark's distinct rated (features, content) keys are sorted and their ratings permuted with
random.Random(f"{tag}|{benchmark}"): tag "shuffle" is the logged control, tag "useless" the
permutation scripts/run_trust.py uses for its useless scenario.
Writes results/runs/<timestamp>_shuffle/ with config.json (provenance, written first),
per_pair.csv and summary.json.
"""

import argparse
import collections
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_ratings as rr  # noqa: E402
from run_harness import provenance, write  # noqa: E402
from run_pooling import _mean, _paired  # noqa: E402

TAGS = ("shuffle", "useless")
SEEDS = (0, 1, 2)


def shuffled_tables(real):
    dataset = rr._state["dataset"]
    groups = collections.defaultdict(set)
    for benchmark, item_ids in rr._state["rated_items"].items():
        for i in item_ids:
            groups[benchmark].add((dataset.items[i]["item_features"], dataset.items[i]["item_content"]))
    tables = {"real": real}
    for tag in TAGS:
        table = {}
        for benchmark, keys in groups.items():
            keys = sorted(keys)
            values = [real[k] for k in keys]
            random.Random(f"{tag}|{benchmark}").shuffle(values)
            table.update(zip(keys, values))
        tables[f"shuffled ({tag})"] = table
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()

    run_dir = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}_shuffle"
    run_dir.mkdir(parents=True, exist_ok=False)
    write(run_dir / "config.json", json.dumps({"tags": TAGS, "seeds": SEEDS, "variant": "version3_rated",
                                               "partners": 0, "min_pair_items": rr.MIN_PAIR_ITEMS,
                                               "provenance": provenance(None)}, indent=1) + "\n")
    started = time.time()
    rr._load()
    tables = shuffled_tables(dict(rr._state["by_content"]))
    rows = [{"ratings": "none (version 3)", **r} for seed in SEEDS for r in rr._task("version3", 0, seed)]
    for name, table in tables.items():
        rr._state["by_content"] = table
        rows += [{"ratings": name, **r} for seed in SEEDS for r in rr._task("version3_rated", 0, seed)]

    reference = [r for r in rows if r["ratings"] == "none (version 3)"]
    summary = {}
    for name in tables:
        group = [r for r in rows if r["ratings"] == name]
        summary[name] = {"alc_pair_mean": _mean([r["alc"] for r in group]),
                         "vs_version3": _paired(group, reference),
                         "by_benchmark": {b: _paired([r for r in group if r["benchmark"] == b],
                                                     [r for r in reference if r["benchmark"] == b])["alc_diff"]
                                          for b in rr.RATED}}
    with (run_dir / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write(run_dir / "summary.json", json.dumps({"table": summary, "elapsed_seconds": round(time.time() - started, 1)},
                                               indent=1) + "\n")
    for name, entry in summary.items():
        versus = entry["vs_version3"]
        print(f"{name:20s} vs version 3: {versus['alc_diff']:+.5f} (SE {versus['alc_diff_se']:.5f})  "
              + " ".join(f"{b} {v:+.4f}" for b, v in entry["by_benchmark"].items()))
    shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    print(f"saved to {shown.as_posix()}")


if __name__ == "__main__":
    main()
