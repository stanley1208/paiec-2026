"""Rebuild the items sent for LLM rating and their Rasch difficulties from the pinned data.

    python scripts/build_rating_samples.py --check
    python scripts/build_rating_samples.py --out results/llm_rating

These steps first ran in scratch folders (2026-10-02 and 2026-10-06); this script was recovered
from the session transcripts on 2026-10-11 and regenerates the committed sample_ids.json and
sample_more_ids.json exactly (same items in the same order; b bitwise equal with scipy 1.14.1 and
numpy 1.26.4, as pinned in requirements.txt).

1. Rasch difficulties. For each benchmark separately, a penalized joint-maximum-likelihood Rasch
   model logit P(correct) = theta_subject - b_item on binomial counts aggregated by (subject,
   item key), every response and trial counted, with priors N(0, 2^2) on b and N(0, 3^2) on
   theta, fitted by L-BFGS-B from zeros (gtol 1e-6, at most 2000 iterations), from all
   subjects. b is then centered to mean 0 over the benchmark's item keys. MathArena items are
   keyed by competition and problem index (their item ids differ by prompt wrapper); other
   benchmarks by item id. Scales are not linked across benchmarks.
2. Items. An item is "informative" when no other key of the same benchmark has the same content.
3. Samples. sample.json: 50 keys per benchmark drawn with random.Random(f"llm-rating|{benchmark}")
   from the sorted keys of informative items, one item (the lowest item id) per key.
   sample_more.json: every other informative MathArena problem and REAL task, and Multi-SWE-bench
   items up to a pool of 400 (the 50 already rated plus 350 drawn with
   random.Random("llm-rating-pool|multi_swebench")), minus the items already rated.

--out writes sample.json and sample_more.json (with item content, as rate.py and rate_more.py
read them) and rasch_b.json (all fitted difficulties); these are not committed.
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paiec import data  # noqa: E402

RATINGS = ROOT / "results" / "llm_rating"
FIT_ORDER = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
RATED = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
SAMPLE_PER_BENCHMARK = 50
MULTI_SWE_POOL = 400


def problem_key(dataset, matharena_alias, item_id):
    """MathArena item ids differ by prompt wrapper; its problems are keyed by competition and index."""
    item = dataset.items[item_id]
    if item["benchmark_id"] != matharena_alias:
        return item_id
    fields = dict(kv.split("=", 1) for kv in item["item_features"].split(";") if "=" in kv)
    return "ma:" + fields["competition"] + ":" + fields["problem_idx"]


def fit_rasch(counts, sd_b=2.0, sd_theta=3.0, with_theta=False):
    """Penalized joint ML Rasch fit on aggregated binomial counts; returns b by key, centered
    (and, with with_theta, also theta by subject on the same centered scale)."""
    subjects, si = np.unique(counts["subject_id"].to_numpy(), return_inverse=True)
    keys, ki = np.unique(counts["key"].to_numpy(), return_inverse=True)
    k = counts["sum"].to_numpy(float)
    n = counts["size"].to_numpy(float)
    n_subjects, n_keys = len(subjects), len(keys)

    def objective(x):
        theta, b = x[:n_subjects], x[n_subjects:]
        eta = theta[si] - b[ki]
        nll = (np.sum(n * np.logaddexp(0, eta) - k * eta)
               + 0.5 * np.sum(b ** 2) / sd_b ** 2 + 0.5 * np.sum(theta ** 2) / sd_theta ** 2)
        residual = n * expit(eta) - k
        grad_theta = np.bincount(si, residual, n_subjects) + theta / sd_theta ** 2
        grad_b = -np.bincount(ki, residual, n_keys) + b / sd_b ** 2
        return nll, np.concatenate([grad_theta, grad_b])

    result = minimize(objective, np.zeros(n_subjects + n_keys), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=2000, gtol=1e-6))
    b = result.x[n_subjects:]
    if with_theta:
        theta = result.x[:n_subjects]
        return dict(zip(keys, b - b.mean())), dict(zip(subjects, theta - b.mean()))
    return dict(zip(keys, b - b.mean()))


def difficulties(dataset, key):
    """Rasch b by item key for each benchmark, fitted from all subjects."""
    out = {}
    for benchmark in FIT_ORDER:
        responses = dataset.responses[dataset.responses["benchmark"] == benchmark]
        counts = (responses.assign(key=responses["item_id"].map(key))
                  .groupby(["subject_id", "key"])["label"].agg(["sum", "size"]).reset_index())
        out[benchmark] = fit_rasch(counts)
    return out


def item_table(dataset, key, b_by_key):
    """benchmark -> rows (item_id, key, features, content, b, informative), sorted by item id."""
    table = {}
    for benchmark in RATED:
        items = sorted(dataset.responses.loc[dataset.responses["benchmark"] == benchmark, "item_id"].unique())
        keys_by_content = collections.defaultdict(set)
        for i in items:
            keys_by_content[dataset.items[i]["item_content"]].add(key(i))
        table[benchmark] = [{"item_id": i, "key": key(i), "features": dataset.items[i]["item_features"],
                             "content": dataset.items[i]["item_content"], "b": float(b_by_key[benchmark][key(i)]),
                             "informative": len(keys_by_content[dataset.items[i]["item_content"]]) == 1}
                            for i in items]
    return table


def entry(benchmark, row):
    return {"benchmark": benchmark, "item_id": row["item_id"], "key": row["key"], "features": row["features"],
            "content": row["content"], "b": row["b"]}


def one_per_key(rows):
    seen, out = set(), []
    for row in rows:  # rows are sorted by item id, so the lowest item id of each key is kept
        if row["key"] not in seen:
            seen.add(row["key"])
            out.append(row)
    return out


def first_sample(table):
    sample = []
    for benchmark in RATED:
        rows = one_per_key([r for r in table[benchmark] if r["informative"]])
        chosen = set(random.Random(f"llm-rating|{benchmark}").sample(sorted(r["key"] for r in rows),
                                                                      SAMPLE_PER_BENCHMARK))
        sample += [entry(benchmark, r) for r in rows if r["key"] in chosen]
    return sample


def more_sample(table, first):
    """Every other informative MathArena problem and REAL task; Multi-SWE-bench up to a pool of 400."""
    rated = {s["item_id"]: s for s in first}
    more = []

    def add(rows, benchmark):
        more.extend(entry(benchmark, r) for r in rows if r["item_id"] not in rated)

    rated_keys = {s["key"] for s in first if s["benchmark"] == "matharena"}
    add([r for r in one_per_key([r for r in table["matharena"] if r["informative"]]) if r["key"] not in rated_keys],
        "matharena")
    add([r for r in table["real_webagents"] if r["informative"]], "real_webagents")
    informative = [r for r in table["multi_swebench"] if r["informative"]]
    rated_ms = {s["item_id"] for s in first if s["benchmark"] == "multi_swebench"}
    others = sorted({r["item_id"] for r in informative} - rated_ms)
    pool = rated_ms | set(random.Random("llm-rating-pool|multi_swebench").sample(others, MULTI_SWE_POOL - len(rated_ms)))
    add([r for r in informative if r["item_id"] in pool], "multi_swebench")
    return more


def ids(sample):
    return [{k: s[k] for k in ("benchmark", "item_id", "key", "b")} for s in sample]


def compare(name, built, committed):
    same_items = ([(s["benchmark"], s["item_id"], s["key"]) for s in built]
                  == [(s["benchmark"], s["item_id"], s["key"]) for s in committed])
    gaps = [abs(a["b"] - c["b"]) for a, c in zip(built, committed)]
    exact = sum(a["b"] == c["b"] for a, c in zip(built, committed))
    print(f"{name}: {len(built)} items built, {len(committed)} committed; same items in the same order: "
          f"{same_items}; largest b difference {max(gaps):.3g}; bitwise-equal b {exact}/{len(committed)}")
    return same_items and max(gaps) <= 1e-6


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="write sample.json, sample_more.json and rasch_b.json here")
    parser.add_argument("--check", action="store_true", help="compare with the committed sample id files")
    args = parser.parse_args()

    dataset = data.load()
    matharena_alias = data.alias("benchmark", "matharena")
    cache = {}

    def key(item_id):
        if item_id not in cache:
            cache[item_id] = problem_key(dataset, matharena_alias, item_id)
        return cache[item_id]

    b_by_key = difficulties(dataset, key)
    table = item_table(dataset, key, b_by_key)
    first = first_sample(table)
    more = more_sample(table, first)
    print(f"sample: {len(first)} {dict(collections.Counter(s['benchmark'] for s in first))}; "
          f"sample_more: {len(more)} {dict(collections.Counter(s['benchmark'] for s in more))}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, sample in (("sample.json", first), ("sample_more.json", more)):
            (args.out / name).write_text(json.dumps(sample), encoding="utf-8")
        (args.out / "rasch_b.json").write_text(
            json.dumps({b: {k: float(v) for k, v in d.items()} for b, d in b_by_key.items()}), encoding="utf-8")
    if args.check:
        ok = compare("sample_ids.json", ids(first), json.loads((RATINGS / "sample_ids.json").read_text(encoding="utf-8")))
        ok &= compare("sample_more_ids.json", ids(more),
                      json.loads((RATINGS / "sample_more_ids.json").read_text(encoding="utf-8")))
        print("REPRODUCED" if ok else "NOT REPRODUCED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
