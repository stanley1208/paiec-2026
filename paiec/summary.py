"""Aggregate per-pair results and simulate Codabench-style draws.

With per-pair evidence, a pair's score does not depend on which other pairs are
evaluated, so draws can be resampled from saved results without re-running predictors.
"""

import math
import random
import statistics

from .scoring import BUDGETS


def _mean(values):
    return math.fsum(values) / len(values)


def summarize_seed(rows):
    by_benchmark = {}
    for row in rows:
        by_benchmark.setdefault(row["benchmark"], []).append(row)
    return {
        "n_pairs": len(rows),
        "alc_pair_mean": _mean([r["alc"] for r in rows]),
        "alc_benchmark_mean": _mean([_mean([r["alc"] for r in rs]) for rs in by_benchmark.values()]),
        "ece_alc_pair_mean": _mean([r["ece_alc"] for r in rows]),
        "brier_by_budget": {b: _mean([r[f"brier_{b}"] for r in rows]) for b in BUDGETS},
        "folds": {b: {"n_pairs": len(rs), "alc_mean": _mean([r["alc"] for r in rs])}
                  for b, rs in sorted(by_benchmark.items())},
    }


def _draw_stats(means, sizes, cap):
    means = sorted(means)
    n = len(means)
    return {"n_draws": n, "cap_items": cap, "alc_mean": _mean(means),
            "alc_std": statistics.pstdev(means), "alc_p05": means[int(0.05 * (n - 1))],
            "alc_p95": means[int(0.95 * (n - 1))], "pairs_per_draw_mean": _mean(sizes)}


def _draw(rows, n_draws, cap, seed):
    """Mean ALC and pair count of each simulated formative draw.

    Like the platform: whole pairs are added while they fit under `cap` distinct items,
    preferring benchmarks and then subjects that are less represented in the draw, with
    random tie-breaking; pairs that no longer fit are skipped.
    """
    rng = random.Random(seed)
    means, sizes = [], []
    for _ in range(n_draws):
        remaining, chosen = cap, []
        bench_count, subject_count = {}, {}
        candidates = list(rows)
        rng.shuffle(candidates)
        while True:
            fitting = [r for r in candidates if r["n_items"] <= remaining]
            if not fitting:
                break
            pick = min(fitting, key=lambda r: (bench_count.get(r["benchmark"], 0),
                                               subject_count.get(r["subject_id"], 0)))
            candidates.remove(pick)
            chosen.append(pick)
            remaining -= pick["n_items"]
            bench_count[pick["benchmark"]] = bench_count.get(pick["benchmark"], 0) + 1
            subject_count[pick["subject_id"]] = subject_count.get(pick["subject_id"], 0) + 1
        means.append(_mean([r["alc"] for r in chosen]))
        sizes.append(len(chosen))
    return means, sizes


def simulate_draws(rows, n_draws, cap=1000, seed=0):
    """Summary statistics of simulated formative draws for one seed's results."""
    return _draw_stats(*_draw(rows, n_draws, cap, seed), cap)


def summarize(rows, n_draws=1000, cap=1000):
    seeds = sorted({r["seed"] for r in rows})
    per_seed = {s: summarize_seed([r for r in rows if r["seed"] == s]) for s in seeds}
    headline = {}
    for metric in ("alc_pair_mean", "alc_benchmark_mean", "ece_alc_pair_mean"):
        values = [per_seed[s][metric] for s in seeds]
        headline[metric] = _mean(values)
        headline[f"{metric}_seed_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    draws = {}
    if n_draws:
        pooled_means, pooled_sizes = [], []
        for s in seeds:
            means, sizes = _draw([r for r in rows if r["seed"] == s], n_draws, cap, s)
            draws[s] = _draw_stats(means, sizes, cap)
            pooled_means += means
            pooled_sizes += sizes
        headline["draws"] = _draw_stats(pooled_means, pooled_sizes, cap)  # all seeds pooled
    return {"seeds": seeds, "headline": headline, "per_seed": per_seed, "draws": draws}
