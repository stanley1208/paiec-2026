"""Descriptive comparisons for the option C pre-check, written after the run (not part of the rule).

    python results/subject_check/compare.py

1. Claude's model match recomputed with numpy and with other Rasch priors.
2. The same measure for the version 1 metadata starting guess (fitted on the other benchmarks,
   as in the harness folds), which version 3 dropped after formative feedback.
3. The same measure for the release year alone, on the subjects that have one.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402
from build_rating_samples import fit_rasch  # noqa: E402
from paiec import baseline_core, baselines, data  # noqa: E402


def centered_corr(x, y, groups):
    x, y, groups = np.array(x, float), np.array(y, float), np.array(groups)
    for g in set(groups.tolist()):
        x[groups == g] -= x[groups == g].mean()
        y[groups == g] -= y[groups == g].mean()
    return float(np.corrcoef(x, y)[0, 1])


def logit(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))


def show(label, per):
    print(f"{label}: " + ", ".join(f"{b} {v:.3f}" for b, v in per.items()) + f"; mean {np.mean(list(per.values())):.3f}")


def main():
    rows = json.loads((HERE / "per_subject.json").read_text(encoding="utf-8"))
    dataset = data.load()
    key = check.item_key(dataset)
    by_benchmark = {b: [r for r in rows if r["benchmark"] == b] for b in check.BENCHMARKS}

    for sd_b, sd_theta in ((2.0, 3.0), (4.0, 3.0), (1.0, 1.5)):
        per = {}
        for b, rb in by_benchmark.items():
            responses = dataset.responses[dataset.responses["benchmark"] == b]
            counts = (responses.assign(key=responses["item_id"].map(key))
                      .groupby(["subject_id", "key"])["label"].agg(["sum", "size"]).reset_index())
            theta = fit_rasch(counts, sd_b=sd_b, sd_theta=sd_theta, with_theta=True)[1]
            per[b] = centered_corr(logit([r["guess"] / 100 for r in rb]), [theta[r["subject_id"]] for r in rb],
                                   [r["group"] for r in rb])
        show(f"Claude, Rasch priors sd_b={sd_b} sd_theta={sd_theta}", per)

    benchmarks = sorted(dataset.responses["benchmark"].unique())
    per = {}
    for b, rb in by_benchmark.items():
        params = baselines.fit_params(dataset.subset([x for x in benchmarks if x != b]), prior="subject")
        x = logit([baseline_core.subject_prior(dataset.subjects[r["subject_id"]], params) for r in rb])
        per[b] = 0.0 if np.ptp(x) == 0 else centered_corr(x, [r["theta"] for r in rb], [r["group"] for r in rb])
    show("version 1 metadata starting guess", per)

    for b, rb in by_benchmark.items():
        years = [baseline_core._year(dataset.subjects[r["subject_id"]]["release_date"]) for r in rb]
        sub = [(r, y) for r, y in zip(rb, years) if y is not None]
        rs, ys = [r for r, _ in sub], [y for _, y in sub]
        groups = [r["group"] for r in rs]
        print(f"release year, {b}: {len(sub)} of {len(rb)} subjects have one; year "
              f"{centered_corr(ys, [r['theta'] for r in rs], groups):.3f}, Claude on the same subjects "
              f"{centered_corr(logit([r['guess'] / 100 for r in rs]), [r['theta'] for r in rs], groups):.3f}")


if __name__ == "__main__":
    main()
