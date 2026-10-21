"""Phase 3 baselines: a global mean, a subject-metadata prior, and Beta-Binomial updates.

The metadata prior is a weighted ridge regression on each training pair's smoothed
logit success rate, with one fixed effect per training benchmark and standardized
subject features. Subjects already seen in training also get a shrunk residual.
"""

from collections import Counter

import numpy as np

from . import baseline_core


def _logit(p):
    return np.log(p / (1 - p))


def fit_params(train, prior="subject", strength=None, ridge=5.0, min_provider_subjects=3,
               residual_shrink=1.0):
    responses = train.responses
    pairs = (responses.groupby(["benchmark", "subject_id"])
             .agg(n=("label", "size"), s=("label", "sum"), items=("item_id", "nunique"))
             .reset_index())
    rates = pairs["s"] / pairs["n"]
    params = {"version": 1, "prior": prior, "strength": strength,
              "global_mean": float(rates.groupby(pairs["benchmark"]).mean().mean())}
    if prior == "global":
        return params

    subjects = [train.subjects[s] for s in pairs["subject_id"]]
    counts = Counter(train.subjects[s]["provider"].strip().lower()
                     for s in pairs["subject_id"].unique())
    years = [y for y in (baseline_core._year(s["release_date"]) for s in train.subjects.values())
             if y is not None]
    spec = {"providers": sorted(p for p, c in counts.items() if p and c >= min_provider_subjects),
            "release_fill": float(np.mean(years)) if years else 0.0}
    raw = np.array([baseline_core.raw_features(s, spec) for s in subjects])
    means, stds = raw.mean(axis=0), raw.std(axis=0)
    stds[stds == 0] = 1.0
    features = np.clip((raw - means) / stds, -baseline_core.CLIP, baseline_core.CLIP)

    benchmarks = sorted(pairs["benchmark"].unique())
    onehot = (pairs["benchmark"].to_numpy()[:, None] == np.array(benchmarks)[None, :]).astype(float)
    design = np.hstack([onehot, features])
    target = _logit((pairs["s"].to_numpy() + 0.5) / (pairs["n"].to_numpy() + 1.0))
    weight = np.minimum(pairs["items"].to_numpy(), 200).astype(float)
    weight /= weight.mean()
    penalty = np.diag([0.0] * len(benchmarks) + [ridge] * features.shape[1])
    gram = design.T @ (weight[:, None] * design) + penalty
    beta = np.linalg.solve(gram, design.T @ (weight * target))
    effects, coefficients = beta[:len(benchmarks)], beta[len(benchmarks):]

    residual = target - design @ beta
    known = {}
    for subject_id, idx in pairs.groupby("subject_id").indices.items():
        key = baseline_core.subject_key(train.subjects[subject_id])
        known[key] = float((weight[idx] * residual[idx]).sum() / (weight[idx].sum() + residual_shrink))

    spec.update(means=means.tolist(), stds=stds.tolist())
    params.update(spec=spec, intercept=float(effects.mean()), weights=coefficients.tolist(),
                  benchmark_offsets=(effects - effects.mean()).tolist(), known_subjects=known)
    return params


class Baseline:
    """fit() builds params on the training benchmarks; predict() uses baseline_core."""

    def __init__(self, prior="subject", strength=None, **options):
        self.options = dict(prior=prior, strength=strength, **options)
        self.params = None

    def fit(self, train):
        self.params = fit_params(train, **self.options)

    def predict(self, input, labeled=None):
        return baseline_core.predict(input, labeled or [], self.params)
