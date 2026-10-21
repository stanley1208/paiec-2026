"""Inference for the Phase 4 model, standard library only.

A subject's success rate p on an unseen benchmark gets a logit-normal prior,

    logit p ~ N(m, sd^2),  m = scale * (metadata score - mean score over training subjects),

with m clipped to +-LIMIT. Before any label the prediction stays near 0.5, nudged by the
subject's metadata, and it does not inherit the training benchmarks' average difficulty
(the Phase 3 prior did, and was miscalibrated on the hidden benchmarks). The metadata
score is the Phase 3 regression without its intercept and benchmark effects, plus the
known-subject residual. Each prediction is the exact posterior mean of p given this
pair's own k successes in n labels, computed on a fixed logit grid (accurate to 1e-6 for
sd up to 2); every item of a pair gets the same prediction.

params come from paiec.logit_normal.fit_params. In a submission folder this module sits
next to an unchanged copy of baseline_core.py.
"""

import functools
import math

try:
    from . import baseline_core
except ImportError:  # loaded as a top-level module from a submission folder
    import baseline_core

LIMIT = 6.0  # prior means are kept well inside the grid
_GRID = [-14.0 + 0.01 * i for i in range(2801)]


def _log_sigmoid(z):
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))


_LOG_P = [_log_sigmoid(z) for z in _GRID]
_LOG_Q = [_log_sigmoid(-z) for z in _GRID]
_P = [math.exp(v) for v in _LOG_P]


def metadata_score(subject, params):
    """The Phase 3 metadata logit of a subject, without intercept or benchmark effect."""
    spec = params["spec"]
    standardized = [min(max((v - m) / s, -baseline_core.CLIP), baseline_core.CLIP) for v, m, s in
                    zip(baseline_core.raw_features(subject, spec), spec["means"], spec["stds"])]
    score = math.fsum(w * x for w, x in zip(params["weights"], standardized))
    return score + params["known_subjects"].get(baseline_core.subject_key(subject), 0.0)


def prior_mean(subject, params):
    """m, the prior mean of the subject's logit success rate on a new benchmark."""
    m = params["scale"] * (metadata_score(subject, params) - params["center"])
    return min(max(m, -LIMIT), LIMIT)


@functools.lru_cache(maxsize=65536)
def posterior_mean(m, sd, k, n):
    """E[p | k successes in n labels] for logit p ~ N(m, sd^2)."""
    log_w = [-0.5 * ((z - m) / sd) ** 2 + k * lp + (n - k) * lq
             for z, lp, lq in zip(_GRID, _LOG_P, _LOG_Q)]
    top = max(log_w)
    w = [math.exp(v - top) for v in log_w]
    return math.fsum(wi * p for wi, p in zip(w, _P)) / math.fsum(w)


def predict(input, labeled, params):
    """Posterior mean for this subject and benchmark; metadata problems fall back to m = 0."""
    subject, item = input
    try:
        m = prior_mean(subject, params)
    except Exception:
        m = 0.0
    own = baseline_core.own_labels(subject, item.get("benchmark_id", ""), labeled)
    return posterior_mean(m, params["sd"], sum(own), len(own))
