"""Inference for the Phase 4 model with item difficulties pooled from other subjects, stdlib only.

The platform passes every label bought in the run, including other subjects' labels on the
same benchmark. When those subjects answered an item, their labels say whether it is easier
or harder than average. A Rasch model on that benchmark,

    P(subject s answers item i) = sigmoid(kappa0 * theta_s - b_i),  b_i ~ N(0, item_sd^2),

is fitted in two stages (as in a two-stage calibration):
1. for each label of another subject, that subject's theta from its other labels on the
   benchmark, with the Phase 4 prior and all items treated as unknown (the Phase 4
   posterior; leaving the label out keeps it from counting twice);
2. each item's b from the other subjects' labels on it, on a grid, using those thetas.
The subject being predicted then gets the Phase 4 posterior, except that its own labels on
items with a known b use that b, and the prediction for an item with a known b uses it.
theta is on the Phase 4 rate-logit scale; kappa0 = sqrt(1 + pi * item_sd^2 / 8) makes an
item of unknown difficulty give about sigmoid(theta) (the probit approximation), and with
no other subjects' labels on the relevant items the prediction is exactly logit_normal_core's.

Items carry no id, so they are matched by (item_features, item_content) within one process.
"""

import functools
import math

try:
    from . import baseline_core, logit_normal_core
except ImportError:  # loaded as a top-level module from a submission folder
    import baseline_core
    import logit_normal_core

_THETA = logit_normal_core._GRID
_B_GRID = [-12.0 + 0.02 * i for i in range(1201)]
_item_ids = {}


def _log_sigmoid(z):
    return logit_normal_core._log_sigmoid(z)


def _item_id(item):
    """A process-local id that is equal exactly when features and content are equal."""
    key = (item.get("item_features", ""), item.get("item_content", ""))
    return _item_ids.setdefault(key, len(_item_ids))


def _moments(log_w, grid):
    top = max(log_w)
    w = [math.exp(v - top) for v in log_w]
    total = math.fsum(w)
    mean = math.fsum(wi * x for wi, x in zip(w, grid)) / total
    var = math.fsum(wi * (x - mean) ** 2 for wi, x in zip(w, grid)) / total
    return mean, var


@functools.lru_cache(maxsize=65536)
def _theta_moments(m, sd, k, n):
    """Mean and variance of theta under the Phase 4 posterior (items unknown)."""
    log_w = [-0.5 * ((z - m) / sd) ** 2 + k * lp + (n - k) * lq
             for z, lp, lq in zip(_THETA, logit_normal_core._LOG_P, logit_normal_core._LOG_Q)]
    return _moments(log_w, _THETA)


@functools.lru_cache(maxsize=65536)
def _item_posterior(observations, item_sd):
    """Mean and variance of b given ((kappa0 * theta mean, scale, label), ...) of other subjects."""
    log_w = []
    for b in _B_GRID:
        terms = [-0.5 * (b / item_sd) ** 2]
        for centre, scale, label in observations:
            z = (centre - b) / scale
            terms.append(_log_sigmoid(z) if label else _log_sigmoid(-z))
        log_w.append(math.fsum(terms))
    return _moments(log_w, _B_GRID)


@functools.lru_cache(maxsize=512)
def _target_weights(m, sd, k, n, informed, kappa0):
    """Normalized posterior weights of theta on the grid: k of n labels on unknown items, plus
    informed = ((b, kappa, label), ...) on items of known difficulty."""
    log_w = []
    for z, lp, lq in zip(_THETA, logit_normal_core._LOG_P, logit_normal_core._LOG_Q):
        terms = [-0.5 * ((z - m) / sd) ** 2, k * lp, (n - k) * lq]
        for b, kappa, label in informed:
            x = (kappa0 * z - b) / kappa
            terms.append(_log_sigmoid(x) if label else _log_sigmoid(-x))
        log_w.append(math.fsum(terms))
    top = max(log_w)
    w = [math.exp(v - top) for v in log_w]
    total = math.fsum(w)
    return tuple(wi / total for wi in w)


@functools.lru_cache(maxsize=65536)
def _predictive(m, sd, k, n, informed, kappa0, b, kappa):
    weights = _target_weights(m, sd, k, n, informed, kappa0)
    if b is None:
        return math.fsum(w * p for w, p in zip(weights, logit_normal_core._P))
    return math.fsum(w * math.exp(_log_sigmoid((kappa0 * z - b) / kappa))
                     for w, z in zip(weights, _THETA))


def predict(input, labeled, params):
    """Phase 4 prediction, sharpened by item difficulties learned from other subjects' labels."""
    subject, item = input
    benchmark = item.get("benchmark_id", "")
    own, others = [], {}
    for entry in labeled:
        try:
            (s, i), label = entry
            if i.get("benchmark_id", "") != benchmark or label not in (0, 1):
                continue
            if s == subject:
                own.append((_item_id(i), int(label)))
            else:
                others.setdefault(baseline_core.subject_key(s), (s, []))[1].append(
                    (_item_id(i), int(label)))
        except Exception:
            continue
    sd = params["sd"]
    try:
        m = logit_normal_core.prior_mean(subject, params)
    except Exception:
        m = 0.0
    k, n = sum(label for _, label in own), len(own)
    if not others:
        return logit_normal_core.posterior_mean(m, sd, k, n)

    item_sd = params["item_sd"]
    kappa0 = math.sqrt(1 + math.pi * item_sd ** 2 / 8)
    observed = {}
    for s, labels in others.values():
        try:
            m_other = logit_normal_core.prior_mean(s, params)
        except Exception:
            m_other = 0.0
        k_other, n_other = sum(y for _, y in labels), len(labels)
        for item_id, label in labels:
            mean, var = _theta_moments(m_other, sd, k_other - label, n_other - 1)
            scale = math.sqrt(1 + math.pi * kappa0 ** 2 * var / 8)
            observed.setdefault(item_id, []).append((kappa0 * mean, scale, label))

    def known(item_id):
        if item_id not in observed:
            return None
        b, var = _item_posterior(tuple(sorted(observed[item_id])), item_sd)
        return b, math.sqrt(1 + math.pi * var / 8)

    informed = []
    for item_id, label in own:
        b = known(item_id)
        if b is not None:
            informed.append((b[0], b[1], label))
            k, n = k - label, n - 1
    target = known(_item_id(item))
    if not informed and target is None:
        return logit_normal_core.posterior_mean(m, sd, k, n)
    b, kappa = target if target is not None else (None, None)
    return _predictive(m, sd, k, n, tuple(sorted(informed)), kappa0, b, kappa)
