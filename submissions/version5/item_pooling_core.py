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

Two optional settings, both off unless present in params:
- agent_scale: the metadata scale for subjects with a non-empty harness (agents), whose
  results depend more on the harness than on the model's release date or provider.
- group_sd: a subject-specific effect u ~ N(0, group_sd^2) for each group of items sharing
  the first key=value field of item_features (such as a competition, website or paper),
  learned from the subject's own labels in that group (a testlet model). A group's labels
  enter as their likelihood with u integrated out on a fixed grid, scaled like kappa0 so a
  single label in an otherwise unseen group still counts about as sigmoid(theta).
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
_U = [-10.0 + 0.2 * i for i in range(101)]  # standard normal grid for group effects
_item_ids = {}


def _log_sigmoid(z):
    return logit_normal_core._log_sigmoid(z)


def _item_id(item):
    """A process-local id that is equal exactly when features and content are equal."""
    key = (item.get("item_features", ""), item.get("item_content", ""))
    return _item_ids.setdefault(key, len(_item_ids))


def _group(item):
    """The item's first key=value feature, such as its competition, website or paper."""
    features = item.get("item_features", "")
    first = features.split(";")[0].strip() if isinstance(features, str) else ""
    return first if "=" in first else None


def _prior_mean(subject, params):
    agent_scale = params.get("agent_scale")
    if agent_scale is not None and baseline_core._field(subject, "harness").strip():
        params = dict(params, scale=agent_scale)
    return logit_normal_core.prior_mean(subject, params)


def _prior(subject, item, params, guess):
    """(m, sd) of a subject's theta: from an LLM's guess of its success rate on the benchmark when
    there is one (option C), otherwise the metadata prior with params["sd"]."""
    if guess is not None and "guess_slope" in params:
        try:
            value = guess(subject, item)
            p = None if value is None else float(value)
        except Exception:
            p = None
        if p is not None and p == p:  # a NaN guess counts as no guess
            p = min(max(p, 0.01), 0.99)
            m = params["guess_intercept"] + params["guess_slope"] * math.log(p / (1 - p))
            return min(max(m, -logit_normal_core.LIMIT), logit_normal_core.LIMIT), params["guess_sd"]
    try:
        return _prior_mean(subject, params), params["sd"]
    except Exception:
        return 0.0, params["sd"]


@functools.lru_cache(maxsize=4)
def _group_tables(group_sd):
    kappa = math.sqrt(1 + math.pi * group_sd ** 2 / 8)
    log_weights = [-0.5 * u * u for u in _U]
    total = math.log(math.fsum(math.exp(v) for v in log_weights))
    log_weights = [v - total for v in log_weights]
    shifted = [[kappa * z + group_sd * u for z in _THETA] for u in _U]
    return (log_weights, [[_log_sigmoid(x) for x in row] for row in shifted],
            [[_log_sigmoid(-x) for x in row] for row in shifted])


@functools.lru_cache(maxsize=4096)
def _log_group_likelihood(k, n, group_sd):
    """log P(k of n labels in one group | theta) on the grid, with the group effect integrated out."""
    log_weights, log_p, log_q = _group_tables(group_sd)
    out = []
    for t in range(len(_THETA)):
        terms = [lw + k * lp[t] + (n - k) * lq[t] for lw, lp, lq in zip(log_weights, log_p, log_q)]
        top = max(terms)
        out.append(top + math.log(sum(math.exp(v - top) for v in terms)))
    return tuple(out)


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
def _item_posterior(observations, item_sd, prior_mean=0.0):
    """Mean and variance of b ~ N(prior_mean, item_sd^2) given ((kappa0 * theta mean, scale,
    label), ...) of other subjects."""
    log_w = []
    for b in _B_GRID:
        terms = [-0.5 * ((b - prior_mean) / item_sd) ** 2]
        for centre, scale, label in observations:
            z = (centre - b) / scale
            terms.append(_log_sigmoid(z) if label else _log_sigmoid(-z))
        log_w.append(math.fsum(terms))
    return _moments(log_w, _B_GRID)


@functools.lru_cache(maxsize=512)
def _target_weights(m, sd, k, n, informed, kappa0, grouped=(), group_sd=0.0):
    """Normalized posterior weights of theta on the grid: k of n labels on unknown items,
    informed = ((b, kappa, label), ...) on items of known difficulty, and
    grouped = ((k_g, n_g), ...) for groups of items with a group effect."""
    group_terms = [_log_group_likelihood(k_g, n_g, group_sd) for k_g, n_g in grouped]
    log_w = []
    for t, (z, lp, lq) in enumerate(zip(_THETA, logit_normal_core._LOG_P, logit_normal_core._LOG_Q)):
        terms = [-0.5 * ((z - m) / sd) ** 2, k * lp, (n - k) * lq]
        for b, kappa, label in informed:
            x = (kappa0 * z - b) / kappa
            terms.append(_log_sigmoid(x) if label else _log_sigmoid(-x))
        terms.extend(g[t] for g in group_terms)
        log_w.append(math.fsum(terms))
    top = max(log_w)
    w = [math.exp(v - top) for v in log_w]
    total = math.fsum(w)
    return tuple(wi / total for wi in w)


@functools.lru_cache(maxsize=65536)
def _predictive(m, sd, k, n, informed, kappa0, b, kappa, grouped=(), group_sd=0.0):
    weights = _target_weights(m, sd, k, n, informed, kappa0, grouped, group_sd)
    if b is None:
        return math.fsum(w * p for w, p in zip(weights, logit_normal_core._P))
    return math.fsum(w * math.exp(_log_sigmoid((kappa0 * z - b) / kappa))
                     for w, z in zip(weights, _THETA))


@functools.lru_cache(maxsize=65536)
def _group_predictive(m, sd, k, n, informed, kappa0, grouped, group_sd, k_group, n_group):
    """Prediction for an item in a group where the subject has k_group of n_group labels;
    grouped holds the subject's other groups."""
    weights = _target_weights(m, sd, k, n, informed, kappa0, grouped, group_sd)
    hit = _log_group_likelihood(k_group + 1, n_group + 1, group_sd)
    seen = _log_group_likelihood(k_group, n_group, group_sd)
    top = max(seen)
    return (math.fsum(w * math.exp(h - top) for w, h in zip(weights, hit))
            / math.fsum(w * math.exp(s - top) for w, s in zip(weights, seen)))


def _ratings_reversed(own, others, ratings):
    """True when, over the benchmark's rated labels (at least 3, with both outcomes), failed
    items are rated easier on average than solved ones."""
    rated = [(ratings.get(item_id), label) for item_id, _, label in own]
    rated += [(ratings.get(item_id), label) for _, labels in others.values() for item_id, label in labels]
    rated = [(r, y) for r, y in rated if r is not None]
    failed = [r for r, y in rated if y == 0]
    solved = [r for r, y in rated if y == 1]
    if len(rated) < 3 or not failed or not solved:
        return False
    return math.fsum(failed) / len(failed) - math.fsum(solved) / len(solved) < 0


def predict(input, labeled, params, rating=None, guess=None):
    """Phase 4 prediction, sharpened by item difficulties learned from other subjects' labels.

    rating, if given with params["rating_slope"] and params["rating_sd"], maps an item dict to
    a difficulty rating (or None). An item's difficulty then gets the prior
    N(rating_slope * (rating - mean rating of the labeled items), rating_sd^2), so ratings
    only rank items against those already labeled on the benchmark; with no rated labeled
    item they are not used. With params["rating_gate"], they are also not used while the
    benchmark's rated labels show failed items rated easier than solved ones.

    guess, if given with params["guess_intercept"], params["guess_slope"] and
    params["guess_sd"], maps (subject dict, item dict) to an LLM's guess of the subject's
    success rate on the item's benchmark (or None). A subject with a guess g gets the prior
    theta ~ N(guess_intercept + guess_slope * logit(g), guess_sd^2) instead of the metadata
    prior, for the target subject and for the other subjects whose labels are pooled.
    """
    subject, item = input
    benchmark = item.get("benchmark_id", "")
    use_ratings = rating is not None and "rating_slope" in params
    ratings = {}

    def item_key(i):
        item_id = _item_id(i)
        if use_ratings and item_id not in ratings:
            try:
                value = rating(i)
                ratings[item_id] = None if value is None else float(value)
            except Exception:
                ratings[item_id] = None
        return item_id

    own, others = [], {}
    for entry in labeled:
        try:
            (s, i), label = entry
            if i.get("benchmark_id", "") != benchmark or label not in (0, 1):
                continue
            if s == subject:
                own.append((item_key(i), _group(i), int(label)))
            else:
                others.setdefault(baseline_core.subject_key(s), (s, []))[1].append(
                    (item_key(i), int(label)))
        except Exception:
            continue
    labeled_ratings = [ratings[x] for x in ratings if ratings[x] is not None]
    rating_mean = math.fsum(labeled_ratings) / len(labeled_ratings) if labeled_ratings else None
    if rating_mean is not None and params.get("rating_gate") and _ratings_reversed(own, others, ratings):
        rating_mean = None  # the ratings disagree with this benchmark's labels: do not use them
    target_id = item_key(item)
    m, sd = _prior(subject, item, params, guess)
    group_sd = params.get("group_sd") or 0.0
    if not others and not group_sd and rating_mean is None:
        return logit_normal_core.posterior_mean(m, sd, sum(y for _, _, y in own), len(own))

    kappa0, observed = 1.0, {}
    if others or rating_mean is not None:
        item_sd = params["item_sd"]
        kappa0 = math.sqrt(1 + math.pi * item_sd ** 2 / 8)

    def rated_prior(item_id):
        """(mean, sd) of a rated item's difficulty, or None for an unrated item."""
        if rating_mean is None or ratings.get(item_id) is None:
            return None
        return params["rating_slope"] * (ratings[item_id] - rating_mean), params["rating_sd"]
    for s, labels in others.values():
        m_other, sd_other = _prior(s, item, params, guess)
        k_other, n_other = sum(y for _, y in labels), len(labels)
        for item_id, label in labels:
            mean, var = _theta_moments(m_other, sd_other, k_other - label, n_other - 1)
            scale = math.sqrt(1 + math.pi * kappa0 ** 2 * var / 8)
            observed.setdefault(item_id, []).append((kappa0 * mean, scale, label))

    def known(item_id):
        rated = rated_prior(item_id)
        if item_id in observed and rated is None:
            b, var = _item_posterior(tuple(sorted(observed[item_id])), item_sd)
        elif item_id in observed:
            b, var = _item_posterior(tuple(sorted(observed[item_id])), rated[1], rated[0])
        elif rated is not None:
            b, var = rated[0], rated[1] ** 2
        else:
            return None
        return b, math.sqrt(1 + math.pi * var / 8)

    informed, plain, groups = [], [], {}
    for item_id, group, label in own:
        b = known(item_id)
        if b is not None:
            informed.append((b[0], b[1], label))
        elif group_sd and group is not None:
            groups.setdefault(group, []).append(label)
        else:
            plain.append(label)
    k, n = sum(plain), len(plain)
    informed = tuple(sorted(informed))
    target = known(target_id)
    target_group = groups.get(_group(item)) if group_sd and target is None else None
    grouped = tuple(sorted((sum(v), len(v)) for g, v in groups.items()
                           if target_group is None or g != _group(item)))
    if target_group is not None:
        return _group_predictive(m, sd, k, n, informed, kappa0, grouped, group_sd,
                                 sum(target_group), len(target_group))
    if not informed and target is None and not grouped:
        return logit_normal_core.posterior_mean(m, sd, k, n)
    b, kappa = target if target is not None else (None, None)
    return _predictive(m, sd, k, n, informed, kappa0, b, kappa, grouped, group_sd)
