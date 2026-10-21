"""Inference for the Phase 3 baselines, standard library only.

The harness imports this module, and scripts/export_submission.py copies it unchanged
into a submission folder, so offline and hosted predictions share one implementation.
params are produced by paiec.baselines.fit_params.

Only subject features that vary within more than one public benchmark are used, so
leave-one-benchmark-out evaluation actually tests how they transfer.
"""

import json
import math
import re

FEATURE_NAMES = ("release_year", "release_missing", "thinking_variant")
CLIP = 3.0  # standardized features beyond +-3 are capped


def _year(text):
    match = re.match(r"^\s*(\d{4})(?:[-/.](\d{1,2}))?", text)
    if not match:
        return None
    month = int(match.group(2) or 1)
    return int(match.group(1)) + (min(max(month, 1), 12) - 1) / 12


def _field(subject, name):
    value = subject.get(name, "")
    return value if isinstance(value, str) else ""


def subject_key(subject):
    return json.dumps(subject, sort_keys=True, separators=(",", ":"))


def raw_features(subject, spec):
    """Unstandardized subject features: FEATURE_NAMES, then one indicator per provider."""
    release = _year(_field(subject, "release_date"))
    provider = _field(subject, "provider").strip().lower()
    text = f"{_field(subject, 'normalized_name')} {_field(subject, 'subject_features_extra')}".lower()
    values = [spec["release_fill"] if release is None else release, float(release is None),
              float("think" in text or "reason" in text)]
    return values + [float(provider == p) for p in spec["providers"]]


def _sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def subject_prior(subject, params):
    """Success probability before any label: the metadata model averaged over the
    training benchmarks' difficulty, since the test benchmark's difficulty is unknown."""
    if params["prior"] == "global":
        return params["global_mean"]
    spec = params["spec"]
    standardized = [min(max((v - m) / s, -CLIP), CLIP) for v, m, s in
                    zip(raw_features(subject, spec), spec["means"], spec["stds"])]
    z = params["intercept"] + math.fsum(w * x for w, x in zip(params["weights"], standardized))
    z += params["known_subjects"].get(subject_key(subject), 0.0)
    offsets = params["benchmark_offsets"]
    return math.fsum(_sigmoid(z + a) for a in offsets) / len(offsets)


def own_labels(subject, benchmark, labeled):
    """0/1 labels revealed for this subject and benchmark; malformed entries are skipped."""
    own = []
    for entry in labeled:
        try:
            (s, i), label = entry
            if s == subject and i.get("benchmark_id", "") == benchmark and label in (0, 1):
                own.append(int(label))
        except Exception:
            continue
    return own


def predict(input, labeled, params):
    """Prior, optionally updated Beta-Binomial style with this pair's revealed labels."""
    subject, item = input
    try:
        prior = subject_prior(subject, params)
    except Exception:
        prior = params["global_mean"]
    strength = params["strength"]
    if strength is None:
        return prior
    own = own_labels(subject, item.get("benchmark_id", ""), labeled)
    return (strength * prior + sum(own)) / (strength + len(own))
