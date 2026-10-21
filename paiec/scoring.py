"""Scoring as described by the organizers: Brier score, ECE and Brier ALC.

Within a subject-benchmark pair every recorded response counts; pairs are then
averaged with equal weight.
"""

import math

BUDGETS = (0, 1, 3, 7, 15, 31)
# Normalized trapezoid area on log2(1 + budget): endpoints 0.1, interior points 0.2.
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)


def brier(predictions, labels):
    """Mean squared difference between probabilities and 0/1 outcomes."""
    if not labels:
        raise ValueError("brier() needs at least one outcome")
    return math.fsum((p - y) ** 2 for p, y in zip(predictions, labels, strict=True)) / len(labels)


def ece(predictions, labels, bins=10):
    """Count-weighted mean |mean prediction - positive rate| over [0, 0.1), ..., [0.9, 1]."""
    totals = [[0, 0.0, 0.0] for _ in range(bins)]
    for p, y in zip(predictions, labels, strict=True):
        cell = totals[min(int(p * bins), bins - 1)]
        cell[0] += 1
        cell[1] += p
        cell[2] += y
    return math.fsum(abs(s_p - s_y) for n, s_p, s_y in totals if n) / len(labels)


def alc(values):
    """Weighted area under the learning curve over the six budgets."""
    if len(values) != len(ALC_WEIGHTS):
        raise ValueError(f"alc() needs {len(ALC_WEIGHTS)} values, one per budget")
    return math.fsum(w * v for w, v in zip(ALC_WEIGHTS, values))
