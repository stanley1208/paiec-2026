import math

import pytest

from paiec import scoring


def test_brier_constant_half_is_a_quarter():
    assert scoring.brier([0.5] * 4, [1, 0, 1, 0]) == 0.25


def test_brier_hand_worked_examples():
    # (0.9-1)^2 + (0.2-0)^2 + (0.6-1)^2 + (0.7-0)^2 = 0.01 + 0.04 + 0.16 + 0.49
    assert scoring.brier([0.9, 0.2, 0.6, 0.7], [1, 0, 1, 0]) == pytest.approx(0.175)
    # Every row counts: three right, one wrong at 0.8 -> (3 * 0.04 + 0.64) / 4
    assert scoring.brier([0.8] * 4, [1, 1, 1, 0]) == pytest.approx(0.19)


def test_brier_rejects_empty_and_mismatched_inputs():
    with pytest.raises(ValueError):
        scoring.brier([], [])
    with pytest.raises(ValueError):
        scoring.brier([0.5, 0.5], [1])


def test_alc_weights_sum_to_one_and_constant_curve_is_flat():
    assert math.fsum(scoring.ALC_WEIGHTS) == 1.0
    assert scoring.alc([0.25] * 6) == 0.25


def test_alc_matches_the_competition_website_example():
    # Rules page: Brier 0.22, 0.18, 0.145, 0.12, 0.1, 0.09 gives Brier ALC 0.140.
    assert scoring.alc([0.22, 0.18, 0.145, 0.12, 0.1, 0.09]) == pytest.approx(0.140)


def test_ece_of_constant_half_is_distance_from_positive_rate():
    # A constant prediction falls in one bin: ECE = |0.5 - 11/40| = 0.225.
    labels = [1] * 11 + [0] * 29
    assert scoring.ece([0.5] * 40, labels) == pytest.approx(0.225)


def test_ece_hand_worked_example_and_bin_edges():
    # Bins [0,0.1): |0.05-0|; [0.1,0.2): |0.15-1|; [0.9,1]: |1.9-1| -> 1.8 / 4
    assert scoring.ece([0.05, 0.15, 0.95, 0.95], [0, 1, 1, 0]) == pytest.approx(0.45)
    assert scoring.ece([1.0], [1]) == 0.0  # 1.0 falls in the last bin
    assert scoring.ece([0.1, 0.1], [0, 1]) == pytest.approx(0.4)
