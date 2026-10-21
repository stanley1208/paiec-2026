from paiec import summary
from paiec.scoring import BUDGETS


def rows(alc_by_pair):
    return [{"seed": 0, "benchmark": b, "subject_id": s, "n_items": n, "alc": a, "ece_alc": 0.0}
            for b, s, n, a in alc_by_pair]


def test_draws_respect_the_item_cap_and_spread_across_benchmarks():
    pairs = rows([("A", "a1", 300, 0.1), ("A", "a2", 300, 0.1), ("A", "a3", 300, 0.1),
                  ("B", "b1", 300, 0.3), ("B", "b2", 300, 0.3), ("B", "b3", 300, 0.3)])
    result = summary.simulate_draws(pairs, n_draws=200, cap=1000, seed=0)
    assert result["pairs_per_draw_mean"] == 3  # three 300-item pairs fit under 1,000
    # Two of one benchmark and one of the other: mean ALC is 1/6 or 7/30.
    assert 0.1 < result["alc_p05"] <= result["alc_p95"] < 0.3


def test_draws_are_deterministic():
    pairs = rows([("A", f"a{i}", 100 + 10 * i, 0.01 * i) for i in range(12)])
    assert summary.simulate_draws(pairs, 50, seed=3) == summary.simulate_draws(pairs, 50, seed=3)


def test_summary_balances_benchmarks():
    pairs = rows([("A", "a1", 100, 0.1), ("A", "a2", 100, 0.1), ("A", "a3", 100, 0.1),
                  ("B", "b1", 100, 0.3)])
    pairs = [dict(r, **{f"brier_{b}": r["alc"] for b in BUDGETS}) for r in pairs]
    result = summary.summarize_seed(pairs)
    assert abs(result["alc_pair_mean"] - 0.15) < 1e-12  # (3 * 0.1 + 0.3) / 4
    assert abs(result["alc_benchmark_mean"] - 0.2) < 1e-12  # (0.1 + 0.3) / 2
