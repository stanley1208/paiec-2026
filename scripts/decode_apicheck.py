"""Read the answer of submissions/apicheck from its Codabench feedback (saved as a text file).

    python scripts/decode_apicheck.py private/codabench/<id>_scoring_output.txt
    python scripts/decode_apicheck.py --selftest

The probe predicts one constant c per checkpoint (see submissions/apicheck/model.py). For a
constant c and a pair whose share of solved items is r, Brier = c^2 + r(1 - 2c) and ECE =
|c - r|, so Brier - ECE^2 = r(1 - r) is the same at every budget of a pair: a budget where it is
not was a mixture of codes (the workers disagreed). Otherwise one code per budget and one r per
pair must fit every pair's Brier and ECE; with two or more pairs of different r the answer is
unique. A code that equals 0.5 everywhere gives Brier exactly 0.250000 at every budget.
"""

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paiec import feedback  # noqa: E402

BUDGETS = (0, 1, 3, 7, 15, 31)
MEANING = {0.5: "works: API answered 401, folder writable, earlier checkpoints' files visible",
           0.45: "API and folder work; only markers from outside the prediction workers are visible",
           0.4: "API and folder work, but earlier checkpoints' files are not visible",
           0.3: "the probe itself failed",
           0.2: "the API works, the folder is not writable",
           0.1: "the API did not answer 401 within 6 seconds"}
TOLERANCE = 2e-6  # the feedback rounds to 6 decimals


def _rates(brier, ece, c):
    return [r for r in (c - ece, c + ece)
            if -TOLERANCE <= r <= 1 + TOLERANCE and abs(brier - (c * c + r * (1 - 2 * c))) <= 4 * TOLERANCE]


def decode(pairs):
    """(budgets that were mixtures, budgets read, list of code tuples that fit those budgets)."""
    mixed = set()
    for pair in pairs.values():
        q = {b: pair["brier"][b] - pair["ece"][b] ** 2 for b in BUDGETS}
        middle = sorted(q.values())[len(q) // 2]
        mixed |= {b for b in BUDGETS if abs(q[b] - middle) > 4 * TOLERANCE}
    read = [b for b in BUDGETS if b not in mixed]

    def fits(pair, combo):
        candidates = [_rates(pair["brier"][b], pair["ece"][b], c) for b, c in zip(read, combo)]
        return any(all(any(abs(r - other) <= 4 * TOLERANCE for other in rates) for rates in candidates[1:])
                   for r in candidates[0]) if candidates else True

    solutions = [combo for combo in itertools.product(sorted(MEANING), repeat=len(read))
                 if all(fits(pair, combo) for pair in pairs.values())]
    return sorted(mixed), read, solutions


def report(pairs):
    mixed, read, solutions = decode(pairs)
    lines = [f"{len(pairs)} pairs"]
    if not mixed and all(abs(p["brier"][b] - 0.25) <= TOLERANCE for p in pairs.values() for b in BUDGETS):
        lines.append("ALL GOOD: Brier 0.250000 at every budget of every pair (code 0.5 everywhere)")
    if mixed:
        lines.append(f"budgets {mixed}: the workers disagreed (a mixture of codes); see the other budgets")
    if len(solutions) == 1:
        lines += [f"budget {b:2d}: {c:<4}  {MEANING[c]}" for b, c in zip(read, solutions[0])]
    else:
        lines.append(f"{len(solutions)} code assignments fit budgets {read}: {solutions[:6]}")
    return "\n".join(lines)


def synthetic(codes, pairs=8, seed=0, mix=None):
    """Feedback text for a run whose checkpoints predicted these codes (mix: {budget: (code, share)})."""
    rng = random.Random(seed)
    rows = []
    for k in range(pairs):
        n = rng.randint(44, 113)
        rows.append((f"subject_{100 + k}", f"benchmark_{200 + k}", n,
                     [int(rng.random() < rng.random()) for _ in range(n)]))
    lines = []
    for budget, c in zip(BUDGETS, codes):
        lines.append(f"Label budget: {budget}")
        for subject, benchmark, n, labels in rows:
            predictions = [mix[budget][0] if mix and budget in mix and rng.random() < mix[budget][1] else c
                           for _ in labels]
            brier = sum((p - y) ** 2 for p, y in zip(predictions, labels)) / n
            bins = {}
            for p, y in zip(predictions, labels):
                cell = bins.setdefault(min(int(p * 10), 9), [0.0, 0.0])
                cell[0] += p
                cell[1] += y
            ece = sum(abs(sp - sy) for sp, sy in bins.values()) / n
            lines.append(f"{subject} {benchmark} {n} {brier:.6f} {ece:.6f}")
    return "\n".join(lines)


def main():
    if sys.argv[1:] == ["--selftest"]:
        cases = [("all good", (0.5,) * 6, None), ("copies per checkpoint", (0.5,) + (0.4,) * 5, None),
                 ("copies, with a validation call", (0.5,) + (0.45,) * 5, None),
                 ("API down at budget 0", (0.1,) + (0.5,) * 5, None),
                 ("one worker failed at budget 3", (0.5,) * 6, {3: (0.1, 0.3)})]
        for name, codes, mix in cases:
            print(f"--- {name}\n{report(feedback.parse(synthetic(codes, mix=mix)))}")
        return
    print(report(feedback.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
