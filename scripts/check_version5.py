"""Decide from version 5's Codabench feedback whether it replaces version 4 as the final entry.

    python scripts/check_version5.py private/codabench/<id>_scoring_output.txt

Decision rule, fixed before version 5 was uploaded (2026-10-20). Version 4 predicts exactly 0.5
for every item at budget 0 (its prior is symmetric and ratings are not used before any label),
so its budget-0 Brier score is exactly 0.25 on every pair of every draw. Version 5 differs only by
the subject guesses, which act from budget 0. So the budget-0 rows of version 5's feedback compare
the two versions on the same pairs. Version 5 becomes the final entry only if all of:
1. the feedback has all six budgets and the ALC summary for every pair;
2. the budget-0 Brier score differs from 0.250000 on at least half of the pairs (a pair whose
   guess did not arrive by budget 0 scores exactly 0.25, so this checks that the guesses reached
   most pairs in time);
3. the mean budget-0 Brier score over the pairs is below 0.25.
Otherwise version 4 (Codabench 968769) stays final. This uses formative feedback to choose between
two versions built and tested beforehand. The report says so, as it does for version 3. The
feedback file is private and stays in private/, and this script prints only its own summary.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paiec import feedback, scoring  # noqa: E402

VERSION4_BUDGET0 = 0.25
TOLERANCE = 1e-6  # the feedback prints 6 decimals


def decide(text):
    pairs = feedback.parse(text)
    if not pairs:
        return {"decision": "version 4", "reason": "no feedback rows found"}
    if any(sorted(p["brier"]) != list(scoring.BUDGETS) or "alc" not in p for p in pairs.values()):
        return {"decision": "version 4", "reason": "some pairs lack a budget or the ALC summary"}
    budget0 = [p["brier"][0] for p in pairs.values()]
    mean = sum(budget0) / len(budget0)
    guessed = sum(abs(b - VERSION4_BUDGET0) > TOLERANCE for b in budget0)
    result = {"pairs": len(budget0), "mean_budget0": mean, "pairs_guessed": guessed,
              "pairs_better": sum(b < VERSION4_BUDGET0 - TOLERANCE for b in budget0),
              "mean_alc": sum(p["alc"] for p in pairs.values()) / len(pairs)}
    if 2 * guessed < len(budget0):
        return dict(result, decision="version 4", reason="guesses did not reach half of the pairs by budget 0")
    return dict(result, decision="version 5" if mean < VERSION4_BUDGET0 else "version 4")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path, help="Codabench scoring output saved as a text file")
    args = parser.parse_args()
    result = decide(args.file.read_text(encoding="utf-8"))
    if "mean_budget0" in result:
        print(f"{result['pairs']} pairs; mean budget-0 Brier {result['mean_budget0']:.6f} "
              f"(version 4: {VERSION4_BUDGET0}); guessed on {result['pairs_guessed']} pairs, below 0.25 on "
              f"{result['pairs_better']}; mean Brier ALC {result['mean_alc']:.6f}")
    if "reason" in result:
        print(result["reason"])
    print(f"final entry: {result['decision']}")


if __name__ == "__main__":
    main()
