"""Check that our ALC formula reproduces Codabench's numbers from a saved feedback file.

    python scripts/check_feedback.py private/codabench/951242_scoring_output.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paiec import feedback  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path, help="Codabench scoring output pasted into a text file")
    args = parser.parse_args()
    report, mean = feedback.check(args.file.read_text(encoding="utf-8"))
    for row in report:
        subject, benchmark = row["pair"]
        status = "ok" if row["ok"] else "MISMATCH"
        print(f"{subject} {benchmark} n={row['n']:>4}  reported {row['reported']:.6f}"
              f"  recomputed {row['recomputed']:.6f}  {status}")
    print(f"mean Brier ALC over {len(report)} pairs: {mean:.6f}")
    if not all(row["ok"] for row in report):
        raise SystemExit("some pairs do not match")


if __name__ == "__main__":
    main()
