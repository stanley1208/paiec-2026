"""Fit version 4 on all public binary benchmarks and write it as a submission folder.

    python scripts/export_version4.py submissions/version4

Version 4 is version 3 (item pooling, scale 0) plus LLM difficulty ratings at trust 0.5 (the
level chosen by scripts/run_trust.py), switched off on a benchmark whose rated labels show
failed items rated easier than solved ones (rating_gate). The rating slope and spread are
fitted on all four rated public benchmarks (results/llm_rating/) and shrunk as in
run_trust.py. The folder has no API key; scripts/add_api_key.py makes a private zip with one,
and without it the submission predicts exactly like version 3.
"""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_ratings  # noqa: E402

from paiec import data  # noqa: E402
from paiec.export import write_submission  # noqa: E402
from paiec.item_pooling import ItemPooling  # noqa: E402

TRUST = 0.5
# One API attempt takes about timeout seconds at most; a prediction waits at most
# budget_seconds for ratings; an abandoned lock is taken over after stale_seconds; max_calls
# caps the attempts that may be billed.
RATING_API = {"model": "claude-opus-5-5", "effort": "low", "timeout": 20, "threads": 8,
              "budget_seconds": 4, "stale_seconds": 45, "max_calls": 1000, "max_chars": 150000}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    dataset = data.load()
    model = ItemPooling(scale=0.0)
    model.fit(dataset)
    by_key, _, _ = run_ratings.load_ratings(dataset)
    slope, spread = run_ratings.rating_fit(by_key, held_out=None)
    item_sd = model.params["item_sd"]
    explained = max(item_sd ** 2 - spread ** 2, 0.0)
    params = dict(model.params, model="item_pooling_rated", rating_trust=TRUST, rating_gate=True,
                  rating_slope=TRUST * slope, rating_sd=math.sqrt(item_sd ** 2 - TRUST ** 2 * explained),
                  rating_fit={"slope": slope, "sd": spread}, rating_api=RATING_API,
                  trained_on=f"{data.REPO_ID}@{data.REVISION}, binary-scale items; "
                             f"ratings in results/llm_rating")
    description = ("Version 4: version 3 (item pooling, scale 0) plus LLM difficulty ratings at "
                   f"trust {TRUST}, fitted on all public binary benchmarks.")
    out = write_submission(params, args.out, description)
    print(f"Wrote {out}: {', '.join(sorted(p.name for p in out.iterdir() if p.is_file()))}")
    print(f"rating slope {slope:.3f} x trust {TRUST} = {params['rating_slope']:.3f}; "
          f"rating sd {params['rating_sd']:.3f} (fitted residual sd {spread:.3f})")


if __name__ == "__main__":
    main()
