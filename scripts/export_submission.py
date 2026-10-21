"""Fit a model on all public binary benchmarks and write it as a submission folder.

    python scripts/export_submission.py beta_binomial_subject submissions/baseline
    python scripts/export_submission.py logit_normal submissions/phase4 --options '{"scale": 0.5}'
    python scripts/export_submission.py item_pooling submissions/version2
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paiec import data  # noqa: E402
from paiec.export import submission_kind, write_submission  # noqa: E402
from paiec.predictors import PREDICTORS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictor", choices=[p for p in sorted(PREDICTORS) if p != "constant"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--options", type=json.loads, default={})
    args = parser.parse_args()

    predictor = PREDICTORS[args.predictor](**args.options)
    predictor.fit(data.load())
    params = dict(predictor.params, trained_on=f"{data.REPO_ID}@{data.REVISION}, binary-scale items")
    kind = {"baseline": "Phase 3 baseline", "logit_normal": "Phase 4 model",
            "item_pooling": "Phase 4 model with item pooling"}[submission_kind(params)]
    description = (f"{kind} '{args.predictor}' (options {json.dumps(args.options)}), "
                   f"fitted on all public binary benchmarks.")
    out = write_submission(params, args.out, description)
    print(f"Wrote {out}: {', '.join(sorted(p.name for p in out.iterdir() if p.is_file()))}")


if __name__ == "__main__":
    main()
