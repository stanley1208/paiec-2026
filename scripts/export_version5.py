"""Write version 5 (version 4 plus Claude's guess of each subject's success rate) as a submission folder.

    python scripts/export_version5.py submissions/version5

Version 5 keeps every parameter of the uploaded version 4 (read from submissions/version4/
params.json) and adds the subject guesses of option C at the guess trust chosen by
scripts/run_subject_prior.py (0.75, results/runs/20261017-021515_subject_prior). The calibration
a, c, s is the same least-squares fit, now on all four rated public benchmarks, using the guesses
in results/subject_prior/guesses.jsonl, which were asked with the request a hosted run sends
(paiec/guess_core.py). The folder has no API key; scripts/add_api_key.py makes a private zip with
one, and without it the submission predicts exactly like version 3.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_subject_prior as rsp  # noqa: E402

from paiec import data  # noqa: E402
from paiec.export import write_submission  # noqa: E402

GUESS_TRUST = 0.75
RUN = ROOT / "results" / "runs" / "20261017-021515_subject_prior"
VERSION4 = ROOT / "submissions" / "version4" / "params.json"


def version5_params(dataset):
    chosen = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))["decision"]["chosen_trust"]
    if chosen != GUESS_TRUST:
        raise SystemExit(f"the test chose trust {chosen}, not {GUESS_TRUST}")
    params = json.loads(VERSION4.read_text(encoding="utf-8"))
    fit = rsp.fit_guess(dataset, rsp.load_guesses(), held_out=None)
    t, sd0 = GUESS_TRUST, params["sd"]
    params.update(model="item_pooling_guessed", guess_trust=t, guess_fit=fit,
                  guess_intercept=t * fit["a"], guess_slope=t * fit["c"],
                  guess_sd=math.sqrt((1 - t ** 2) * sd0 ** 2 + t ** 2 * fit["s"] ** 2),
                  trained_on=params["trained_on"] + "; subject guesses in results/subject_prior")
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    params = version5_params(data.load())
    description = ("Version 5: version 4 plus Claude's guess of each subject's success rate as its "
                   f"prior, at guess trust {GUESS_TRUST}, fitted on all public binary benchmarks.")
    out = write_submission(params, args.out, description)
    fit = params["guess_fit"]
    print(f"Wrote {out}: {', '.join(sorted(p.name for p in out.iterdir() if p.is_file()))}")
    print(f"guess fit a {fit['a']:+.3f}, c {fit['c']:.3f}, s {fit['s']:.3f} on {fit['fit_pairs']} pairs; "
          f"prior N({params['guess_intercept']:+.3f} + {params['guess_slope']:.3f} logit(g), "
          f"{params['guess_sd']:.3f}^2)")


if __name__ == "__main__":
    main()
