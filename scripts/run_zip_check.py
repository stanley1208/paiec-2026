"""Score the packaged version 4 and version 5 folders end to end, with the saved ratings and guesses.

    python scripts/run_zip_check.py

A check of the packaging, not a model comparison: each submission folder is copied to a scratch
folder with a dummy key file and a rating_cache/ pre-filled with the saved Claude replies
(results/llm_rating ratings by item, results/subject_prior guesses by subject and benchmark), so
the submission's own code runs exactly as hosted but finds every reply cached and calls no API
(the script fails if any API attempt is made). Both run through the harness as submissions (fresh
modules every checkpoint) on the pairs of scripts/run_final.py with K = 0, seeds 0-2. Their
parameters were fitted on all public benchmarks, so the scores are in-sample and differ a little
from the leave-one-benchmark-out test (scripts/run_subject_prior.py); what should carry over is
version 5 beating version 4 by about the tested margin. Writes nothing in the repository.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ratings as rr  # noqa: E402
import run_subject_prior as rsp  # noqa: E402
from run_pooling import _mean, _paired  # noqa: E402

from paiec import data, guess_core, llm_rating_core, protocol  # noqa: E402
from paiec.predictors import SubmissionPredictor  # noqa: E402

FOLDERS = {"version4": ROOT / "submissions" / "version4", "version5": ROOT / "submissions" / "version5"}
DUMMY_KEY = "sk-ant-api03-" + "DUMMYKEYFORCACHEDRUNS" * 3 + "AA"


def prepare(name, scratch, dataset, by_content, guesses):
    folder = scratch / name
    shutil.copytree(FOLDERS[name], folder)
    (folder / llm_rating_core.KEY_FILE).write_text(DUMMY_KEY, encoding="utf-8")
    cache = folder / "rating_cache"
    (cache / "calls").mkdir(parents=True)
    for (features, content), rating in by_content.items():
        key = llm_rating_core.item_key({"item_features": features, "item_content": content})
        (cache / f"{key}.json").write_text(json.dumps({"rating": rating}), encoding="utf-8")
    for key, row in guesses.items():
        (cache / f"guess-{key}.json").write_text(json.dumps({"rating": row["guess"]}), encoding="utf-8")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    dataset = data.load()
    _, rated_items, by_content = rr.load_ratings(dataset)
    rr._state.update(dataset=dataset, rated_items=rated_items)
    guesses = rsp.load_guesses()
    scratch = Path(tempfile.mkdtemp(prefix="paiec_zip_check_"))
    rows = {}
    started = time.time()
    try:
        for name in FOLDERS:
            folder = prepare(name, scratch, dataset, by_content, guesses)
            rows[name] = []
            for seed in args.seeds:
                settings = protocol.Settings(seed=seed, evidence="partners", partners=0,
                                             min_pair_items=rr.MIN_PAIR_ITEMS)
                for benchmark in rr.RATED:
                    responses = dataset.responses
                    keep = (responses["benchmark"] != benchmark) | responses["item_id"].isin(rated_items[benchmark])
                    cut = data.Dataset(responses[keep].reset_index(drop=True), dataset.subjects, dataset.items)
                    rows[name] += protocol.run_fold(cut, benchmark, lambda: SubmissionPredictor(folder), settings)
                print(f"{name} seed {seed}: {len(rows[name])} pair results, {time.time() - started:.0f}s", flush=True)
            calls = len(list((folder / "rating_cache" / "calls").iterdir()))
            if calls:
                raise SystemExit(f"{name} made {calls} API attempts; the cache was incomplete")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    v4, v5 = rows["version4"], rows["version5"]
    print(f"version 4 zip: ALC {_mean([r['alc'] for r in v4]):.4f}, budget-0 Brier {_mean([r['brier_0'] for r in v4]):.4f}")
    print(f"version 5 zip: ALC {_mean([r['alc'] for r in v5]):.4f}, budget-0 Brier {_mean([r['brier_0'] for r in v5]):.4f}")
    diff = _paired(v5, v4)
    print(f"version 5 minus version 4: {diff['alc_diff']:+.5f} (SE {diff['alc_diff_se']:.5f}); by benchmark: "
          + " ".join(f"{b} {_paired([r for r in v5 if r['benchmark'] == b], [r for r in v4 if r['benchmark'] == b])['alc_diff']:+.4f}"
                     for b in rr.RATED))
    print("no API attempts in either run")


if __name__ == "__main__":
    main()
