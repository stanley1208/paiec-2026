"""Score a predictor with the offline leave-one-benchmark-out harness.

    python scripts/run_harness.py --predictor constant
    python scripts/run_harness.py --submission submissions/constant
    python scripts/run_harness.py --config configs/full.json --predictor constant

Writes results/runs/<timestamp>_<name>/ with config.json (settings and provenance,
written before the run), per_pair.csv and summary.json.
"""

import argparse
import csv
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paiec import data, protocol, summary  # noqa: E402
from paiec.predictors import PREDICTORS, SubmissionPredictor  # noqa: E402

SETTING_KEYS = ("min_pair_items", "max_pair_items", "evidence", "partners", "split", "strict_copies",
                "pool_items")
CONFIG_KEYS = {"description", "seeds", "draws", *SETTING_KEYS}


def git(*args):
    """Raw stdout bytes of a git command run in the repository."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True).stdout


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def provenance(submission):
    status = git("status", "--porcelain").decode("utf-8", "replace").splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").decode("utf-8", "replace")
    files = {}
    if submission:
        files = {p.relative_to(submission).as_posix(): file_sha256(p)
                 for p in sorted(submission.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts}
    return {
        "git_commit": git("rev-parse", "HEAD").decode().strip(),
        "git_status": status,
        "git_diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest() if status else None,
        "untracked_files": {f: file_sha256(ROOT / f) for f in untracked.splitlines()},
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("pandas", "pyarrow", "numpy")},
        "data_repo": data.REPO_ID,
        "data_revision": data.REVISION,
        "data_manifest": data.verify(),
        "submission_files": files,
        "command": sys.argv,
    }


def write(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictor", choices=sorted(PREDICTORS))
    group.add_argument("--submission", type=Path, help="directory containing model.py")
    parser.add_argument("--options", type=json.loads, default={},
                        help='predictor keyword arguments as JSON, e.g. \'{"strength": 8}\'')
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        parser.error(f"unknown config keys: {sorted(unknown)}")
    settings = [protocol.Settings(seed=seed, **{k: config[k] for k in SETTING_KEYS if k in config})
                for seed in config["seeds"]]
    if args.submission:
        submission = args.submission.resolve()
        name = f"submission-{submission.name}"
        make_predictor = lambda: SubmissionPredictor(submission)  # noqa: E731
    else:
        submission, name = None, args.predictor
        make_predictor = functools.partial(PREDICTORS[args.predictor], **args.options)
        make_predictor()  # fail fast on bad options
        name += "".join(f"-{k}{v}" for k, v in sorted(args.options.items()))
    in_sample = bool(submission)

    payload = {"predictor": name, "options": args.options, "in_sample": in_sample,
               "config_file": str(args.config),
               "config": config, "settings": [dataclasses.asdict(s) for s in settings],
               "provenance": provenance(submission)}
    run_dir = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write(run_dir / "config.json", json.dumps(payload, indent=1) + "\n")
    if in_sample:
        print("warning: submission artifacts were trained offline, so these scores are in-sample")

    started = time.time()
    dataset = data.load()
    rows = []
    for seed_settings in settings:
        rows.extend(protocol.run(dataset, make_predictor, seed_settings))
    n_draws = config.get("draws", 0) if settings[0].evidence == "pair" else 0
    result = summary.summarize(rows, n_draws=n_draws)
    result["in_sample"] = in_sample
    result["elapsed_seconds"] = round(time.time() - started, 1)

    with (run_dir / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write(run_dir / "summary.json", json.dumps(result, indent=1) + "\n")

    headline = result["headline"]
    print(f"{name}: {len(rows)} pair results over seeds {result['seeds']}"
          f" in {result['elapsed_seconds']:.0f}s{' (IN-SAMPLE)' if in_sample else ''}")
    if "draws" in headline:
        draws = headline["draws"]
        print(f"  estimated Codabench score (simulated draws, all seeds): {draws['alc_mean']:.6f},"
              f" 90% range [{draws['alc_p05']:.6f}, {draws['alc_p95']:.6f}],"
              f" {draws['pairs_per_draw_mean']:.1f} pairs per draw")
    print(f"  Brier ALC, mean over benchmarks: {headline['alc_benchmark_mean']:.6f}"
          f" (seed std {headline['alc_benchmark_mean_seed_std']:.6f})")
    print(f"  Brier ALC, mean over all pairs (pool-weighted): {headline['alc_pair_mean']:.6f}"
          f" (seed std {headline['alc_pair_mean_seed_std']:.6f})")
    shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    print(f"  saved to {shown.as_posix()}")


if __name__ == "__main__":
    main()
