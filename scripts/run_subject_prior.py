"""Does Claude's guess of each subject's success rate improve version 4 offline? (option C, full test)

    python scripts/run_subject_prior.py

Decision rule, fixed before the first run (2026-10-17).
Option C is version 4 (rating trust 0.5 with the gate) whose subject prior comes from Claude Opus
5.5's guess g of the subject's success rate on the benchmark (paiec/guess_core.py: the subject's
fields and one of its tasks; guesses in results/subject_prior/guesses.jsonl), for the target
subject and for the partner subjects whose labels are pooled. Calibration: an ordinary
least-squares fit of logit(success rate) on logit(g), both clipped to [0.01, 0.99], over the
subjects with 40 to 1,000 items on the rated benchmarks other than the held-out one, gives
intercept a, slope c and root mean squared residual s. With trust t, a subject with a guess gets
theta ~ N(t * (a + c * logit(g)), (1 - t^2) * 1.5^2 + t^2 * s^2), so t = 0 is version 4 and t = 1
trusts the calibration fully (the same interpolation as the rating trust of scripts/run_trust.py).
A subject without a guess keeps version 4's N(0, 1.5^2). Every scored pair must have a guess, or the
script stops before running.
Pairs: those of scripts/run_final.py (held-out benchmark cut to rated items, pairs of at least 40
items, every item rated, seeds 0-2).
Step 1, trust. With K = 0 partner pairs, each t in {0, 0.25, 0.5, 0.75, 1} runs under four guess
scenarios for the held-out benchmark: real; useless (its guesses shuffled among its subjects);
high and low (every guess moved by +1 and by -1 on the logit scale, a benchmark-wide level error).
a, c and s always come from the real guesses of the other benchmarks. The chosen t minimises the
largest regret over the four scenarios, regret being pair-mean Brier ALC minus the best t's ALC in
that scenario (ties go to the larger t).
Step 2, GO to building a hosted version of option C only if the chosen t is above 0 and, with the
real guesses at that t:
1. with K = 0, option C beats version 4 by at least 0.003 pair-mean Brier ALC and by more than
   2 standard errors (clustered by subject-benchmark pair);
2. it also beats version 4 (any margin) with K = 1 and with K = 2;
3. with K = 0, it beats version 4 on at least 3 of the 4 benchmarks.
Otherwise version 4 stays the final submission.
Descriptive only: everything else in summary.json (Brier by budget, all trust levels, the fits).
Writes results/runs/<timestamp>_subject_prior/ with config.json (provenance, written first),
per_pair.csv and summary.json.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_gate as rg  # noqa: E402
import run_ratings as rr  # noqa: E402
import run_trust as rt  # noqa: E402
from run_harness import provenance, write  # noqa: E402
from run_pooling import _mean, _paired  # noqa: E402

from paiec import data, guess_core, item_pooling_core, protocol, scoring  # noqa: E402

GUESSES = ROOT / "results" / "subject_prior" / "guesses.jsonl"
TRUST = (0.25, 0.5, 0.75, 1.0)  # t = 0 is version 4
SCENARIOS = ("real", "useless", "high", "low")
PARTNERS = (0, 1, 2)
SEEDS = (0, 1, 2)
SHIFT = 1.0
MIN_GAIN, MIN_Z = 0.003, 2.0


def _logit(p):
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + math.exp(-z))


def load_guesses():
    """The latest valid guess by guess key, with its benchmark and subject."""
    if not GUESSES.exists():
        raise SystemExit(f"{GUESSES} not found: run results/subject_prior/ask.py first")
    out = {}
    for line in GUESSES.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["guess"] is not None:
            out[row["key"]] = row
    return out


def fit_guess(dataset, guesses, held_out):
    """(a, c, s): least squares of logit(success rate) on logit(guess) over the other rated benchmarks."""
    rows = {(r["benchmark"], r["subject_id"]): r["guess"] for r in guesses.values()}
    settings = protocol.Settings(min_pair_items=rr.MIN_PAIR_ITEMS)
    xs, ys = [], []
    for benchmark in rr.RATED:
        if benchmark == held_out:
            continue
        for pair in protocol.eligible_pairs(dataset, benchmark, settings):
            if (benchmark, pair.subject_id) in rows:
                labels = [y for outcomes in pair.responses.values() for _, y in outcomes]
                xs.append(_logit(rows[benchmark, pair.subject_id]))
                ys.append(_logit(_mean(labels)))
    mx, my = _mean(xs), _mean(ys)
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise SystemExit(f"all calibration guesses are equal (held out: {held_out})")
    slope = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    sd = math.sqrt(_mean([(y - intercept - slope * x) ** 2 for x, y in zip(xs, ys)]))
    return {"a": intercept, "c": slope, "s": sd, "fit_pairs": len(xs)}


def guess_tables(guesses):
    """Guess by key for each scenario; only the held-out benchmark's keys are ever looked up."""
    real = {k: r["guess"] for k, r in guesses.items()}
    useless = {}
    for benchmark in rr.RATED:
        keys = sorted(k for k, r in guesses.items() if r["benchmark"] == benchmark)
        values = [real[k] for k in keys]
        random.Random(f"useless-guesses|{benchmark}").shuffle(values)
        useless.update(zip(keys, values))
    return {"real": real, "useless": useless,
            "high": {k: _sigmoid(_logit(g) + SHIFT) for k, g in real.items()},
            "low": {k: _sigmoid(_logit(g) - SHIFT) for k, g in real.items()}}


def coverage(dataset, guesses):
    """Scored pairs (cut benchmark) by benchmark, and those without a guess."""
    out = {}
    settings = protocol.Settings(min_pair_items=rr.MIN_PAIR_ITEMS)
    _, rated_items, _ = rr.load_ratings(dataset)
    for benchmark in rr.RATED:
        responses = dataset.responses
        keep = (responses["benchmark"] != benchmark) | responses["item_id"].isin(rated_items[benchmark])
        cut = data.Dataset(responses[keep].reset_index(drop=True), dataset.subjects, dataset.items)
        pairs = protocol.eligible_pairs(cut, benchmark, settings)
        missing = [p.subject_id for p in pairs if guess_core.guess_key(
            *cut.make_input(p.subject_id, next(iter(p.responses)))) not in guesses]
        out[benchmark] = {"pairs": len(pairs), "missing": missing}
    return out


def _load():
    rt._load()
    rr._state["by_content"] = rr._state["tables"]["real"]
    guesses = load_guesses()
    rr._state["guess_tables"] = guess_tables(guesses)
    rr._state["guess_fits"] = {b: fit_guess(rr._state["dataset"], guesses, b) for b in rr.RATED}


class GuessedItemPooling(rg.GatedItemPooling):
    """Version 4 with the subject prior from Claude's guesses at trust t (calibrated without held_out)."""

    def __init__(self, held_out, table, guess_trust):
        super().__init__(0.0, held_out, 0.5, True)  # self.trust is version 4's rating trust
        self.table, self.guess_trust = table, guess_trust

    def fit(self, train):
        super().fit(train)
        reference = rg.GatedItemPooling(0.0, self.held_out, 0.5, True)
        reference.fit(train)
        if self.params != reference.params:
            raise AssertionError("option C must keep every version 4 parameter")
        fit, t, sd0 = rr._state["guess_fits"][self.held_out], self.guess_trust, self.params["sd"]
        self.params = dict(self.params, guess_intercept=t * fit["a"], guess_slope=t * fit["c"],
                           guess_sd=math.sqrt((1 - t ** 2) * sd0 ** 2 + t ** 2 * fit["s"] ** 2))

    def predict(self, input, labeled=None):
        return item_pooling_core.predict(input, labeled or [], self.params, rr._rating, self._guess)

    def _guess(self, subject, item):
        return self.table.get(guess_core.guess_key(subject, item))


def _task(scenario, trust, partners, seed):
    dataset = rr._state["dataset"]
    settings = protocol.Settings(seed=seed, evidence="partners", partners=partners,
                                 min_pair_items=rr.MIN_PAIR_ITEMS)
    rows = []
    for benchmark in rr.RATED:
        responses = dataset.responses
        keep = (responses["benchmark"] != benchmark) | responses["item_id"].isin(rr._state["rated_items"][benchmark])
        cut = data.Dataset(responses[keep].reset_index(drop=True), dataset.subjects, dataset.items)
        if trust == 0:
            factory = lambda: rg.GatedItemPooling(0.0, benchmark, 0.5, True)  # noqa: E731
        else:
            table = rr._state["guess_tables"][scenario]
            factory = lambda: GuessedItemPooling(benchmark, table, trust)  # noqa: E731
        for row in protocol.run_fold(cut, benchmark, factory, settings):
            rows.append({"scenario": scenario, "trust": trust, "partners": partners, **row})
    return rows


def tasks():
    out = [("real", 0.0, k, seed) for k in PARTNERS for seed in SEEDS]  # version 4, the same in every scenario
    out += [(s, t, 0, seed) for s in SCENARIOS for t in TRUST for seed in SEEDS]
    out += [("real", t, k, seed) for t in TRUST for k in PARTNERS[1:] for seed in SEEDS]
    return out


def _summary(rows):
    def group(scenario, trust, partners):
        if trust == 0:
            scenario = "real"
        return [r for r in rows if (r["scenario"], r["trust"], r["partners"]) == (scenario, trust, partners)]

    levels = (0.0,) + TRUST
    table, alc = {}, {}
    for scenario in SCENARIOS:
        reference = group(scenario, 0.0, 0)
        for t in levels:
            rows_t = group(scenario, t, 0)
            alc[scenario, t] = _mean([r["alc"] for r in rows_t])
            table[f"k0 {scenario} t={t:g}"] = {
                "pairs": len(rows_t), "alc_pair_mean": alc[scenario, t],
                "vs_version4": _paired(rows_t, reference),
                "by_benchmark": {b: _paired([r for r in rows_t if r["benchmark"] == b],
                                            [r for r in reference if r["benchmark"] == b])["alc_diff"] for b in rr.RATED},
                "brier": {b: _mean([r[f"brier_{b}"] for r in rows_t]) for b in scoring.BUDGETS}}
    for k in PARTNERS[1:]:
        reference = group("real", 0.0, k)
        for t in levels:
            rows_t = group("real", t, k)
            table[f"k{k} real t={t:g}"] = {
                "pairs": len(rows_t), "alc_pair_mean": _mean([r["alc"] for r in rows_t]),
                "vs_version4": _paired(rows_t, reference),
                "brier": {b: _mean([r[f"brier_{b}"] for r in rows_t]) for b in scoring.BUDGETS}}
    regret = {t: {s: alc[s, t] - min(alc[s, u] for u in levels) for s in SCENARIOS} for t in levels}
    worst = {t: max(r.values()) for t, r in regret.items()}
    chosen = min(levels, key=lambda t: (worst[t], -t))
    decision = {"regret": {f"{t:g}": r for t, r in regret.items()}, "worst_regret": {f"{t:g}": w for t, w in worst.items()},
                "chosen_trust": chosen}
    if chosen == 0:
        decision.update(checks={}, go=False)
        return table, decision
    main = table[f"k0 real t={chosen:g}"]
    checks = {
        "k0_gain": main["vs_version4"]["alc_diff"] <= -MIN_GAIN
        and main["vs_version4"]["alc_diff"] + MIN_Z * main["vs_version4"]["alc_diff_se"] < 0,
        "k1_k2_better": all(table[f"k{k} real t={chosen:g}"]["vs_version4"]["alc_diff"] < 0 for k in PARTNERS[1:]),
        "three_of_four": sum(v < 0 for v in main["by_benchmark"].values()) >= 3,
    }
    decision.update(checks=checks, go=all(checks.values()))
    return table, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=3)  # each loads the data (~1 GB)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()

    dataset = data.load()
    guesses = load_guesses()
    covered = coverage(dataset, guesses)
    missing = {b: c["missing"] for b, c in covered.items() if c["missing"]}
    if missing:
        raise SystemExit(f"scored pairs without a guess (rerun ask.py): {missing}")
    fits = {b: fit_guess(dataset, guesses, b) for b in rr.RATED}
    del dataset

    run_dir = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}_subject_prior"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {"trust": TRUST, "scenarios": SCENARIOS, "shift": SHIFT, "partners": PARTNERS, "seeds": SEEDS,
               "min_pair_items": rr.MIN_PAIR_ITEMS, "rule": {"min_gain": MIN_GAIN, "min_z": MIN_Z},
               "guesses": len(guesses), "coverage": {b: c["pairs"] for b, c in covered.items()}, "guess_fits": fits,
               "provenance": provenance(None)}
    write(run_dir / "config.json", json.dumps(payload, indent=1) + "\n")

    started = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_load) as pool:
        rows = [row for chunk in pool.map(_task, *zip(*tasks())) for row in chunk]
    table, decision = _summary(rows)

    with (run_dir / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write(run_dir / "summary.json", json.dumps(
        {"table": table, "decision": decision, "elapsed_seconds": round(time.time() - started, 1)}, indent=1) + "\n")

    print("guess calibration by held-out benchmark (a, c, s, pairs):")
    for b, f in fits.items():
        print(f"  {b:18s} {f['a']:+.3f} {f['c']:.3f} {f['s']:.3f} {f['fit_pairs']}")
    print(f"{'':22s} {'pairs':>5s} {'ALC':>7s} {'vs version 4 (SE)':>20s}  by benchmark")
    for name, entry in table.items():
        versus = entry["vs_version4"]
        benchmarks = " ".join(f"{b} {v:+.4f}" for b, v in entry.get("by_benchmark", {}).items())
        print(f"{name:22s} {entry['pairs']:5d} {entry['alc_pair_mean']:.4f} "
              f"{versus['alc_diff']:+.5f} ({versus['alc_diff_se']:.5f})  {benchmarks}")
    print("worst regret:", {t: round(w, 4) for t, w in decision["worst_regret"].items()},
          "chosen trust:", decision["chosen_trust"])
    print("checks:", decision["checks"], "->", "GO" if decision["go"] else "NO-GO")
    shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    print(f"saved to {shown.as_posix()}")


if __name__ == "__main__":
    main()
