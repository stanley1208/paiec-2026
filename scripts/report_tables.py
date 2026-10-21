"""Numbers for the technical report's data table (Table 1), computed from the pinned data.

    python scripts/report_tables.py

Prints one row per benchmark: binary items, responses, subjects, mean label, eligible pairs in
the default setting (80 to 1000 items) and in the rated setting (rated items only, at least 40),
and items with an LLM rating. Also writes docs/report/tables/table1.tex.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ratings  # noqa: E402

from paiec import data, item_pooling_core, protocol  # noqa: E402

NAMES = {"matharena": "MathArena", "multi_swebench": "Multi-SWE-bench", "real_webagents": "REAL",
         "researchcodebench": "ResearchCodeBench", "swe_rebench": "SWE-rebench"}


def worked_example():
    """V4 against V3 for one target after one solved label, with the submitted parameters."""
    params = json.loads((ROOT / "submissions" / "version4" / "params.json").read_text(encoding="utf-8"))
    subject = dict.fromkeys(data.SUBJECT_FIELDS, "")
    subject["normalized_name"] = "example-model"

    def item(text):
        return {"item_content": text, "item_features": "", "interactors": "", "benchmark_id": "benchmark_000001"}
    ratings = {"easy item": 3, "hard item": 8}
    labeled = [[[subject, item("easy item")], 1]]
    target = [subject, item("hard item")]
    with_ratings = item_pooling_core.predict(target, labeled, params, lambda i: ratings.get(i["item_content"]))
    without = item_pooling_core.predict(target, labeled, params)
    print(f"worked example (one solved label on an item rated 3; target rated 8): "
          f"V4 {with_ratings:.6f}, V3 {without:.6f}")


def probit_error(tau=2.0, low=-6.0, high=6.0):
    """Largest |E_b[sigma(kappa0*theta - b)] - sigma(theta)| over theta, b ~ N(0, tau^2)."""
    kappa0 = math.sqrt(1 + math.pi * tau ** 2 / 8)
    nodes = [-8 + 0.001 * k for k in range(16001)]  # standard normal grid
    weights = [math.exp(-0.5 * z * z) for z in nodes]
    total = math.fsum(weights)
    worst = 0.0
    for step in range(1201):
        theta = low + (high - low) * step / 1200
        mean = math.fsum(w / (1 + math.exp(-(kappa0 * theta - tau * z))) for w, z in zip(weights, nodes)) / total
        worst = max(worst, abs(mean - 1 / (1 + math.exp(-theta))))
    print(f"probit approximation, tau = {tau}: kappa0 = {kappa0:.5f}, largest error over theta in "
          f"[{low:g}, {high:g}] = {worst:.4f}")


def main():
    worked_example()
    probit_error()
    dataset = data.load()
    _, rated_items, _ = run_ratings.load_ratings(dataset)
    responses = dataset.responses
    rows = []
    for slug, name in NAMES.items():
        part = responses[responses["benchmark"] == slug]
        default_pairs = len(protocol.eligible_pairs(dataset, slug, protocol.Settings()))
        rated = len(rated_items.get(slug, ()))
        if rated:
            keep = (responses["benchmark"] != slug) | responses["item_id"].isin(rated_items[slug])
            cut = data.Dataset(responses[keep].reset_index(drop=True), dataset.subjects, dataset.items)
            rated_pairs = len(protocol.eligible_pairs(cut, slug, protocol.Settings(min_pair_items=run_ratings.MIN_PAIR_ITEMS)))
        else:
            rated_pairs = 0
        rows.append((name, part["item_id"].nunique(), len(part), part["subject_id"].nunique(),
                     part["label"].mean(), default_pairs, rated_pairs, rated))
    total = (responses["item_id"].nunique(), len(responses), responses["subject_id"].nunique())
    print(f"{'benchmark':18s} {'items':>6s} {'responses':>9s} {'subjects':>8s} {'mean':>6s} {'pairs':>5s} {'rated pairs':>11s} {'rated items':>11s}")
    for r in rows:
        print(f"{r[0]:18s} {r[1]:6d} {r[2]:9d} {r[3]:8d} {r[4]:6.3f} {r[5]:5d} {r[6]:11d} {r[7]:11d}")
    print(f"{'total':18s} {total[0]:6d} {total[1]:9d} {total[2]:8d}")

    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Benchmark & Items & Responses & Subjects & Mean label & Pairs & Rated items & Rated pairs \\",
             r"\midrule"]
    for name, items, n, subjects, mean, pairs, rated_pairs, rated in rows:
        lines.append(f"{name} & {items:,} & {n:,} & {subjects} & {mean:.2f} & {pairs} & "
                     f"{rated if rated else '--'} & {rated_pairs if rated else '--'} \\\\")
    lines += [r"\midrule", f"Total & {total[0]:,} & {total[1]:,} & {total[2]} & & {sum(r[5] for r in rows)} & "
              f"{sum(r[7] for r in rows):,} & {sum(r[6] for r in rows)} \\\\", r"\bottomrule", r"\end{tabular}"]
    out = ROOT / "docs" / "report" / "tables" / "table1.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
