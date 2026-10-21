"""Numbers, tables and the figure of the technical report's results section, from committed runs.

    python scripts/report_results.py

Writes docs/report/tables/table_offline.tex (offline results in three blocks, one per setting),
docs/report/tables/table_formative.tex (formative uploads) and docs/report/figures/budgets.pdf
(Brier score by budget in the rated setting), and prints every number the text quotes. The
formative table needs the private feedback files in private/codabench/; when they are present it
also writes results/formative_summary.json, which holds only per-upload aggregates (no per-pair
rows or IDs), so the table can be rebuilt without them.
"""

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pooling import _mean, _paired  # noqa: E402

from paiec import feedback, scoring  # noqa: E402

RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "docs" / "report" / "tables"
FIGURES = ROOT / "docs" / "report" / "figures"
FORMATIVE = ROOT / "results" / "formative_summary.json"
BENCHMARKS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
NAMES = {"matharena": "MathArena", "multi_swebench": "Multi-SWE-bench", "real_webagents": "REAL",
         "researchcodebench": "ResearchCodeBench"}
# Codabench uploads that scored (id, date as Codabench shows it in UTC+8, version).
UPLOADS = [("951242", "2026-09-29", "constant 0.5"), ("954621", "2026-10-01", "Baseline"),
           ("958131", "2026-10-03", "V1"), ("963204", "2026-10-06", "V2"), ("963789", "2026-10-06", "V3"),
           ("968769", "2026-10-08", "V4"), ("1005477", "2026-10-20", "V5")]


def rows(run, **where):
    with (RUNS / run / "per_pair.csv").open(encoding="utf-8") as handle:
        out = []
        for row in csv.DictReader(handle):
            if all(str(row.get(k)) == str(v) for k, v in where.items()):
                for key in ("alc",) + tuple(f"brier_{b}" for b in scoring.BUDGETS):
                    row[key] = float(row[key])
                row["seed"] = int(row["seed"])
                out.append(row)
    if not out:
        raise SystemExit(f"no rows in {run} for {where}")
    return out


def fmt(x, digits=4):
    return f"{x:.{digits}f}"


def se_and_estimate(d, se):
    """The SE to two significant figures and the estimate to the same decimal place."""
    places = max(4, 1 - math.floor(math.log10(se))) if se > 0 else 4
    return f"${d:+.{places}f}$ ({se:.{places}f})"


def delta(group, reference):
    d = _paired(group, reference)
    return d["alc_diff"], d["alc_diff_se"]


def offline():
    """Three blocks: default setting (K = 0), hidden-sized with one partner, rated setting (K = 0)."""
    blocks = []
    default = [("Baseline", rows("20260930-233206_beta_binomial_subject")),
               ("V1 ($\\lambda = 0.5$)", rows("20261002-005443_logit_normal-scale0.5")),
               ("V3 ($\\lambda = 0$)", rows("20261005-213837_item_pooling-scale0.0"))]
    blocks.append(("Default setting, $K = 0$", default))
    pooling = "20261005-013412_pooling"
    hidden = [("V1 ($\\lambda = 0.5$)", rows(pooling, setting="hidden", variant="phase4")),
              ("V2 ($\\lambda = 0.5$, pooling)", rows(pooling, setting="hidden", variant="pooled_k1")),
              ("V3 ($\\lambda = 0$, pooling)", rows("20261005-213957_item_pooling-scale0.0-hidden-partners1"))]
    blocks.append(("Hidden-sized setting, $K = 1$", hidden))
    final, prior = "20261009-003522_final", "20261017-021515_subject_prior"
    rated = [("V3", rows(final, model="version3", partners=0)),
             ("V4 (ratings, $t = 0.5$, gate)", rows(final, model="version4", partners=0, coverage="1.0")),
             ("V5 (V4 + subject guesses, $t_g = 0.75$)", rows(prior, scenario="real", trust="0.75", partners=0))]
    blocks.append(("Rated setting, $K = 0$", rated))
    lines, numbers = [], {}
    for title, entries in blocks:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\")
        previous = None
        for name, group in entries:
            alc = _mean([r["alc"] for r in group])
            if previous is None:
                cell = "--"
            else:
                d, se = delta(group, previous)
                cell = se_and_estimate(d, se)
                numbers[f"{title} | {name} vs previous"] = (round(d, 5), round(se, 5))
            numbers[f"{title} | {name}"] = round(alc, 4)
            lines.append(f"\\quad {name} & {len(group)} & {fmt(alc)} & {cell} \\\\")
            previous = group
        lines.append("\\addlinespace")
    body = "\n".join(lines[:-1])
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_offline.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\nPredictor & Pair results & Brier ALC & $\\Delta$ to row above (SE) \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8", newline="\n")
    return numbers, rated


def rated_details(rated):
    """V5 against V4 in the rated setting: by benchmark, by partner count, by scenario, by budget."""
    prior = "20261017-021515_subject_prior"
    out = {}
    v4, v5 = rated[1][1], rated[2][1]
    out["v5 vs v4 by benchmark"] = {NAMES[b]: round(delta([r for r in v5 if r["benchmark"] == b],
                                                        [r for r in v4 if r["benchmark"] == b])[0], 4) for b in BENCHMARKS}
    for k in (1, 2):
        out[f"v5 vs v4, K={k}"] = tuple(round(x, 5) for x in delta(
            rows(prior, scenario="real", trust="0.75", partners=k), rows(prior, scenario="real", trust="0.0", partners=k)))
    v4_prior = rows(prior, scenario="real", trust="0.0", partners=0)
    for scenario in ("useless", "high", "low"):
        out[f"v5 vs v4, {scenario} guesses"] = tuple(round(x, 5) for x in delta(
            rows(prior, scenario=scenario, trust="0.75", partners=0), v4_prior))
    summary = json.loads((RUNS / prior / "summary.json").read_text(encoding="utf-8"))
    out["worst regret by guess trust"] = {t: round(v, 4) for t, v in summary["decision"]["worst_regret"].items()}
    out["brier by budget"] = {name: [round(_mean([r[f"brier_{b}"] for r in group]), 4) for b in scoring.BUDGETS]
                              for name, group in (("V3", rated[0][1]), ("V4", v4), ("V5", v5))}
    weights = dict(zip(scoring.BUDGETS, (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)))
    gain = {b: weights[b] * (_mean([r[f"brier_{b}"] for r in v5]) - _mean([r[f"brier_{b}"] for r in v4]))
            for b in scoring.BUDGETS}
    total = sum(gain.values())
    out["share of the v5 gain by budget"] = {b: round(g / total, 2) for b, g in gain.items()}
    pairs_gain = {b: sum(r5["alc"] - r4["alc"] for r5, r4 in zip(
        sorted((r for r in v5 if r["benchmark"] == b), key=lambda r: (r["seed"], r["subject_id"])),
        sorted((r for r in v4 if r["benchmark"] == b), key=lambda r: (r["seed"], r["subject_id"])))) for b in BENCHMARKS}
    out["share of the v5 pair-mean gain by benchmark"] = {
        NAMES[b]: round(g / sum(pairs_gain.values()), 2) for b, g in pairs_gain.items()}
    useless = rows(prior, scenario="useless", trust="0.75", partners=0)
    out["shuffled guesses vs v4 by benchmark"] = {
        NAMES[b]: round(delta([r for r in useless if r["benchmark"] == b], [r for r in v4_prior if r["benchmark"] == b])[0], 4)
        for b in BENCHMARKS}
    final = "20261009-003522_final"
    v3 = rows(final, model="version3", partners=0)
    for coverage in ("0.9", "0.7"):
        out[f"v4 vs v3, coverage {coverage}"] = tuple(round(x, 5) for x in delta(
            rows(final, model="version4", partners=0, coverage=coverage), v3))
    for k in (1, 2):
        out[f"v4 vs v3, K={k}"] = tuple(round(x, 5) for x in delta(
            rows(final, model="version4", partners=k, coverage="1.0"), rows(final, model="version3", partners=k)))
    trust = json.loads((RUNS / "20261006-000259_trust" / "summary.json").read_text(encoding="utf-8"))
    out["worst regret by rating trust"] = {t: round(v, 4) for t, v in trust["decision"]["worst_regret"].items()}
    pooling = "20261005-013412_pooling"
    out["pooling with one shared split, K=1"] = tuple(round(x, 5) for x in delta(
        rows(pooling, setting="hidden_shared_split", variant="pooled_k1"),
        rows(pooling, setting="hidden_shared_split", variant="phase4")))
    out["pooling by benchmark, K=1"] = {
        NAMES.get(b, b): round(delta([r for r in rows(pooling, setting="hidden", variant="pooled_k1") if r["benchmark"] == b],
                                     [r for r in rows(pooling, setting="hidden", variant="phase4") if r["benchmark"] == b])[0], 4)
        for b in BENCHMARKS}
    return out


def figure(rated):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 1.85))
    x = [math.log2(1 + b) for b in scoring.BUDGETS]
    styles = {"V3": ("0.6", "o", "--"), "V4": ("0.35", "s", "-."), "V5": ("0.0", "^", "-")}
    for label, group in (("V3", rated[0][1]), ("V4", rated[1][1]), ("V5", rated[2][1])):
        color, marker, line = styles[label]
        ax.plot(x, [_mean([r[f"brier_{b}"] for r in group]) for b in scoring.BUDGETS], color=color,
                marker=marker, linestyle=line, linewidth=1, markersize=3.5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in scoring.BUDGETS])
    ax.set_xlabel("Labels acquired (budget)")
    ax.set_ylabel("Mean Brier score")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "budgets.pdf")


def formative():
    """Per-upload aggregates from the private feedback, or from the saved summary when it is absent."""
    private = ROOT / "private" / "codabench"
    if all((private / f"{uid}_scoring_output.txt").exists() for uid, _, _ in UPLOADS):
        summary = []
        for uid, date, version in UPLOADS:
            pairs = feedback.parse((private / f"{uid}_scoring_output.txt").read_text(encoding="utf-8"))
            gaps = [p["alc"] - (p["brier"][0] - p["ece"][0] ** 2) for p in pairs.values()]
            mean_gap = _mean(gaps)
            sd = math.sqrt(sum((g - mean_gap) ** 2 for g in gaps) / (len(gaps) - 1))
            summary.append({"id": uid, "date": date, "version": version, "pairs": len(pairs),
                            "score": round(_mean([p["alc"] for p in pairs.values()]), 6),
                            "budget0_brier": round(_mean([p["brier"][0] for p in pairs.values()]), 6),
                            "score_minus_floor": round(mean_gap, 6), "se": round(sd / math.sqrt(len(gaps)), 6)})
        FORMATIVE.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    summary = json.loads(FORMATIVE.read_text(encoding="utf-8"))
    lines = [f"{s['id']} & {s['date']} & {s['version']} & {s['pairs']} & {s['score']:.4f} & {s['budget0_brier']:.3f} & "
             f"{s['score_minus_floor']:.3f} ({s['se']:.3f}) \\\\" for s in summary]
    (TABLES / "table_formative.tex").write_text(
        "\\begin{tabular}{llcrrrr}\n\\toprule\nCodabench & Date & Version & Pairs & Score & Budget-0 Brier & "
        "Score $-$ floor (SE) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8", newline="\n")
    return summary


def main():
    numbers, rated = offline()
    for key, value in numbers.items():
        print(f"{key}: {value}")
    for key, value in rated_details(rated).items():
        print(f"{key}: {value}")
    figure(rated)
    for s in formative():
        print("formative", s)
    print("wrote docs/report/tables/table_offline.tex, table_formative.tex, docs/report/figures/budgets.pdf")


if __name__ == "__main__":
    main()
