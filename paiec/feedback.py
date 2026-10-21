"""Parse Codabench formative feedback and recompute Brier ALC from its per-budget tables."""

import re

from .scoring import BUDGETS, alc

ROW = re.compile(r"^(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$")


def parse(text):
    """Return {(subject_id, benchmark_id): {"n": int, "alc": float, "ece_alc": float,
    "brier": {budget: float}, "ece": {budget: float}}} from the scoring output text."""
    pairs, section = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("ALC summary"):
            section = "alc"
        elif line.startswith("Label budget:"):
            section = int(line.split(":")[1])
        match = ROW.match(line)
        if not match or section is None:
            continue
        key = (match.group(1), match.group(2))
        pair = pairs.setdefault(key, {"n": int(match.group(3)), "brier": {}, "ece": {}})
        value, ece = float(match.group(4)), float(match.group(5))
        if section == "alc":
            pair["alc"], pair["ece_alc"] = value, ece
        else:
            pair["brier"][section], pair["ece"][section] = value, ece
    return pairs


def check(text, tolerance=1e-6):
    """Recompute each pair's ALC from its six per-budget Brier scores.

    The tables print 6 decimals, so recomputed and reported values can differ by
    rounding (at most 5e-7 here); larger differences mean our formula is wrong.
    """
    pairs = parse(text)
    if not pairs:
        raise ValueError("no feedback rows found")
    report = []
    for key, pair in sorted(pairs.items()):
        if sorted(pair["brier"]) != list(BUDGETS):
            raise ValueError(f"{key}: budgets {sorted(pair['brier'])} missing from feedback")
        recomputed = alc([pair["brier"][b] for b in BUDGETS])
        report.append({"pair": key, "n": pair["n"], "reported": pair["alc"],
                       "recomputed": recomputed, "ok": abs(recomputed - pair["alc"]) <= tolerance})
    mean = sum(r["reported"] for r in report) / len(report)
    return report, mean
