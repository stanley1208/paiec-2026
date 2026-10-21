"""Option C pre-check: does Claude know how strong each AI system is on a benchmark?

Pre-registered (2026-10-15), before any API call:
Claude Opus 5.5 (same model, effort and server-side fallback as version 4) is shown one AI
system's details, exactly the subject fields a hosted predictor receives, and three example
tasks of a benchmark (no benchmark name, no labels), and guesses the percentage of the
benchmark's tasks that the system completes correctly. One request per eligible subject of the
default setting (80 to 1,000 items): 81 MathArena, 54 Multi-SWE-bench, 26 REAL and 31
ResearchCodeBench subjects, 192 in all; identical prompts are asked once.

Groups: within a benchmark, subjects that are linked through shared items (the connected
components of the subject-item graph) form a group. Only Multi-SWE-bench has two groups (36
eligible subjects answered only the python items, 18 only the other languages), and the data
cannot compare abilities across them. The three examples are drawn once per group, from its own
items, leaving out MathArena problems given only as an image; long tasks show their first and
last 750 characters. So within a group the prompts differ only in the subject's details.

Primary measure ("model match"): for each benchmark, the Pearson correlation between
logit(guess), with the guess clipped to [1, 99] percent, and the subject's Rasch ability theta
(penalized joint ML, fitted per benchmark from all subjects with fit_rasch of
scripts/build_rating_samples.py), both centered within each group; then the plain mean over the
four benchmarks. A benchmark whose centered guesses are all equal counts as 0.
Rule: GO to the full option C test only if model match >= 0.70; otherwise drop option C.
A request without a valid guess is asked again by running --run again. If a benchmark still has
valid guesses for fewer than 80% of its subjects, the result is INCOMPLETE and no decision is
taken; otherwise subjects without a guess are left out and counted.

The threshold is the "well" case of the headroom simulation of 2026-10-15: model match 0.7 and
question match 0.6 gave about -0.035 Brier ALC against version 3 in the default setting, about
-0.02 against version 4.

Everything else is descriptive: Spearman correlation with theta (centered within groups),
Pearson correlation of logit(guess) with the logit of the subject's own success rate, mean guess
against mean success rate, and the Brier score of predicting the guess for every response of the
subject (a constant 0.5 scores 0.25). On MathArena subjects were run on different competitions,
so even exact success rates match theta at only about 0.85.

These are public benchmarks and Claude may remember published leaderboards, so a GO here is
optimistic; a NO-GO is reason enough to stop.

    python results/subject_check/check.py --dry-run             # build the requests, no API call
    .venv/Scripts/python results/subject_check/check.py --run   # ask Claude (needs ANTHROPIC_API_KEY)
    python results/subject_check/check.py --analyze             # apply the rule (pinned scipy 1.14.1)

The API key is read from ANTHROPIC_API_KEY, or from the user's saved Windows environment
variable; it is never printed or written anywhere.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_rating_samples import problem_key  # noqa: E402
from paiec import data, protocol  # noqa: E402

BENCHMARKS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
EXAMPLES = 3
EXAMPLE_CHARS = 1500
THRESHOLD = 0.70
MIN_SHARE = 0.80
MODEL = "claude-opus-5-5"
PRICES = {"claude-opus-5-5": (4.00 / 1e6, 20.00 / 1e6), "claude-opus-4-8": (5.00 / 1e6, 25.00 / 1e6)}
BUDGET = 8.0  # stop starting new requests once the estimated spend reaches this
OUT = HERE / "guesses.jsonl"

SYSTEM = """You estimate how well AI systems perform on benchmarks. An AI system is a language \
model, possibly run inside an agent harness, with its settings. You will see the details recorded \
for one AI system (some fields may be empty) and three example tasks from one benchmark.

Estimate the percentage of this benchmark's tasks that this AI system completes correctly, from \
0 (none) to 100 (all). Use what you know about the system: its developer, release date, size, \
harness and settings. Judge how hard the benchmark is from the example tasks. Do not try to solve \
the tasks.

Reply with only the integer percentage."""

SCHEMA = {"type": "object", "properties": {"success_percent": {"type": "integer"}},
          "required": ["success_percent"], "additionalProperties": False}


def item_key(dataset):
    matharena = data.alias("benchmark", "matharena")
    cache = {}

    def key(item_id):
        if item_id not in cache:
            cache[item_id] = problem_key(dataset, matharena, item_id)
        return cache[item_id]
    return key


def groups(responses, key):
    """Connected components of the subject-item graph: subject -> group, item key -> group.
    A group is named by its smallest item key."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = {(s, key(i)) for s, i in zip(responses["subject_id"], responses["item_id"])}
    for s, k in edges:
        a, b = find(("s", s)), find(("k", k))
        if a != b:
            parent[a] = b
    members = {}
    for node in list(parent):
        members.setdefault(find(node), []).append(node)
    subject_group, key_group = {}, {}
    for nodes in members.values():
        name = min(n[1] for n in nodes if n[0] == "k")
        for kind, value in nodes:
            (subject_group if kind == "s" else key_group)[value] = name
    return subject_group, key_group


def subject_text(subject):
    lines = [f"{field}: {subject[field]}" for field in data.SUBJECT_FIELDS if str(subject[field]).strip()]
    return "\n".join(lines) or "(no details recorded)"


def shown(content):
    if len(content) <= EXAMPLE_CHARS:
        return content
    half = EXAMPLE_CHARS // 2
    return content[:half] + "\n[...]\n" + content[-half:]


def example_text(dataset, item_ids):
    parts = []
    for n, item_id in enumerate(item_ids, 1):
        item = dataset.items[item_id]
        parts.append(f"Example task {n}\nFeatures: {item['item_features'] or '(none)'}\n"
                     f"{shown(str(item['item_content']))}")
    return "\n\n".join(parts)


def image_only(dataset, item_id):
    return str(dataset.items[item_id]["item_content"]).rstrip().endswith("See image.")


def requests(dataset):
    """One request per eligible subject of the default setting, in a fixed order."""
    settings = protocol.Settings()
    key = item_key(dataset)
    out = []
    for benchmark in BENCHMARKS:
        responses = dataset.responses[dataset.responses["benchmark"] == benchmark]
        subject_group, key_group = groups(responses, key)
        items = sorted(responses["item_id"].unique())
        examples = {}
        for group in sorted(set(subject_group.values())):
            pool = [i for i in items if key_group[key(i)] == group and not image_only(dataset, i)]
            draw = random.Random(f"subject-check|{benchmark}|{group}").sample(pool, EXAMPLES)
            examples[group] = example_text(dataset, draw)
        for pair in protocol.eligible_pairs(dataset, benchmark, settings):
            group = subject_group[pair.subject_id]
            prompt = (f"AI system:\n{subject_text(dataset.subjects[pair.subject_id])}\n\n"
                      f"Example tasks from the benchmark:\n\n{examples[group]}")
            out.append({"benchmark": benchmark, "subject_id": pair.subject_id, "group": group,
                        "prompt": prompt, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()})
    return out


def api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
        return winreg.QueryValueEx(handle, "ANTHROPIC_API_KEY")[0]


def load_records():
    """All records by prompt hash, oldest first."""
    records = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            records.setdefault(record["prompt_sha"], []).append(record)
    return records


def latest_guesses(records):
    """The latest valid guess by prompt hash."""
    out = {}
    for sha, recs in records.items():
        valid = [r["guess"] for r in recs if r["guess"] is not None]
        if valid:
            out[sha] = valid[-1]
    return out


def dry_run(reqs):
    unique = {r["prompt_sha"]: r for r in reqs}
    chars = sum(len(SYSTEM) + len(r["prompt"]) for r in unique.values())
    for benchmark in BENCHMARKS:
        bench = [r for r in reqs if r["benchmark"] == benchmark]
        sizes = {}
        for r in bench:
            sizes[r["group"]] = sizes.get(r["group"], 0) + 1
        print(f"{benchmark:18s} {len(bench):3d} subjects, {len({r['prompt_sha'] for r in bench}):3d} distinct "
              f"prompts, groups {sorted(sizes.values(), reverse=True)}")
    print(f"{len(reqs)} subjects, {len(unique)} distinct prompts, about {chars / 4 / 1e3:.0f}k input tokens")


def run(reqs):
    import anthropic

    done = latest_guesses(load_records())
    todo = list({r["prompt_sha"]: r for r in reqs if r["prompt_sha"] not in done}.values())
    print(f"{len(todo)} prompts to ask")
    client = anthropic.Anthropic(api_key=api_key(), max_retries=4)
    lock = threading.Lock()
    spent = {"usd": 0.0}

    def ask(req):
        with lock:
            if spent["usd"] >= BUDGET:
                return None
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": "claude-opus-4-8"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": req["prompt"]}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            guess = int(json.loads(text)["success_percent"]) if response.stop_reason != "refusal" else None
        except (ValueError, KeyError, TypeError):
            guess = None
        if guess is not None and not 0 <= guess <= 100:
            guess = None
        price_in, price_out = PRICES.get(response.model, PRICES["claude-opus-4-8"])
        usd = response.usage.input_tokens * price_in + response.usage.output_tokens * price_out
        attempts = len(getattr(response.usage, "iterations", None) or []) or 1
        record = {"prompt_sha": req["prompt_sha"], "benchmark": req["benchmark"], "guess": guess,
                  "text": text[:200], "stop_reason": response.stop_reason, "model": response.model,
                  "attempts": attempts, "input_tokens": response.usage.input_tokens,
                  "output_tokens": response.usage.output_tokens, "usd": usd}
        with lock:
            spent["usd"] += usd
            with OUT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        return record

    skipped = 0
    with cf.ThreadPoolExecutor(4) as pool:
        for record in pool.map(ask, todo):
            if record is None:
                skipped += 1
                continue
            print(f"{record['benchmark'][:12]:12s} guess {record['guess']} ({record['model']}, "
                  f"{record['input_tokens']} in / {record['output_tokens']} out, ${record['usd']:.3f})", flush=True)
    print(f"spent this run: ${spent['usd']:.2f}; budget stop at ${BUDGET:.2f}; "
          f"{skipped} prompts not asked because of the budget stop")


def centered(values, labels):
    import numpy as np
    out = np.array(values, float)
    labels = np.array(labels)
    for g in set(labels.tolist()):
        out[labels == g] -= out[labels == g].mean()
    return out


def analyze(dataset, reqs):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    from build_rating_samples import fit_rasch

    records = load_records()
    guesses = latest_guesses(records)
    key = item_key(dataset)
    rows, results = [], {}
    for benchmark in BENCHMARKS:
        responses = dataset.responses[dataset.responses["benchmark"] == benchmark]
        counts = (responses.assign(key=responses["item_id"].map(key))
                  .groupby(["subject_id", "key"])["label"].agg(["sum", "size"]).reset_index())
        _, theta = fit_rasch(counts, with_theta=True)
        labels = responses.groupby("subject_id")["label"].agg(list)
        bench = [r for r in reqs if r["benchmark"] == benchmark]
        have = [r for r in bench if r["prompt_sha"] in guesses]
        result = {"subjects": len(bench), "with_guess": len(have),
                  "never_asked": sum(r["prompt_sha"] not in records for r in bench),
                  "invalid_only": sum(r["prompt_sha"] in records and r["prompt_sha"] not in guesses for r in bench),
                  "groups": len({r["group"] for r in bench})}
        results[benchmark] = result
        if len(have) < MIN_SHARE * len(bench):
            continue
        g = np.array([guesses[r["prompt_sha"]] for r in have], float)
        p = np.clip(g / 100, 0.01, 0.99)
        group = [r["group"] for r in have]
        x = centered(np.log(p / (1 - p)), group)
        th = centered([theta[r["subject_id"]] for r in have], group)
        y = [np.array(labels[r["subject_id"]], float) for r in have]
        rate = np.array([v.mean() for v in y])
        clipped_rate = np.clip(rate, 0.01, 0.99)
        constant = bool(np.ptp(x) == 0)
        result.update({
            "constant_guesses": constant,
            "pearson_logit_guess_theta": 0.0 if constant else float(pearsonr(x, th)[0]),
            "spearman_guess_theta": 0.0 if constant else float(spearmanr(x, th)[0]),
            "pearson_logit_guess_logit_rate": 0.0 if constant else float(pearsonr(
                x, centered(np.log(clipped_rate / (1 - clipped_rate)), group))[0]),
            "mean_guess": float(p.mean()), "mean_success_rate": float(rate.mean()),
            "brier_guess": float(np.mean([np.mean((pi - v) ** 2) for pi, v in zip(p, y)])),
            "brier_half": 0.25, "brier_known_rate": float(np.mean(rate * (1 - rate))),
            "fallback_answers": sum(any(rec["model"] != MODEL for rec in records[r["prompt_sha"]]) for r in have),
        })
        rows += [{"benchmark": benchmark, "subject_id": r["subject_id"], "group": r["group"], "guess": gi,
                  "theta": float(theta[r["subject_id"]]), "success_rate": float(sr)}
                 for r, gi, sr in zip(have, g, rate)]
    incomplete = [b for b in BENCHMARKS if "pearson_logit_guess_theta" not in results[b]]
    match = None if incomplete else float(np.mean([results[b]["pearson_logit_guess_theta"] for b in BENCHMARKS]))
    decision = "INCOMPLETE" if incomplete else ("GO" if match >= THRESHOLD else "NO-GO")
    usd = sum(rec["usd"] for recs in records.values() for rec in recs)
    summary = {"model_match": match, "threshold": THRESHOLD, "decision": decision, "incomplete": incomplete,
               "missing": sum(results[b]["subjects"] - results[b]["with_guess"] for b in BENCHMARKS),
               "requests_made": sum(len(recs) for recs in records.values()), "estimated_usd": usd,
               "benchmarks": results}
    (HERE / "analysis.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    (HERE / "per_subject.json").write_text(json.dumps(rows, indent=0) + "\n", encoding="utf-8", newline="\n")
    print(f"{'benchmark':18s} {'n':>4s} {'pearson':>8s} {'spearman':>8s} {'p.rate':>7s} "
          f"{'guess':>6s} {'actual':>6s} {'brier':>6s} {'known':>6s}")
    for b in BENCHMARKS:
        r = results[b]
        if b in incomplete:
            print(f"{b:18s} {r['with_guess']:4d} of {r['subjects']}: too few guesses")
            continue
        print(f"{b:18s} {r['with_guess']:4d} {r['pearson_logit_guess_theta']:8.3f} {r['spearman_guess_theta']:8.3f} "
              f"{r['pearson_logit_guess_logit_rate']:7.3f} {r['mean_guess']:6.3f} {r['mean_success_rate']:6.3f} "
              f"{r['brier_guess']:6.4f} {r['brier_known_rate']:6.4f}")
    shown_match = "n/a" if match is None else f"{match:.3f}"
    print(f"\nmodel match {shown_match} (rule: GO if >= {THRESHOLD}); missing guesses {summary['missing']}; "
          f"{summary['requests_made']} requests, about ${usd:.2f}: {decision}")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    dataset = data.load()
    reqs = requests(dataset)
    if args.dry_run:
        dry_run(reqs)
    elif args.run:
        run(reqs)
    else:
        analyze(dataset, reqs)


if __name__ == "__main__":
    main()
