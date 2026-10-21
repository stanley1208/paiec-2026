"""Ask Claude for the subject guesses used by scripts/run_subject_prior.py (option C, full test).

    .venv/Scripts/python results/subject_prior/ask.py           # needs ANTHROPIC_API_KEY
    python results/subject_prior/ask.py --dry-run               # list the requests, no API call

One request per subject and rated benchmark for every subject the test needs: those with 40 to
1,000 items on the full benchmark (the pairs the guess calibration is fitted on) and those with 40
to 1,000 items once the benchmark is cut to its rated items (the pairs scripts/run_final.py and
scripts/run_subject_prior.py score; some Multi-SWE-bench subjects have more than 1,000 items in
full but fewer after the cut). Each request is built by paiec/guess_core.py exactly as a hosted
submission would build it. The example task is one of the subject's own items on the benchmark,
drawn with random.Random(f"subject-prior|{benchmark}|{subject_id}") from its rated items (the
held-out benchmark is cut to them). Writes guesses.jsonl here; a rerun asks only what has no
valid guess yet. The API key is read from ANTHROPIC_API_KEY, or from the user's saved Windows
environment variable; it is never printed or written anywhere.
"""
import argparse
import concurrent.futures as cf
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

import run_ratings as rr  # noqa: E402

from paiec import data, guess_core, protocol  # noqa: E402

SETTINGS = {"model": "claude-opus-5-5", "effort": "low"}
PRICES = {"claude-opus-5-5": (4.00 / 1e6, 20.00 / 1e6), "claude-opus-4-8": (5.00 / 1e6, 25.00 / 1e6)}
BUDGET = 5.0  # stop starting new requests once the estimated spend reaches this
OUT = HERE / "guesses.jsonl"


def cut_dataset(dataset, rated_items, benchmark):
    """The dataset with this benchmark cut to its rated items, as scripts/run_final.py builds it."""
    responses = dataset.responses
    keep = (responses["benchmark"] != benchmark) | responses["item_id"].isin(rated_items[benchmark])
    return data.Dataset(responses[keep].reset_index(drop=True), dataset.subjects, dataset.items)


def needed_subjects(dataset, rated_items, benchmark):
    """Subjects of the calibration pairs (full benchmark) and of the scored pairs (cut benchmark)."""
    settings = protocol.Settings(min_pair_items=rr.MIN_PAIR_ITEMS)
    full = {p.subject_id for p in protocol.eligible_pairs(dataset, benchmark, settings)}
    cut = {p.subject_id for p in protocol.eligible_pairs(cut_dataset(dataset, rated_items, benchmark), benchmark, settings)}
    return full, cut


def requests(dataset):
    _, rated_items, _ = rr.load_ratings(dataset)
    out = []
    for benchmark in rr.RATED:
        full, cut = needed_subjects(dataset, rated_items, benchmark)
        responses = dataset.responses[dataset.responses["benchmark"] == benchmark]
        own_items = responses.groupby("subject_id")["item_id"].agg(lambda s: sorted(set(s)))
        for subject_id in sorted(full | cut):
            own = own_items[subject_id]
            rated = [i for i in own if i in rated_items[benchmark]]
            example = random.Random(f"subject-prior|{benchmark}|{subject_id}").choice(rated or own)
            subject, item = dataset.make_input(subject_id, example)
            out.append({"benchmark": benchmark, "subject_id": subject_id, "example": example,
                        "rated_example": bool(rated), "in_full": subject_id in full, "in_cut": subject_id in cut,
                        "key": guess_core.guess_key(subject, item),
                        "args": guess_core.request_args(subject, item, SETTINGS)})
    return out


def api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
        return winreg.QueryValueEx(handle, "ANTHROPIC_API_KEY")[0]


def done_keys():
    if not OUT.exists():
        return set()
    return {r["key"] for r in map(json.loads, OUT.read_text(encoding="utf-8").splitlines()) if r["guess"] is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reqs = requests(data.load())
    keys = {r["key"] for r in reqs}
    if len(keys) != len(reqs):
        raise SystemExit("two requests share a key")
    chars = sum(len(r["args"]["system"]) + len(r["args"]["messages"][0]["content"]) for r in reqs)
    counts = {b: sum(r["benchmark"] == b for r in reqs) for b in rr.RATED}
    print(f"{len(reqs)} requests {counts}, {sum(not r['rated_example'] for r in reqs)} with an unrated example, "
          f"about {chars / 4 / 1e3:.0f}k input tokens")
    if args.dry_run:
        print(reqs[0]["args"]["messages"][0]["content"][:800])
        return

    import anthropic
    done = done_keys()
    todo = [r for r in reqs if r["key"] not in done]
    print(f"{len(todo)} to ask")
    client = anthropic.Anthropic(api_key=api_key(), max_retries=4)
    lock = threading.Lock()
    spent = {"usd": 0.0}

    def ask(req):
        with lock:
            if spent["usd"] >= BUDGET:
                return None
        response = client.beta.messages.create(**req["args"])
        text = "".join(block.text for block in response.content if block.type == "text")
        price_in, price_out = PRICES.get(response.model, PRICES["claude-opus-4-8"])
        usd = response.usage.input_tokens * price_in + response.usage.output_tokens * price_out
        record = {"key": req["key"], "benchmark": req["benchmark"], "subject_id": req["subject_id"],
                  "example": req["example"], "guess": guess_core.parse(text, response.stop_reason),
                  "text": text[:200], "stop_reason": response.stop_reason, "model": response.model,
                  "attempts": len(getattr(response.usage, "iterations", None) or []) or 1,
                  "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                  "usd": usd}
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
    print(f"spent this run: ${spent['usd']:.2f}; budget stop at ${BUDGET:.2f}; {skipped} not asked (budget)")


if __name__ == "__main__":
    main()
