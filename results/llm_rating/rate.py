"""Option A upper bound: can a frontier LLM rate item difficulty zero-shot?

Pre-registered (2026-10-06): Claude Opus 5.5 rates 200 public items (50 per benchmark, seeded)
from 1 to 10 with a fixed rubric, seeing only the item's content and features. GO to testing a
deployable open model only if the Spearman correlation between rating and Rasch difficulty
(fitted from all subjects) is >= 0.30 on at least 3 of the 4 benchmarks and > 0 on all 4.

Run with the project venv: C:/AIMS/.venv/Scripts/python rate.py [--limit N]
The API key is read from ANTHROPIC_API_KEY, or from the user's saved Windows environment
variable; it is never printed or written anywhere.
"""
import argparse
import concurrent.futures as cf
import json
import os

import threading
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4.00 / 1e6, 20.00 / 1e6  # USD per token (Claude Opus 5.5)
BUDGET = 15.0  # stop starting new requests once spend reaches this

SYSTEM = """You estimate how hard benchmark tasks are for current AI systems: large language \
models and LLM-based agents released 2023-2026, from small open models to frontier models.

Rate the task you are given on a scale from 1 to 10:
1 = almost every such AI system completes it correctly
5 = about half of them do
10 = almost none of them do

Judge from the task itself: the knowledge it needs, the number of reasoning or action steps, \
how long and intricate the required output is, whether it needs tools, code execution or \
interaction with an environment, and how ambiguous the success criterion is. Do not try to \
solve the task.

Reply with only the integer rating."""

SCHEMA = {"type": "object", "properties": {"rating": {"type": "integer"}},
          "required": ["rating"], "additionalProperties": False}


def api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
        return winreg.QueryValueEx(handle, "ANTHROPIC_API_KEY")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sample = json.loads((HERE / "sample.json").read_text(encoding="utf-8"))[:args.limit]
    out_path = HERE / "ratings.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["item_id"] for line in out_path.read_text(encoding="utf-8").splitlines()}
    todo = [s for s in sample if s["item_id"] not in done]

    client = anthropic.Anthropic(api_key=api_key(), max_retries=4)
    lock = threading.Lock()
    spent = {"usd": 0.0}

    def rate(item):
        with lock:
            if spent["usd"] >= BUDGET:
                return None
        prompt = f"Task features: {item['features'] or '(none)'}\n\nTask:\n{item['content']}"
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": "claude-opus-4-8"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            rating = int(json.loads(text)["rating"]) if response.stop_reason != "refusal" else None
        except (ValueError, KeyError, TypeError):
            rating = None
        if rating is not None and not 1 <= rating <= 10:
            rating = None
        usd = response.usage.input_tokens * PRICE_IN + response.usage.output_tokens * PRICE_OUT
        record = {"item_id": item["item_id"], "benchmark": item["benchmark"], "rating": rating,
                  "text": text[:200], "stop_reason": response.stop_reason, "model": response.model,
                  "input_tokens": response.usage.input_tokens,
                  "output_tokens": response.usage.output_tokens, "usd": usd}
        with lock:
            spent["usd"] += usd
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        return record

    with cf.ThreadPoolExecutor(4) as pool:
        for record in pool.map(rate, todo):
            if record is not None:
                print(f"{record['benchmark'][:12]:12s} rating {record['rating']} "
                      f"({record['input_tokens']} in / {record['output_tokens']} out, ${record['usd']:.3f})",
                      flush=True)
    print(f"spent this run: ${spent['usd']:.2f}; budget stop at ${BUDGET:.2f}")


if __name__ == "__main__":
    main()
