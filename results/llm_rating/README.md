# LLM difficulty ratings

Claude Opus 5.5 (`claude-opus-5-5`) rated 1,049 public measurement-db items from 1 (almost every
AI system succeeds) to 10 (almost none do), seeing only each item's content and features, at low
effort with a JSON-schema output (prompt in `rate.py`, identical to `paiec/llm_rating_core.py`).
All 1,049 replies came from that model; the server-side fallback was never used.

## Step 1: upper-bound test (2026-10-06)

200 items, 50 per benchmark. The decision rule, fixed before the run, is in `rate.py`;
`analyze.py` applies it. Estimated cost $9.16 (from token counts at assumed prices, not billing).
Spearman correlation between rating and Rasch difficulty: 0.87 (MathArena), 0.50
(Multi-SWE-bench), 0.56 (REAL), 0.77 (ResearchCodeBench): GO under the rule.

## Step 2: more ratings (2026-10-06)

849 more items: every other informative MathArena problem (316) and REAL task (183), and 350
Multi-SWE-bench items. Estimated cost $2.92 (total for both steps: $12.08). `rate_more.py` is a
copy of `rate.py` with other file names and a $10 cap; its docstring still describes step 1.

Spearman correlations on the step 2 items: 0.81 (MathArena, 316), 0.26 (Multi-SWE-bench, 350),
0.41 (REAL, 183). On all rated items: 0.82 (366), 0.28 (400), 0.44 (233) and 0.77 (50). The
larger samples show that the step 1 figures for Multi-SWE-bench and REAL were optimistic.

## How the items and difficulties were chosen

`python scripts/build_rating_samples.py --check` regenerates both samples and their Rasch
difficulties from the pinned data and compares them with the files here (it reproduces them
exactly). In short: a Rasch model fitted separately per benchmark from all subjects (penalized
joint maximum likelihood); items whose content is shared by another problem are left out ("not
informative"), so MathArena keeps 366 of its 636 problems; one item per problem; seeded draws.
The script's docstring gives the details. The rating scripts read the item contents from
`sample.json` and `sample_more.json`, which `python scripts/build_rating_samples.py --out
results/llm_rating` writes (they are not committed).

## Files

- `sample_ids.json`, `sample_more_ids.json`: the rated items and their Rasch difficulties.
- `ratings.jsonl`, `ratings_more.jsonl`: one rating per item, with token counts and estimated cost.
- `rate.py`, `rate_more.py`: as run (they need an `ANTHROPIC_API_KEY`). `analyze.py` needs no key
  but reads `sample.json`; write it first with `python scripts/build_rating_samples.py --out
  results/llm_rating`. The correlations on all 1,049 rated items follow from the committed
  `sample_*_ids.json` and `ratings*.jsonl` files; they were computed inline, not by a script.

These are public benchmarks, so the model may partly remember how hard their items are; hidden
benchmarks may correlate less.
