# Option C full test: version 4 with Claude's guess as each subject's prior

Run on 2026-10-17. The decision rule is the docstring of `scripts/run_subject_prior.py` (committed
in 45eb1b3 before any guess was requested; a89cb9d fixed a bug without changing the rule).
These hashes refer to the development repository, whose history is not public (see the top-level
README).

## Guesses

`ask.py` asked Claude Opus 5.5 (`claude-opus-5-5`, low effort, JSON-schema output, same
server-side fallback as version 4) for one guess per subject and rated benchmark, using the
request of `paiec/guess_core.py`: the subject's recorded fields and one of its own tasks. That is
232 requests for every subject the test scores or calibrates on. All 232 replies were valid and
came from Claude Opus 5.5 (no fallback). Estimated cost $0.80, from token counts at list prices,
not billing.

On the 192 subjects of the pre-check, these guesses match Rasch ability at 0.771 (0.749, 0.742,
0.764, 0.831). The pre-check's three-example prompt matched at 0.818.

## Result (run `results/runs/20261017-021515_subject_prior`)

Pair-mean Brier ALC on the 198 pairs of `scripts/run_final.py`, seeds 0-2. Version 4 scores
0.1930 with K = 0 partner pairs.

| Guess trust t | 0 (version 4) | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| Real guesses, K = 0 | 0.1930 | 0.1877 | 0.1832 | **0.1795** | 0.1782 |
| Shuffled guesses | 0.1930 | 0.1904 | 0.1893 | 0.1906 | 0.2001 |
| Guesses +1 logit | 0.1930 | 0.1897 | 0.1880 | 0.1886 | 0.1955 |
| Guesses -1 logit | 0.1930 | 0.1865 | 0.1822 | 0.1804 | 0.1857 |
| Worst regret | 0.0148 | 0.0094 | 0.0050 | **0.0013** | 0.0108 |

The chosen trust is 0.75. Against version 4, option C scores -0.0135 (SE 0.0013) with K = 0,
-0.0132 with K = 1 and -0.0132 with K = 2. By benchmark (K = 0): MathArena -0.0077,
Multi-SWE-bench -0.0217, REAL -0.0101 and ResearchCodeBench -0.0109. All three checks pass:
**GO** to building a hosted version. An independent recomputation from `per_pair.csv` reproduced
every number. The version 4 rows are identical to those of `results/runs/20261009-003522_final`.

## Caveats (from a skeptical review of the run)

- Most of the gain comes early: budget 0 gives 34% and budget 1 gives 40%. These are the budgets
  where version 1's metadata starting guess helped offline but hurt in all three hosted runs
  (budget-0 Brier 0.283-0.303 against 0.25). Option C's starting guess comes from a different
  source, but nothing offline shows that it transfers to the hidden benchmarks.
- The stress scenarios are milder than that failure. Shuffled guesses keep the benchmark's level,
  and shuffling was never combined with a level shift. At t = 0.75, the budget-0 Brier change
  against 0.25 is -0.046 with the real guesses but +0.006 with shuffled guesses shifted +1 logit,
  and +0.021 with the ranking reversed.
- Multi-SWE-bench gives 57% of the gain, and about half of its gain comes from the benchmark's
  level (it still gains -0.0107 with shuffled guesses). MathArena has the largest downside: +0.0081
  with shuffled guesses.
- The standard error treats subjects as the unit. Over the four benchmarks as units it is about
  0.003. The benchmarks are public, so Claude may remember their leaderboards.

## Files

- `ask.py`, `guesses.jsonl`: the requests and the 232 replies.
- The run folder holds `config.json` (provenance), `per_pair.csv` and `summary.json`.
- The first run, `results/runs/20261017-015838_subject_prior`, is invalid: the guess trust
  overwrote version 4's rating trust. It is kept locally and not committed.
