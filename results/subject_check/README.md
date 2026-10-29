# Option C pre-check: does Claude know how strong each AI system is?

Run on 2026-10-16 (UTC+8). The decision rule is in the docstring of `check.py` and was committed
(a5100b6) before the run. Commit hashes in this folder refer to the development repository,
whose history is not public (see the top-level README).

Claude Opus 5.5 (`claude-opus-5-5`, low effort, JSON-schema output, same server-side fallback as
version 4) saw one subject's recorded fields, exactly as a hosted predictor receives them, plus
three example tasks of the benchmark (no benchmark name and no labels). It guessed the
percentage of the benchmark's tasks that the subject completes. There was one request per
eligible subject of the default setting: 192 requests, no invalid replies, no fallback answers,
and an estimated cost of $1.37 (from token counts at list prices, not billing).

## Result

| Benchmark | Subjects | Pearson with theta | Spearman | Mean guess | Mean success | Brier of guess | Brier of known rate |
|---|---|---|---|---|---|---|---|
| MathArena | 81 | 0.957 | 0.966 | 0.607 | 0.624 | 0.2149 | 0.1968 |
| Multi-SWE-bench | 54 | 0.767 | 0.758 | 0.365 | 0.298 | 0.1998 | 0.1853 |
| REAL | 26 | 0.661 | 0.566 | 0.353 | 0.340 | 0.2064 | 0.1723 |
| ResearchCodeBench | 31 | 0.886 | 0.856 | 0.284 | 0.353 | 0.2202 | 0.2048 |

Model match (mean Pearson) is 0.818, against a threshold of 0.70: **GO** under the rule. A
constant 0.5 scores Brier 0.25. "Known rate" is the Brier score of predicting each subject's
own success rate exactly.

## Comparisons (`compare.py`, written after the run)

- The result does not depend on the Rasch priors: 0.817 and 0.816 with two other choices.
- The version 1 metadata starting guess, fitted on the other benchmarks, scores 0.611 on the same
  measure (0.527, 0.682, 0.617, 0.619). Version 3 dropped it after formative feedback.
- Release year alone scores 0.815, 0.440, 0.323 and 0.368. On the same subjects, Claude scores
  0.966, 0.764, 0.639 and 0.886.

## Caveats

These benchmarks are public and Claude may remember published results, MathArena's per-model
leaderboard in particular, so the hidden benchmarks will probably show a lower match. A GO here
only justifies the full option C test; on its own it is not evidence that option C helps.

## Files

- `check.py`: the pre-registered check (`--dry-run`, `--run`, `--analyze`).
- `guesses.jsonl`: one record per request (guess, model that answered, tokens, estimated cost).
- `analysis.json`, `per_subject.json`: written by `--analyze`.
- `compare.py`: the comparisons above.
