# LLM priors on item difficulty and subject ability (PAIEC 2026)

Code, data products and results of my entry to the NeurIPS 2026 Predictive AI Evaluation
Competition (PAIEC). The predictor estimates the probability that an AI system (a subject)
answers an item of an unseen benchmark correctly, from the subject's recorded fields, the
item's text and the labels acquired so far. It chains small Bayesian models (a normal prior on
ability with a grid posterior, item difficulties pooled from other subjects' labels) with two
priors elicited from Claude Opus 5.5: a rating of each item's difficulty and a guess of each
subject's success rate.

- Final entry: version 5, Codabench submission **1005477** (`submissions/version5`).
- Repository: <https://github.com/stanley1208/paiec-2026>.
- Technical report: `docs/report/` (LaTeX sources); the PDF is submitted through OpenReview
  (NeurIPS.cc/2026/Workshop/PAIEC).

## Layout

| Path | Contents |
| --- | --- |
| `paiec/` | data loading, the offline harness that simulates the hosted protocol, scoring, predictors, the stdlib inference cores copied into submissions, and the LLM client |
| `scripts/` | experiment runs (each `run_*.py` states its question in its docstring, and the decided tests their decision rule), report tables, submission export and packaging |
| `submissions/` | the uploaded submission folders (constant, baseline, phase4 = V1, version2 to version5) and the platform probes (apicheck, netcheck, sdkcheck) |
| `results/runs/` | committed experiment runs: `config.json` (provenance), `per_pair.csv`, `summary.json` |
| `results/llm_rating/`, `results/subject_check/`, `results/subject_prior/` | the LLM ratings and guesses, the scripts that requested them, and READMEs |
| `results/formative_summary.json` | per-upload aggregates of the private Codabench feedback |
| `docs/report/` | the technical report |
| `docs/ai_assistance_log.md` | dated log of the AI assistance behind this entry |
| `tests/` | unit tests |

## Setup

1. Python 3.13 and `pip install -r requirements.txt`. The submissions themselves use only the
   standard library. numpy 1.26.4, the version the committed runs used (needed for bit-identical
   Rasch fits), has no Python 3.13 wheel, so pip builds it from source; install a C compiler first
   (on Windows, the Visual Studio Build Tools).
2. Data: measurement-db is a gated Hugging Face dataset. Accept its terms at
   <https://huggingface.co/datasets/aims-foundations/measurement-db>, log in
   (`huggingface-cli login`), then run `python scripts/download_data.py`. It fetches revision
   `bc8204d811823da849c6686bf124d4ca9f82e4de` into `data/` and records each file's SHA-256;
   every run folder records the same hashes for comparison.
3. Starting kit: the harness reuses the acquisition code of the official starting kit, which has
   no license and is therefore not included. Download `starting_kit.zip` from the Codabench
   competition page (competition 17828, build `5f7b7bf`, SHA-256 `a870970a7b9e82385929db37435aac29e3f0ee648cce7df143787b88ac4770f6`)
   and copy its `tools/` and `sample_data/` folders into `third_party/paiec_starting_kit/`.

## Reproducing the report

| Report item | Command |
| --- | --- |
| Tests | `python -m pytest -q` |
| Table 1 (data), the worked example, the probit error | `python scripts/report_tables.py` |
| Tables 2-3 and Figure 1 | `python scripts/report_results.py` (Table 3 is rebuilt from `results/formative_summary.json`) |
| Baselines and V1 settings (default setting) | `python scripts/run_harness.py --predictor beta_binomial_subject`, `--predictor logit_normal --options '{"scale": 0.5}'`, and so on; each run folder's `config.json` records its command |
| V3 in the hidden-sized setting with one partner | `python scripts/run_harness.py --predictor item_pooling --options '{"scale": 0.0}' --config configs/hidden_partners1.json` (reproduces `results/runs/20261005-213957_item_pooling-scale0.0-hidden-partners1`, a folder renamed by hand whose `config.json` records a scratch copy of this config) |
| Item pooling, agent rule and question groups | `python scripts/run_pooling.py`, `python scripts/run_ideas.py` |
| LLM rating test (200 items) | `python scripts/build_rating_samples.py --out results/llm_rating`, then `python results/llm_rating/analyze.py` |
| Ratings, rating trust, gate, final V3 vs V4 comparison | `python scripts/run_ratings.py`, `run_trust.py`, `run_gate.py`, `run_final.py` |
| Subject-strength check | `python results/subject_check/check.py --analyze` |
| V5 test (guessed subject priors) | `python scripts/run_subject_prior.py` |
| Packaged V4 and V5 run through the harness with cached replies | `python scripts/run_zip_check.py` |
| Final zip | `python scripts/export_version5.py submissions/version5` then `python scripts/build_submission.py submissions/version5`; `dist/version5.zip` has SHA-256 `b7357ae7bc4459ea4751a234c43f7b7cab2ec48ebd1baf0f833e9061884c578e` |

The commands use POSIX shell quoting (bash, zsh, Git Bash); in Windows PowerShell write
`--options '{\"scale\": 0.5}'`. None of them needs an API key: the LLM ratings and guesses are
saved in `results/`. Three analyses cannot be rerun because their scratch scripts were not kept: the
text-embedding test, the scenario analysis behind V1's prior scale and the later V5 runs with
shifted-and-shuffled or reversed guesses (all described in the report). The rating correlations on all 1,049 rated items were computed inline; they follow from
the committed `sample_*_ids.json` and `ratings*.jsonl` files.

The run folders record the commit of the development repository they ran from. This public
repository starts from a fresh history, because the development history contains private
Codabench feedback, so those commit hashes do not exist here.

## The API during scoring

V4 and V5 call the Anthropic API while Codabench scores them, but only when the submission
folder contains `anthropic_api_key.txt`. `scripts/add_api_key.py` writes a private copy of a
built zip with the key; never commit a key or share that copy. Without a key, V5 predicts
exactly as V3. Requesting new ratings or guesses (`results/llm_rating/rate.py`,
`results/subject_check/check.py --run`, `results/subject_prior/ask.py`) needs `ANTHROPIC_API_KEY`
and the `anthropic` package (pinned in `requirements-lm.txt`), costs money, and gives somewhat
different replies.

## AI assistance

Claude Code with Claude Opus 5.5 wrote most of this code, ran the experiments and drafted the
report under my direction; commits carry its co-author line and `docs/ai_assistance_log.md`
records who did what.

## License

Code: MIT (`LICENSE`), except `docs/report/neurips_2026.sty` (the unmodified NeurIPS 2026 style
file, by its authors) and the starting kit's three-line default acquisition rule, which
`paiec/protocol.py` (`random_query`) reproduces so that the harness matches the platform. The LLM
ratings and guesses and the other data products derived from measurement-db (run results, Rasch
difficulties, fitted parameters): CC BY-SA 4.0, like the measurement-db curation
(<https://huggingface.co/datasets/aims-foundations/measurement-db>). measurement-db itself and
the starting kit are not redistributed here. `NOTICE.md` repeats these exceptions.
