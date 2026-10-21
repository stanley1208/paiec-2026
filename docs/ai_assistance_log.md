# AI assistance log

Running record of work done with an LLM coding assistant (Claude Code, model Claude Opus 5.5),
for disclosure in the technical report. Each phase is reviewed and approved by the author
before the next one begins. Files named below that are absent from the public copy of this
repository (the vendored starting kit and the internal report-writing notes) were left out of
the release on purpose.

## 2026-09-27 to 2026-09-29: Phase 0, investigation (no repository code; throwaway analysis scripts in scratch only)

- Read the competition rules page, Codabench metadata, the official starting kit (build
  `5f7b7bf`), the organizers' baseline repository (`paiec_baseline` @ `82d330d`), the
  measurement-db builders, and `torch_measure`; compared them against the author's notes.
- Downloaded measurement-db at Hugging Face revision `bc8204d8` (scratch only) and computed
  per-benchmark statistics; a second, independent script recomputed them, and a separate
  pass audited label parsing and ID consistency.
- Searched public sources (website and its archived copy, organizer documents, Codabench,
  Hugging Face, GitHub) and, read-only, the AIMS Discord and Codabench forums for answers to
  open questions; drafted questions for the author to post.
- Assistant sub-agents were used for parallel reading and independent re-checks of claims.

## 2026-09-29: Phase 1, constant-0.5 submission

- Set up the repository; vendored the starting-kit `tools/` and `sample_data/` unchanged,
  with a provenance note (`third_party/paiec_starting_kit/NOTICE.md`).
- Wrote `submission/model.py` (always returns 0.5) and `scripts/build_submission.py`
  (reproducible zip).
- Checks run: kit zip checker (OK); kit smoke test sequentially on Windows, Python 3.13.2
  (OK), and with isolated workers in WSL Ubuntu 24.04, Python 3.12.3 (OK); direct calls in
  the positional, keyword and no-label forms return a float 0.5.
- Two assistant reviewer sub-agents independently rebuilt the zip from a fresh clone and
  re-ran the checks; their findings led to a platform-independent zip (same hash on Windows
  and Linux) and wording fixes in this log.
- Artifact: `dist/submission.zip`, sha256 `817135d909942846077a1c17fe0fd60c66bfda968595b95c9f8d509e19df7705`.
- Result: uploaded by the author to Codabench (submission 951242, 2026-09-29);
  Brier ALC 0.250000, as expected for a constant 0.5 on binary outcomes.

## 2026-09-30: Phase 2, offline evaluation harness

- Wrote `paiec/` (pinned data loader with integrity and hash checks, Brier/ECE/ALC scoring,
  a leave-one-benchmark-out simulator of the streaming protocol that reuses the starting
  kit's `AcquisitionHook` and `prediction_input_key`, result summaries and simulated
  Codabench-style draws), `scripts/download_data.py`, `scripts/run_harness.py`, two
  configs and 30 tests.
- Checks: the constant 0.5 predictor scores exactly 0.25 on all 192 default pairs (221 in
  the full setting); hand-worked scoring examples, including the rules-page ALC example and
  one row of the author's Codabench feedback, reproduce exactly.
- Three assistant reviewer sub-agents audited the harness: a replay through the kit's own
  streaming client matched acquisition decisions exactly, and an independent
  re-implementation matched every per-pair score to 1e-12. Their findings (a training-data
  aliasing leak, shared evidence between calls, provenance gaps, submission-loading
  differences) were fixed with new tests; the tests for the leak and the shared evidence
  were shown to fail on the old code.

## 2026-09-30 to 2026-10-01: Phase 3, baselines

- Wrote `paiec/baseline_core.py` (stdlib-only inference shared by the harness and the
  exported submission), `paiec/baselines.py` (global mean; metadata prior from a weighted
  ridge regression with benchmark fixed effects; Beta-Binomial update from the pair's own
  revealed labels), `paiec/export.py`, `paiec/feedback.py`, the export and feedback-check
  scripts, and tests.
- Offline results (default config, 3 seeds, estimated Codabench score): global mean 0.253,
  metadata prior 0.237, Beta-Binomial from the global mean 0.217, Beta-Binomial from the
  metadata prior 0.211 (benchmark mean 0.213; about 0.214 after accounting for choosing the
  update strength on the same folds).
- Two assistant reviewer sub-agents re-derived the fit independently, checked for leakage,
  fuzzed the exported submission and tested it on Linux. Following their findings, features
  that vary within only one public benchmark were dropped (no offline change) and the
  submission's error handling was made finer-grained.
- Result: uploaded by the author to Codabench (submission 954621, 2026-10-01); Brier ALC
  0.232172, inside the offline 90% range (0.158 to 0.255). `scripts/check_feedback.py`
  reproduced every pair's ALC from the per-budget tables exactly, and their mean equals the
  Codabench score.

## 2026-10-02: Phase 4 step 1, cautious starting guess

- Wrote `paiec/logit_normal_core.py` (stdlib-only inference: a logit-normal prior centred at
  0.5 plus the subject's metadata score relative to the training subjects, then the exact
  posterior mean given the pair's own labels, on a fixed grid), `paiec/logit_normal.py`
  (fitted on the training benchmarks only, reusing the Phase 3 metadata regression),
  `paiec/stress.py` and `scripts/run_stress.py` (held-out benchmarks made 1 or 2 logits
  easier or harder; shuffled and reversed metadata as controls), and tests.
- Offline results (default config, 3 seeds): mean over pairs 0.2142 against 0.2161 for
  Phase 3 (difference -0.0019, standard error 0.0010 clustered by pair); mean over
  benchmarks 0.2137 against 0.2129; simulated Codabench draws 0.2115 for both. It is better
  on MathArena and slightly worse on the other three benchmarks, so there is no clear gain
  when the held-out benchmark is as hard as the training ones. With shifted benchmarks the
  budget-0 Brier stays between 0.231 and 0.250 pooled (Phase 3: 0.209 to 0.294); the ALC is
  0.013 to 0.024 better when benchmarks are easier and about 0.005 worse when harder.
  Controls: no metadata 0.2188, shuffled 0.2180, reversed 0.2291.
- Four assistant reviewer sub-agents (an independent re-implementation with a different
  integration method, a leakage and interface audit, code review with mutation testing, and
  a stress-test audit), each followed by a sceptical verifier sub-agent. The
  re-implementation matched every per-pair score to 2e-15 and no leakage was found. Their
  findings, all minor, led to stronger tests (11 of 11 deliberate code mutations are now
  caught), corrected docstrings, per-benchmark results and paired standard errors in the
  stress summary, and the more cautious wording above. A scratch scenario analysis (subject
  metadata that does not carry over, subjects newer than any in training) found that a
  half-strength metadata nudge has about a third of the full nudge's worst-case loss with
  the same average; two of its scenarios were chosen to resemble the per-budget curve in the
  Phase 3 Codabench feedback, and the conclusion holds without them. The choice is left to
  the author.

## 2026-10-02: Phase 4 step 2, packaged submission

- The author chose the half-strength metadata nudge (scale 0.5).
- Extended `paiec/export.py` and `scripts/export_submission.py` to the Phase 4 model (the
  Phase 3 export was checked to be byte-identical), exported `submissions/phase4` (fitted
  on all public binary data) and built `dist/phase4.zip`, sha256
  `06da4037e444f3f7860e67a0d456c24eafd9082074c71eda6278cc4da0937208`.
- Checks: 57 tests; the kit's zip checker and smoke test on Windows and in WSL Ubuntu
  24.04 (Python 3.12.3), sequentially and with 4 isolated workers; a Linux rebuild gives
  the same hash; 60,000 fuzzed malformed calls all returned a valid probability; a harness
  run through the exported folder, reloaded at every checkpoint, completed. The offline
  estimate for this setting is 0.2158 over pairs, 0.2154 over benchmarks and 0.2131 on
  simulated Codabench draws.
- Three assistant reviewer sub-agents (a fresh-clone rebuild, hosted-platform robustness
  including an end-to-end run of the kit's streaming client with JSON round trips, and
  equivalence of the shipped model with the evaluated one), followed by a sceptical
  verifier, found no blocking or major issue; the shipped model matched the evaluated
  predictor exactly on every public pair.
- Result: uploaded by the author to Codabench (submission 958131, 2026-10-03); Brier ALC
  0.202502, inside the offline 90% range. This upload was scored on a different draw of 9
  pairs from the same 7 benchmarks (1 pair in common with submission 954621), so the change
  from 0.232172 is partly due to the draw. `scripts/check_feedback.py` reproduced every
  pair's ALC exactly. As agreed beforehand, nothing was retuned on this feedback.

## 2026-10-03: Phase 4 step 3, language-model difficulty check (negative result)

- With the author's approval, set up a project-local `.venv` (CPU PyTorch,
  sentence-transformers; versions in `requirements-lm.txt`) and downloaded two small text
  embedding models to `data/models/` at pinned revisions: all-MiniLM-L6-v2 and
  bge-small-en-v1.5.
- Check, with the decision rule fixed before looking at results: embed every binary item's
  text, fit a ridge regression from the embeddings to the item difficulties estimated from
  other subjects (Rasch fits from the Phase 4 plan review), train on the other benchmarks,
  choose the ridge strength on the training benchmarks only, and measure the Spearman
  correlation with the true difficulties on the held-out benchmark. Continue only if the
  mean over the 4 benchmarks with eligible pairs reached 0.30.
- Result: no-go. Mean correlation -0.06 to 0.04 across the 4 embedding variants, against
  0.06 for the earlier word-count features. Within a single benchmark the same embeddings
  do rank difficulty (0.20 to 0.51, about the same as word counts), so the pipeline works
  and the failure is in transferring to an unseen benchmark. Removing each benchmark's
  mean embedding first did not help (-0.06 to 0.04). The language-model route was
  stopped and the Phase 4 submission stays the candidate. The scripts are scratch
  analyses for now; they will be moved into the repository with the other report
  experiments.

## 2026-10-03 to 2026-10-05: version 2, item difficulties pooled from other subjects

- Assistant sub-agents surveyed the organizers' public work (public professional sources
  only; nothing about the hidden test). The idea that fit best was a two-stage ("online")
  calibration: on the platform, `labeled` also carries other subjects' labels, which show
  how hard each item is.
- Wrote `paiec/item_pooling_core.py` (stdlib only: other subjects' abilities from their
  own labels, each item's difficulty from their labels on it, then the Phase 4 posterior
  using those difficulties; exactly the Phase 4 prediction when no other subject labeled a
  relevant item), `paiec/item_pooling.py`, a "partners" evidence mode and an item-pool
  option in the harness, `scripts/run_pooling.py` with a decision rule fixed before the
  first run, and tests.
- Offline (pairs cut to about Codabench size, one partner pair per target, 3 seeds):
  -0.00116 Brier ALC (standard error 0.00015, clustered by pair), better on all 4
  benchmarks that can have partners; -0.0022 with two partners and -0.0043 with four;
  -0.0009 on full-size pairs; about zero if the platform gives all subjects the same item
  split, which is not documented. The expected change at the density of the formative draws
  is about -0.0004, too small to see in one upload.
- Four assistant reviewer sub-agents (an independent re-implementation of the maths,
  leakage and platform fidelity, hosted robustness, statistics), each followed by a
  sceptical verifier, found no leakage. Their findings led to leaving each partner label out
  of that partner's own ability, to an exporter that refuses unknown models (before, it
  would have packaged version 2 as the Phase 3 baseline without warning), and to reporting
  the plain mean over pairs.
- Packaged `submissions/version2` as `dist/version2.zip`, sha256
  `5e798499907b1ec1c1c2201fa5e4c51f8a19af0f0f5f586132abe0821a4f6cf1`. Checks: 70 tests;
  the kit's checker and smoke test on Windows and in WSL with 4 isolated workers; a Linux
  rebuild gives the same hash; a 12,000-call fuzz with several subjects returned a valid
  probability every time; a harness run through the submission folder with partner labels
  matched the in-repository model. Three more reviewer sub-agents with verifiers (fresh
  rebuild, the kit's worker machinery on multi-subject draws, equivalence on real data)
  found no blocking issue.
- Result: the author's first upload (submission 961774) stayed in "Preparing" for over 12
  hours while later submissions by others finished, so the author cancelled it and uploaded
  the same zip again (submission 963204, 2026-10-05): Brier ALC 0.187682. This draw had 8
  pairs, and only 2 of them shared a benchmark, so on the other 6 version 2 made exactly the
  Phase 4 predictions. One pair that was also in the Phase 4 draw scored identically both
  times. The change from 0.202502 therefore mostly reflects the draw, not the model.
  `scripts/check_feedback.py` reproduced every pair's ALC exactly; nothing was retuned.

## 2026-10-05: two more ideas from the organizers' work (negative results)

- Added two optional settings to `paiec/item_pooling_core.py`, both off by default (the
  uploaded version 2 folder is kept as submitted, and a test checks that the current code
  still predicts exactly like it): an "agent rule" (no metadata nudge for subjects with a
  harness, after the organizers' finding that the harness matters more than the model) and
  "question groups" (a subject-specific effect per group of items sharing their first
  feature, such as a competition, website or paper, after the testlet model in
  torch_measure). `scripts/run_ideas.py` compares each with version 2 on the same pairs,
  with a decision rule fixed before the run.
- Agent rule: worse, +0.0023 Brier ALC on Codabench-sized pairs with one partner and
  +0.0008 on full-size pairs, all on Multi-SWE-bench (the only public benchmark with
  harness subjects) and mostly at budget 0. No-go.
- Question groups: -0.0004 (standard error 0.0004) on Codabench-sized pairs and -0.0009 on
  full-size pairs. Better on MathArena, REAL and ResearchCodeBench (groups are competitions,
  websites and papers), worse on Multi-SWE-bench by 0.0019 (groups are programming
  languages). No-go under the rule. Changing the group definition after seeing this would
  be selected on the same benchmarks, so it was not tried; it is noted for the report.
  Version 2 stays the candidate.

## 2026-10-05: version 3, first guess at 0.5 (a decision informed by formative feedback)

- Disclosure: in all three formative uploads of the Phase 4 family, the budget-0 Brier
  score was above 0.25 (the score of a constant 0.5) in most pairs. Knowing that offline
  evidence pointed the other way, and that this departs from the project's own rule of not
  tuning on formative feedback, the author chose to remove the metadata nudge (scale 0, so
  every subject starts at 0.5). This is the only setting in the project chosen with
  formative test feedback.
- Offline, version 3 is worse than version 2 on all 4 public benchmarks: +0.0030 Brier ALC
  on full-size pairs and +0.0046 on Codabench-sized pairs with one partner
  (`results/runs/20261005-213837_item_pooling-scale0.0` and
  `...-hidden-partners1`). Budget 0 becomes exactly 0.25.
- Packaged `submissions/version3` as `dist/version3.zip`, sha256
  `c9ec6a9e65dac43979697897be0b3f398da3f21c5ad6971cedd612f33db28056`; its params differ from
  version 2's only in scale. Checks: 74 tests, the kit's checker
  and smoke tests on Windows and in WSL with 4 workers, a Linux rebuild with the same hash, a
  12,000-call fuzz, and a harness run through the submission folder matching the
  in-repository model exactly.
- Before the upload the author agreed that version 3 would become the final submission if
  it ran, whatever its score, so that the choice would not depend on one draw.
- Result: Codabench submission 963789 (2026-10-05), Brier ALC 0.155874, a new draw of 9
  pairs (none seen in earlier uploads). Budget 0 was 0.25 in every pair. Setting budget 0
  to 0.25 in the two previous uploads would have lowered them by 0.003 and 0.005, so most
  of the change from 0.1877 reflects an easier draw. `scripts/check_feedback.py`
  reproduced every pair's ALC (to 1e-6, the rounding of the tables).

## 2026-10-06: can a frontier LLM rate item difficulty? (option A, upper-bound test)

- With the author's approval and API key (set by the author as a Windows user environment
  variable; read by the script, never printed or stored), Claude Opus 5.5 rated 200 public
  items (50 per benchmark, seeded) from 1 to 10 with a fixed rubric, at low effort with a
  JSON-schema output. Cost $9.16 (estimate was about $7; some ResearchCodeBench items were
  twice as long as assumed). Two short trial runs with a first-number parser were discarded
  after the parser was found to be unreliable.
- Spearman correlation with the Rasch difficulty, by benchmark: 0.87, 0.50, 0.56, 0.77. The
  rule fixed before the run (at least 0.30 on 3 of 4 benchmarks and positive on all) passed.
  Caveat: these benchmarks are public, so the model may partly remember how hard items are.
- The starting kit says API-based submissions are supported (network allowlist, up to 16
  workers), and the organizers' own BLE baseline calls an LLM API with credentials placed in
  a private config inside the submission zip. Which API hosts are allowlisted is not stated.
  Results and scripts: `results/llm_rating/`.
- Step 2: rated 849 more public items ($2.92; every MathArena problem and REAL task, 400
  Multi-SWE-bench tasks), then added an optional rating prior to
  `paiec/item_pooling_core.py` (off by default; tests check that the uploaded version 2
  and version 3 folders still predict exactly like the code): a rated item's difficulty gets
  the prior N(slope * (rating - mean rating of the labeled items), sd^2), so ratings only rank
  items against those already labeled and budget 0 is unchanged. Slope and sd are fitted on
  the other rated benchmarks only. Ratings are reused only for items whose text shows the
  problem (MathArena's system-prompt-only items are excluded).
- `scripts/run_ratings.py` (rule fixed before the run; held-out benchmarks cut to rated
  items, pairs of at least 40 items, seeds 0-2): version 3 with ratings beats version 3 by
  -0.0188 Brier ALC (standard error 0.0021), -0.028 MathArena, -0.0047 Multi-SWE-bench,
  -0.0086 REAL, -0.041 ResearchCodeBench, all from budget 1 on. GO. A control with the
  ratings shuffled within each benchmark is +0.027 worse, so the gain comes from the
  ratings' information, and misleading ratings would cost about as much as good ones gain.
- Safety tuning (`scripts/run_trust.py`, rule fixed before the run): a trust level t scales
  the fitted slope (keeping each item's total prior variance), tested at t = 0 to 1 under
  real, half-noisy and useless (shuffled) ratings. Minimising the largest regret picked
  t = 0.5: -0.0150 with real ratings, -0.0049 with half-noisy ratings, +0.0065 with useless
  ratings (full trust: -0.0188, +0.0027, +0.0236).
- Prepared `submissions/netcheck` (`dist/netcheck.zip`, no credentials): predicts 0.5 and
  fails on purpose if api.anthropic.com is unreachable from the scoring environment, to check
  the network allowlist with one upload by the author.
- Network check result: Codabench submission 964074 (2026-10-06) produced predictions for
  every pair at every budget (all 0.25, as a constant 0.5 must), so api.anthropic.com was
  reachable from the scoring environment.
- Built version 4 (`scripts/export_version4.py`, `paiec/llm_rating_core.py`,
  `submissions/version4`): version 3 plus Claude Opus 5.5 difficulty ratings at trust 0.5,
  requested during scoring with the same prompt and request as the offline test. The zip holds
  no key; the author runs `scripts/add_api_key.py` to make a private copy with one. Without a
  key, or if the SDK, the folder or the API fails, it predicts exactly like version 3.
- An independent review (several Claude agents) found one blocker and several important
  issues, all fixed before any upload: a prediction could wait 90 s for ratings (now at most
  4 s, with requests on background threads and ratings used once ready); SDK retries were not
  counted against the call cap (now no SDK retries, every attempt counted, cap 1000 calls);
  a fallback cache in each worker's own temp folder could multiply calls (removed); failed
  requests were cached as permanent (now retried once, never cached); abandoned lock files
  blocked items (taken over after 60 s); no logging (one stderr line per worker now says
  whether ratings are on); the key format check and the private zip's location (now the
  user's home folder). It also found that reversed ratings would cost +0.027, so a gate
  switches ratings off on a benchmark whose rated labels show failed items rated easier
  than solved ones.
- `scripts/run_gate.py` (rule fixed before the run): with the gate, real ratings -0.0144 (was
  -0.0150), useless +0.0048 (was +0.0065), reversed +0.0110 (was +0.0268) against version 3.
  The gate lowers the largest regret (0.011, against 0.027 without it and 0.015 for version
  3), so it stays on.
- Checks on the final zip: 87 tests; the kit checker and smoke tests on Windows and Linux
  (Linux rebuild gives the same zip); through the offline harness without a key it scores
  0.199368, exactly version 3; a live test on 3 public items reproduced the saved ratings
  and the core model's prediction (about 1 cent); a wrong key and a hanging API both give
  version 3's predictions with no prediction slower than the 4 s budget.
- Prepared `submissions/sdkcheck` (`dist/sdkcheck.zip`, dummy key only): predicts 0.5 and
  fails on purpose unless the pinned SDK installs, the submission folder is writable and a
  request built like version 4's reaches the API (401 with the dummy key), to check the
  hosted setup before the author uploads version 4 with a real key.
- A second, focused review (16 Claude agents: four reviewers, each finding checked by a
  skeptic; ratings simulated with fake clients through the kit's own worker pool, no
  network) found no blocker. Fixed: replies that the API is busy (429, 529) used up the call
  cap with no backoff (now free, and every worker pauses for the retry-after time); a
  rejected key now stops ratings in every worker at once; a request the API refuses (400)
  is marked unratable for free; after two failures a worker ignored ratings that other
  workers wrote later; lock files left by workers killed at the end of a checkpoint blocked
  items for 60 s (on Linux a lock whose owner process has ended is now taken over at once;
  otherwise after 45 s); requests started at budget 0 were always killed (budget 0 now
  waits up to 4 s too, so ratings are ready for budget 1); later predictions did not wait
  for ratings already being fetched; background threads are daemon threads;
  `scripts/build_submission.py` now refuses to pack a key file, a private zip or a rating
  cache; the key check lives in one place; the instructions now say that spending limits
  are set per workspace (not per key) and to revoke the key only after final results.
  Refuted: a too-short 20 s timeout for long items (the offline run timed them at about
  5 s) and a harmful gate on pooled labels.
- In the simulations (560 predictions per checkpoint), the share of predictions with all
  their labels rated at budgets 7 / 15 / 31 rose from 56% / 0% / 0% to 96% / 98% / 91% with
  a healthy API, and a 90-second outage no longer exhausts the cap (702 of 1000 attempts,
  none lost, against 1000 and 189 items lost). No prediction took longer than 4.2 s.
  Unchanged checks on the final zip: 99 tests, kit checker and smoke tests on Windows and
  Linux, 0.199368 without a key, live test (about 1 cent), wrong key and hanging API.
- Still unknown: whether all six checkpoints of a hosted run share one copy of the
  submission folder (if not, the cache and the call cap work per checkpoint, about 4 times
  the API cost). `submissions/sdkcheck` now logs each worker's role and the markers earlier
  workers left, which answers this.
- A final check of that change (8 Claude agents, real SDK errors through a mocked network,
  lock races reproduced on Linux) found no blocker or major issue. Fixed five minor ones:
  two workers could take over the same abandoned lock at once and both pay for a call (each
  lock now holds a unique token that is read back, and a worker only removes its own lock;
  the cache is read again once the lock is held), so the reproductions went from 4-19 of 60
  double owners and 2-3 extra calls per 64 items to none; a request that never left (no
  connection) counted against the cap (now free, with a pause); a billing error (402, or a
  400 about the credit balance or usage limit) marked every item asked as unratable (now it
  stops new requests in every worker, like a refused key, and writes nothing); a single 401
  or 403 discarded ratings already paid for (now they are still used). Also: the private
  zip is owner-only from the moment it is created, and `sdk_check_markers/` is git-ignored.
  Final zip `dist/version4.zip` (sha256 cf5dc20a...): 106 tests, kit checks on Windows and
  Linux, 0.199368 without a key, live test; in the simulations 736 API calls for 736 ratings
  with a healthy API, nothing lost in a 90-second outage, no prediction slower than 4.2 s.
- Codabench submission 968358 (`dist/sdkcheck.zip`, dummy key) failed with "No module named
  'anthropic'": the platform did not install requirements.txt. The netcheck 964074 also shows
  Failed now, although it produced predictions at every budget ("Prediction completed"; raw
  stdout is hidden for this competition, so our stderr lines are not visible either).
- So version 4 no longer needs the SDK: `llm_rating_core.Client` sends the same request with
  only the standard library (POST /v1/messages?beta=true with the same body and API headers,
  through the HTTPS proxy the environment names, as urllib and the SDK would). A comparison
  with the SDK through a mock transport gave an identical body, path and API headers and the
  same handling of every reply (rating, 401, 429 with retry-after, 400 credit, 529); a live
  test without the SDK reproduced the saved ratings. requirements.txt is gone from the zip.
  `scripts/add_api_key.py` now checks the key with a free request (the list of models) and
  refuses to write the private zip unless the API accepts it and offers the model.
- Since logs are hidden, `submissions/apicheck` (`dist/apicheck.zip`, dummy key) reports
  through its scores: every prediction is 0.5 when a version 4 request reaches the API (401
  for the dummy key), the folder is writable and later checkpoints see earlier files; 0.4,
  0.2 or 0.1 mark which of these fails. Locally (Windows and Linux, no SDK) it gives 0.5.
- A review of that change (8 Claude agents, a local fake proxy and TLS API, the kit's worker
  code on Linux) found no blocker or major issue. Fixed: the probe's API check now takes at
  most 6 seconds in all (up to 3 tries, so one network blip in one worker no longer reads as
  a blocked API); it writes its marker before the API check and records whether a
  prediction worker wrote it, with a separate code (0.45) when only other processes left
  markers; failures before a request is sent (a bad proxy address or proxy reply) are free.
  Simulated through the kit flow, every case gives a distinct answer: all good 0.5; a copy
  of the folder per checkpoint 0.5 then 0.4 (0.45 with a validation call); API down at
  budget 0 0.1 then 0.5; a read-only folder 0.2; a slow or silent API 0.1 within 6.2 s.
  `scripts/decode_apicheck.py` reads the codes from the feedback; on the constant-0.5
  upload 951242 it reads 0.5 at every budget, which also confirms how the platform's ECE
  treats a constant prediction. Final zips: `dist/version4.zip` (sha256 f538d851...) and
  `dist/apicheck.zip` (84f41a5f...); 130 tests; live test with the real key reproduced the
  saved ratings without the SDK.
- API check result: Codabench submission 968649 (`dist/apicheck.zip`, dummy key) finished with
  Brier 0.250000 at every budget of all 9 pairs; `scripts/decode_apicheck.py` reads code 0.5
  at every budget (the only assignment that fits). So in the scoring environment a request
  built exactly like version 4's reaches the API without the SDK, the submission folder is
  writable, and later checkpoints see files written by earlier checkpoints' workers: version
  4's shared cache and call cap work across the whole run.
- The author's first run of `scripts/add_api_key.py` refused their new key: the format check
  only accepted sk-ant-apiNN- keys, but the author's keys are of the sk-ant-usr- form, which
  the API accepts (the stored key used in all live tests has it). The same check ran inside
  version 4 at start-up, so on Codabench it would have ignored the key and predicted as
  version 3. Earlier live tests had passed the key to the client directly and so skipped
  that check. Fixed: any sk-ant- key of key characters passes, admin keys and login tokens
  are still refused, the script says why a pasted value was refused (without showing it),
  and the free check against the API remains the real test. A new live test feeds the real
  key through the key-file path from memory, exactly as on the platform: ratings on, saved
  ratings reproduced. `dist/version4.zip` is now sha256 55d46699...
- Version 4 result: Codabench submission 968769 (the author's private zip of
  `dist/version4.zip` 55d46699... plus their key) finished with 0.170402 over 8 pairs; the
  per-pair ALC recomputes exactly and budget 0 is exactly 0.25 for every pair, as designed.
  No pair repeats an earlier model upload, so there is no paired comparison. As a rough
  correction for the draw, budget-0 Brier minus ECE squared gives each pair's r(1 - r), the
  score of the best constant guess for it: mean 0.140 on this draw (version 3's draw: 0.121,
  version 2's: 0.143, Phase 4's: 0.166). Score minus that floor: version 4 0.031 (SE 0.012),
  version 3 0.034 (0.008), Phase 4 0.036 (0.011), version 2 0.045 (0.011), baseline 0.049
  (0.013). Consistent with the offline gain of about 0.014 but far from proving it: one draw
  of 8 pairs cannot. Per the rule fixed before the upload, version 4 becomes the final entry
  if it ran with ratings on, which the author checks on the workspace usage page (logs are
  hidden).
- The author's workspace usage page shows $9.49 for the version 4 run (the workspace and its
  key were used for nothing else; the free key check costs nothing), close to the estimate of
  about $8 per draw with the shared cache. So the ratings were used, and by the rule fixed
  before the upload, version 4 (Codabench 968769) is the final entry.
- 2026-10-08: the author asked the organizers whether the Anthropic API may be
  used during scoring; the answer was that Anthropic and OpenAI APIs are allowed. So version 4's
  use of the Anthropic API is within the rules ("network access during scoring is limited to
  approved inference APIs").
- Checks on the final `dist/version4.zip` (sha256 55d46699..., the zip in Codabench 968769
  plus the key), run after the key-format fix: kit checker and smoke test on Windows; kit
  checker, both smoke tests and a Linux rebuild with the same hash under WSL; the offline
  harness without a key gives 0.199368 (version 3); a wrong key gives version 3's predictions;
  the live test through the key-file path reproduced the saved ratings. The probe uploaded as
  968649 is the build of commit c1ab665 (sha256 84f41a5f..., rebuilt from that commit with the
  same hash); `dist/apicheck.zip` was later rebuilt (2e449efd...) only because its copy of
  llm_rating_core.py got the new key-format check, which the probe does not use.
- Report preparation: Claude sub-agents (a workflow of four readers, an outline writer and a
  critic) built `docs/report/factsheet.json` (194 sourced facts) and
  `docs/report/outline_draft1.md`. The critic found that `tests/test_scoring.py` used one row
  of our private formative feedback as an ECE example; it was replaced by a synthetic example.
- With the author's approval, downloaded the official NeurIPS 2026 template
  (https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip,
  20,259 bytes, sha256 82473931...; linked as "Paper template" from the NeurIPS 2026 call for
  papers) and added its `neurips_2026.sty` unchanged as `docs/report/neurips_2026.sty`. The
  report uses `[sglblindworkshop, final]` (named authors); that layout needs no LaTeX packages
  beyond those already installed (the submission layout would need lineno, which is missing).
- `scripts/run_final.py` (descriptive, committed before the run): version 4 against version 3
  on the 594 pair results of run_gate.py, with partner pairs and partly missing ratings. With no
  partners and every rating it reproduces run_gate.py (-0.01437, SE 0.00106). With 1 or 2
  partner pairs the gain stays the same (-0.01443 and -0.01406). With 90% of items rated it
  is -0.01399; with 70%, -0.00971. All four benchmarks improve in every setting. Budget 0 is
  unchanged by design; the gain is about -0.012 to -0.015 at budget 1 and about -0.017 from
  budget 7 on. Run 20261009-003522_final (419 s).
- LaTeX packages: checking which packages exist with MiKTeX's kpsewhich triggered its
  on-the-fly installer (an unintended side effect of the check). geometry, units, algorithms
  and cleveref were installed that way; the author cancelled lineno and microtype. With the
  author's approval, microtype was then installed with `miktex packages install microtype`;
  lineno stays uninstalled (the report's layout does not use line numbers). Later checks look
  for the files directly, which installs nothing.
- Report deep dive (a workflow of Claude agents with web access): requirements and grading
  criteria (competition page, NeurIPS 2026 E&D reviewer guidelines, paper checklist, Main
  Track Handbook LLM policy), example competition reports, a reference list (47 entries, each
  checked against its arXiv/DOI/publisher page) and published evidence on what makes text read
  as LLM-generated. Outputs: `docs/report/writing_guide.md`, `docs/report/references.bib`,
  `docs/report/references_checked.json`, `docs/report/research_notes.json` and a style checker
  `docs/report/ai_tell_check.py` that prints counts for review (not a verdict).
- Section 2 check (a workflow of Claude agents: two fact checkers, a reviewer comparing with
  published papers, an AI-tone reviewer and a judge): every number in the draft and in Table 1
  was correct, but nine statements were wrong or misleading (for example SWE-rebench "only in
  training", the compared models being "reloaded", the gate rule "committed before the first
  run", and leaving out that version 3's scale 0 was chosen from feedback against offline
  evidence). Claude re-verified the new facts with scripts (93,764 subject-item combinations;
  222 of 271 interval MathArena items with only 0/1 scores; version 3 minus version 2 offline
  +0.0030, SE 0.0005, default and +0.0046, SE 0.0006, hidden-sized with K = 1), revised the
  draft, moved details to an appendix and corrected two facts in the fact sheet. A checker's
  test compile made MiKTeX install the cm-super fonts on the fly (58 MB); with them the PDF
  embeds only Type 1 fonts, as the template requires.
- Recovered provenance (a Claude agent searched the earlier session transcripts and scratch
  folders): the rated items were drawn with string-seeded random.Random draws from items whose
  content belongs to a single problem, one item per problem, and their Rasch difficulties come
  from a penalized joint-maximum-likelihood fit per benchmark from all subjects.
  `scripts/build_rating_samples.py` now regenerates `sample_ids.json` and
  `sample_more_ids.json` exactly (all 1,049 items in order, b bitwise equal with scipy 1.14.1,
  now pinned). The scratch files are backed up in the git-ignored private folder.
- The "+0.027" shuffled-ratings control quoted above came from an inline script whose output was
  never saved. `scripts/run_shuffle_control.py` reran it (run 20261011-011524_shuffle): shuffled
  ratings +0.02702 (SE 0.00169) against version 3, at full trust without the gate, own labels
  only, in the rated setting; real ratings -0.01878 (SE 0.00210). run_trust.py's "useless"
  scenario is the same setting with a different permutation (+0.02359).
- Corrections to earlier entries: "every MathArena problem" was rated means every problem whose
  text is its own (366 of 636; the rest share content with other problems). Over all rated
  items the Spearman correlations are 0.82, 0.28, 0.45 and 0.77; the 50-item figures for
  Multi-SWE-bench (0.50) and REAL (0.56) were optimistic. `results/llm_rating/README.md` now
  documents both rating steps.
- Section 3 check (a workflow of Claude agents: formula checker, skeptical reviewer, comparison
  with published method sections, AI-tone reviewer and a judge): every formula matched the code;
  the errors were an incomplete description of the metadata prior, the claim that each version
  "adds one change", a table caption about scores the table did not show, and REAL's
  correlation, which rounds to 0.44, not 0.45 (also corrected in `results/llm_rating/README.md`;
  the entry above that gives 0.45 is wrong). Claude revised the draft with the organizers'
  sigma(phi - z) notation, the probit approximation (largest error 0.0101 for tau = 2), a
  worked example (0.417 under V4 against 0.647 under V3) and a model-details appendix; both
  numbers are printed by `scripts/report_tables.py`.

## 2026-10-15: option C pre-check, does Claude know how strong each AI system is?

- Headroom (an inline scratch script, not committed): on the default-setting pairs a label-free
  predictor that knew each subject's Rasch ability and each item's difficulty exactly would score
  Brier 0.087 (in-sample, so optimistic), against 0.25 for a constant 0.5 and a V3 Brier ALC of
  0.219. A simulation with noisy knowledge put "model match 0.7, question match 0.6" at about
  -0.035 Brier ALC against V3 (about -0.02 against V4).
- Claude wrote `results/subject_check/check.py`, which asks Claude Opus 5.5 once per eligible
  subject (192) for the share of a benchmark's tasks that subject completes, and fixed the
  decision rule in its docstring (GO to a full option C test only if the mean within-benchmark
  Pearson correlation with the subject's Rasch ability is at least 0.70). `fit_rasch` in
  `scripts/build_rating_samples.py` can now also return the abilities (`--check` still
  reproduces the committed samples bit for bit).
- Before the run, a workflow of three Claude reviewers (statistics, faithfulness to the hosted
  inputs, API safety) checked the script with synthetic guesses and no API calls. Fixed:
  Multi-SWE-bench consists of two groups of subjects that share no items, so abilities are
  compared within groups and each group gets its own examples; the rule now says what constant
  guesses and missing guesses mean; long examples show their start and end, and image-only
  MathArena problems are not used as examples; cost is priced by the model that answered.
- With the author's approval, the check ran with the author's API key (read from the Windows
  environment, never printed or stored): 192 requests, no invalid replies, no fallback answers,
  about $1.37. Model match 0.818 (0.957, 0.767, 0.661, 0.886), so GO under the rule. After the
  run Claude wrote `results/subject_check/compare.py`: the version 1 metadata starting guess
  scores 0.611 on the same measure, and other Rasch priors change the match by at most 0.002.
  Caveat: public benchmarks, so Claude may remember published results.

## 2026-10-17: option C full test, version 4 with Claude's guess as each subject's prior

- The author approved building and testing option C offline. Claude designed it as version 4
  plus one guess per subject and benchmark (the subject's prior), not one request per subject
  and item: version 4 already asks about each item, and one request per subject is about 100
  times cheaper and fast enough for budget 0.
- Claude wrote `paiec/guess_core.py` (the guess request, standard library only),
  `results/subject_prior/ask.py` (the guesses for the test), `scripts/run_subject_prior.py` (the
  harness run, decision rule in its docstring), an optional `guess` argument for
  `paiec/item_pooling_core.py` (without it predictions are unchanged; tests compare against the
  frozen version 4 copy) and `tests/test_option_c.py`.
- Before the run, a workflow of three Claude reviewers (harness and statistics, model code,
  faithfulness to hosted runs) found that 28 of the 70 Multi-SWE-bench test pairs would have had
  no guess (they have more than 1,000 items before the cut to rated items), that a malformed
  guess value could crash a prediction, and that the rule ignored the risk that guesses do not
  transfer to a new benchmark. Fixed before committing: requests for every scored subject (232)
  and a coverage check that stops the run if any is missing; malformed guesses count as no
  guess; and a trust level for the guesses chosen by minimax regret over real, shuffled and
  level-shifted (+1 and -1 logit) guesses, as for the ratings.
- The rule was committed (45eb1b3), then Claude Opus 5.5 answered all 232 guess requests
  ($0.80, no invalid replies, no fallback answers).
- The first run (results/runs/20261017-015838_subject_prior, kept locally, not committed) is
  invalid. Claude noticed that guess trust 0.25 scored worse than version 4 at budget 31, traced
  one pair, and found a bug: the script stored the guess trust in the attribute that holds
  version 4's rating trust, so every option C model also had its rating trust changed (to 0.25,
  0.75 or 1 instead of 0.5). That run said GO at trust 0.75 (-0.0163 against version 4), but it
  did not test the pre-registered model. Fixed by renaming the attribute; the script now stops
  if option C's version 4 parameters differ from version 4's. The rule is unchanged, and the
  test was rerun.
- Valid run (results/runs/20261017-021515_subject_prior): chosen guess trust 0.75; option C minus
  version 4 -0.0135 (SE 0.0013) with K = 0, -0.0132 with K = 1 and K = 2, better on all four
  benchmarks: GO to building a hosted version. A workflow of three Claude agents verified it: an
  independent recomputation matched every number, the version 4 rows are identical to the
  committed final run, and the trust-0.5 rows are identical to the invalid run's (the bug did
  not touch them). A skeptical review found that 74% of the gain comes at budgets 0 and 1, where
  version 1's metadata starting guess failed in hosted runs. It also found that the stress
  scenarios are milder than that failure, that Multi-SWE-bench gives 57% of the gain, and that
  these guesses match subject ability at 0.771, not the pre-check's 0.818. These caveats are in
  `results/subject_prior/README.md`.

## 2026-10-20: version 5, option C as a submission

- The author approved building version 5 (version 4 plus option C), with a hard stop: if it is
  not uploaded by the end of 2026-10-22, version 4 stays final. Claude generalized
  `paiec/llm_rating_core.py` so the same cache, locks, call cap and failure handling also fetch
  one guess per subject and benchmark (sent before ratings). Claude added the version 5 model
  template to `paiec/export.py` and wrote `scripts/export_version5.py`, which writes
  `submissions/version5`: version 4's parameters plus the guess calibration at trust 0.75,
  refitted on all four rated benchmarks (a +0.040, c 0.851, s 0.811 on 204 pairs). Tests are in
  `tests/test_version5.py`. The uploaded version 4 files stay frozen; tests check them by SHA-256.
- Selection rule (`scripts/check_version5.py`, committed before the upload): version 4 scores
  exactly 0.25 at budget 0 on every pair (checked on its own and version 3's feedback), so
  version 5 becomes final only if its feedback shows a guess on at least half of the pairs and a
  mean budget-0 Brier below 0.25. Otherwise version 4 stays. This choice uses formative feedback
  and must be disclosed in the report.
- A workflow of three Claude reviewers checked version 5 before the zip was built. Hosted
  robustness: no problem found. With a hanging API every prediction took at most about 4.1 s, and
  a WSL simulation that kills 4 workers at every checkpoint gave valid predictions and one guess
  per key. Equivalence: all 232 offline guess requests rebuild identically, and rating behaviour
  is unchanged. On the platform, though, the example task is whichever item comes first, which
  can be uninformative (for example, a problem given only as an image). The offline guesses never
  saw such examples. The selection-rule reviewer asked for the commit before the upload and the
  minimum share of guessed pairs (both done). It also listed what the report must disclose: the
  second choice made from feedback; that subject fields are sent to the API during scoring; that
  Claude may remember public results; the history of metadata priors; and that guesses are
  fetched live.
- Checks on `dist/version5.zip` (sha256 b7357ae7..., built from commit 5ea16cc):
  - 156 tests pass.
  - The kit checker and smoke test pass on Windows; in WSL, the kit checker, both smoke tests
    (sequential and 4 isolated workers) pass, and a Linux rebuild gives the same hash.
  - Without a key, the offline harness matches version 3 exactly on all 576 pair results of the
    default setting (largest difference 0.0).
  - A live test with the author's key, read in memory and never written, used the submission's
    own code on 6 public items: valid guesses (0.45 and 0.70), predictions equal to the core
    model with those guesses and ratings, and the key in no file. 8 billed attempts, a few cents.
  - In that test, the first budget-0 predictions of each pair came before the guess arrived, so
    they fell back to 0.5. Timed directly, guesses take 1.4-3.1 s and ratings 1.3-1.7 s.
  - The author asked whether the packaged zip had been scored locally. It had not, with guesses
    on. Claude wrote `scripts/run_zip_check.py`, which runs the version 4 and version 5 folders
    as submissions through the harness, with their caches pre-filled with the saved replies, so
    no API is called (none was). On the pairs of `scripts/run_final.py` with K = 0, the version 4
    zip scored 0.1901 and the version 5 zip 0.1752: -0.0150 (SE 0.0012), better on all four
    benchmarks. Budget-0 Brier was 0.1996 against 0.2500. These are in-sample, since the
    parameters were fitted on all public benchmarks. They are consistent with the tested -0.0135.
- The author uploaded `version5_private.zip` (Codabench 1005477). The feedback is saved in
  `private/codabench/` and is not committed. Formative score 0.140778 on 8 pairs, version 4's
  0.170402 on a different draw. Our ALC formula reproduces every pair. The pre-registered rule
  (`scripts/check_version5.py`) gives: guesses on 8 of 8 pairs, mean budget-0 Brier 0.180
  (version 4: exactly 0.25), below 0.25 on 6 of 8 pairs, so **version 5 (1005477) is the final
  entry**.
- Draw-adjusted, the score minus each pair's budget-0 floor r(1 - r) (approximate for version 5,
  whose first budget-0 predictions may have been 0.5 before the guess arrived) is 0.009 (SE
  0.007). It is 0.031 for version 4, 0.034 for version 3 and 0.045-0.049 for version 2 and the
  baseline. On the public leaderboard (55 entries; our own entries are not listed), 0.1408 would
  rank 7th and 0.1704 27th. Formative draws differ from upload to upload, so these ranks are only
  indicative.

## 2026-10-21: technical report, full draft by Claude

- Project work stopped at the planned freeze (2026-10-21), with version 5 final. The author then
  changed the report process. Under the earlier plan (option B), Claude drafted each section and
  the author was to rewrite it, but no author rewrite was made: `docs/report/rewrite` holds only
  Claude's drafts. Now Claude writes the whole report and the author reads and edits it at the
  end. Claude told the author that the report's AI-use statement must then say that Claude
  drafted the text, and the draft's disclosure paragraph (Section 8) says so.
- Claude wrote `scripts/report_results.py`, which builds the results tables and the figure from
  committed runs, and `results/formative_summary.json`, which holds only per-upload aggregates. It
  drafted sections 1 and 4-9, the abstract, the appendices (decision rules, prompts, engineering,
  data card and licenses) and the NeurIPS checklist (questions taken from neurips.cc), and updated
  sections 2 and 3 and the model appendix for version 5 (commit c3d42a4). Items only the author
  can supply are marked [TO CONFIRM]: repository URL and license, the V5 run's cost, funding,
  and the license of the released outputs.
- Review round 1: four Claude agents (two fact checkers, a style reviewer and a referee) found
  115 issues: 14 wrong statements, 28 misleading ones, 13 omissions and style points. Examples:
  the ALC weights were misdescribed; two citations were described more broadly than the papers
  support; the disclosure claimed the author had already read the report; a missing rating and a
  missing guess had their fallbacks swapped. Claude revised the draft, added the rating-trust
  regrets, coverage and partner results and the pooling split assumption, renamed symbols that
  had two meanings, moved the broader-impact paragraph and the commands to new appendices, read
  Anthropic's Commercial Terms of Service (21 October 2026) for the data card, and trimmed the main
  text to about 8.5 pages (commit 84e7614).
- Review round 2: two Claude agents rechecked the revision. They found that it had fixed the
  first-round problems but introduced two errors of its own ("232 subjects" for 232 guesses; the
  rules name no API providers), and that some passages were repeated or out of order. Claude fixed
  these in the next commit.

## 2026-10-22: public code release prepared

- The rules (re-read 2026-10-22) require the codebase and the report for final evaluation, with
  the selected Codabench ID entered in the OpenReview form and the report linking the released
  code; no email to the organizers is required (an earlier note said so). With the author's
  go-ahead and Claude's defaults (code under MIT; the unlicensed starting kit left out), Claude
  wrote README.md, LICENSE (MIT, with third-party exceptions; derived data under CC BY-SA 4.0),
  configs/hidden_partners1.json and scripts/export_public.py. That script writes the committed
  files, minus the starting kit and the internal report-writing notes, to dist/public, which has
  its own fresh history (author "Stanley Mei <smei@stanford.edu>"), because the development
  history contains private Codabench feedback.
- Three Claude agents checked the public copy. From a clean clone, with the kit copied in and the
  data linked, the tests passed and the tables were rebuilt byte for byte. The rebuilt final zip
  had the README's SHA-256, and the subject check gave 0.818 again. No real key, private feedback
  ID or personal data was found. Their documentation fixes were applied (commit de537af) and the
  copy was re-exported. Nothing has been pushed; the author pushes.
