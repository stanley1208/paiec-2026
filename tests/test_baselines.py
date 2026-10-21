import json
import sys
from pathlib import Path

import pytest
from synthetic import make_dataset, with_metadata

from paiec import baseline_core, feedback, protocol
from paiec.baselines import Baseline, fit_params
from paiec.export import write_submission
from paiec.predictors import PREDICTORS, SubmissionPredictor

ROOT = Path(__file__).resolve().parents[1]


def test_global_mean_averages_pair_rates_per_benchmark():
    train = make_dataset().subset(["train"])
    rates = train.responses.groupby("subject_id")["label"].mean()
    assert fit_params(train, prior="global")["global_mean"] == pytest.approx(rates.mean())


def test_params_are_json_and_priors_are_valid_probabilities():
    dataset = with_metadata(make_dataset())
    params = fit_params(dataset.subset(["train", "held"]), prior="subject", strength=4.0)
    assert json.loads(json.dumps(params)) == params
    for subject in dataset.subjects.values():
        assert 0 < baseline_core.subject_prior(subject, params) < 1
    assert baseline_core.subject_key(dataset.subjects["s1"]) in params["known_subjects"]


def test_beta_binomial_uses_only_this_pairs_labels():
    dataset = with_metadata(make_dataset())
    params = fit_params(dataset.subset(["train"]), prior="global", strength=4.0)
    own = dataset.make_input("s2", "held-0")
    other_subject = dataset.make_input("s1", "held-1")
    other_benchmark = dataset.make_input("s2", "train-1")
    labeled = [[own, 1], [own, 1], [own, 0], [other_subject, 1], [other_benchmark, 0]]
    expected = (4 * params["global_mean"] + 2) / (4 + 3)
    assert baseline_core.predict(own, labeled, params) == pytest.approx(expected)
    assert baseline_core.predict(own, [], params) == params["global_mean"]


def test_malformed_fields_and_evidence_degrade_gracefully():
    dataset = with_metadata(make_dataset())
    params = fit_params(dataset.subset(["train", "held"]), prior="subject", strength=4.0)
    x = dataset.make_input("s1", "held-0")
    good = [[x, 1], [x, 0]]
    junk = [[x[0], None], [[x[0], None], 1], [x, "1"], [x, 2], "junk"]
    assert baseline_core.predict(x, good + junk, params) == baseline_core.predict(x, good, params)
    odd_subject = [dict(x[0], release_date=None, provider=["list"]), x[1]]
    assert 0 < baseline_core.predict(odd_subject, [], params) < 1
    assert baseline_core._year("2025/06/01") == baseline_core._year(" 2025-06-01") == 2025 + 5 / 12


@pytest.mark.parametrize("name", ["beta_binomial", "beta_binomial_subject"])
def test_baselines_score_identically_with_or_without_other_pairs_evidence(name):
    dataset = with_metadata(make_dataset())
    runs = [protocol.run_fold(dataset, "held", PREDICTORS[name],
                              protocol.Settings(evidence=evidence)) for evidence in ("pair", "benchmark")]
    assert runs[0] == runs[1]


def test_exported_submission_predicts_exactly_like_the_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    dataset = with_metadata(make_dataset())
    baseline = Baseline(prior="subject", strength=4.0)
    baseline.fit(dataset.subset(["train"]))
    write_submission(baseline.params, tmp_path, "test export")
    submission = SubmissionPredictor(tmp_path)
    labeled = [[dataset.make_input("s1", "held-3"), 1], [dataset.make_input("s1", "held-4"), 0]]
    for subject_id in ("s1", "s2", "s3"):
        x = dataset.make_input(subject_id, "held-0")
        assert submission.predict(x, labeled) == baseline.predict(x, labeled)
        assert submission.predict_acquisition(x, labeled) == baseline.predict(x, labeled)
    # Malformed input falls back to the global mean instead of raising.
    assert submission.predict("not an input", None) == baseline.params["global_mean"]


def test_committed_submission_copy_of_baseline_core_is_current():
    copy = ROOT / "submissions" / "baseline" / "baseline_core.py"
    if not copy.exists():
        pytest.skip("submissions/baseline not exported yet")
    assert copy.read_bytes() == Path(baseline_core.__file__).read_bytes()


def test_feedback_check_recomputes_alc_from_budget_tables():
    lines = ["ALC summary", "subject_000001  benchmark_000001  50  0.140000  0.100000"]
    for budget, brier in zip(protocol.scoring.BUDGETS, (0.22, 0.18, 0.145, 0.12, 0.1, 0.09)):
        lines += [f"Label budget: {budget}", f"subject_000001  benchmark_000001  50  {brier:.6f}  0.1"]
    report, mean = feedback.check("\n".join(lines))
    assert report[0]["ok"] and report[0]["recomputed"] == pytest.approx(0.14)
    bad = "\n".join(lines).replace("0.140000", "0.150000")
    assert not feedback.check(bad)[0][0]["ok"]
