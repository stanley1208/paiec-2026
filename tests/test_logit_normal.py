import importlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from synthetic import make_dataset, with_metadata

from paiec import baseline_core, logit_normal_core, protocol, stress
from paiec.export import write_submission
from paiec.logit_normal import LogitNormal, fit_params
from paiec.predictors import PREDICTORS, SubmissionPredictor

ROOT = Path(__file__).resolve().parents[1]


def reference_posterior_mean(m, sd, k, n):
    """Independent check: a much finer and wider grid centred on the prior."""
    z = np.linspace(m - 12 * sd, m + 12 * sd, 400001)
    p = 1 / (1 + np.exp(-z))
    log_w = -0.5 * ((z - m) / sd) ** 2 + k * np.log(p) + (n - k) * np.log1p(-p)
    w = np.exp(log_w - log_w.max())
    return float((w * p).sum() / w.sum())


@pytest.mark.parametrize("m, sd, k, n", [(0.0, 1.5, 0, 0), (1.2, 1.5, 0, 7), (-2.0, 1.0, 31, 31),
                                         (0.4, 2.0, 9, 15), (5.9, 1.5, 0, 31), (6.0, 2.0, 0, 0),
                                         (-6.0, 2.0, 0, 31), (6.0, 2.0, 31, 31)])
def test_posterior_mean_matches_a_finer_integration(m, sd, k, n):
    assert logit_normal_core.posterior_mean(m, sd, k, n) == pytest.approx(
        reference_posterior_mean(m, sd, k, n), abs=1e-6)


def test_posterior_mean_is_centred_symmetric_and_moves_with_labels():
    pm = logit_normal_core.posterior_mean
    assert pm(0.0, 1.5, 0, 0) == pytest.approx(0.5, abs=1e-12)
    assert pm(0.7, 1.5, 3, 7) == pytest.approx(1 - pm(-0.7, 1.5, 4, 7), abs=1e-12)
    assert [pm(0.0, 1.5, k, 7) for k in range(8)] == sorted(pm(0.0, 1.5, k, 7) for k in range(8))
    assert pm(0.0, 1.5, 30, 31) > 0.85


def test_starting_guess_keeps_the_metadata_score_but_not_the_training_average():
    dataset = with_metadata(make_dataset())
    train = dataset.subset(["train", "held"])  # s1 is in both, so pairs != subjects
    params, half = fit_params(train), fit_params(train, sd=1.75, scale=0.5)
    assert (half["sd"], half["scale"]) == (1.75, 0.5)
    assert any(abs(v) > 1e-3 for v in params["known_subjects"].values())
    means = []
    for subject in train.subjects.values():
        score = logit_normal_core.metadata_score(subject, params)
        phase3 = baseline_core.subject_prior(subject, dict(params, intercept=0.0,
                                                           benchmark_offsets=[0.0]))
        assert 1 / (1 + math.exp(-score)) == pytest.approx(phase3, abs=1e-12)
        means.append(logit_normal_core.prior_mean(subject, params))
        assert means[-1] == pytest.approx(score - params["center"], abs=1e-12)
        assert logit_normal_core.prior_mean(subject, half) == pytest.approx(0.5 * means[-1])
    assert math.fsum(means) == pytest.approx(0.0, abs=1e-12)
    flat = fit_params(train, scale=0.0)
    for subject_id in ("s1", "s2", "s3"):
        x = dataset.make_input(subject_id, "held-0")
        assert logit_normal_core.predict(x, [], flat) == pytest.approx(0.5, abs=1e-12)


def test_uses_only_this_pairs_labels_and_ignores_malformed_ones():
    dataset = with_metadata(make_dataset())
    model = LogitNormal()
    model.fit(dataset.subset(["train"]))
    own = dataset.make_input("s2", "held-0")
    other_subject = dataset.make_input("s1", "held-1")
    other_benchmark = dataset.make_input("s2", "train-1")
    junk = [[own[0], None], [own, "1"], [own, 2], "junk", [[own[0], None], 1]]
    labeled = [[own, 1], [own, 1], [own, 0], [other_subject, 1], [other_benchmark, 0]] + junk
    m = logit_normal_core.prior_mean(own[0], model.params)
    assert model.predict(own, labeled) == logit_normal_core.posterior_mean(m, 1.5, 2, 3)
    assert model.predict(own, None) == logit_normal_core.posterior_mean(m, 1.5, 0, 0)
    odd_subject = [dict(own[0], release_date=None, provider=["list"]), own[1]]
    assert 0 < model.predict(odd_subject, []) < 1


def test_items_without_benchmark_id_use_the_subjects_labels_without_one():
    dataset = with_metadata(make_dataset())
    model = LogitNormal()
    model.fit(dataset.subset(["train"]))
    x, y = dataset.make_input("s2", "held-0"), dataset.make_input("s2", "held-1")
    del x[1]["benchmark_id"], y[1]["benchmark_id"]
    m = logit_normal_core.prior_mean(x[0], model.params)
    labeled = [[y, 1], [dataset.make_input("s2", "held-2"), 0]]
    assert model.predict(x, labeled) == logit_normal_core.posterior_mean(m, 1.5, 1, 1)


def test_scores_identically_with_or_without_other_pairs_evidence():
    dataset = with_metadata(make_dataset())
    runs = [protocol.run_fold(dataset, "held", PREDICTORS["logit_normal"],
                              protocol.Settings(evidence=evidence)) for evidence in ("pair", "benchmark")]
    assert runs[0] == runs[1]


def test_core_also_works_from_a_flat_submission_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + sys.path)
    monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items()
                                         if k not in ("baseline_core", "logit_normal_core")})
    for module in (baseline_core, logit_normal_core):
        shutil.copyfile(module.__file__, tmp_path / f"{module.__name__.split('.')[-1]}.py")
    flat = importlib.import_module("logit_normal_core")
    assert flat.__file__.startswith(str(tmp_path))
    assert flat.baseline_core.__file__.startswith(str(tmp_path))
    dataset = with_metadata(make_dataset())
    params = json.loads(json.dumps(fit_params(dataset.subset(["train"]))))
    x = dataset.make_input("s1", "held-0")
    labeled = [[dataset.make_input("s1", "held-3"), 1], [dataset.make_input("s1", "held-4"), 0]]
    assert flat.predict(x, labeled, params) == logit_normal_core.predict(x, labeled, params)


def test_shift_benchmark_moves_only_the_held_out_rates_one_way():
    dataset = make_dataset()
    before = dataset.responses.copy()
    assert stress.shift_benchmark(dataset, "held", 0.0).responses.equals(before)
    for delta in (2.0, -2.0):
        shifted = stress.shift_benchmark(dataset, "held", delta, seed=1)
        assert dataset.responses.equals(before)
        held = before["benchmark"] == "held"
        assert shifted.responses[~held].equals(before[~held])
        old, new = before.loc[held, "label"], shifted.responses.loc[held, "label"]
        assert ((new >= old) if delta > 0 else (new <= old)).all()
        y = old.groupby(before.loc[held, "subject_id"]).mean()
        moved = new.groupby(before.loc[held, "subject_id"]).mean()
        target = 1 / (1 + np.exp(-(np.log(y / (1 - y)) + delta)))
        assert (moved - target).abs().max() < 0.15


def test_shuffled_metadata_permutes_the_starting_guesses_and_uses_own_labels():
    dataset = with_metadata(make_dataset())
    subjects = [dataset.make_input(s, "held-0")[0] for s in ("s1", "s2", "s3")]
    shuffled, plain = stress.ShuffledMetadata(subjects, seed=3), LogitNormal()
    for model in (shuffled, plain):
        model.fit(dataset.subset(["train"]))
    inputs = [dataset.make_input(s, "held-0") for s in ("s1", "s2", "s3")]
    before, after = [plain.predict(x) for x in inputs], [shuffled.predict(x) for x in inputs]
    assert sorted(after) == pytest.approx(sorted(before), abs=1e-15)
    assert after != pytest.approx(before, abs=1e-6)
    x = inputs[0]
    labeled = [[dataset.make_input("s1", "held-1"), 1], [dataset.make_input("s1", "held-2"), 1],
               [dataset.make_input("s2", "held-1"), 0]]
    m = shuffled.means[baseline_core.subject_key(x[0])]
    assert shuffled.predict(x, labeled) == logit_normal_core.posterior_mean(m, 1.5, 2, 2)


def test_exported_submission_predicts_exactly_like_the_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    dataset = with_metadata(make_dataset())
    model = LogitNormal(scale=0.5)
    model.fit(dataset.subset(["train"]))
    write_submission(model.params, tmp_path, "test export")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "baseline_core.py", "logit_normal_core.py", "model.py", "params.json"]
    submission = SubmissionPredictor(tmp_path)
    labeled = [[dataset.make_input("s1", "held-3"), 1], [dataset.make_input("s1", "held-4"), 0]]
    no_benchmark = dataset.make_input("s2", "held-0")
    del no_benchmark[1]["benchmark_id"]
    for x in [dataset.make_input(s, "held-0") for s in ("s1", "s2", "s3")] + [no_benchmark]:
        assert submission.predict(x, labeled) == model.predict(x, labeled)
        assert submission.predict_acquisition(x, labeled) == model.predict(x, labeled)
    for broken in ("not an input", None, [None, None]):
        assert submission.predict(broken, None) == 0.5


def test_committed_phase4_submission_is_current():
    folder = ROOT / "submissions" / "phase4"
    if not folder.exists():
        pytest.skip("submissions/phase4 not exported yet")
    for module in (baseline_core, logit_normal_core):
        name = module.__name__.split(".")[-1] + ".py"
        assert (folder / name).read_bytes() == Path(module.__file__).read_bytes()
    params = json.loads((folder / "params.json").read_text(encoding="utf-8"))
    assert (params["model"], params["sd"], params["scale"]) == ("logit_normal", 1.5, 0.5)
