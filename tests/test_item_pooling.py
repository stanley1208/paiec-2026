import copy
import math
import random

import pytest
from synthetic import make_dataset, with_metadata

from paiec import item_pooling_core, logit_normal_core, protocol
from paiec.export import write_submission
from paiec.item_pooling import ItemPooling, fit_params
from paiec.logit_normal import LogitNormal
from paiec.predictors import PREDICTORS, SubmissionPredictor


@pytest.fixture
def fitted():
    dataset = with_metadata(make_dataset())
    model = ItemPooling()
    model.fit(dataset.subset(["train"]))
    return dataset, model


def test_without_other_subjects_it_is_exactly_phase4(fitted):
    dataset, model = fitted
    phase4 = LogitNormal(scale=0.5)
    phase4.fit(dataset.subset(["train"]))
    x = dataset.make_input("s1", "held-0")
    other_benchmark = dataset.make_input("s2", "train-1")
    for labels in ([], [1], [1, 0, 0], [1] * 7):
        own = [[dataset.make_input("s1", f"held-{10 + j}"), y] for j, y in enumerate(labels)]
        assert model.predict(x, own + [[other_benchmark, 1]]) == phase4.predict(x, own)


def test_other_subjects_labels_on_the_item_move_the_prediction(fitted):
    dataset, model = fitted
    x = dataset.make_input("s1", "held-0")
    base = model.predict(x, [])
    strong = [[dataset.make_input("s2", f"held-{20 + j}"), 1] for j in range(6)]
    weak = [[dataset.make_input("s2", f"held-{20 + j}"), 0] for j in range(6)]
    # A strong subject failing the item makes it look hard; a weak one solving it, easy.
    assert model.predict(x, strong + [[dataset.make_input("s2", "held-0"), 0]]) < base
    assert model.predict(x, weak + [[dataset.make_input("s2", "held-0"), 1]]) > base
    # Other subjects' labels on unrelated items do not change a fresh subject's prediction.
    assert model.predict(x, strong) == pytest.approx(base, abs=1e-15)


def test_known_difficulty_of_own_items_updates_ability(fitted):
    dataset, model = fitted
    x = dataset.make_input("s1", "held-0")
    own = [[dataset.make_input("s1", "held-5"), 1]]
    plain = model.predict(x, own)
    # s3 (weak: fails everything else) solved held-5, so held-5 looks easy and s1 solving it
    # says less about s1's ability.
    s3 = [[dataset.make_input("s3", f"held-{30 + j}"), 0] for j in range(6)]
    easy = s3 + [[dataset.make_input("s3", "held-5"), 1]]
    assert model.predict(x, own + easy) < plain


def test_labels_are_matched_by_content_and_order_does_not_matter(fitted):
    dataset, model = fitted
    x = dataset.make_input("s1", "held-0")
    partner = [[dataset.make_input("s2", f"held-{20 + j}"), j % 2] for j in range(8)]
    partner.append([dataset.make_input("s2", "held-0", interactors="other"), 0])
    own = [[dataset.make_input("s1", f"held-{40 + j}"), 1] for j in range(3)]
    labeled = own + partner + [[x[0], None], "junk", [[x[0], x[1]], 2]]
    value = model.predict(x, labeled)
    assert value != pytest.approx(model.predict(x, own), abs=1e-6)
    shuffled = labeled[:]
    random.Random(1).shuffle(shuffled)
    assert model.predict(x, shuffled) == value
    assert model.predict(copy.deepcopy(x), copy.deepcopy(labeled)) == value
    changed = copy.deepcopy(x)
    changed[1]["item_content"] += " (edited)"
    assert model.predict(changed, labeled) == model.predict(changed, own)


def test_partners_evidence_passes_other_pairs_labels():
    dataset = with_metadata(make_dataset())
    seen = []

    class Recorder:
        def fit(self, train):
            pass

        def predict(self, input, labeled=None):
            seen.append({protocol.json.dumps(s, sort_keys=True) for (s, _), _ in labeled})
            return 0.5

    protocol.run_fold(dataset, "held", Recorder, protocol.Settings(evidence="partners", partners=1))
    assert max(len(s) for s in seen) == 2  # own subject plus exactly one partner
    with pytest.raises(ValueError):
        protocol.Settings(partners=1)


def test_logit_normal_ignores_partners_and_item_pooling_matches_it_without_them():
    dataset = with_metadata(make_dataset())
    phase4 = lambda: LogitNormal(scale=0.5)  # noqa: E731
    pair = protocol.run_fold(dataset, "held", phase4, protocol.Settings())
    partners = protocol.run_fold(dataset, "held", phase4, protocol.Settings(evidence="partners", partners=2))
    pooled = protocol.run_fold(dataset, "held", PREDICTORS["item_pooling"], protocol.Settings())
    assert pair == partners == pooled


def test_item_pool_keeps_only_the_chosen_items_of_the_held_out_benchmark():
    dataset = make_dataset()
    settings = protocol.Settings(pool_items=30)
    small = protocol._item_pool(dataset, "held", settings)
    held = small.responses[small.responses["benchmark"] == "held"]
    assert held["item_id"].nunique() == 30
    train = dataset.responses[dataset.responses["benchmark"] == "train"]
    assert small.responses[small.responses["benchmark"] == "train"].reset_index(drop=True).equals(
        train.reset_index(drop=True))
    assert set(protocol._item_pool(dataset, "held", settings).responses["item_id"]) == set(
        small.responses["item_id"])


def test_core_imports_from_a_flat_folder(tmp_path, monkeypatch):
    import importlib
    import shutil
    import sys
    from paiec import baseline_core
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + sys.path)
    monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items() if k not in (
        "baseline_core", "logit_normal_core", "item_pooling_core")})
    for module in (baseline_core, logit_normal_core, item_pooling_core):
        shutil.copyfile(module.__file__, tmp_path / f"{module.__name__.split('.')[-1]}.py")
    flat = importlib.import_module("item_pooling_core")
    assert flat.logit_normal_core.__file__.startswith(str(tmp_path))
    assert flat.baseline_core.__file__.startswith(str(tmp_path))


def test_a_partner_label_is_left_out_of_that_partners_own_ability(fitted):
    dataset, model = fitted
    x = dataset.make_input("s1", "held-0")
    partner = dataset.make_input("s2", "held-0")
    params, sd, item_sd = model.params, model.params["sd"], model.params["item_sd"]
    kappa0 = math.sqrt(1 + math.pi * item_sd ** 2 / 8)
    mean, var = item_pooling_core._theta_moments(logit_normal_core.prior_mean(partner[0], params), sd, 0, 0)
    b, var_b = item_pooling_core._item_posterior(
        ((kappa0 * mean, math.sqrt(1 + math.pi * kappa0 ** 2 * var / 8), 0),), item_sd)
    expected = item_pooling_core._predictive(logit_normal_core.prior_mean(x[0], params), sd, 0, 0, (),
                                             kappa0, b, math.sqrt(1 + math.pi * var_b / 8))
    assert model.predict(x, [[partner, 0]]) == expected


def test_fit_params_defaults_are_the_submitted_choice():
    params = fit_params(with_metadata(make_dataset()).subset(["train"]))
    assert (params["model"], params["scale"], params["sd"], params["item_sd"]) == (
        "item_pooling", 0.5, 1.5, 2.0)


def test_exported_version2_predicts_exactly_like_the_harness(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    dataset = with_metadata(make_dataset())
    model = ItemPooling()
    model.fit(dataset.subset(["train"]))
    write_submission(model.params, tmp_path, "test export")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "baseline_core.py", "item_pooling_core.py", "logit_normal_core.py", "model.py", "params.json"]
    submission = SubmissionPredictor(tmp_path)
    labeled = [[dataset.make_input("s1", f"held-{40 + j}"), j % 2] for j in range(4)]
    labeled += [[dataset.make_input(s, f"held-{j}"), (j + len(s)) % 2] for s in ("s2", "s3") for j in range(12)]
    no_benchmark = dataset.make_input("s1", "held-1")
    del no_benchmark[1]["benchmark_id"]
    pooled_at_least_once = False
    for x in [dataset.make_input("s1", f"held-{j}") for j in range(6)] + [no_benchmark]:
        assert submission.predict(x, labeled) == model.predict(x, labeled)
        pooled_at_least_once |= model.predict(x, labeled) != model.predict(x, labeled[:4])
    assert pooled_at_least_once
    for broken in ("not an input", None, [None, None]):
        assert submission.predict(broken, None) == 0.5


def test_export_refuses_a_model_it_has_no_template_for(tmp_path):
    with pytest.raises(ValueError):
        write_submission({"model": "item_pool", "sd": 1.5}, tmp_path, "typo")
    with pytest.raises(ValueError):
        write_submission({"global_mean": 0.4}, tmp_path, "no model and no prior")
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name, scale", [("version2", 0.5), ("version3", 0.0)])
def test_committed_submissions_still_predict_like_the_code(name, scale, tmp_path, monkeypatch):
    """Uploaded submission folders are frozen; with the later options off, the current core
    must predict exactly like each folder's copy of item_pooling_core.py."""
    import importlib
    import json
    import shutil
    import sys
    from pathlib import Path
    from paiec import baseline_core
    folder = Path(__file__).resolve().parents[1] / "submissions" / name
    if not folder.exists():
        pytest.skip(f"submissions/{name} not exported yet")
    for module in (baseline_core, logit_normal_core):
        module_name = module.__name__.split(".")[-1] + ".py"
        assert (folder / module_name).read_bytes() == Path(module.__file__).read_bytes()
    params = json.loads((folder / "params.json").read_text(encoding="utf-8"))
    assert (params["model"], params["sd"], params["scale"], params["item_sd"]) == (
        "item_pooling", 1.5, scale, 2.0)
    assert not {"agent_scale", "group_sd", "rating_slope"} & set(params)
    for file_name in ("baseline_core.py", "logit_normal_core.py", "item_pooling_core.py"):
        shutil.copyfile(folder / file_name, tmp_path / file_name)
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + sys.path)
    monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items() if k not in (
        "baseline_core", "logit_normal_core", "item_pooling_core")})
    frozen = importlib.import_module("item_pooling_core")
    assert frozen.__file__.startswith(str(tmp_path))
    dataset = with_metadata(make_dataset())
    for item in dataset.items.values():
        item["item_features"] = "group=" + item["item_content"][-1]
    rng = random.Random(0)
    for _ in range(40):
        labeled = [[dataset.make_input(rng.choice(["s1", "s2", "s3"]), f"held-{rng.randrange(60)}"),
                    rng.randint(0, 1)] for _ in range(rng.randrange(25))]
        x = dataset.make_input(rng.choice(["s1", "s2", "s3"]), f"held-{rng.randrange(60)}")
        assert frozen.predict(x, labeled, params) == item_pooling_core.predict(x, labeled, params)

def test_agent_scale_changes_only_subjects_with_a_harness():
    dataset = with_metadata(make_dataset())
    dataset.subjects["s2"]["harness"] = "SWE-agent"
    plain, agent_rule = ItemPooling(), ItemPooling(agent_scale=0.0)
    for model in (plain, agent_rule):
        model.fit(dataset.subset(["train"]))
    agent, other = dataset.make_input("s2", "held-0"), dataset.make_input("s1", "held-0")
    assert agent_rule.predict(agent, []) == pytest.approx(0.5, abs=1e-12)
    assert plain.predict(agent, []) != pytest.approx(0.5, abs=1e-6)
    assert agent_rule.predict(other, []) == plain.predict(other, [])


def test_group_effects_reduce_to_phase4_without_grouped_labels_and_learn_within_groups():
    dataset = with_metadata(make_dataset())
    for item_id, item in dataset.items.items():
        item["item_features"] = f"topic={int(item_id.split('-')[1]) % 3};idx={item_id}"
    plain, grouped = ItemPooling(), ItemPooling(group_sd=1.0)
    for model in (plain, grouped):
        model.fit(dataset.subset(["train"]))
    x = dataset.make_input("s1", "held-0")  # topic=0
    assert grouped.predict(x, []) == plain.predict(x, [])
    same = [[dataset.make_input("s1", f"held-{j}"), 1] for j in (3, 6, 9, 12)]  # topic=0
    other = [[dataset.make_input("s1", f"held-{j}"), 1] for j in (1, 4, 7, 10)]  # topic=1
    # Successes in the item's own group count more than successes elsewhere,
    assert grouped.predict(x, same) > grouped.predict(x, other)
    # and successes elsewhere count less than in the plain model, which ignores groups.
    assert plain.predict(x, []) < grouped.predict(x, other) < plain.predict(x, other)


def test_group_likelihood_matches_a_dense_integral():
    import numpy as np
    sd = 1.0
    kappa = math.sqrt(1 + math.pi * sd ** 2 / 8)
    u = np.linspace(-14, 14, 280001)
    density = np.exp(-0.5 * (u / sd) ** 2)
    density /= density.sum()
    for k, n in ((0, 1), (3, 5), (7, 7), (31, 31)):
        table = item_pooling_core._log_group_likelihood(k, n, sd)
        for t in (800, 1000, 1400, 1800, 2000):  # |theta| <= 6, where the posterior can be
            z = kappa * item_pooling_core._THETA[t] + u
            ref = np.log(np.sum(density * np.exp(-k * np.logaddexp(0, -z) - (n - k) * np.logaddexp(0, z))))
            assert table[t] == pytest.approx(ref, abs=1e-5)


def test_ratings_rank_items_against_labeled_ones_and_are_unused_at_budget_zero():
    dataset = with_metadata(make_dataset())
    model = ItemPooling(scale=0.0)
    model.fit(dataset.subset(["train"]))
    params = dict(model.params, rating_slope=0.5, rating_sd=1.0)
    level = {f"held-{k}": 5 for k in range(120)}
    level.update({"held-1": 9, "held-2": 2})
    by_content = {dataset.items[i]["item_content"]: r for i, r in level.items()}
    rating = lambda item: by_content.get(item["item_content"])  # noqa: E731
    hard, easy = dataset.make_input("s1", "held-1"), dataset.make_input("s1", "held-2")
    # No labeled item yet: ratings cannot be centred, so the prediction is version 3's.
    assert item_pooling_core.predict(hard, [], params, rating) == item_pooling_core.predict(hard, [], params)
    labeled = [[dataset.make_input("s1", f"held-{j}"), j % 2] for j in range(10, 16)]
    p_hard = item_pooling_core.predict(hard, labeled, params, rating)
    p_easy = item_pooling_core.predict(easy, labeled, params, rating)
    p_plain = item_pooling_core.predict(hard, labeled, params)
    assert p_hard < p_plain < p_easy
    # A failing or silent rating source falls back to the unrated prediction.
    def broken(item):
        raise RuntimeError("no rating")
    assert item_pooling_core.predict(hard, labeled, params, broken) == p_plain
    assert item_pooling_core.predict(hard, labeled, params, lambda item: None) == p_plain
    # Without rating_slope in params, a rating source is ignored.
    assert item_pooling_core.predict(hard, labeled, model.params, rating) == p_plain


def test_rating_gate_switches_ratings_off_when_they_disagree_with_the_labels():
    dataset = with_metadata(make_dataset())
    model = ItemPooling(scale=0.0)
    model.fit(dataset.subset(["train"]))
    params = dict(model.params, rating_slope=0.5, rating_sd=1.0, rating_gate=True)
    hard = dataset.make_input("s1", "held-1")
    labeled = [[dataset.make_input("s1", f"held-{j}"), j % 2] for j in range(10, 16)]
    p_plain = item_pooling_core.predict(hard, labeled, params)

    def source(failed, solved):
        level = {f"held-{j}": (failed if j % 2 == 0 else solved) for j in range(10, 16)}
        level["held-1"] = 9
        by_content = {dataset.items[i]["item_content"]: r for i, r in level.items()}
        return lambda item: by_content.get(item["item_content"])
    agreeing, reversed_ = source(failed=8, solved=3), source(failed=3, solved=8)
    assert item_pooling_core.predict(hard, labeled, params, agreeing) < p_plain
    assert item_pooling_core.predict(hard, labeled, params, reversed_) == p_plain
    # Without the gate, reversed ratings are used; with too few or one-sided labels, the gate waits.
    ungated = dict(params, rating_gate=False)
    assert item_pooling_core.predict(hard, labeled, ungated, reversed_) != p_plain
    for few in (labeled[:2], [entry for entry in labeled if entry[1] == 1]):
        assert (item_pooling_core.predict(hard, few, params, reversed_)
                == item_pooling_core.predict(hard, few, ungated, reversed_)
                != item_pooling_core.predict(hard, few, params))
