import json
import math

import pytest
from synthetic import make_dataset, with_metadata

from paiec import data, guess_core, item_pooling_core, logit_normal_core
from paiec.item_pooling import ItemPooling

GUESS = {"guess_intercept": -0.2, "guess_slope": 0.8, "guess_sd": 0.9}


@pytest.fixture
def setup():
    dataset = with_metadata(make_dataset())
    model = ItemPooling(scale=0.0)
    model.fit(dataset.subset(["train"]))
    return dataset, dict(model.params, **GUESS)


def by_name(values):
    return lambda subject, item: values.get(subject["normalized_name"])


def test_without_a_guess_option_c_is_version4(setup):
    dataset, params = setup
    x = dataset.make_input("s1", "held-0")
    labeled = [[dataset.make_input(s, f"held-{j}"), (j + len(s)) % 2] for s in ("s1", "s2") for j in range(3, 9)]
    for evidence in ([], labeled):
        plain = item_pooling_core.predict(x, evidence, params)
        assert item_pooling_core.predict(x, evidence, params, guess=by_name({})) == plain

        def broken(subject, item):
            raise RuntimeError("no guess")
        assert item_pooling_core.predict(x, evidence, params, guess=broken) == plain
        # Without guess_slope in params, a guess source is ignored.
        bare = {k: v for k, v in params.items() if k not in GUESS}
        assert item_pooling_core.predict(x, evidence, bare, guess=by_name({"s1": 0.9})) == plain


def test_malformed_guesses_count_as_no_guess(setup):
    dataset, params = setup
    x = dataset.make_input("s1", "held-0")
    labeled = [[dataset.make_input("s2", "held-0"), 1], [dataset.make_input("s1", "held-4"), 0]]
    for evidence in ([], labeled):
        plain = item_pooling_core.predict(x, evidence, params, guess=by_name({"s2": 0.4}))
        for bad in ("abc", "", [0.3], {"p": 0.3}, 1j, 10 ** 400, float("nan"), "nan"):
            p = item_pooling_core.predict(x, evidence, params, guess=by_name({"s1": bad, "s2": 0.4}))
            assert p == plain
        if evidence:  # a partner's malformed guess falls back to the partner's version 4 prior
            no_partner_guess = item_pooling_core.predict(x, evidence, params, guess=by_name({}))
            assert item_pooling_core.predict(x, evidence, params, guess=by_name({"s2": "nan"})) == no_partner_guess


def test_a_guess_sets_the_starting_prediction(setup):
    dataset, params = setup
    x = dataset.make_input("s1", "held-0")
    for g in (0.05, 0.3, 0.8):
        m = GUESS["guess_intercept"] + GUESS["guess_slope"] * math.log(g / (1 - g))
        expected = logit_normal_core.posterior_mean(m, GUESS["guess_sd"], 0, 0)
        assert item_pooling_core.predict(x, [], params, guess=by_name({"s1": g})) == expected
    low, high = (item_pooling_core.predict(x, [], params, guess=by_name({"s1": g})) for g in (0.2, 0.7))
    assert low < 0.5 < high
    # Guesses outside (0.01, 0.99) are clipped, and other subjects' guesses do not matter without labels.
    assert (item_pooling_core.predict(x, [], params, guess=by_name({"s1": 1.0}))
            == item_pooling_core.predict(x, [], params, guess=by_name({"s1": 0.99, "s2": 0.1})))


def test_partner_guesses_change_pooled_item_difficulty(setup):
    dataset, params = setup
    x = dataset.make_input("s1", "held-0")
    # s2 solved the target item. If s2 is weak, the item looks easy; if strong, it says less.
    labeled = [[dataset.make_input("s2", "held-0"), 1]]
    weak = item_pooling_core.predict(x, labeled, params, guess=by_name({"s1": 0.5, "s2": 0.1}))
    strong = item_pooling_core.predict(x, labeled, params, guess=by_name({"s1": 0.5, "s2": 0.9}))
    assert weak > strong


def test_guess_prompt_holds_only_platform_inputs():
    assert guess_core.FIELDS == data.SUBJECT_FIELDS
    dataset = with_metadata(make_dataset())
    subject, item = dataset.make_input("s2", "held-3")
    args = guess_core.request_args(subject, item, {"model": "claude-opus-5-5", "effort": "low"})
    text = args["messages"][0]["content"]
    assert item["benchmark_id"] not in text and "s2" in text and item["item_content"] in text
    assert args["model"] == "claude-opus-5-5" and args["output_config"]["effort"] == "low"
    assert guess_core.shown("x" * 2000).count("x") == guess_core.EXAMPLE_CHARS
    # One key per subject and benchmark: the item does not matter, the benchmark does.
    same = guess_core.guess_key(subject, dataset.make_input("s2", "held-7")[1])
    assert guess_core.guess_key(subject, item) == same
    assert guess_core.guess_key(subject, dataset.make_input("s2", "train-1")[1]) != same
    assert guess_core.guess_key(dataset.make_input("s3", "held-3")[0], item) != same


def test_guess_parsing():
    assert guess_core.parse(json.dumps({"success_percent": 37}), "end_turn") == 0.37
    for text, stop in ((json.dumps({"success_percent": 37}), "refusal"), ("37", "end_turn"),
                       (json.dumps({"success_percent": 140}), "end_turn"), ("{}", "end_turn"),
                       ('{"success_percent": Infinity}', "end_turn"), ('{"success_percent": 1e400}', "end_turn"),
                       ("[" * 100000, "end_turn")):
        assert guess_core.parse(text, stop) is None
