import importlib
import json
import math
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from synthetic import make_dataset, with_metadata
from test_version4 import FAKE_KEY, SETTINGS, FakeClient, export, fake_rating, items, make_rater

from paiec import guess_core, item_pooling_core, llm_rating_core, logit_normal_core
from paiec.export import write_submission

ROOT = Path(__file__).resolve().parents[1]
GUESS = {"guess_intercept": 0.03, "guess_slope": 0.64, "guess_sd": 1.16}


def fake_percent(request):
    return sum(map(ord, request["messages"][0]["content"].split("Example task")[0])) % 101


def fake_guess(subject, item):
    """The guess FakeGuessClient gives: it depends only on the subject's fields."""
    request = guess_core.request_args(subject, item, SETTINGS)
    return fake_percent(request) / 100


class FakeGuessClient(FakeClient):
    """FakeClient that also answers guess requests (or refuses them, with refuse_guesses)."""

    def __init__(self, refuse_guesses=False, **options):
        super().__init__(**options)
        self.refuse_guesses = refuse_guesses

    def create(self, **request):
        if request["system"] != guess_core.SYSTEM:
            return super().create(**request)
        with self.lock:
            self.calls.append(request)
            self.times.append(time.monotonic())
        time.sleep(self.delay)
        if self.refuse_guesses:
            return types.SimpleNamespace(stop_reason="refusal", content=[])
        block = types.SimpleNamespace(type="text", text=json.dumps({"success_percent": fake_percent(request)}))
        return types.SimpleNamespace(stop_reason="end_turn", content=[block])


def subject(name):
    return {f: "" for f in guess_core.FIELDS} | {"normalized_name": name}


def item(k, benchmark="benchmark_1"):
    return {"item_content": f"q {k}", "item_features": "x=1", "interactors": "", "benchmark_id": benchmark}


def guess_calls(client):
    return [c for c in client.calls if c["system"] == guess_core.SYSTEM]


def test_one_guess_per_subject_and_benchmark_cached_on_disk_and_shared(tmp_path, monkeypatch):
    client = FakeGuessClient()
    first = make_rater(tmp_path, monkeypatch, client)
    a, b = subject("model a"), subject("model b")
    jobs = [first.guess_job(s, item(k)) for s in (a, b) for k in range(4)]
    first.request_jobs(jobs + jobs, wait=5)
    assert len(guess_calls(client)) == 2
    assert first.lookup_guess(a, item(9)) == fake_guess(a, item(0))
    assert first.lookup_guess(a, item(0, "benchmark_2")) is None  # another benchmark: not asked
    second = make_rater(tmp_path, monkeypatch, client)  # a restarted worker reads the files
    assert second.lookup_guess(b, item(5)) == fake_guess(b, item(0))
    second.request_jobs([second.guess_job(b, item(7))], wait=5)
    assert len(guess_calls(client)) == 2
    request = guess_calls(client)[0]
    assert request["model"] == "claude-opus-5-5" and request["output_config"]["effort"] == "low"
    assert request["output_config"]["format"]["schema"] == guess_core.SCHEMA


def test_guesses_are_sent_before_ratings(tmp_path, monkeypatch):
    client = FakeGuessClient(delay=0.05)
    rater = make_rater(tmp_path, monkeypatch, client, threads=1)
    jobs = [rater.rating_job(i) for i in items(6)] + [rater.guess_job(subject("m"), item(0))]
    rater.request_jobs(jobs, wait=5)
    order = [c["system"] == guess_core.SYSTEM for c in client.calls]
    assert len(order) == 7 and order.index(True) <= 1  # at most the rating already taken goes first


def test_a_refused_guess_is_none_and_not_asked_again(tmp_path, monkeypatch):
    client = FakeGuessClient(refuse_guesses=True)
    rater = make_rater(tmp_path, monkeypatch, client)
    job = rater.guess_job(subject("m"), item(0))
    rater.request_jobs([job], wait=5)
    rater.request_jobs([job], wait=5)
    assert rater.lookup_guess(subject("m"), item(1)) is None and len(guess_calls(client)) == 1


@pytest.mark.parametrize("stored", [7, -0.1, 1.5, True, "0.4", None, [0.3]])
def test_only_probabilities_on_disk_count_as_guesses(tmp_path, monkeypatch, stored):
    rater = make_rater(tmp_path, monkeypatch, FakeGuessClient())
    key = rater.guess_job(subject("m"), item(0))[0]
    (tmp_path / "rating_cache" / f"{key}.json").write_text(json.dumps({"rating": stored}), encoding="utf-8")
    assert rater.lookup_guess(subject("m"), item(0)) is None


def test_guesses_share_the_call_cap(tmp_path, monkeypatch):
    client = FakeGuessClient()
    rater = make_rater(tmp_path, monkeypatch, client, max_calls=3, threads=1)
    jobs = [rater.rating_job(i) for i in items(4)] + [rater.guess_job(subject(f"m{k}"), item(0)) for k in range(2)]
    rater.request_jobs(jobs, wait=5)
    assert len(client.calls) == 3 and len(guess_calls(client)) == 2  # the guesses went first


def test_benchmark_subjects_lists_the_target_and_its_partners_once():
    a, b, c = subject("a"), subject("b"), subject("c")
    labeled = [[(b, item(1)), 1], [(b, item(2)), 0], [(c, item(3, "benchmark_2")), 1], [(a, item(4)), 1],
               "malformed", [(None, item(5)), 1], [(subject("d"), "not an item"), 0]]
    assert llm_rating_core.benchmark_subjects((a, item(0)), labeled) == [a, b]
    assert llm_rating_core.benchmark_subjects((a, item(0)), []) == [a]


def export5(tmp_path, dataset):
    params = export(tmp_path, dataset)
    params = dict(params, model="item_pooling_guessed", **GUESS)
    write_submission(params, tmp_path, "test export")
    return params


def load_model(folder, monkeypatch, client=None):
    monkeypatch.setattr(sys, "path", [str(folder)] + sys.path)
    monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items() if k not in (
        "baseline_core", "logit_normal_core", "item_pooling_core", "llm_rating_core", "guess_core", "model")})
    if client is not None:
        (folder / llm_rating_core.KEY_FILE).write_text(FAKE_KEY, encoding="utf-8")
        core = importlib.import_module("llm_rating_core")
        monkeypatch.setattr(core.Rater, "_make_client", lambda self, key: (client, "fake"))
    module = importlib.import_module("model")
    assert module.llm_rating_core.__file__.startswith(str(folder))
    assert module.llm_rating_core.guess_core.__file__.startswith(str(folder))
    return module


def cases(dataset, seed=0, n=30):
    import random
    rng = random.Random(seed)
    for _ in range(n):
        labeled = [[dataset.make_input(rng.choice(["s1", "s2", "s3"]), f"held-{rng.randrange(60)}"),
                    rng.randint(0, 1)] for _ in range(rng.randrange(20))]
        yield dataset.make_input(rng.choice(["s1", "s2", "s3"]), f"held-{rng.randrange(60)}"), labeled


def test_exported_version5_without_a_key_predicts_exactly_like_version3(tmp_path, monkeypatch):
    dataset = with_metadata(make_dataset())
    params = export5(tmp_path, dataset)
    assert {"guess_core.py", "llm_rating_core.py", "prediction_config.json"} <= {p.name for p in tmp_path.iterdir()}
    model = load_model(tmp_path, monkeypatch)
    for x, labeled in cases(dataset):
        assert model.predict(x, labeled) == item_pooling_core.predict(x, labeled, params)
    assert not (tmp_path / "rating_cache").exists()


def test_exported_version5_uses_guesses_for_target_and_partners(tmp_path, monkeypatch):
    dataset = with_metadata(make_dataset())
    params = export5(tmp_path, dataset)
    model = load_model(tmp_path, monkeypatch, FakeGuessClient())
    changed = 0
    for x, labeled in cases(dataset, seed=1):
        expected = item_pooling_core.predict(x, labeled, params, fake_rating, fake_guess)
        assert model.predict(x, labeled) == pytest.approx(expected, abs=1e-15)
        changed += expected != item_pooling_core.predict(x, labeled, params, fake_rating)
    assert changed > 0
    for broken in ("not an input", None, [None, None]):
        assert model.predict(broken, None) == 0.5


def test_exported_version5_starts_from_the_guess_at_budget_zero(tmp_path, monkeypatch):
    dataset = with_metadata(make_dataset())
    params = export5(tmp_path, dataset)
    model = load_model(tmp_path, monkeypatch, FakeGuessClient())
    x = dataset.make_input("s2", "held-3")
    g = min(max(fake_guess(*x), 0.01), 0.99)
    m = GUESS["guess_intercept"] + GUESS["guess_slope"] * math.log(g / (1 - g))
    assert model.predict(x, []) == logit_normal_core.posterior_mean(m, GUESS["guess_sd"], 0, 0)


def test_exported_version5_without_guesses_is_version4(tmp_path, monkeypatch):
    dataset = with_metadata(make_dataset())
    params = export5(tmp_path, dataset)
    model = load_model(tmp_path, monkeypatch, FakeGuessClient(refuse_guesses=True))
    for x, labeled in cases(dataset, seed=2):
        expected = item_pooling_core.predict(x, labeled, params, fake_rating)
        assert model.predict(x, labeled) == pytest.approx(expected, abs=1e-15)


def test_a_slow_guess_waits_at_most_the_budget_and_is_used_later(tmp_path, monkeypatch):
    dataset = with_metadata(make_dataset())
    params = export5(tmp_path, dataset)
    params["rating_api"] = dict(params["rating_api"], budget_seconds=0.2)  # a copy: SETTINGS is shared
    write_submission(params, tmp_path, "test export")
    model = load_model(tmp_path, monkeypatch, FakeGuessClient(delay=1.0))
    x = dataset.make_input("s1", "held-3")
    start = time.monotonic()
    assert model.predict(x, []) == item_pooling_core.predict(x, [], params)  # no guess yet: version 4
    assert time.monotonic() - start < 0.8
    time.sleep(1.5)
    assert model.predict(x, []) == item_pooling_core.predict(x, [], params, fake_rating, fake_guess)


def test_committed_version5_submission_is_current():
    folder = ROOT / "submissions" / "version5"
    if not folder.exists():
        pytest.skip("submissions/version5 not exported yet")
    from paiec import baseline_core
    for module in (baseline_core, logit_normal_core, item_pooling_core, llm_rating_core, guess_core):
        name = module.__name__.split(".")[-1] + ".py"
        assert (folder / name).read_bytes() == Path(module.__file__).read_bytes()
    assert {p.name for p in folder.iterdir()} == {
        "baseline_core.py", "guess_core.py", "item_pooling_core.py", "llm_rating_core.py",
        "logit_normal_core.py", "model.py", "params.json", "prediction_config.json"}
    v4 = json.loads((ROOT / "submissions" / "version4" / "params.json").read_text(encoding="utf-8"))
    v5 = json.loads((folder / "params.json").read_text(encoding="utf-8"))
    added = {"guess_trust", "guess_fit", "guess_intercept", "guess_slope", "guess_sd"}
    assert set(v5) - set(v4) == added
    assert {k for k in v4 if v4[k] != v5[k]} == {"model", "trained_on"} and v5["model"] == "item_pooling_guessed"
    t, fit = v5["guess_trust"], v5["guess_fit"]
    assert t == 0.75 and v5["guess_intercept"] == t * fit["a"] and v5["guess_slope"] == t * fit["c"]
    assert v5["guess_sd"] == math.sqrt((1 - t ** 2) * v4["sd"] ** 2 + t ** 2 * fit["s"] ** 2)


def feedback_text(budget0, missing_budget=None, alc_summary=True):
    lines = []
    for budget in (0, 1, 3, 7, 15, 31):
        if budget == missing_budget:
            continue
        lines.append(f"Label budget: {budget}")
        for k, b0 in enumerate(budget0):
            lines.append(f"subject_{k} benchmark_1 100 {b0 if budget == 0 else 0.2:.6f} 0.050000")
    if alc_summary:
        lines.append("ALC summary")
        lines += [f"subject_{k} benchmark_1 100 0.200000 0.050000" for k in range(len(budget0))]
    return "\n".join(lines)


def test_version5_replaces_version4_only_if_its_budget0_brier_is_below_a_quarter():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import check_version5
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    assert check_version5.decide(feedback_text([0.20, 0.27, 0.24]))["decision"] == "version 5"
    assert check_version5.decide(feedback_text([0.25, 0.25]))["decision"] == "version 4"  # no guesses arrived
    assert check_version5.decide(feedback_text([0.20, 0.31]))["decision"] == "version 4"
    assert check_version5.decide(feedback_text([0.2, 0.2], missing_budget=7))["decision"] == "version 4"
    assert check_version5.decide(feedback_text([0.2, 0.2], alc_summary=False))["decision"] == "version 4"
    assert check_version5.decide("no rows")["decision"] == "version 4"
    # Guesses must reach at least half of the pairs by budget 0.
    assert check_version5.decide(feedback_text([0.249999] + [0.25] * 7))["decision"] == "version 4"
    assert check_version5.decide(feedback_text([0.2] * 3 + [0.25] * 5))["decision"] == "version 4"
    assert check_version5.decide(feedback_text([0.2] * 4 + [0.25] * 4))["decision"] == "version 5"
