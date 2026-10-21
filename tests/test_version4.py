import hashlib
import importlib
import json
import os
import random
import sys
import threading
import time
import types
import zipfile
from pathlib import Path

import pytest
from synthetic import make_dataset, with_metadata

from paiec import baseline_core, item_pooling_core, llm_rating_core, logit_normal_core
from paiec.export import write_submission
from paiec.item_pooling import ItemPooling

ROOT = Path(__file__).resolve().parents[1]
# SHA-256 of llm_rating_core.py as uploaded in version 4 and in the API check.
VERSION4_RATING_CORE = "a494867b1fdd579d16015648970301baf3741bfbfdc4ec9b4c36491ca3cd2d59"
FAKE_KEY = "sk-ant-api03-" + "CANARY" * 15 + "AA"
SETTINGS = {"model": "claude-opus-5-5", "effort": "low", "timeout": 20, "threads": 4,
            "budget_seconds": 4, "stale_seconds": 1, "max_calls": 1000, "max_chars": 150000}


class StatusError(Exception):
    """Like the SDK's APIStatusError: a status code and the response headers."""

    def __init__(self, status, retry_after=None, message=None):
        super().__init__(message or f"HTTP {status}")
        self.message = message or f"HTTP {status}"
        self.status_code = status
        self.type = None
        self.response = types.SimpleNamespace(headers={"retry-after": retry_after} if retry_after else {})


class FakeClient:
    """Stands in for anthropic.Anthropic: rates an item from its prompt, counts calls. errors
    are raised by the first calls, in order; always is raised by every call."""

    def __init__(self, delay=0.0, errors=(), always=None):
        self.calls, self.times, self.delay = [], [], delay
        self.errors, self.always = list(errors), always
        self.lock = threading.Lock()
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self.create))

    def create(self, **request):
        with self.lock:
            self.calls.append(request)
            self.times.append(time.monotonic())
            error = self.errors.pop(0) if self.errors else self.always
        time.sleep(self.delay)
        if error is not None:
            raise error
        rating = sum(map(ord, request["messages"][0]["content"])) % 10 + 1
        block = types.SimpleNamespace(type="text", text=json.dumps({"rating": rating}))
        return types.SimpleNamespace(stop_reason="end_turn", content=[block])


def fake_rating(item):
    content = f"Task features: {item.get('item_features', '') or '(none)'}\n\nTask:\n{item['item_content']}"
    return sum(map(ord, content)) % 10 + 1


def make_rater(folder, monkeypatch, client, key=FAKE_KEY, **settings):
    (folder / llm_rating_core.KEY_FILE).write_text(key, encoding="utf-8")
    monkeypatch.setattr(llm_rating_core.Rater, "_make_client", lambda self, key: (client, "fake"))
    return llm_rating_core.Rater(folder, dict(SETTINGS, **settings))


def items(n, tag="q"):
    return [{"item_content": f"{tag} {k}", "item_features": "x=1"} for k in range(n)]


def markers(folder):
    return len(os.listdir(folder / "rating_cache" / "calls"))


def on_disk(folder, item):
    return (folder / "rating_cache" / f"{llm_rating_core.item_key(item)}.json").exists()


def test_without_a_key_file_there_are_no_ratings_and_nothing_is_written(tmp_path, capsys):
    rater = llm_rating_core.Rater(tmp_path, SETTINGS)
    rater.request(items(2), wait=1)
    assert not rater.active and rater.lookup(items(1)[0]) is None
    assert list(tmp_path.iterdir()) == []
    assert "no API key file" in capsys.readouterr().err


def test_a_malformed_key_turns_ratings_off_without_showing_it(tmp_path, monkeypatch, capsys):
    client = FakeClient()
    rater = make_rater(tmp_path, monkeypatch, client, key="sk-ant-api03-MALFORMED KEY WITH SPACES")
    rater.request(items(2), wait=1)
    assert not rater.active and client.calls == []
    err = capsys.readouterr().err
    assert "not hold a regular Anthropic API key" in err and "MALFORMED" not in err


def test_ratings_are_cached_on_disk_and_shared_between_workers(tmp_path, monkeypatch, capsys):
    client = FakeClient()
    first = make_rater(tmp_path, monkeypatch, client)
    batch = items(6)
    first.request(batch + batch, wait=5)
    assert len(client.calls) == 6 and not list((tmp_path / "rating_cache").glob("*.lock"))
    assert [first.lookup(i) for i in batch] == [fake_rating(i) for i in batch]
    second = make_rater(tmp_path, monkeypatch, client)  # a restarted worker reads the files
    assert [second.lookup(i) for i in batch] == [fake_rating(i) for i in batch]
    second.request(batch, wait=5)
    assert len(client.calls) == 6 and markers(tmp_path) == 6
    request = client.calls[0]
    assert request["model"] == "claude-opus-5-5" and request["system"] == llm_rating_core.SYSTEM
    assert request["output_config"]["format"]["schema"] == llm_rating_core.SCHEMA
    assert FAKE_KEY not in capsys.readouterr().err
    assert all(FAKE_KEY not in p.read_text(encoding="utf-8", errors="replace")
               for p in (tmp_path / "rating_cache").rglob("*") if p.is_file())


def test_a_prediction_waits_at_most_the_budget_and_ratings_arrive_later(tmp_path, monkeypatch):
    rater = make_rater(tmp_path, monkeypatch, FakeClient(delay=1.0))
    batch = items(3)
    start = time.monotonic()
    rater.request(batch, wait=0.2)
    assert time.monotonic() - start < 0.8
    assert rater.lookup(batch[0]) is None
    time.sleep(1.5)
    assert rater.lookup(batch[0]) == fake_rating(batch[0])


def test_a_later_prediction_also_waits_for_ratings_requested_earlier(tmp_path, monkeypatch):
    client = FakeClient(delay=1.0)
    rater = make_rater(tmp_path, monkeypatch, client)
    item = items(1)[0]
    rater.request([item], wait=0)  # an earlier prediction started it
    start = time.monotonic()
    rater.request([item], wait=3)
    assert 0.5 < time.monotonic() - start < 2 and rater.lookup(item) == fake_rating(item)
    assert len(client.calls) == 1


def test_a_rating_on_disk_is_used_at_once_even_when_all_threads_are_busy(tmp_path, monkeypatch):
    rater = make_rater(tmp_path, monkeypatch, FakeClient(delay=2.0), threads=1)
    rater.request(items(1, "slow"), wait=0)  # occupies the only thread
    cached = items(1, "cached")[0]
    (tmp_path / "rating_cache" / f"{llm_rating_core.item_key(cached)}.json").write_text(
        json.dumps({"rating": 7}), encoding="utf-8")
    start = time.monotonic()
    rater.request([cached], wait=4)
    assert time.monotonic() - start < 0.5 and rater.lookup(cached) == 7


def test_failures_are_retried_once_and_a_rating_written_later_by_another_worker_is_used(tmp_path, monkeypatch):
    failing = FakeClient(always=ConnectionError("network down"))
    rater = make_rater(tmp_path, monkeypatch, failing)
    item = items(1)[0]
    rater.request([item], wait=2)
    assert not on_disk(tmp_path, item) and not list((tmp_path / "rating_cache").glob("*.lock"))
    rater.request([item], wait=2)  # second attempt in this process
    rater.request([item], wait=2)  # given up: no more calls
    assert len(failing.calls) == llm_rating_core.ATTEMPTS_PER_ITEM == markers(tmp_path)
    assert rater.lookup(item) is None
    (tmp_path / "rating_cache" / f"{llm_rating_core.item_key(item)}.json").write_text(
        json.dumps({"rating": 3}), encoding="utf-8")
    assert rater.lookup(item) == 3


def test_busy_replies_are_free_and_pause_every_worker(tmp_path, monkeypatch, capsys):
    client = FakeClient(errors=[StatusError(429, "1"), StatusError(529, "1")])
    rater = make_rater(tmp_path, monkeypatch, client)
    other = make_rater(tmp_path, monkeypatch, client)  # another worker, same folder
    item = items(1)[0]
    rater.request([item], wait=3)  # 429
    assert (tmp_path / "rating_cache" / "pause_until").exists() and rater.lookup(item) is None
    other.request([item], wait=3)  # waits for the pause, then 529
    rater.request([item], wait=3)  # waits for the pause, then rated
    assert rater.lookup(item) == fake_rating(item) == other.lookup(item)
    assert len(client.calls) == 3 and markers(tmp_path) == 1  # only the billed attempt counts
    gaps = [b - a for a, b in zip(client.times, client.times[1:])]
    assert min(gaps) > 0.8


def NotSent():
    return llm_rating_core.APIFailure("no connection (ConnectionRefusedError)", not_sent=True)


@pytest.mark.parametrize("error", [StatusError(401), StatusError(403), StatusError(402),
                                   StatusError(400, message="Your credit balance is too low")],
                         ids=["401", "403", "402", "400-credit"])
def test_a_refused_key_or_account_stops_new_requests_but_keeps_ratings(tmp_path, monkeypatch, capsys, error):
    client = FakeClient(errors=[None, error], always=None)
    rater = make_rater(tmp_path, monkeypatch, client, threads=1)
    first, second, third = items(3)
    rater.request([first], wait=3)  # rated and paid for
    rater.request([second, third], wait=3)  # refused: no new requests from now on
    assert rater.active and rater.lookup(first) == fake_rating(first)
    assert rater.lookup(second) is None and not on_disk(tmp_path, second) and not on_disk(tmp_path, third)
    assert markers(tmp_path) == 1 and (tmp_path / "rating_cache" / "stopped").exists()
    calls = len(client.calls)
    later = make_rater(tmp_path, monkeypatch, client)  # a worker at the next checkpoint
    later.request(items(3, "new"), wait=1)
    assert len(client.calls) == calls and later.lookup(first) == fake_rating(first)
    err = capsys.readouterr().err
    assert "refused this key or account" in err and "earlier in this run: cached ratings only" in err


def test_a_request_that_never_left_is_free_and_retried_after_a_pause(tmp_path, monkeypatch):
    client = FakeClient(errors=[NotSent()])
    rater = make_rater(tmp_path, monkeypatch, client)
    item = items(1)[0]
    rater.request([item], wait=1)
    assert markers(tmp_path) == 0 and (tmp_path / "rating_cache" / "pause_until").exists()
    rater.pause_until = time.time() + 0.5  # shorten the 10 s default pause for the test
    (tmp_path / "rating_cache" / "pause_until").write_text(repr(rater.pause_until), encoding="utf-8")
    rater.request([item], wait=3)
    assert rater.lookup(item) == fake_rating(item) and markers(tmp_path) == 1


def test_failure_kinds_follow_the_error_attributes():
    kind = llm_rating_core.failure_kind
    billing = StatusError(400)
    billing.type = "billing_error"
    assert [kind(StatusError(s)) for s in (429, 529, 401, 403, 402, 400, 404, 413, 422, 500, 503)] == [
        "busy", "busy", "refused", "refused", "refused", "unratable", "unratable", "unratable",
        "unratable", "transient", "transient"]
    assert kind(StatusError(400, message="You have reached your specified API usage limits")) == "refused"
    assert kind(billing) == "refused"
    assert kind(NotSent()) == "busy" and kind(TimeoutError("read timed out")) == "transient"


def test_a_worker_never_removes_a_lock_another_worker_took_over(tmp_path, monkeypatch):
    rater = make_rater(tmp_path, monkeypatch, FakeClient())
    key = llm_rating_core.item_key(items(1)[0])
    token = rater._take_lock(key)
    assert token and rater._take_lock(key) is None  # held: a second taker waits
    lock = rater.cache / f"{key}.lock"
    lock.write_text("12345 another-worker", encoding="utf-8")  # taken over meanwhile
    rater._release(key, token)
    assert lock.read_text(encoding="utf-8") == "12345 another-worker"


def test_a_rating_written_just_before_the_lock_was_taken_is_not_paid_for_again(tmp_path, monkeypatch):
    client = FakeClient()
    rater = make_rater(tmp_path, monkeypatch, client)
    item = items(1)[0]
    key = llm_rating_core.item_key(item)
    take = rater._take_lock

    def take_after_another_worker_finished(k):
        (rater.cache / f"{k}.json").write_text(json.dumps({"rating": 6}), encoding="utf-8")
        return take(k)
    monkeypatch.setattr(rater, "_take_lock", take_after_another_worker_finished)
    rater.request([item], wait=3)
    assert rater.lookup(item) == 6 and client.calls == [] and not (rater.cache / f"{key}.lock").exists()


def test_a_request_the_api_refuses_to_process_is_unrated_and_free(tmp_path, monkeypatch):
    client = FakeClient(always=StatusError(400))
    rater = make_rater(tmp_path, monkeypatch, client)
    item = items(1)[0]
    rater.request([item], wait=3)
    rater.request([item], wait=3)
    assert len(client.calls) == 1 and markers(tmp_path) == 0
    assert on_disk(tmp_path, item) and rater.lookup(item) is None and rater.active


def test_call_cap_counts_billed_attempts_and_then_stops(tmp_path, monkeypatch, capsys):
    client = FakeClient()
    rater = make_rater(tmp_path, monkeypatch, client, max_calls=2, threads=1)
    batch = items(5)
    rater.request(batch, wait=5)
    rater.request(batch, wait=5)
    assert len(client.calls) == 2 == markers(tmp_path)
    assert sum(on_disk(tmp_path, i) for i in batch) == 2  # the others are not marked unratable
    assert "call cap of 2 reached" in capsys.readouterr().err


def test_an_abandoned_lock_is_taken_over(tmp_path, monkeypatch):
    client = FakeClient()
    rater = make_rater(tmp_path, monkeypatch, client)
    item = items(1)[0]
    lock = rater.cache / f"{llm_rating_core.item_key(item)}.lock"
    lock.write_text("", encoding="utf-8")
    old = time.time() - 10
    os.utime(lock, (old, old))
    rater.request([item], wait=3)
    assert rater.lookup(item) == fake_rating(item) and len(client.calls) == 1


def test_a_worker_takes_over_as_soon_as_another_releases_the_lock(tmp_path, monkeypatch):
    client = FakeClient()
    rater = make_rater(tmp_path, monkeypatch, client, stale_seconds=30)
    item = items(1)[0]
    lock = rater.cache / f"{llm_rating_core.item_key(item)}.lock"
    lock.write_text("", encoding="utf-8")  # another worker is rating it, then fails and lets go
    threading.Timer(0.5, lock.unlink).start()
    start = time.monotonic()
    rater.request([item], wait=5)
    assert rater.lookup(item) == fake_rating(item) and time.monotonic() - start < 2


@pytest.mark.skipif(os.name != "posix", reason="owner checks only run on POSIX, like the platform")
def test_a_lock_left_by_a_killed_worker_is_taken_over_at_once(tmp_path, monkeypatch):
    import subprocess
    killed = subprocess.Popen([sys.executable, "-c", "pass"])
    killed.wait()  # its PID no longer exists
    client = FakeClient()
    rater = make_rater(tmp_path, monkeypatch, client, stale_seconds=30)
    item = items(1)[0]
    (rater.cache / f"{llm_rating_core.item_key(item)}.lock").write_text(str(killed.pid), encoding="utf-8")
    start = time.monotonic()
    rater.request([item], wait=5)
    assert rater.lookup(item) == fake_rating(item) and time.monotonic() - start < 1


def test_an_unwritable_folder_turns_ratings_off(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(llm_rating_core.Rater, "_cache_dir", lambda self: None)
    rater = make_rater(tmp_path, monkeypatch, FakeClient())
    assert not rater.active
    assert "not writable" in capsys.readouterr().err


def export(tmp_path, dataset):
    model = ItemPooling(scale=0.0)
    model.fit(dataset.subset(["train"]))
    params = dict(model.params, model="item_pooling_rated", rating_slope=0.4, rating_sd=1.9,
                  rating_gate=True, rating_api=SETTINGS)
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
    return module


def random_cases(dataset, seed=0, n=30):
    rng = random.Random(seed)
    for _ in range(n):
        labeled = [[dataset.make_input(rng.choice(["s1", "s2", "s3"]), f"held-{rng.randrange(60)}"),
                    rng.randint(0, 1)] for _ in range(rng.randrange(20))]
        yield dataset.make_input(rng.choice(["s1", "s2", "s3"]), f"held-{rng.randrange(60)}"), labeled


def test_exported_version4_without_a_key_predicts_exactly_like_version3(tmp_path, monkeypatch):
    dataset = with_metadata(make_dataset())
    params = export(tmp_path, dataset)
    files = {p.name for p in tmp_path.iterdir()}
    assert {"prediction_config.json", "llm_rating_core.py"} <= files and "requirements.txt" not in files
    model = load_model(tmp_path, monkeypatch)
    for x, labeled in random_cases(dataset):
        assert model.predict(x, labeled) == item_pooling_core.predict(x, labeled, params)
    assert not (tmp_path / "rating_cache").exists()


def test_exported_version4_with_ratings_matches_the_core(tmp_path, monkeypatch):
    dataset = with_metadata(make_dataset())
    params = export(tmp_path, dataset)
    model = load_model(tmp_path, monkeypatch, FakeClient())
    changed = 0
    for x, labeled in random_cases(dataset, seed=1):
        expected = item_pooling_core.predict(x, labeled, params, fake_rating)
        assert model.predict(x, labeled) == pytest.approx(expected, abs=1e-15)
        changed += expected != item_pooling_core.predict(x, labeled, params)
    assert changed > 0
    for broken in ("not an input", None, [None, None]):
        assert model.predict(broken, None) == 0.5


def test_exported_version4_rates_the_target_at_budget_zero_for_later_checkpoints(tmp_path, monkeypatch):
    dataset = with_metadata(make_dataset())
    params = export(tmp_path, dataset)
    model = load_model(tmp_path, monkeypatch, FakeClient(delay=0.3))
    x = dataset.make_input("s1", "held-3")
    assert model.predict(x, []) == item_pooling_core.predict(x, [], params)  # unchanged at budget 0
    assert model.RATER.lookup(x[1]) == fake_rating(x[1])


def test_add_api_key_writes_a_private_copy_and_rejects_non_keys(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import add_api_key
    source = tmp_path / "version4.zip"
    with zipfile.ZipFile(source, "w") as handle:
        handle.writestr("model.py", "def predict(input, labeled=None):\n    return 0.5\n")
        handle.writestr("params.json", json.dumps({"rating_api": {"model": "claude-opus-5-5"}}))
    out = tmp_path / "private"
    out.mkdir()
    monkeypatch.setattr(sys, "argv", ["add_api_key.py", str(source), "--out", str(out)])
    checked = []
    monkeypatch.setattr(add_api_key, "check_key", lambda key, model: checked.append((key, model)))
    for bad in ("not-a-key", "sk-ant-", "sk-ant-admin01-" + "x" * 30, FAKE_KEY + " x"):
        monkeypatch.setattr(add_api_key.getpass, "getpass", lambda prompt, bad=bad: bad)
        with pytest.raises(SystemExit):
            add_api_key.main()
    assert list(out.iterdir()) == []
    assert checked == []  # malformed keys are refused before any request
    monkeypatch.setattr(add_api_key.getpass, "getpass", lambda prompt: FAKE_KEY)
    monkeypatch.setattr(add_api_key, "check_key", lambda key, model: "the API rejected this key (HTTP 401)")
    with pytest.raises(SystemExit, match="Key check failed"):
        add_api_key.main()
    assert list(out.iterdir()) == []
    monkeypatch.setattr(add_api_key, "check_key", lambda key, model: checked.append((key, model)))
    add_api_key.main()
    assert checked == [(FAKE_KEY, "claude-opus-5-5")]
    with zipfile.ZipFile(out / "version4_private.zip") as handle:
        assert sorted(handle.namelist()) == ["anthropic_api_key.txt", "model.py", "params.json"]
        assert handle.read("anthropic_api_key.txt").decode() == FAKE_KEY + "\n"


@pytest.mark.parametrize("extra", ["anthropic_api_key.txt", "version4_private.zip",
                                   "rating_cache/abc.json", "sdk_check_markers/x"])
def test_build_refuses_a_folder_with_a_key_or_runtime_files(tmp_path, monkeypatch, extra):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_submission
    folder = tmp_path / "version4_refusal_test"
    (folder / extra).parent.mkdir(parents=True, exist_ok=True)
    (folder / "model.py").write_text("def predict(input, labeled=None):\n    return 0.5\n", encoding="utf-8")
    (folder / extra).write_text("secret", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to build"):
        build_submission.build(folder)
    assert not (ROOT / "dist" / "version4_refusal_test.zip").exists()


def test_committed_version4_submission_is_current(tmp_path, monkeypatch):
    """The uploaded folder is frozen: its helpers match the code byte for byte, and its copy of
    item_pooling_core (which later gained the optional subject guess) predicts exactly like the
    current core without a guess."""
    folder = ROOT / "submissions" / "version4"
    if not folder.exists():
        pytest.skip("submissions/version4 not exported yet")
    for module in (baseline_core, logit_normal_core):
        name = module.__name__.split(".")[-1] + ".py"
        assert (folder / name).read_bytes() == Path(module.__file__).read_bytes()
    # llm_rating_core gained subject guesses in version 5; the uploaded copy stays as it was.
    assert hashlib.sha256((folder / "llm_rating_core.py").read_bytes()).hexdigest() == VERSION4_RATING_CORE
    for name in ("baseline_core.py", "logit_normal_core.py", "item_pooling_core.py"):
        (tmp_path / name).write_bytes((folder / name).read_bytes())
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + sys.path)
    monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items() if k not in (
        "baseline_core", "logit_normal_core", "item_pooling_core")})
    frozen = importlib.import_module("item_pooling_core")
    assert frozen.__file__.startswith(str(tmp_path))
    frozen_params = json.loads((folder / "params.json").read_text(encoding="utf-8"))
    dataset = with_metadata(make_dataset())
    rating = lambda item: sum(map(ord, item["item_content"])) % 10 + 1  # noqa: E731
    rng = random.Random(0)
    for _ in range(40):
        labeled = [[dataset.make_input(rng.choice(["s1", "s2", "s3"]), f"held-{rng.randrange(60)}"),
                    rng.randint(0, 1)] for _ in range(rng.randrange(25))]
        x = dataset.make_input(rng.choice(["s1", "s2", "s3"]), f"held-{rng.randrange(60)}")
        assert (frozen.predict(x, labeled, frozen_params, rating)
                == item_pooling_core.predict(x, labeled, frozen_params, rating))
    params = json.loads((folder / "params.json").read_text(encoding="utf-8"))
    assert (params["model"], params["scale"], params["rating_trust"], params["rating_gate"]) == (
        "item_pooling_rated", 0.0, 0.5, True)
    api = params["rating_api"]
    assert api["model"] == "claude-opus-5-5" and api["budget_seconds"] <= 5
    assert api["timeout"] < api["stale_seconds"] and api["max_calls"] <= 1000
    assert {p.name for p in folder.iterdir()} == {
        "baseline_core.py", "item_pooling_core.py", "llm_rating_core.py", "logit_normal_core.py",
        "model.py", "params.json", "prediction_config.json"}


class FakeConnection:
    """Stands in for http.client.HTTPSConnection: records the request, replies as told."""

    sent = []

    def __init__(self, host, port=None, timeout=None, context=None, reply=(200, {}, {}), fail=None):
        self.host, self.port, self.timeout, self.reply, self.fail = host, port, timeout, reply, fail
        self.tunnel = None

    def set_tunnel(self, host, port=None, headers=None):
        self.tunnel = (host, port, headers)

    def connect(self):
        if self.fail == "connect":
            raise ConnectionRefusedError("refused")
        if self.fail == "proxy-garbage":  # a proxy that does not answer CONNECT with HTTP
            raise llm_rating_core.http.client.BadStatusLine("\x05\x00")

    def request(self, method, path, body=None, headers=None):
        FakeConnection.sent.append((self.host, method, path, json.loads(body) if body else None, headers))
        FakeConnection.routes.append((self.host, self.port, self.tunnel))

    def getresponse(self):
        if self.fail == "read":
            raise TimeoutError("timed out")
        status, body, headers = self.reply
        return types.SimpleNamespace(status=status, read=lambda: json.dumps(body).encode(),
                                     getheaders=lambda: list(headers.items()))

    def close(self):
        pass


def connection(monkeypatch, proxies=None, **behaviour):
    FakeConnection.sent, FakeConnection.routes = [], []
    monkeypatch.setattr(llm_rating_core.urllib.request, "getproxies", lambda: dict(proxies or {}))
    monkeypatch.setattr(llm_rating_core.http.client, "HTTPSConnection",
                        lambda host, port=None, timeout=None, context=None:
                        FakeConnection(host, port, timeout, context, **behaviour))


def test_the_client_goes_through_the_proxy_the_environment_names(monkeypatch):
    reply = (200, {"content": [{"type": "text", "text": "{\"rating\": 2}"}], "stop_reason": "end_turn"}, {})
    connection(monkeypatch, proxies={"https": "http://user:p%40ss@proxy.example:3128"}, reply=reply)
    monkeypatch.setattr(llm_rating_core.urllib.request, "proxy_bypass", lambda host: False)
    llm_rating_core.Client(FAKE_KEY, 20).beta.messages.create(**llm_rating_core.request_args(items(1)[0], SETTINGS))
    host, port, tunnel = FakeConnection.routes[0]
    assert (host, port, tunnel[:2]) == ("proxy.example", 3128, ("api.anthropic.com", 443))
    assert tunnel[2] == {"Proxy-Authorization": "Basic dXNlcjpwQHNz"}  # user:p@ss
    connection(monkeypatch, proxies={"https": "http://proxy.example:3128"}, reply=reply)
    monkeypatch.setattr(llm_rating_core.urllib.request, "proxy_bypass", lambda host: True)  # NO_PROXY
    llm_rating_core.Client(FAKE_KEY, 20).beta.messages.create(**llm_rating_core.request_args(items(1)[0], SETTINGS))
    assert FakeConnection.routes[0] == ("api.anthropic.com", None, None)


@pytest.mark.parametrize("proxies, fail", [({"https": "http://proxy.example:3128 "}, None),  # bad port
                                           ({"https": "http://proxy.example:3128"}, "proxy-garbage")])
def test_failures_before_anything_is_sent_are_free(monkeypatch, proxies, fail):
    connection(monkeypatch, proxies=proxies, fail=fail)
    monkeypatch.setattr(llm_rating_core.urllib.request, "proxy_bypass", lambda host: False)
    with pytest.raises(llm_rating_core.APIFailure) as caught:
        llm_rating_core.Client(FAKE_KEY, 20).beta.messages.create(**llm_rating_core.request_args(items(1)[0], SETTINGS))
    assert caught.value.not_sent and llm_rating_core.failure_kind(caught.value) == "busy"
    assert FakeConnection.sent == [] and FAKE_KEY not in str(caught.value)


def test_the_standard_library_client_sends_the_tested_request_and_reads_the_reply(monkeypatch):
    connection(monkeypatch, reply=(200, {"content": [{"type": "text", "text": "{\"rating\": 8}"}],
                                         "stop_reason": "end_turn"}, {}))
    item = items(1)[0]
    response = llm_rating_core.Client(FAKE_KEY, 20).beta.messages.create(**llm_rating_core.request_args(item, SETTINGS))
    assert response.stop_reason == "end_turn" and [b.text for b in response.content] == ['{"rating": 8}']
    host, method, path, body, headers = FakeConnection.sent[0]
    assert (host, method, path) == ("api.anthropic.com", "POST", "/v1/messages?beta=true")
    assert headers["x-api-key"] == FAKE_KEY and headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-beta"] == "server-side-fallback-2026-06-01"
    assert body["model"] == "claude-opus-5-5" and body["system"] == llm_rating_core.SYSTEM
    assert body["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": llm_rating_core.SCHEMA}}
    assert body["messages"][0]["content"] == f"Task features: x=1\n\nTask:\n{item['item_content']}"
    assert "betas" not in body


@pytest.mark.parametrize("behaviour, expected", [
    ({"reply": (401, {"error": {"type": "authentication_error", "message": "invalid x-api-key"}}, {})},
     ("refused", 401, "authentication_error")),
    ({"reply": (429, {"error": {"type": "rate_limit_error", "message": "slow"}}, {"Retry-After": "7"})},
     ("busy", 429, "rate_limit_error")),
    ({"reply": (502, "<html>bad gateway</html>", {})}, ("transient", 502, None)),
    ({"fail": "connect"}, ("busy", None, None)),
    ({"fail": "read"}, ("transient", None, None))])
def test_the_standard_library_client_reports_failures_without_the_key(monkeypatch, behaviour, expected):
    connection(monkeypatch, **behaviour)
    with pytest.raises(Exception) as caught:
        llm_rating_core.Client(FAKE_KEY, 20).beta.messages.create(**llm_rating_core.request_args(items(1)[0], SETTINGS))
    error = caught.value
    assert (llm_rating_core.failure_kind(error), getattr(error, "status_code", None),
            getattr(error, "type", None)) == expected
    assert FAKE_KEY not in str(error) and FAKE_KEY not in repr(error)
    if expected[1] == 429:
        assert error.response.headers["retry-after"] == "7"


def test_no_submission_code_needs_the_anthropic_package():
    for path in [ROOT / "paiec" / "llm_rating_core.py", *(ROOT / "submissions" / "version4").glob("*.py"),
                 *(ROOT / "submissions" / "apicheck").glob("*.py")]:
        assert "import anthropic" not in path.read_text(encoding="utf-8"), path


def load_probe(folder, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(folder)] + sys.path)
    monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items()
                                         if k not in ("llm_rating_core", "model")})
    return importlib.import_module("model")


@pytest.fixture
def probe_folder(tmp_path):
    folder = tmp_path / "apicheck"
    folder.mkdir()
    for name in ("model.py", "llm_rating_core.py", "prediction_config.json"):
        (folder / name).write_bytes((ROOT / "submissions" / "apicheck" / name).read_bytes())
    return folder


@pytest.mark.parametrize("status, expected", [(401, 0.5), (200, 0.1), (403, 0.1), (None, 0.1)])
def test_the_api_check_encodes_the_api_reply_in_its_predictions(probe_folder, monkeypatch, status, expected):
    monkeypatch.setenv("PAIEC_PREDICTION_WORKER_ID", "0")
    probe = load_probe(probe_folder, monkeypatch)
    if status is None:
        connection(monkeypatch, fail="connect")
    else:
        connection(monkeypatch, reply=(status, {"error": {"type": "x", "message": "y"}}, {}))
    start = time.monotonic()
    assert probe.predict(["subject", {}], []) == expected
    assert time.monotonic() - start < probe.DEADLINE + 0.5
    if status == 401:
        assert len(FakeConnection.sent) == 1 and FakeConnection.sent[0][2] == "/v1/messages?beta=true"
        assert FakeConnection.sent[0][4]["x-api-key"] == probe.DUMMY_KEY
    else:
        assert len(FakeConnection.sent) == (0 if status is None else 3)  # three tries
    assert len(list((probe_folder / "api_check_markers").iterdir())) == 1  # whatever the API did


def test_the_api_check_retries_a_quick_failure_and_never_takes_long(probe_folder, monkeypatch):
    probe = load_probe(probe_folder, monkeypatch)
    replies = [False, True]
    monkeypatch.setattr(probe, "_api_once", lambda: replies.pop(0))
    assert probe._api_answers_401() is True  # one blip, then 401
    monkeypatch.setattr(probe, "_api_once", lambda: time.sleep(5) or True)  # a path that hangs
    start = time.monotonic()
    assert probe._api_answers_401(deadline=0.5) is False
    assert time.monotonic() - start < 1


def test_the_api_check_tells_whether_later_checkpoints_see_earlier_files(probe_folder, monkeypatch):
    monkeypatch.setenv("PAIEC_PREDICTION_WORKER_ID", "1")
    probe = load_probe(probe_folder, monkeypatch)
    monkeypatch.setattr(probe, "_api_answers_401", lambda: True)
    assert probe.predict(["s", {}], []) == 0.5  # checkpoint 0 (no labels)
    assert probe.predict(["s", {}], [None] * 8) == 0.5  # a later checkpoint, same folder
    markers_dir = probe_folder / "api_check_markers"
    for marker in markers_dir.iterdir():
        marker.unlink()  # as if the next checkpoint got a fresh copy of the folder
    assert probe.predict(["s", {}], [None] * 24) == 0.4
    for marker in markers_dir.iterdir():
        marker.unlink()
    (markers_dir / "0~p~123~abcd1234").write_text("", encoding="utf-8")  # the parent's validation call
    assert probe.predict(["s", {}], [None] * 56) == 0.45
    monkeypatch.setattr(probe, "_folder", lambda labels: (False, ""))
    probe._codes.clear()
    assert probe.predict(["s", {}], [None] * 120) == 0.2


def test_the_api_check_never_raises(probe_folder, monkeypatch):
    probe = load_probe(probe_folder, monkeypatch)

    def broken(labels):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(probe, "code", broken)
    assert probe.predict(None, None) == 0.3


def test_committed_api_check_uses_the_version4_client_and_settings():
    folder = ROOT / "submissions" / "apicheck"
    assert hashlib.sha256((folder / "llm_rating_core.py").read_bytes()).hexdigest() == VERSION4_RATING_CORE
    assert {p.name for p in folder.iterdir()} == {"model.py", "llm_rating_core.py", "prediction_config.json"}
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import export_version4
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    source = (folder / "model.py").read_text(encoding="utf-8")
    api = export_version4.RATING_API
    assert (f'SETTINGS = {{"model": "{api["model"]}", "effort": "{api["effort"]}", '
            f'"max_chars": {api["max_chars"]}}}') in source


@pytest.mark.parametrize("codes, mix, expected_mixed", [
    ((0.5,) * 6, None, []), ((0.5,) + (0.4,) * 5, None, []), ((0.5,) + (0.45,) * 5, None, []),
    ((0.1,) + (0.5,) * 5, None, []), ((0.2,) * 6, None, []), ((0.5,) * 6, {7: (0.1, 0.3)}, [7])])
def test_the_api_check_feedback_decodes_to_the_codes_it_was_made_from(monkeypatch, codes, mix, expected_mixed):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import decode_apicheck
    from paiec import feedback
    mixed, read, solutions = decode_apicheck.decode(feedback.parse(decode_apicheck.synthetic(codes, mix=mix)))
    assert mixed == expected_mixed
    assert solutions == [tuple(c for b, c in zip(decode_apicheck.BUDGETS, codes) if b in read)]


def test_the_key_check_accepts_both_api_key_styles_and_refuses_admin_keys_and_tokens(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import add_api_key
    body = "Ab3_-" * 19
    for key in (f"sk-ant-api03-{body}", f"sk-ant-usr-{body}", FAKE_KEY):
        assert llm_rating_core.KEY_PATTERN.fullmatch(key), key[:12]
    refused = {"": "nothing was pasted", "sk-ant-": "characters long", f"sk-ant-admin01-{body}": "admin key",
               f"sk-ant-oat01-{body}": "login token", f"sk-ant-api03-{body} x": "spaces",
               f"api03-{body}": "does not start with sk-ant-", "": "does not start with sk-ant-"}
    for key, reason in refused.items():
        assert not llm_rating_core.KEY_PATTERN.fullmatch(key)
        message = add_api_key.why_not(key)
        assert reason in message and (len(key) < 8 or key not in message)
