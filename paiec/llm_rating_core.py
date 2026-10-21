"""Item difficulty ratings from Claude for version 4, with a shared cache, a call cap and a time budget.

The prompt, model and request are the ones tested offline (results/llm_rating/rate.py), sent
with only the Python standard library, since the platform does not install packages (Client
sends what the anthropic SDK sends for client.beta.messages.create). The API key is read from
anthropic_api_key.txt in the submission folder, which the author adds to a private copy of the
zip; it is never logged or copied.

Ratings are kept in memory and in one file per item under rating_cache/ in the submission
folder, which the platform's worker processes share (they restart at every checkpoint), so an
item is rated once per run. A lock file per item stops parallel workers from rating the same
item twice (a lock is taken over once its owner process has ended, or after stale_seconds),
and a marker file per API attempt that may be billed caps the total number of such attempts.
Requests run on background daemon threads: a prediction waits at most budget_seconds for them
and then uses whatever ratings are ready. Replies that the API is busy (429, 529) and requests
that never left (no connection) are free and not counted; every worker then pauses. When the
API refuses the key or the account (401, 402, 403, billing errors), no worker sends new
requests, and ratings already obtained are still used. Without a key file, a writable
submission folder there are no ratings, and item_pooling_core predicts exactly as without
them. Lines on stderr say which case applies (the platform hides them from participants).

Version 5 also asks, through the same cache, locks, call cap and failure handling, for one guess
per subject and benchmark of the subject's success rate (guess_core.py). Guesses are sent before
ratings, since they are used from budget 0.
"""

import base64
import concurrent.futures
import hashlib
import http.client
import itertools
import json
import os
import queue
import re
import ssl
import sys
import threading
import time
import types
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

try:
    from . import guess_core
except ImportError:  # loaded as a top-level module from a submission folder
    import guess_core

SYSTEM = """You estimate how hard benchmark tasks are for current AI systems: large language \
models and LLM-based agents released 2023-2026, from small open models to frontier models.

Rate the task you are given on a scale from 1 to 10:
1 = almost every such AI system completes it correctly
5 = about half of them do
10 = almost none of them do

Judge from the task itself: the knowledge it needs, the number of reasoning or action steps, \
how long and intricate the required output is, whether it needs tools, code execution or \
interaction with an environment, and how ambiguous the success criterion is. Do not try to \
solve the task.

Reply with only the integer rating."""

SCHEMA = {"type": "object", "properties": {"rating": {"type": "integer"}},
          "required": ["rating"], "additionalProperties": False}

KEY_FILE = "anthropic_api_key.txt"
# An Anthropic API key (sk-ant-api03-..., sk-ant-usr-..., ...), never an admin key or a login token.
KEY_PATTERN = re.compile(r"sk-ant-(?!admin|oat|ort)[A-Za-z0-9_\-]{40,300}")
ATTEMPTS_PER_ITEM = 2  # per process: failures that may be billed (timeouts, dropped connections, 5xx)
BUSY_PER_ITEM = 10  # per process: free failures (busy replies, requests that never left)
BUSY = (429, 529)  # rate limited or overloaded
REFUSED = (401, 402, 403)  # the key or the account cannot be used
UNRATABLE = (400, 404, 413, 422)  # this request can never work: the item stays unrated
MAX_PAUSE = 60.0
LOCK_CHECK_SECONDS = 0.05  # after creating a lock, read it back to settle simultaneous takeovers


def item_key(item):
    text = f"{item.get('item_features', '')}\x00{item.get('item_content', '')}"
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _log(message):
    worker = os.environ.get("PAIEC_PREDICTION_WORKER_ID", "parent")
    print(f"[version4 worker {worker}] {message}", file=sys.stderr, flush=True)


def _status(error):
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def request_args(item, settings):
    """The rating request for one item, as keyword arguments of client.beta.messages.create."""
    content = str(item.get("item_content", ""))[:settings["max_chars"]]
    features = item.get("item_features", "") or "(none)"
    return dict(
        model=settings["model"],
        max_tokens=4000,
        betas=["server-side-fallback-2026-06-01"],
        fallbacks=[{"model": "claude-opus-4-8"}],
        output_config={"effort": settings["effort"], "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Task features: {features}\n\nTask:\n{content}"}],
    )


class APIFailure(Exception):
    """A request that got no usable reply: the reply's status_code, error type, message and
    headers, or not_sent when no connection could be made (so nothing reached the API)."""

    def __init__(self, message, status_code=None, type=None, headers=None, not_sent=False):
        super().__init__(message)
        self.message, self.status_code, self.type, self.not_sent = message, status_code, type, not_sent
        self.response = types.SimpleNamespace(headers=headers or {})


class Client:
    """client.beta.messages.create with only the standard library: the same POST, path, body
    and API headers as the anthropic SDK, no retries, and timeout seconds per socket operation."""

    HOST = "api.anthropic.com"

    def __init__(self, key, timeout):
        self.key, self.timeout = key, timeout
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self.create))

    def create(self, *, betas=(), **body):
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json", "accept": "application/json"}
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        connection = None
        try:
            try:
                connection = self._connection()
                connection.connect()
            except (OSError, http.client.HTTPException, ValueError) as error:  # nothing was sent yet:
                # DNS, refused, unreachable, TLS, timeout, a bad proxy address or proxy reply
                raise APIFailure(f"no connection ({type(error).__name__})", not_sent=True) from None
            connection.request("POST", "/v1/messages?beta=true", body=json.dumps(body).encode("utf-8"),
                               headers=headers)
            response = connection.getresponse()
            data = response.read()
            reply_headers = {name.lower(): value for name, value in response.getheaders()}
        finally:
            if connection is not None:
                connection.close()
        try:
            reply = json.loads(data.decode("utf-8"))
        except ValueError:
            reply = None
        if response.status != 200:
            error = reply.get("error") if isinstance(reply, dict) else None
            error = error if isinstance(error, dict) else {}
            raise APIFailure(str(error.get("message") or f"HTTP {response.status}"), response.status,
                             error.get("type"), reply_headers)
        blocks = [types.SimpleNamespace(type=block.get("type"), text=block.get("text", ""))
                  for block in reply["content"]]
        return types.SimpleNamespace(stop_reason=reply.get("stop_reason"), content=blocks)

    def _connection(self):
        """A connection to the API, through the HTTPS proxy that the environment names (as
        urllib and the SDK would use it), unless the API host is exempt from it."""
        context = ssl.create_default_context()
        proxies = urllib.request.getproxies()
        proxy = proxies.get("https") or proxies.get("all")
        if not proxy or urllib.request.proxy_bypass(self.HOST):
            return http.client.HTTPSConnection(self.HOST, timeout=self.timeout, context=context)
        parts = urllib.parse.urlsplit(proxy if "://" in proxy else f"http://{proxy}")
        headers = {}
        if parts.username is not None:
            user = f"{urllib.parse.unquote(parts.username)}:{urllib.parse.unquote(parts.password or '')}"
            headers["Proxy-Authorization"] = "Basic " + base64.b64encode(user.encode("utf-8")).decode("ascii")
        connection = http.client.HTTPSConnection(parts.hostname, parts.port or 80, timeout=self.timeout,
                                                 context=context)
        connection.set_tunnel(self.HOST, 443, headers)
        return connection


def failure_kind(error):
    """How to treat a failed request: "busy" (free: pause, then retry), "refused" (free: no new
    requests in any worker), "unratable" (free: the item stays unrated) or "transient" (may be
    billed: retry once)."""
    status = _status(error)
    text = str(getattr(error, "message", error)).lower()
    if status in BUSY or getattr(error, "not_sent", False):
        return "busy"
    if (status in REFUSED or getattr(error, "type", None) == "billing_error"
            or (status == 400 and ("credit balance" in text or "usage limit" in text))):
        return "refused"
    if status in UNRATABLE:
        return "unratable"
    return "transient"


class Rater:
    """lookup(item) gives a known rating in 1..10 or None; request(items, wait) fetches ratings.
    lookup_guess(subject, item) and request_jobs(jobs, wait) do the same for subject guesses."""

    GUESS, RATING = 0, 1  # queue priorities: guesses first

    def __init__(self, folder, settings):
        self.folder = Path(folder)
        self.settings = settings
        self.memory = {}  # key -> rating, or None when the item cannot be rated
        self.pending = {}  # key -> future of a request in progress in this process
        self.given_up = set()  # not requested again by this process; other workers may rate them
        self.failures = {}
        self.busy = {}
        self.counts = {"attempts": 0, "rated": 0, "failed": 0, "busy": 0}
        self.lock = threading.Lock()
        self.pause_until = 0.0
        self.active = self.stopped = self.capped = False
        self.client = self.cache = self.queue = None
        try:
            key = (self.folder / KEY_FILE).read_text(encoding="utf-8").strip()
        except Exception:
            key = ""
        if not key:
            _log("no API key file: predicting as version 3")
            return
        if not KEY_PATTERN.fullmatch(key):
            _log("the API key file does not hold a regular Anthropic API key: predicting as version 3")
            return
        self.cache = self._cache_dir()
        if self.cache is None:
            _log("submission folder not writable: predicting as version 3")
            return
        if (self.cache / "stopped").exists():
            self.active = self.stopped = True
            _log("the API refused this key or account earlier in this run: cached ratings only")
            return
        try:
            self.client, version = self._make_client(key)
        except Exception as error:
            _log(f"no API client ({type(error).__name__}): predicting as version 3")
            return
        self.queue = queue.PriorityQueue()
        self.order = itertools.count()  # first in, first out within a priority
        for _ in range(self.settings["threads"]):
            threading.Thread(target=self._work, daemon=True).start()  # daemon: never delays exit
        self.active = True
        _log(f"ratings on ({version} client, role {os.environ.get('PAIEC_STREAM_ROLE', '-')}, "
             f"{self._calls()} earlier API attempts in this folder)")

    def _make_client(self, key):
        return Client(key, self.settings["timeout"]), "standard library"

    def _cache_dir(self):
        try:
            path = self.folder / "rating_cache"
            (path / "calls").mkdir(parents=True, exist_ok=True)
            probe = path / f".probe-{uuid.uuid4().hex}"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return path
        except Exception:
            return None

    def _calls(self):
        try:
            return len(os.listdir(self.cache / "calls"))
        except Exception:
            return 0

    @staticmethod
    def rating_job(item):
        return item_key(item), ("rating", item)

    @staticmethod
    def guess_job(subject, item):
        """The guess for this subject on the item's benchmark, asked with this item as example."""
        return "guess-" + guess_core.guess_key(subject, item), ("guess", subject, item)

    def lookup(self, item):
        """The rating if already known in this process or on disk; never waits."""
        return self._lookup_key(item_key(item))

    def lookup_guess(self, subject, item):
        """The subject's guessed success rate on the item's benchmark if known, else None."""
        value = self._lookup_key(self.guess_job(subject, item)[0])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            return None
        return value

    def _lookup_key(self, key):
        if not self.active:
            return None
        with self.lock:
            if key in self.memory:
                return self.memory[key]
        found, rating = self._read(key)
        if found:
            self._remember(key, rating)
        return rating

    def request(self, items, wait):
        """Start fetching the ratings not known yet, and wait at most wait seconds for them."""
        self.request_jobs([self.rating_job(item) for item in items], wait)

    def request_jobs(self, jobs, wait):
        """Start fetching what jobs ((key, job) from rating_job or guess_job) ask for, if not known
        yet, and wait at most wait seconds for them."""
        if not self.active or self.queue is None:
            return
        futures = []
        for key, job in jobs:
            with self.lock:
                if key in self.memory:
                    continue
                if key in self.pending:  # requested by an earlier prediction: wait for it too
                    futures.append(self.pending[key])
                    continue
            found, rating = self._read(key)  # rated by another worker or an earlier checkpoint
            if found:
                self._remember(key, rating)
                continue
            with self.lock:
                if self.stopped or self.capped or key in self.given_up:
                    continue
                future = self.pending[key] = concurrent.futures.Future()
            priority = self.GUESS if job[0] == "guess" else self.RATING
            self.queue.put((priority, next(self.order), future, key, job))
            futures.append(future)
        if futures and wait > 0:
            concurrent.futures.wait(futures, timeout=wait)

    def _work(self):
        while True:
            _, _, future, key, job = self.queue.get()
            try:
                self._fetch(key, job)
            except Exception:
                pass
            finally:
                with self.lock:
                    self.pending.pop(key, None)
                future.set_result(None)

    def _read(self, key):
        try:
            return True, json.loads((self.cache / f"{key}.json").read_text(encoding="utf-8"))["rating"]
        except Exception:
            return False, None

    def _write(self, key, rating):
        partial = self.cache / f".{key}.{uuid.uuid4().hex}.tmp"
        partial.write_text(json.dumps({"rating": rating}), encoding="utf-8")
        os.replace(partial, self.cache / f"{key}.json")

    def _remember(self, key, rating):
        with self.lock:
            self.memory[key] = rating

    @staticmethod
    def _token(lock):
        try:
            return lock.read_text(encoding="utf-8")
        except Exception:
            return None

    def _take_lock(self, key):
        """This thread's token if it now holds the item's lock, else None. An abandoned lock is
        taken over; when two workers take one over at once, the token read back after a short
        delay is the last one written, and only that worker keeps the lock."""
        lock = self.cache / f"{key}.lock"
        token = f"{os.getpid()} {uuid.uuid4().hex}"
        for _ in range(2):
            try:
                handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(handle, token.encode())
                finally:
                    os.close(handle)
                time.sleep(LOCK_CHECK_SECONDS)
                return token if self._token(lock) == token else None
            except FileExistsError:
                try:
                    if not self._abandoned(lock):
                        return None
                    lock.unlink()  # its owner was killed or gave up: take it over
                except FileNotFoundError:
                    continue
                except Exception:
                    return None
            except Exception:
                return None
        return None

    def _release(self, key, token):
        lock = self.cache / f"{key}.lock"
        if self._token(lock) == token:  # never remove a lock another worker has taken over
            lock.unlink(missing_ok=True)

    def _abandoned(self, lock):
        """True if the lock is older than stale_seconds or, on POSIX, its owner process has
        ended (workers are killed at the end of every checkpoint)."""
        if time.time() - lock.stat().st_mtime >= self.settings["stale_seconds"]:
            return True
        if os.name != "posix":  # os.kill(pid, 0) would end the process on Windows
            return False
        try:
            pid = int(lock.read_text(encoding="utf-8").split()[0])
            if pid == os.getpid():
                return False
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except Exception:
            pass
        return False

    def _wait_for_other(self, key):
        """While another worker holds the item's lock: True once its rating is written or the
        lock is gone or abandoned, False if that has not happened within stale_seconds."""
        lock = self.cache / f"{key}.lock"
        deadline = time.monotonic() + self.settings["stale_seconds"]
        while time.monotonic() < deadline:
            time.sleep(0.25)
            if self._read(key)[0]:
                return True
            try:
                if self._abandoned(lock):
                    return True
            except FileNotFoundError:
                return True
            except Exception:
                return False
        return False

    def _pause(self):
        """Sleep while the API has asked every worker to wait."""
        try:
            until = float((self.cache / "pause_until").read_text(encoding="utf-8"))
        except Exception:
            until = 0.0
        with self.lock:
            until = max(until, self.pause_until)
        delay = min(until - time.time(), MAX_PAUSE)
        if delay > 0:
            time.sleep(delay)

    def _set_pause(self, error):
        try:
            seconds = float(error.response.headers.get("retry-after", 10))
        except Exception:
            seconds = 10.0
        until = time.time() + min(max(seconds, 1.0), MAX_PAUSE)
        with self.lock:
            self.pause_until = max(self.pause_until, until)
        try:
            partial = self.cache / f".pause.{uuid.uuid4().hex}.tmp"
            partial.write_text(repr(until), encoding="utf-8")
            os.replace(partial, self.cache / "pause_until")
        except Exception:
            pass

    def _fetch(self, key, job):
        for _ in range(3):
            self._pause()
            found, rating = self._read(key)
            if found:
                return self._remember(key, rating)
            token = self._take_lock(key)
            if token:
                break
            if not self._wait_for_other(key):  # another worker is rating it
                return None  # not remembered: a later request can try again
        else:
            return None
        outcome = "transient"
        try:
            found, rating = self._read(key)  # rated just before this thread took the lock
            outcome = rating if found else self._ask(job)
            if outcome not in ("busy", "transient", "capped", "stopped"):
                if not found:
                    self._write(key, outcome)
                self._remember(key, outcome)
        finally:
            self._release(key, token)  # done, or let others retry later
        with self.lock:
            if outcome == "transient":
                self.failures[key] = self.failures.get(key, 0) + 1
                if self.failures[key] >= ATTEMPTS_PER_ITEM:
                    self.given_up.add(key)
            elif outcome == "busy":
                self.busy[key] = self.busy.get(key, 0) + 1
                if self.busy[key] >= BUSY_PER_ITEM:
                    self.given_up.add(key)
        return None

    def _ask(self, job):
        """A rating (or a guess, for a guess job); None if it cannot be had; "busy", "transient",
        "capped" or "stopped" if none was obtained this time."""
        if self.stopped or (self.cache / "stopped").exists():
            self.stopped = True
            return "stopped"
        try:
            if self._calls() >= self.settings["max_calls"]:
                with self.lock:
                    first, self.capped = not self.capped, True
                if first:
                    _log(f"call cap of {self.settings['max_calls']} reached: no new ratings")
                return "capped"
            marker = self.cache / "calls" / uuid.uuid4().hex
            marker.write_text("", encoding="utf-8")
        except Exception:
            return "transient"
        self._count("attempts")
        try:
            if job[0] == "guess":
                args = guess_core.request_args(job[1], job[2], self.settings)
            else:
                args = request_args(job[1], self.settings)
            response = self.client.beta.messages.create(**args)
        except Exception as error:
            kind = failure_kind(error)
            if kind != "transient":
                marker.unlink(missing_ok=True)  # refused or never sent: not billed, not counted
            if kind == "busy":
                self._set_pause(error)
                self._count("busy")
                return "busy"
            self._count("failed")
            if kind == "refused":
                self._stop(error)
                return "stopped"
            return None if kind == "unratable" else "transient"
        try:
            if job[0] == "guess":
                text = "".join(block.text for block in response.content if block.type == "text")
                guess = guess_core.parse(text, response.stop_reason)
                self._count("rated")
                return guess
            if response.stop_reason == "refusal":
                return None
            text = "".join(block.text for block in response.content if block.type == "text")
            rating = int(json.loads(text)["rating"])
            self._count("rated")
            return rating if 1 <= rating <= 10 else None
        except Exception:
            return None

    def _stop(self, error):
        status = _status(error)
        reason = f"HTTP {status}" if status else type(error).__name__
        with self.lock:
            first, self.stopped = not self.stopped, True
        if first:
            _log(f"the API refused this key or account ({reason}): no new ratings, cached ones still used")
        try:
            (self.cache / "stopped").write_text(reason, encoding="utf-8")
        except Exception:
            pass

    def _count(self, name):
        with self.lock:
            self.counts[name] += 1
            counts = dict(self.counts)
        if name == "attempts" and counts["attempts"] % 100 == 0:
            _log(f"{counts['attempts']} API attempts: {counts['rated']} rated, {counts['busy']} busy, "
                 f"{counts['failed']} failed")


def benchmark_subjects(input, labeled):
    """The subject to predict and every other subject with labels on its benchmark, once each."""
    subject, item = input
    benchmark = item.get("benchmark_id", "")
    subjects, seen = [subject], {guess_core.guess_key(subject, item)}
    for entry in labeled:
        try:
            (other, other_item), _ = entry
            if other_item.get("benchmark_id", "") != benchmark or not isinstance(other, dict):
                continue
            key = guess_core.guess_key(other, item)
            if key not in seen:
                seen.add(key)
                subjects.append(other)
        except Exception:
            continue
    return subjects


def benchmark_items(input, labeled):
    """The item to predict and every labeled item on its benchmark."""
    subject, item = input
    benchmark = item.get("benchmark_id", "")
    items = [item]
    for entry in labeled:
        try:
            (_, other), _ = entry
            if other.get("benchmark_id", "") == benchmark:
                items.append(other)
        except Exception:
            continue
    return items
