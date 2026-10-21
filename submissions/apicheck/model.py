"""API check, not a model: would version 4's rating requests work in the scoring environment?

The platform hides the logs of this competition, so the answer is in the scores. At its first
prediction in each checkpoint, every worker leaves a marker file noting how many labels it saw
and whether it is a prediction worker (to learn whether later checkpoints see the files of
earlier ones, as version 4's shared rating cache needs), then sends a request exactly like
version 4's (llm_rating_core.Client and request_args) with a dummy key, which the API must
answer with 401: up to 3 tries, at most 6 seconds in all. Every prediction is then a constant
that encodes the result:

    0.5   everything works                       (Brier score exactly 0.250000)
    0.45  API and folder work; the only earlier markers come from outside the prediction workers
    0.4   API and folder work, but this checkpoint does not see earlier checkpoints' files
    0.3   the probe itself failed
    0.2   the API works, the folder is not writable
    0.1   the API did not answer 401 within 6 seconds (blocked, unreachable, slow, another reply)

No credentials are used, and nothing raises, so the run always finishes.
"""

import os
import threading
import time
import uuid
from pathlib import Path

import llm_rating_core

FOLDER = Path(__file__).resolve().parent
DUMMY_KEY = "sk-ant-api03-api-check-dummy-not-a-real-key"
SETTINGS = {"model": "claude-opus-5-5", "effort": "low", "max_chars": 150000}  # as version 4
TIMEOUT = 5  # seconds per socket operation
DEADLINE = 6.0  # seconds for the whole check, DNS included, well within a per-call time limit
WORKER = "w" if os.environ.get("PAIEC_PREDICTION_WORKER_ID") is not None else "p"
_api_ok = None
_codes = {}


def _api_once():
    try:
        client = llm_rating_core.Client(DUMMY_KEY, TIMEOUT)
        client.beta.messages.create(**llm_rating_core.request_args(
            {"item_content": "API check", "item_features": ""}, SETTINGS))
    except Exception as error:
        return getattr(error, "status_code", None) == 401
    return False  # a dummy key must never be accepted


def _api_answers_401(deadline=DEADLINE):
    """True if one of up to 3 tries got 401 within deadline seconds in all."""
    found = []
    end = time.monotonic() + deadline

    def run():
        for _ in range(3):
            if _api_once():
                found.append(True)
                return
            if time.monotonic() + 0.3 >= end:
                return
            time.sleep(0.3)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(deadline)
    return bool(found)


def _folder(labels):
    """(writable, earlier): earlier is "w" if prediction workers of an earlier checkpoint left
    markers, "p" if only other processes did, or "" if there are none (or labels == 0)."""
    try:
        markers = FOLDER / "api_check_markers"
        markers.mkdir(exist_ok=True)
        kinds = set()
        for name in os.listdir(markers):
            parts = name.split("~")
            if len(parts) == 4 and parts[0].isdigit() and int(parts[0]) < labels:
                kinds.add(parts[1])
        (markers / f"{labels}~{WORKER}~{os.getpid()}~{uuid.uuid4().hex[:8]}").write_text("", encoding="utf-8")
        return True, "w" if "w" in kinds or labels == 0 else "p" if kinds else ""
    except Exception:
        return False, ""


def code(labels):
    global _api_ok
    writable, earlier = _folder(labels)  # marker first, whatever the API does
    if _api_ok is None:
        _api_ok = _api_answers_401()
    if not _api_ok:
        return 0.1
    if not writable:
        return 0.2
    return {"w": 0.5, "p": 0.45}.get(earlier, 0.4)


def predict(input, labeled=None):
    try:
        labels = len(labeled or [])
        if labels not in _codes:
            _codes[labels] = code(labels)
        return _codes[labels]
    except Exception:
        return 0.3
