"""SDK check, not a model: would version 4's rating path work in the scoring environment?

It predicts a constant 0.5 and fails on purpose (at import) unless, in every worker, the
anthropic SDK pinned in requirements.txt imports, the submission folder is writable, and a
request built exactly like version 4's reaches api.anthropic.com through the SDK (it must come
back 401, since the key is a dummy). So the submission's status alone answers the question.
Details go to stderr: Python and SDK versions, the worker and its role, and the marker files
that earlier workers left in the folder. If the folder is shared by the whole run, the count
keeps rising over the six checkpoints and lists earlier roles; if each checkpoint gets its own
copy, it stays small. No credentials are used.
"""

import os
import platform
import sys
import time
import uuid
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
WORKER = os.environ.get("PAIEC_PREDICTION_WORKER_ID", "parent")
ROLE = os.environ.get("PAIEC_STREAM_ROLE", "-")


def _log(message):
    print(f"[sdk-check worker {WORKER}, role {ROLE}] {message}", file=sys.stderr, flush=True)


_log(f"python {platform.python_version()} on {platform.platform()}")
import anthropic  # noqa: E402  (fails the run if requirements.txt was not installed)

_log(f"anthropic {anthropic.__version__}")

MARKERS = FOLDER / "sdk_check_markers"
MARKERS.mkdir(exist_ok=True)
EARLIER = [p.name for p in MARKERS.iterdir()]
(MARKERS / f"{ROLE}~{WORKER}~{os.getpid()}~{uuid.uuid4().hex[:8]}").write_text("", encoding="utf-8")
_log(f"submission folder writable; {len(EARLIER)} marker files from earlier workers, roles "
     f"{sorted({name.split('~')[0] for name in EARLIER})}")

_client = anthropic.Anthropic(api_key="sk-ant-api03-sdk-check-dummy-not-a-real-key", max_retries=0, timeout=20)
_start = time.monotonic()
try:
    _client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=4000,
        betas=["server-side-fallback-2026-06-01"],
        fallbacks=[{"model": "claude-opus-4-8"}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {"rating": {"type": "integer"}},
            "required": ["rating"], "additionalProperties": False}}},
        system="sdk check",
        messages=[{"role": "user", "content": "sdk check"}],
    )
    raise RuntimeError("[sdk-check] a request with a dummy key was accepted")
except anthropic.AuthenticationError:
    _log(f"api.anthropic.com answered 401 through the SDK in {time.monotonic() - _start:.2f}s")


def predict(input, labeled=None):
    return 0.5
