"""Make a private copy of a submission zip that contains an Anthropic API key.

    python scripts/add_api_key.py dist/version4.zip

Run this yourself. It asks for the key without showing it, checks that it looks like an
Anthropic API key (not an admin key or a login token), checks with a free request (the list of
models) that the API accepts it and offers the model the submission uses, and writes
<name>_private.zip in your user folder (or --out): the original files plus
anthropic_api_key.txt. The key is not printed or stored anywhere else. Upload the private zip
to Codabench only, never share it and never make the submission public. Spending limits are
set per workspace, not per key: create the key in a workspace of its own with a low monthly
spend limit, and revoke it once the final results are published (organizers may rerun
submissions until then).
"""

import argparse
import getpass
import http.client
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paiec.llm_rating_core import KEY_FILE, KEY_PATTERN, Client  # noqa: E402


def why_not(key):
    """Why key fails KEY_PATTERN, without showing it (only its length and its public prefix)."""
    if not key:
        return "nothing was pasted (paste with a right-click or Ctrl+V, then press Enter)"
    if not key.startswith("sk-ant-"):
        return f"it does not start with sk-ant- ({len(key)} characters were pasted)"
    if key.startswith(("sk-ant-admin", "sk-ant-oat", "sk-ant-ort")):
        return "it is an admin key or a login token; create a key in the workspace's API Keys tab"
    if not all(c.isascii() and (c.isalnum() or c in "_-") for c in key):
        return "it contains spaces or other characters that keys do not have; copy it again"
    return f"it is {len(key)} characters long, which does not fit an API key"


def check_key(key, model):
    """None if the API accepts the key and lists model, else the reason (never the key)."""
    connection = None
    try:
        connection = Client(key, 30)._connection()  # the same route as the submission's requests
        connection.request("GET", "/v1/models?limit=1000",
                           headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        response = connection.getresponse()
        status, data = response.status, response.read()
    except (OSError, http.client.HTTPException, ValueError) as error:
        return f"could not reach api.anthropic.com ({type(error).__name__})"
    finally:
        if connection is not None:
            connection.close()
    if status in (401, 403):
        return f"the API rejected this key (HTTP {status})"
    if status != 200:
        return f"the API answered HTTP {status}"
    try:
        models = {entry["id"] for entry in json.loads(data.decode("utf-8"))["data"]}
    except (ValueError, KeyError, TypeError):
        return "the API's list of models could not be read"
    if model not in models:
        return f"this key cannot use {model}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip", type=Path, help="the zip built by scripts/build_submission.py")
    parser.add_argument("--out", type=Path, default=None, help="folder for the private zip")
    args = parser.parse_args()

    source = args.zip.resolve()
    target = (args.out or Path.home()).resolve() / f"{source.stem}_private.zip"
    with zipfile.ZipFile(source) as original:
        if KEY_FILE in original.namelist():
            sys.exit(f"{source.name} already contains {KEY_FILE}")
        entries = [(info, original.read(info)) for info in original.infolist()]
        model = json.loads(original.read("params.json"))["rating_api"]["model"]

    key = getpass.getpass("Paste the API key for this submission (it will not be shown): ").strip()
    if not KEY_PATTERN.fullmatch(key):
        sys.exit(f"That is not an Anthropic API key: {why_not(key)}. Nothing was written.")
    problem = check_key(key, model)
    if problem:
        sys.exit(f"Key check failed: {problem}. Nothing was written.")
    print(f"Key check passed: the API accepts the key and offers {model}.")

    handle = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # owner only from the start
    os.chmod(target, 0o600)  # also when the file already existed
    with os.fdopen(handle, "wb") as raw, zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as private:
        for info, data in entries:
            private.writestr(info, data)
        info = zipfile.ZipInfo(KEY_FILE, entries[0][0].date_time)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        info.create_system = 3
        private.writestr(info, key + "\n")
    print(f"Wrote {target} with {KEY_FILE}. Upload it to Codabench only; never share or publish it.")


if __name__ == "__main__":
    main()
