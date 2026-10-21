"""Load the public measurement-db release and present it in competition format.

Inputs mirror the runtime: a subject dict with the 8 subject fields and an item dict
with item_content, item_features, interactors and an anonymous benchmark_id. Missing
values become "". Only items whose declared response scale is exactly {0, 1} are kept,
selected by scale and never by observed value.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "aims-foundations/measurement-db"
REVISION = "bc8204d811823da849c6686bf124d4ca9f82e4de"
BENCHMARKS = ("matharena", "mmdocrag", "multi_swebench", "real_webagents",
              "researchcodebench", "swe_rebench")
TABLES = ("benchmarks", "items", "response", "subjects")
DATA_DIR = ROOT / "data" / "measurement-db" / REVISION
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version", "subject_features_extra")
OBSERVATION_KEY = ["subject_id", "item_id", "trial", "test_condition", "interactors"]


def alias(prefix, key):
    """Stable anonymous id such as benchmark_274926, like the platform's aliases."""
    digest = int(hashlib.sha256(f"paiec-harness|{prefix}|{key}".encode()).hexdigest(), 16)
    return f"{prefix}_{100000 + digest % 900000}"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(data_dir=None):
    """Fetch the pinned tables and write a manifest with their SHA-256 hashes."""
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir or DATA_DIR)
    files = {}
    for slug in BENCHMARKS:
        for table in TABLES:
            name = f"{slug}/{table}.parquet"
            path = Path(hf_hub_download(REPO_ID, name, repo_type="dataset",
                                        revision=REVISION, local_dir=data_dir))
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {"repo_id": REPO_ID, "revision": REVISION, "files": files}
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                            encoding="utf-8", newline="\n")
    return manifest


def verify(data_dir=None, benchmarks=BENCHMARKS):
    """Check the downloaded files against manifest.json; returns the manifest."""
    data_dir = Path(data_dir or DATA_DIR)
    path = data_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/download_data.py first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION:
        raise ValueError(f"{path} is for revision {manifest['revision']}, expected {REVISION}")
    for slug in benchmarks:
        for table in TABLES:
            name = f"{slug}/{table}.parquet"
            if _sha256(data_dir / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"{data_dir / name} does not match manifest.json")
    return manifest


def _clean(value, strip=True):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip() if strip else str(value)


def _is_binary_scale(scale):
    return scale.get("kind") == "discrete" and sorted(scale.get("values", [])) == [0.0, 1.0]


def _read(slug, table, data_dir):
    path = Path(data_dir) / slug / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/download_data.py first")
    metadata = pq.read_schema(path).metadata or {}
    if metadata.get(b"measurement_db.schema_version") != b"3":
        raise ValueError(f"{path} is not measurement-db schema version 3")
    return pd.read_parquet(path)


def _check_unique(frame, columns, what):
    duplicated = frame[columns].astype("string").fillna("<NA>").duplicated()
    if duplicated.any():
        raise ValueError(f"{what}: {int(duplicated.sum())} duplicated keys {columns}")


@dataclass
class Dataset:
    """Binary responses plus the competition-format subject and item dictionaries.

    responses has one row per recorded response: benchmark, subject_id, item_id,
    interactors, trial and an int label. subjects maps subject_id to the subject dict;
    items maps item_id to the item dict without interactors, which live on responses.
    """

    responses: pd.DataFrame
    subjects: dict
    items: dict

    def make_input(self, subject_id, item_id, interactors=""):
        """A fresh [subject, item] pair with exactly the platform's keys."""
        subject, item = self.subjects[subject_id], self.items[item_id]
        return [{f: subject[f] for f in SUBJECT_FIELDS},
                {"item_content": item["item_content"], "item_features": item["item_features"],
                 "interactors": interactors, "benchmark_id": item["benchmark_id"]}]

    def subset(self, benchmarks):
        """Only the given benchmarks, with copied dicts, so training cannot touch test inputs."""
        responses = self.responses[self.responses["benchmark"].isin(benchmarks)]
        responses = responses.reset_index(drop=True)
        return Dataset(responses,
                       {s: dict(self.subjects[s]) for s in responses["subject_id"].unique()},
                       {i: dict(self.items[i]) for i in responses["item_id"].unique()})


def load_benchmark(slug, data_dir=None):
    """Binary-scale responses of one benchmark, after integrity checks."""
    tables = {table: _read(slug, table, Path(data_dir or DATA_DIR)) for table in TABLES}
    bench, items, resp, subj = (tables[t] for t in TABLES)

    for name in ("benchmarks", "items", "response"):
        if set(tables[name]["benchmark_id"].unique()) != {slug}:
            raise ValueError(f"{slug}/{name}: unexpected benchmark_id values")
    _check_unique(subj, ["subject_id"], f"{slug}/subjects")
    _check_unique(items, ["item_id"], f"{slug}/items")
    _check_unique(resp, ["response_id"], f"{slug}/response")
    _check_unique(resp, OBSERVATION_KEY, f"{slug}/response")
    if not resp["subject_id"].isin(subj["subject_id"]).all():
        raise ValueError(f"{slug}: responses reference unknown subjects")
    if not resp["item_id"].isin(items["item_id"]).all():
        raise ValueError(f"{slug}: responses reference unknown items")

    bench_scale = json.loads(bench["response_scale"].iloc[0])
    scales = items["grading_criterion"].map(
        lambda g: json.loads(g).get("response_scale", bench_scale))
    binary_items = items[scales.map(_is_binary_scale)]
    resp = resp[resp["item_id"].isin(binary_items["item_id"])]
    if resp["response"].isna().any() or not resp["response"].isin([0.0, 1.0]).all():
        raise ValueError(f"{slug}: binary-scale items have null or non-0/1 responses")

    benchmark_alias = alias("benchmark", slug)
    responses = pd.DataFrame({
        "benchmark": slug,
        "subject_id": resp["subject_id"].to_numpy(),
        "item_id": resp["item_id"].to_numpy(),
        "interactors": [_clean(v) for v in resp["interactors"]],
        "trial": resp["trial"].astype("int64").to_numpy(),
        "label": resp["response"].astype("int64").to_numpy(),
    })
    subjects = {row.subject_id: {f: _clean(getattr(row, f)) for f in SUBJECT_FIELDS}
                for row in subj[subj["subject_id"].isin(responses["subject_id"])].itertuples()}
    item_dicts = {row.item_id: {"item_content": _clean(row.content, strip=False),
                                "item_features": _clean(row.item_features),
                                "benchmark_id": benchmark_alias}
                  for row in binary_items.itertuples()}
    return Dataset(responses, subjects, item_dicts)


def load(data_dir=None, benchmarks=BENCHMARKS):
    """All requested benchmarks merged, after checking file hashes against the manifest.

    Benchmarks without binary-scale items drop out.
    """
    verify(data_dir, benchmarks)
    parts = [load_benchmark(slug, data_dir) for slug in benchmarks]
    subjects, items = {}, {}
    for part in parts:
        for subject_id, fields in part.subjects.items():
            if subjects.setdefault(subject_id, fields) != fields:
                raise ValueError(f"subject {subject_id} has conflicting fields across benchmarks")
        items.update(part.items)
    responses = pd.concat([p.responses for p in parts], ignore_index=True)
    return Dataset(responses, subjects, items)
