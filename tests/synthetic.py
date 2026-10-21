"""A small synthetic dataset shared by the tests."""

import random

import pandas as pd

from paiec.data import SUBJECT_FIELDS, Dataset, alias


def make_dataset(seed=0, duplicate_content=False):
    """Benchmark 'train' plus held-out 'held' with pairs of 101, 90 and 50 items; s1 is in both."""
    rng = random.Random(seed)
    rows, subjects, items = [], {}, {}
    specs_by_bench = {"train": [("t1", 90), ("s1", 85)],
                      "held": [("s1", 101), ("s2", 90), ("s3", 50)]}
    for bench, specs in specs_by_bench.items():
        for k in range(120):
            content = f"{bench} question {k % 10 if duplicate_content else k}"
            items[f"{bench}-{k}"] = {"item_content": content, "item_features": "",
                                     "benchmark_id": alias("benchmark", bench)}
        for subject_id, n_items in specs:
            subjects[subject_id] = {f: "" for f in SUBJECT_FIELDS} | {"normalized_name": subject_id}
            for k in range(n_items):
                for trial in range(1, rng.choice([1, 1, 2, 3]) + 1):
                    rows.append({"benchmark": bench, "subject_id": subject_id,
                                 "item_id": f"{bench}-{k}", "interactors": "", "trial": trial,
                                 "label": rng.randint(0, 1)})
    return Dataset(pd.DataFrame(rows), subjects, items)


def with_metadata(dataset):
    """Give the synthetic subjects some varied metadata, including malformed dates."""
    values = {"t1": ("OpenAI", "2025-03-01", "high"), "s1": ("Anthropic", "2024-06", ""),
              "s2": ("OpenAI", "not a date", "low"), "s3": ("", "", "")}
    for subject_id, (provider, release, effort) in values.items():
        dataset.subjects[subject_id].update(provider=provider, release_date=release,
                                            reasoning_effort=effort)
    return dataset
