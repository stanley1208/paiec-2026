"""Simulate the hosted streaming protocol with leave-one-benchmark-out folds.

For a held-out benchmark, every subject with enough distinct items forms a pair. Each
pair's items are split 50/50 into an acquisition stream and an evaluation pool (odd
counts give the extra item to acquisition). Labels are acquired up to the caps
1, 3, 7, 15, 31; all pairs finish a cap before any pair continues. At each cap the
predictor sees frozen evidence and predicts every distinct evaluation input; every
recorded response of the evaluation items is scored.

An acquired item reveals one of its recorded responses, chosen uniformly with a seeded
RNG, and counts as one label. Items are split by item_id, as on the platform, so items
with identical visible content can land in both pools.
"""

import copy
import hashlib
import json
import math
import numbers
import random
from dataclasses import dataclass

from . import scoring
from .data import Dataset, alias
from .kit import AcquisitionHook, prediction_input_key

MAX_LABELS = 31


@dataclass(frozen=True)
class Settings:
    seed: int = 0
    min_pair_items: int = 80
    max_pair_items: int | None = 1000  # the platform draws pairs under a 1,000-item cap
    # The platform passes every label bought across the sampled pairs. "pair" gives only
    # the pair's own labels, which is exactly what a predictor that filters evidence to its
    # own subject and benchmark sees; "benchmark" adds the other pairs of the fold, so a
    # correctly filtering predictor must score identically under both. "benchmark" with
    # strict_copies is slow (evidence copies grow with the number of pairs squared).
    # "partners" adds the labels of `partners` other pairs of the fold, drawn per pair with a
    # seeded RNG, like the few subjects a Codabench draw puts on one benchmark.
    evidence: str = "pair"
    partners: int = 0
    split: str = "independent"  # "independent" per pair, or "shared" item order per benchmark
    strict_copies: bool = True  # deep-copy evidence for every predict call, like the platform
    # Keep only this many randomly chosen items of the held-out benchmark, so pairs are about
    # as small as on Codabench and subjects share items as in a small benchmark.
    pool_items: int | None = None

    def __post_init__(self):
        if self.evidence not in ("pair", "benchmark", "partners"):
            raise ValueError(f"unknown evidence setting {self.evidence!r}")
        if self.split not in ("independent", "shared"):
            raise ValueError(f"unknown split setting {self.split!r}")
        if self.partners < 0 or (self.partners and self.evidence != "partners"):
            raise ValueError("partners needs evidence='partners' and a count >= 0")


@dataclass
class Pair:
    benchmark: str
    subject_id: str
    n_items: int
    stream: list  # acquisition items in candidate order
    eval_items: list
    responses: dict  # item_id -> [(interactors, label), ...] in trial order


def _rng(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def random_query(context, current_input):
    """Default policy without labeling.py, as in the kit's streaming_ingestion.run_streaming."""
    key = json.dumps([context, current_input], sort_keys=True, separators=(",", ":"))
    uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
    return uniform < min(1, context["labels_remaining"] / context["items_remaining"])


def check_probability(value):
    """At least as strict as the platform: a real number in [0, 1], not bool, NaN or inf."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"predict() returned {value!r}, not a number")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"predict() returned {value!r}, not a probability in [0, 1]")
    return number


def _split(benchmark, subject_id, items, settings):
    if settings.split == "independent":
        order = list(items)
        _rng(settings.seed, benchmark, subject_id, "split").shuffle(order)
    else:
        rank = {i: _rng(settings.seed, benchmark, i, "rank").random() for i in items}
        order = sorted(items, key=lambda i: (rank[i], i))
    n_acquisition = (len(order) + 1) // 2
    return order[:n_acquisition], order[n_acquisition:]


def _item_pool(dataset, benchmark, settings):
    """The dataset with the held-out benchmark cut to settings.pool_items random items."""
    responses = dataset.responses
    held = responses["benchmark"] == benchmark
    items = sorted(responses.loc[held, "item_id"].unique())
    keep = _rng(settings.seed, benchmark, "pool").sample(items, min(settings.pool_items, len(items)))
    kept = responses[~held | responses["item_id"].isin(keep)].reset_index(drop=True)
    return Dataset(kept, dataset.subjects, dataset.items)


def eligible_pairs(dataset, benchmark, settings):
    responses = dataset.responses[dataset.responses["benchmark"] == benchmark]
    pairs = []
    for subject_id, rows in sorted(responses.groupby("subject_id"), key=lambda kv: kv[0]):
        items = sorted(rows["item_id"].unique())
        if len(items) < settings.min_pair_items:
            continue
        if settings.max_pair_items is not None and len(items) > settings.max_pair_items:
            continue
        acquisition, evaluation = _split(benchmark, subject_id, items, settings)
        stream = list(acquisition)
        _rng(settings.seed, benchmark, subject_id, "stream").shuffle(stream)
        by_item = {}
        for row in rows.sort_values(["item_id", "trial"]).itertuples():
            by_item.setdefault(row.item_id, []).append((row.interactors, int(row.label)))
        pairs.append(Pair(benchmark, subject_id, len(items), stream, evaluation, by_item))
    return pairs


def _evaluation_targets(dataset, pair):
    """Distinct inputs (deduplicated with the platform's key) and every scored row."""
    inputs, index, rows, keys = [], {}, [], {}
    for item_id in sorted(pair.eval_items):
        for interactors, label in pair.responses[item_id]:
            if (item_id, interactors) not in keys:
                keys[item_id, interactors] = prediction_input_key(
                    dataset.make_input(pair.subject_id, item_id, interactors))
            key = keys[item_id, interactors]
            if key not in index:
                index[key] = len(inputs)
                inputs.append((item_id, interactors))
            rows.append((index[key], label))
    return inputs, rows


def run_fold(dataset, benchmark, make_predictor, settings):
    """Score one held-out benchmark; returns one result row per pair."""
    if settings.pool_items is not None:
        dataset = _item_pool(dataset, benchmark, settings)
    pairs = eligible_pairs(dataset, benchmark, settings)
    if not pairs:
        return []
    predictor = make_predictor()
    predictor.fit(dataset.subset(sorted(set(dataset.responses["benchmark"]) - {benchmark})))
    hook = AcquisitionHook(getattr(predictor, "acquisition_function", None))
    if hook.mode == "legacy_rank":
        raise NotImplementedError("one-argument acquisition hooks are not simulated")

    order = list(pairs)
    _rng(settings.seed, benchmark, "pair-order").shuffle(order)
    targets = {id(p): _evaluation_targets(dataset, p) for p in pairs}
    cursor = {id(p): 0 for p in pairs}
    acquired = {id(p): [] for p in pairs}
    fold_acquired = []
    curves = {id(p): {"brier": [], "ece": [], "labels": []} for p in pairs}
    partners = {}
    for pair in pairs:
        others = [p for p in pairs if p is not pair]
        _rng(settings.seed, benchmark, pair.subject_id, "partners").shuffle(others)
        partners[id(pair)] = others[:settings.partners]

    def evidence(pair):
        if settings.evidence == "pair":
            return acquired[id(pair)]
        if settings.evidence == "partners":
            return acquired[id(pair)] + [e for p in partners[id(pair)] for e in acquired[id(p)]]
        return fold_acquired

    for cap in scoring.BUDGETS:
        for pair in order:
            _acquire(dataset, pair, cap, predictor, hook, settings, cursor, acquired,
                     fold_acquired, evidence(pair))
        if hasattr(predictor, "reload"):
            predictor.reload()
        for pair in order:
            _evaluate(dataset, pair, predictor, settings, targets[id(pair)],
                      list(evidence(pair)), curves[id(pair)])
            curves[id(pair)]["labels"].append(len(acquired[id(pair)]))

    results = []
    for pair in pairs:
        curve = curves[id(pair)]
        inputs, rows = targets[id(pair)]
        result = {"seed": settings.seed, "benchmark": benchmark,
                  "benchmark_id": alias("benchmark", benchmark), "subject_id": pair.subject_id,
                  "n_items": pair.n_items, "n_eval_items": len(pair.eval_items),
                  "n_eval_rows": len(rows), "n_eval_inputs": len(inputs),
                  "alc": scoring.alc(curve["brier"]), "ece_alc": scoring.alc(curve["ece"])}
        for i, budget in enumerate(scoring.BUDGETS):
            result[f"brier_{budget}"] = curve["brier"][i]
            result[f"ece_{budget}"] = curve["ece"][i]
            result[f"labels_{budget}"] = curve["labels"][i]
        results.append(result)
    return results


def _acquire(dataset, pair, cap, predictor, hook, settings, cursor, acquired, fold_acquired,
             evidence):
    key = id(pair)
    while len(acquired[key]) < cap and cursor[key] < len(pair.stream):
        item_id = pair.stream[cursor[key]]
        current = dataset.make_input(pair.subject_id, item_id, pair.responses[item_id][0][0])
        context = {"subject_id": alias("subject", pair.subject_id),
                   "benchmark_id": alias("benchmark", pair.benchmark),
                   "labels_remaining": MAX_LABELS - len(acquired[key]),
                   "labels_acquired": len(acquired[key]), "max_labels": MAX_LABELS,
                   "items_remaining": len(pair.stream) - cursor[key]}
        if hook.mode == "random":
            query = random_query(context, current)
        else:
            prediction = None
            if hook.needs_prediction:
                predict = getattr(predictor, "predict_acquisition", predictor.predict)
                prediction = check_probability(
                    predict(copy.deepcopy(current), copy.deepcopy(evidence)))
            query = hook(current, prediction, evidence, context)
        cursor[key] += 1
        if query:
            rng = _rng(settings.seed, pair.benchmark, pair.subject_id, item_id, "reveal")
            interactors, label = rng.choice(pair.responses[item_id])
            entry = [dataset.make_input(pair.subject_id, item_id, interactors), label]
            acquired[key].append(entry)
            fold_acquired.append(entry)


def _evaluate(dataset, pair, predictor, settings, targets, evidence, curve):
    inputs, rows = targets
    predictions = []
    for item_id, interactors in inputs:
        labeled = copy.deepcopy(evidence) if settings.strict_copies else evidence
        value = predictor.predict(dataset.make_input(pair.subject_id, item_id, interactors),
                                  labeled)
        predictions.append(check_probability(value))
    row_predictions = [predictions[i] for i, _ in rows]
    labels = [label for _, label in rows]
    curve["brier"].append(scoring.brier(row_predictions, labels))
    curve["ece"].append(scoring.ece(row_predictions, labels))


def run(dataset, make_predictor, settings, benchmarks=None):
    """All leave-one-benchmark-out folds for one seed."""
    results = []
    for benchmark in benchmarks or sorted(set(dataset.responses["benchmark"])):
        results.extend(run_fold(dataset, benchmark, make_predictor, settings))
    return results
