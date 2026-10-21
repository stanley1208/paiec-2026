"""Stress tests for the starting guess, run through the normal protocol.

shift_benchmark makes a held-out benchmark easier or harder than the training ones;
ShuffledMetadata is the Phase 4 model with the subjects' metadata scores shuffled.
"""

import json
import math
import random

from . import baseline_core, logit_normal_core
from .data import Dataset
from .logit_normal import LogitNormal


def shift_benchmark(dataset, benchmark, delta, seed=0):
    """Copy of dataset in which every subject's success rate on benchmark moves by delta logits.

    Per subject, labels are flipped one way only (0 to 1 when delta > 0, 1 to 0 when
    delta < 0), each with the probability that moves the expected rate from y to
    sigmoid(logit(y) + delta), with y clipped to [0.005, 0.995]. Other benchmarks are
    unchanged. Flipping one way with the same probability for every item also narrows the
    spread of item difficulties, so this suits predictors that give every item of a pair the
    same probability, not item-level models.
    """
    responses = dataset.responses.copy()
    held = responses[responses["benchmark"] == benchmark]
    for subject_id, index in held.groupby("subject_id").groups.items():
        labels = responses.loc[index, "label"].tolist()
        y = min(max(sum(labels) / len(labels), 0.005), 0.995)
        target = 1 / (1 + math.exp(-(math.log(y / (1 - y)) + delta)))
        rng = random.Random(f"{seed}|{benchmark}|{subject_id}|{delta}")
        if target >= y:
            q = (target - y) / (1 - y)
            labels = [1 if v == 1 or rng.random() < q else 0 for v in labels]
        else:
            q = (y - target) / y
            labels = [0 if v == 0 or rng.random() < q else 1 for v in labels]
        responses.loc[index, "label"] = labels
    return Dataset(responses, dataset.subjects, dataset.items)


class ShuffledMetadata:
    """LogitNormal, but the held-out subjects' prior means are randomly permuted among them.

    This removes the ranking within the benchmark but keeps the subjects' average prior
    mean. subjects must include every subject it will be asked to predict.
    """

    def __init__(self, subjects, seed=0, **options):
        self.keys = sorted({baseline_core.subject_key(s) for s in subjects})
        self.seed = seed
        self.model = LogitNormal(**options)

    def fit(self, train):
        self.model.fit(train)
        means = [logit_normal_core.prior_mean(json.loads(k), self.model.params) for k in self.keys]
        random.Random(self.seed).shuffle(means)
        self.means = dict(zip(self.keys, means))

    def predict(self, input, labeled=None):
        subject, item = input
        own = baseline_core.own_labels(subject, item.get("benchmark_id", ""), labeled or [])
        return logit_normal_core.posterior_mean(self.means[baseline_core.subject_key(subject)],
                                                self.model.params["sd"], sum(own), len(own))
