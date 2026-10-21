"""Phase 4 model: a cautious starting guess near 0.5, updated with the pair's own labels.

See logit_normal_core for the model. fit_params reuses the Phase 3 metadata regression
(paiec.baselines.fit_params) on the training benchmarks, drops its intercept and benchmark
effects, and centres the subjects' scores on their mean over the training subjects.
"""

import math

from . import baselines, logit_normal_core


def fit_params(train, sd=1.5, scale=1.0, **options):
    params = baselines.fit_params(train, prior="subject", **options)
    scores = [logit_normal_core.metadata_score(s, params) for s in train.subjects.values()]
    params.update(model="logit_normal", center=math.fsum(scores) / len(scores), sd=sd,
                  scale=scale)
    return params


class LogitNormal:
    """fit() builds params on the training benchmarks; predict() uses logit_normal_core."""

    def __init__(self, sd=1.5, scale=1.0, **options):
        self.options = dict(sd=sd, scale=scale, **options)
        self.params = None

    def fit(self, train):
        self.params = fit_params(train, **self.options)

    def predict(self, input, labeled=None):
        return logit_normal_core.predict(input, labeled or [], self.params)
