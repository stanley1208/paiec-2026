"""Phase 4 model plus item difficulties pooled from other subjects' labels in the same run.

See item_pooling_core for the model. fit_params is the Phase 4 fit plus the prior sd of item
difficulties; the default scale is the submitted Phase 4 choice (half metadata nudge). The
optional agent_scale and group_sd settings (ideas tested after version 2) are off by default.
"""

from . import item_pooling_core
from .logit_normal import fit_params as fit_logit_normal


def fit_params(train, item_sd=2.0, sd=1.5, scale=0.5, agent_scale=None, group_sd=None,
               **options):
    params = fit_logit_normal(train, sd=sd, scale=scale, **options)
    params.update(model="item_pooling", item_sd=item_sd)
    # The optional settings are stored only when used, so version 2's params stay unchanged.
    params.update({k: v for k, v in (("agent_scale", agent_scale), ("group_sd", group_sd))
                   if v is not None})
    return params


class ItemPooling:
    """fit() builds params on the training benchmarks; predict() uses item_pooling_core."""

    def __init__(self, item_sd=2.0, sd=1.5, scale=0.5, agent_scale=None, group_sd=None,
                 **options):
        self.options = dict(item_sd=item_sd, sd=sd, scale=scale, agent_scale=agent_scale,
                            group_sd=group_sd, **options)
        self.params = None

    def fit(self, train):
        self.params = fit_params(train, **self.options)

    def predict(self, input, labeled=None):
        return item_pooling_core.predict(input, labeled or [], self.params)
