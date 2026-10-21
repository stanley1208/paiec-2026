"""Predictors the harness can score.

A predictor has fit(train) and predict(input, labeled), plus an optional
acquisition_function(input, prediction=None, labeled=None, context=None). fit receives
a Dataset without the held-out benchmark; a new predictor is built for every fold.
predict must not rely on earlier calls or on module-level state.
"""

import contextlib
import functools
import importlib.util
import os
import sys
from pathlib import Path

from .baselines import Baseline
from .item_pooling import ItemPooling
from .logit_normal import LogitNormal


class ConstantPredictor:
    """Always predicts the same probability (0.5 by default)."""

    def __init__(self, probability=0.5):
        self.probability = probability

    def fit(self, train):
        pass

    def predict(self, input, labeled=None):
        return self.probability


class SubmissionPredictor:
    """A submission directory, loaded close to the way the hosted runtime loads it.

    The parent keeps one model module for acquisition-time predictions and for
    labeling.py (whose `import model` finds it). Evaluation uses a fresh module,
    re-imported with its helper modules before every checkpoint (the harness calls
    reload()), as the platform restarts its workers. Code runs with the submission
    directory as working directory. Unlike the platform, parent and evaluation modules
    share one process, so helpers imported lazily at call time are shared between them;
    the kit's smoke test with --concurrency 4 (Linux or WSL) checks true isolation.
    A submission's artifacts were trained offline, usually on all public benchmarks,
    so its leave-one-benchmark-out scores are in-sample.
    """

    in_sample = True
    _loaded_dirs = set()

    def __init__(self, path):
        self.path = Path(path).resolve()
        SubmissionPredictor._loaded_dirs.add(os.path.normcase(str(self.path)))
        self._purge()
        if str(self.path) not in sys.path:
            sys.path.insert(0, str(self.path))
        self.parent = self._load("model.py", "model")
        self.model = None
        if (self.path / "labeling.py").exists():
            hook = self._load("labeling.py", "participant_labeling").acquisition_function

            @functools.wraps(hook)
            def acquisition_function(*args, **kwargs):
                with self._in_directory():
                    return hook(*args, **kwargs)

            self.acquisition_function = acquisition_function

    def fit(self, train):
        pass  # trained offline

    def reload(self):
        self._purge(keep=("model", "participant_labeling"))
        self.model = self._load("model.py", "participant_model")

    def predict(self, input, labeled=None):
        if self.model is None:
            self.reload()
        with self._in_directory():
            return self.model.predict(input, labeled)

    def predict_acquisition(self, input, labeled=None):
        with self._in_directory():
            return self.parent.predict(input, labeled)

    def _purge(self, keep=()):
        """Forget modules imported from any submission directory, except `keep`."""
        for name, module in list(sys.modules.items()):
            origin = getattr(module, "__file__", None)
            if name in keep or not origin:
                continue
            folder = os.path.normcase(os.path.dirname(os.path.abspath(origin)))
            if any(folder == d or folder.startswith(d + os.sep) for d in self._loaded_dirs):
                del sys.modules[name]

    @contextlib.contextmanager
    def _in_directory(self):
        previous = os.getcwd()
        if previous == str(self.path):
            yield
            return
        os.chdir(self.path)
        try:
            yield
        finally:
            os.chdir(previous)

    def _load(self, filename, name):
        spec = importlib.util.spec_from_file_location(name, self.path / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        with self._in_directory():
            spec.loader.exec_module(module)
        return module


PREDICTORS = {
    "constant": ConstantPredictor,
    "global_mean": functools.partial(Baseline, prior="global"),
    "subject_prior": functools.partial(Baseline, prior="subject"),
    "beta_binomial": functools.partial(Baseline, prior="global", strength=4.0),
    "beta_binomial_subject": functools.partial(Baseline, prior="subject", strength=4.0),
    "logit_normal": LogitNormal,
    "item_pooling": ItemPooling,
}
