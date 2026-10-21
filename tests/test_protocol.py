import sys

import numpy as np
import pytest
from synthetic import make_dataset

from paiec import protocol
from paiec.data import SUBJECT_FIELDS
from paiec.predictors import ConstantPredictor, SubmissionPredictor


class Recorder:
    """Constant predictor that records what it sees at each evaluation checkpoint."""

    def __init__(self):
        self.checkpoints, self.fit_benchmarks = [], None

    def fit(self, train):
        self.fit_benchmarks = set(train.responses["benchmark"])

    def reload(self):
        self.checkpoints.append([])

    def predict(self, input, labeled=None):
        self.checkpoints[-1].append((input, labeled))
        return 0.5


class OwnPairMean(ConstantPredictor):
    """(1 + successes) / (2 + n) over revealed labels of the same subject and benchmark."""

    def predict(self, input, labeled=None):
        subject, item = input
        own = [label for (s, i), label in labeled or []
               if s == subject and i["benchmark_id"] == item["benchmark_id"]]
        return (1 + sum(own)) / (2 + len(own))


def run_held(dataset, predictor, **settings):
    return protocol.run_fold(dataset, "held", lambda: predictor, protocol.Settings(**settings))


def test_constant_predictor_scores_exactly_a_quarter():
    rows = run_held(make_dataset(), ConstantPredictor())
    assert {r["subject_id"] for r in rows} == {"s1", "s2"}  # s3 has fewer than 80 items
    for row in rows:
        assert row["alc"] == 0.25
        assert all(row[f"brier_{b}"] == 0.25 for b in protocol.scoring.BUDGETS)


def test_split_is_half_and_half_with_extra_item_to_acquisition():
    pairs = {p.subject_id: p for p in protocol.eligible_pairs(make_dataset(), "held",
                                                              protocol.Settings())}
    assert (len(pairs["s1"].stream), len(pairs["s1"].eval_items)) == (51, 50)
    assert (len(pairs["s2"].stream), len(pairs["s2"].eval_items)) == (45, 45)
    for pair in pairs.values():
        assert not set(pair.stream) & set(pair.eval_items)
        assert set(pair.stream) | set(pair.eval_items) == set(pair.responses)


def test_training_data_excludes_the_held_out_benchmark():
    recorder = Recorder()
    run_held(make_dataset(), recorder)
    assert recorder.fit_benchmarks == {"train"}


def test_evidence_is_nested_capped_and_never_from_evaluation_items():
    dataset = make_dataset()
    recorder = Recorder()
    run_held(dataset, recorder)
    pairs = {p.subject_id: p for p in protocol.eligible_pairs(dataset, "held", protocol.Settings())}
    content_to_item = {v["item_content"]: k for k, v in dataset.items.items()}
    previous = {s: set() for s in pairs}
    for cap, calls in zip(protocol.scoring.BUDGETS, recorder.checkpoints):
        seen = {}
        for input, labeled in calls:
            subject_id = input[0]["normalized_name"]
            revealed = set()
            for (subject, item), label in labeled:
                assert subject["normalized_name"] == subject_id  # own labels only
                assert label in (0, 1) and isinstance(label, int)
                revealed.add(content_to_item[item["item_content"]])
            seen.setdefault(subject_id, revealed)
            assert revealed == seen[subject_id]  # same frozen evidence for every call
        for subject_id, revealed in seen.items():
            assert len(revealed) == cap  # the default policy fills each cap
            assert revealed <= set(pairs[subject_id].stream)
            assert previous[subject_id] <= revealed
            previous[subject_id] = revealed


def test_results_are_deterministic_and_depend_on_the_seed():
    dataset = make_dataset()
    def without_seed(rows):
        return [{k: v for k, v in row.items() if k != "seed"} for row in rows]

    first = run_held(dataset, OwnPairMean())
    assert first == run_held(dataset, OwnPairMean())
    assert len({r["brier_31"] for r in first} | {r["brier_0"] for r in first}) > 2
    assert without_seed(first) != without_seed(run_held(dataset, OwnPairMean(), seed=1))


def test_filtering_predictor_scores_identically_with_or_without_other_pairs_evidence():
    dataset = make_dataset()
    assert run_held(dataset, OwnPairMean()) == run_held(dataset, OwnPairMean(), evidence="benchmark")


def test_mutating_labeled_or_input_cannot_change_later_calls():
    class Mutator(OwnPairMean):
        def predict(self, input, labeled=None):
            probability = super().predict(input, labeled)
            if labeled:
                labeled[0][1] = 1 - labeled[0][1]
                labeled.append(labeled[0])
            input[0]["normalized_name"] = "changed"
            return probability

    dataset = make_dataset()
    assert run_held(dataset, Mutator()) == run_held(dataset, OwnPairMean())


def test_fit_cannot_modify_test_inputs():
    class Leaky(Recorder):
        def fit(self, train):
            super().fit(train)
            for fields in train.subjects.values():
                fields["normalized_name"] = "leaked"
                fields["extra"] = "leaked"
            for item in train.items.values():
                item["item_content"] = "leaked"

    recorder = Leaky()
    run_held(make_dataset(), recorder)
    for calls in recorder.checkpoints:
        for (subject, item), _ in calls:
            assert tuple(subject) == SUBJECT_FIELDS
            assert "leaked" not in subject.values() and item["item_content"] != "leaked"


def test_identical_inputs_are_predicted_once_but_every_row_is_scored():
    dataset = make_dataset(duplicate_content=True)
    recorder = Recorder()
    rows = run_held(dataset, recorder)
    for row in rows:
        assert row["n_eval_inputs"] < row["n_eval_items"] <= row["n_eval_rows"]
    assert len(recorder.checkpoints[0]) == sum(r["n_eval_inputs"] for r in rows)


class Hooked(ConstantPredictor):
    def __init__(self, decide):
        super().__init__()
        self.decide, self.contexts = decide, []

    def acquisition_function(self, input, prediction=None, labeled=None, context=None):
        self.contexts.append((prediction, dict(context)))
        return self.decide(context)


def test_hook_must_return_a_real_bool():
    with pytest.raises(ValueError):
        run_held(make_dataset(), Hooked(lambda context: 1))


def test_hook_receives_prediction_and_context():
    predictor = Hooked(lambda context: True)
    rows = run_held(make_dataset(), predictor)
    prediction, first = predictor.contexts[0]
    assert prediction == 0.5
    assert first["labels_remaining"] == 31 and first["labels_acquired"] == 0
    assert first["max_labels"] == 31 and first["items_remaining"] in (45, 51)
    assert all(r[f"labels_{b}"] == b for r in rows for b in protocol.scoring.BUDGETS)


def test_hook_that_never_queries_leaves_every_budget_empty():
    rows = run_held(make_dataset(), Hooked(lambda context: False))
    assert all(r[f"labels_{b}"] == 0 for r in rows for b in protocol.scoring.BUDGETS)


def test_benchmark_evidence_includes_other_pairs():
    recorder = Recorder()
    run_held(make_dataset(), recorder, evidence="benchmark")
    _, labeled = recorder.checkpoints[-1][0]
    assert {subject["normalized_name"] for (subject, _), _ in labeled} == {"s1", "s2"}
    assert len(labeled) == 62


def test_max_pair_items_excludes_large_pairs():
    rows = run_held(make_dataset(), ConstantPredictor(), max_pair_items=95)
    assert {r["subject_id"] for r in rows} == {"s2"}


def test_shared_split_keeps_each_pair_balanced():
    pairs = protocol.eligible_pairs(make_dataset(), "held", protocol.Settings(split="shared"))
    assert [(len(p.stream), len(p.eval_items)) for p in pairs] == [(51, 50), (45, 45)]


def test_check_probability_rejects_invalid_values():
    assert protocol.check_probability(0.3) == 0.3
    for bad in (True, "0.5", np.array(0.5), float("nan"), float("inf"), 1.01, -0.1):
        with pytest.raises(ValueError):
            protocol.check_probability(bad)


def test_submission_directory_loads_like_the_hosted_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    (tmp_path / "p.txt").write_text("0.25")
    (tmp_path / "helper.py").write_text("COUNT = 0\n")
    (tmp_path / "model.py").write_text(
        "import helper\nCALLS = 0\nP = float(open('p.txt').read())\n\n"
        "def predict(input, labeled=None):\n    global CALLS\n    CALLS += 1\n"
        "    helper.COUNT += 1\n    return P\n")
    (tmp_path / "labeling.py").write_text(
        "import model\n\n"
        "def acquisition_function(input, prediction=None, labeled=None, context=None):\n"
        "    return prediction == model.P == float(open('p.txt').read())\n")
    predictor = SubmissionPredictor(tmp_path)
    rows = protocol.run_fold(make_dataset(), "held", lambda: predictor, protocol.Settings())
    assert all(r[f"labels_{b}"] == b for r in rows for b in protocol.scoring.BUDGETS)
    # Evaluation ran in a fresh module: only the last checkpoint's calls are counted.
    assert predictor.model.CALLS == sum(r["n_eval_inputs"] for r in rows)
    assert predictor.parent.CALLS > 0  # acquisition-time predictions used the parent
    # A new predictor for the next fold starts from fresh helper modules.
    SubmissionPredictor(tmp_path)
    assert sys.modules["helper"].COUNT == 0
