"""Checks on the real measurement-db release; skipped until scripts/download_data.py has run."""

import pytest

from paiec import data, protocol
from paiec.predictors import ConstantPredictor

pytestmark = pytest.mark.skipif(not (data.DATA_DIR / "manifest.json").exists(),
                                reason="measurement-db not downloaded")


@pytest.fixture(scope="module")
def dataset():
    return data.load()


def test_binary_counts_match_the_phase_0_analysis(dataset):
    responses = dataset.responses
    assert len(responses) == 225_764
    assert responses.groupby("benchmark")["item_id"].nunique().to_dict() == {
        "matharena": 1484, "multi_swebench": 2126, "real_webagents": 233,
        "researchcodebench": 212, "swe_rebench": 6306}
    per_pair = responses.groupby(["benchmark", "subject_id"])["item_id"].nunique()
    assert int((per_pair >= 80).sum()) == 221
    assert int(((per_pair >= 80) & (per_pair <= 1000)).sum()) == 192


def test_labels_and_inputs_have_the_competition_format(dataset):
    assert set(dataset.responses["label"].unique()) == {0, 1}
    for fields in dataset.subjects.values():
        assert tuple(fields) == data.SUBJECT_FIELDS
        assert all(isinstance(v, str) for v in fields.values())
    subject_id, item_id = dataset.responses.iloc[0][["subject_id", "item_id"]]
    subject, item = dataset.make_input(subject_id, item_id)
    assert list(item) == ["item_content", "item_features", "interactors", "benchmark_id"]
    assert item["benchmark_id"].startswith("benchmark_")


def test_constant_predictor_scores_a_quarter_on_every_real_pair(dataset):
    rows = protocol.run(dataset, ConstantPredictor, protocol.Settings(seed=0))
    assert len(rows) == 192
    assert all(r["alc"] == 0.25 for r in rows)
    assert {r["benchmark"] for r in rows} == {"matharena", "multi_swebench", "real_webagents",
                                             "researchcodebench"}
