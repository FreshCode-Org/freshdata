"""Trust score and quality report tests."""

import json

import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise import (
    QualityReport,
    TrustScore,
    TrustScoreWeights,
    build_quality_report,
    compute_trust_score,
)
from freshdata.enterprise.metrics import ColumnTrust


def test_clean_frame_scores_high(already_clean):
    score = compute_trust_score(already_clean)
    assert score.overall > 95
    assert score.grade == "A"
    assert 0 <= score.completeness <= 100


def test_messy_frame_scores_lower_than_clean(messy, already_clean):
    assert compute_trust_score(messy).overall < compute_trust_score(already_clean).overall


def test_empty_frame_is_perfect():
    score = compute_trust_score(pd.DataFrame())
    assert score.overall == 100.0
    assert score.columns == ()


def test_completeness_penalised_by_missing():
    df = pd.DataFrame({"a": [1, 2, None, 4]})  # 3 of 4 present
    assert compute_trust_score(df).completeness == 75.0


def test_validity_flags_whitespace_sentinel_and_unparseable():
    df = pd.DataFrame({
        # whitespace (" a ") + 2 sentinels ("N/A", "??"), padded to 21 rows
        "s": [" a ", "b", "N/A", "??"] + ["ok"] * 17,
        # 20 parseable + 1 unparseable => 95.2% parse, so a numeric conversion is offered
        "n": [str(i) for i in range(20)] + ["x"],
    })
    score = compute_trust_score(df)
    assert score.validity < 100
    issues = {c.name: c.issues for c in score.columns}
    assert any("sentinel" in i for i in issues["s"])
    assert any("unparseable" in i for i in issues["n"])


def test_validity_flags_outliers():
    values = list(range(25)) + [10_000]
    df = pd.DataFrame({"v": values})
    score = compute_trust_score(df)
    assert any("outlier" in i for c in score.columns for i in c.issues)


def test_consistency_penalised_by_constant_and_mixed():
    df = pd.DataFrame({"const": [1, 1, 1, 1], "mixed": [1, "a", 2.0, "b"]})
    assert compute_trust_score(df).consistency < 100


def test_consistency_penalised_by_duplicate_labels():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    assert compute_trust_score(df).consistency < 100


def test_uniqueness_penalised_by_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert compute_trust_score(df).uniqueness < 100


def test_unhashable_cells_do_not_crash_uniqueness():
    df = pd.DataFrame({"x": [[1], [2], [1]]})
    assert 0.0 <= compute_trust_score(df).uniqueness <= 100.0


def test_weights_shift_the_overall_score():
    # 'a' has one missing cell (completeness 87.5%); everything else is clean.
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": ["x", "y", "z", "w"]})
    only_completeness = TrustScoreWeights(completeness=1, validity=0, uniqueness=0, consistency=0)
    only_validity = TrustScoreWeights(completeness=0, validity=1, uniqueness=0, consistency=0)
    low = compute_trust_score(df, weights=only_completeness).overall
    high = compute_trust_score(df, weights=only_validity).overall
    assert low < high


def test_compute_trust_score_accepts_polars():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert compute_trust_score(df).overall > 95


@pytest.mark.parametrize(
    "overall, grade",
    [(95, "A"), (85, "B"), (75, "C"), (65, "D"), (50, "F")],
)
def test_grade_boundaries(overall, grade):
    score = TrustScore(overall, overall, overall, overall, overall, 10, 2,
                       (ColumnTrust("a", 100, 100, 0.0),))
    assert score.grade == grade


def test_trust_score_serialization_and_render():
    score = compute_trust_score(pd.DataFrame({"a": [1, 2, 3]}))
    payload = score.to_dict()
    assert set(payload) >= {"overall", "grade", "dimensions", "columns"}
    assert json.dumps(payload)  # JSON-serializable
    assert "Data Trust Score" in score.to_markdown()
    assert "Data Trust Score" in str(score)
    assert repr(score).startswith("<TrustScore")


def test_quality_report_round_trip(messy):
    cleaned, report = fd.clean(messy, return_report=True, verbose=False)
    quality = build_quality_report(messy, cleaned, report)
    assert quality.percent_clean == round(quality.trust_after.overall, 2)
    assert quality.trust_delta == round(
        quality.trust_after.overall - quality.trust_before.overall, 2
    )
    payload = quality.to_dict()
    assert set(payload) >= {"trust_before", "trust_after", "clean_report", "percent_clean"}
    assert json.loads(quality.to_json())  # valid JSON string
    md = quality.to_markdown()
    assert "# freshdata Quality Report" in md
    assert "## Actions" in md  # messy cleaning produced actions
    assert str(quality) == md


def test_quality_report_without_actions_omits_action_table(already_clean):
    cleaned, report = fd.clean(already_clean, return_report=True, verbose=False)
    quality = build_quality_report(already_clean, cleaned, report)
    assert "## Actions" not in quality.to_markdown()


def test_quality_report_construct_directly_sets_generated_at():
    score = compute_trust_score(pd.DataFrame({"a": [1, 2]}))
    _, report = fd.clean(pd.DataFrame({"a": [1, 2]}), return_report=True, verbose=False)
    q = QualityReport(trust_before=score, trust_after=score, clean_report=report, actor="me")
    assert q.actor == "me"
    assert "T" in q.generated_at  # ISO 8601
