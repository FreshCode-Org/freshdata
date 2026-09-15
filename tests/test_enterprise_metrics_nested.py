"""Trust scoring on frames with nested / unhashable cell values.

Object columns holding Python lists make ``Series.nunique`` and
``DataFrame.duplicated`` raise ``TypeError``; nested Arrow dtypes (list,
large_list, struct, map) raise ``pyarrow.lib.ArrowNotImplementedError``, a
``NotImplementedError`` subclass, because they cannot be hashed or
dictionary-encoded. Both mean "cannot compute": the column is not reported as
constant, and uniqueness is unknown (``NaN``, ``None`` when serialised) and
left out of the overall score instead of counting as a perfect 100.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise import (
    TrustScoreWeights,
    build_quality_report,
    clean_enterprise,
    compute_trust_score,
)
from freshdata.enterprise.metrics import _is_constant

_UNHASHABLE = "unhashable values: duplicate rows not checked"


def _arrow_nested_series(kind: str, values: list) -> pd.Series:
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pd, "ArrowDtype"):
        pytest.skip("pd.ArrowDtype is not available in this pandas version")
    types = {
        "list": lambda: pa.list_(pa.string()),
        "large_list": lambda: pa.large_list(pa.string()),
        "struct": lambda: pa.struct([("k", pa.int64()), ("v", pa.string())]),
        "map": lambda: pa.map_(pa.string(), pa.int64()),
    }
    try:
        return pd.Series(pd.array(values, dtype=pd.ArrowDtype(types[kind]())))
    except (TypeError, ValueError, NotImplementedError) as exc:  # pragma: no cover
        pytest.skip(f"nested ArrowDtype {kind!r} unsupported here: {exc}")


# Two distinct nested payloads per kind; rows 2 and 3 of each frame repeat.
_PAYLOADS = {
    "list": (["x"], ["y", "z"]),
    "large_list": (["x"], ["y", "z"]),
    "struct": ({"k": 1, "v": "x"}, {"k": 2, "v": "y"}),
    "map": ([("k", 1)], [("j", 2)]),
}
_KINDS = sorted(_PAYLOADS)


def _nested_frame(kind: str) -> pd.DataFrame:
    first, second = _PAYLOADS[kind]
    return pd.DataFrame({
        "a": [1, 2, 2],
        "tags": _arrow_nested_series(kind, [first, second, second]),
    })


def _renormalised(score, weights: TrustScoreWeights | None = None) -> float:
    w = (weights or TrustScoreWeights()).normalized()
    measured = w["completeness"] + w["validity"] + w["consistency"]
    return (
        w["completeness"] * score.completeness
        + w["validity"] * score.validity
        + w["consistency"] * score.consistency
    ) / measured


def _assert_uniqueness_unknown(score) -> None:
    assert math.isnan(score.uniqueness)
    assert not score.uniqueness_assessed
    assert score.overall == pytest.approx(_renormalised(score))
    payload = score.to_dict()
    assert payload["dimensions"]["uniqueness"] is None
    assert json.loads(json.dumps(payload)) == payload  # strict JSON, no NaN
    assert "uniqueness n/a" in str(score)
    assert "| Uniqueness | n/a |" in score.to_markdown()
    by_name = {c.name: c for c in score.columns}
    assert _UNHASHABLE in by_name["tags"].issues
    assert _UNHASHABLE not in by_name["a"].issues


def test_trust_score_arrow_list_column_repro():
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pd, "ArrowDtype"):
        pytest.skip("pd.ArrowDtype is not available in this pandas version")
    tags = pd.array([["x"], ["y", "z"], ["y", "z"]],
                    dtype=pd.ArrowDtype(pa.list_(pa.string())))
    df = pd.DataFrame({"a": [1, 2, 2], "tags": tags})
    score = compute_trust_score(df)
    assert score.n_rows == 3
    assert score.completeness == 100.0
    _assert_uniqueness_unknown(score)  # duplicate detection impossible


@pytest.mark.parametrize("kind", _KINDS)
def test_trust_score_arrow_nested_column(kind):
    df = _nested_frame(kind)
    snapshot = df.copy(deep=True)
    score = compute_trust_score(df)
    assert (score.n_rows, score.n_cols) == (3, 2)
    assert score.completeness == 100.0
    assert score.validity == 100.0
    _assert_uniqueness_unknown(score)
    assert 0.0 <= score.overall <= 100.0
    by_name = {c.name: c for c in score.columns}
    assert "constant column" not in by_name["tags"].issues
    pd.testing.assert_frame_equal(df, snapshot)  # scoring never mutates


@pytest.mark.parametrize("kind", _KINDS)
def test_identical_nested_values_are_not_reported_constant(kind):
    first, _ = _PAYLOADS[kind]
    s = _arrow_nested_series(kind, [first, first, first])
    assert _is_constant(s, len(s)) is False


def test_object_list_column_matches_nested_fallback():
    df = pd.DataFrame({"a": [1, 2, 2], "tags": [["x"], ["y", "z"], ["y", "z"]]})
    score = compute_trust_score(df)
    assert score.completeness == 100.0
    assert score.validity == 100.0
    assert score.consistency == 50.0  # "tags" infers as mixed
    _assert_uniqueness_unknown(score)
    # (0.3 * 100 + 0.3 * 100 + 0.2 * 50) / 0.8; previously 90.0 with uniqueness 100.
    assert score.overall == pytest.approx(87.5)
    by_name = {c.name: c for c in score.columns}
    assert "constant column" not in by_name["tags"].issues
    same = pd.Series([["x"], ["x"], ["x"]])
    assert _is_constant(same, len(same)) is False


def test_unknown_uniqueness_carries_no_weight():
    df = pd.DataFrame({"a": [1, None, 2], "tags": [["x"], ["y"], ["y"]]})
    heavy = TrustScoreWeights(completeness=1.0, validity=0.0, uniqueness=5.0, consistency=0.0)
    score = compute_trust_score(df, weights=heavy)
    assert score.overall == pytest.approx(score.completeness)
    only_uniqueness = TrustScoreWeights(completeness=0.0, validity=0.0, uniqueness=1.0,
                                        consistency=0.0)
    score = compute_trust_score(df, weights=only_uniqueness)
    assert not score.uniqueness_assessed
    assert score.overall == pytest.approx(
        (score.completeness + score.validity + score.consistency) / 3)


def test_quality_report_markdown_marks_unknown_uniqueness():
    df = pd.DataFrame({"a": [1, 2, 2], "tags": [["x"], ["y", "z"], ["y", "z"]]})
    cleaned, report = fd.clean(df, return_report=True, verbose=False)
    quality = build_quality_report(df, cleaned, report)
    assert "| Uniqueness | n/a | n/a |" in quality.to_markdown()
    payload = json.loads(quality.to_json())
    assert payload["trust_before"]["dimensions"]["uniqueness"] is None


def test_plain_frame_scores_unchanged():
    df = pd.DataFrame({
        "a": [1, 2, 2, None],
        "b": ["x", "x", "x", "x"],
        "m": [1, "a", "a", 2.5],
    })
    score = compute_trust_score(df)
    assert score.completeness == pytest.approx(100.0 * 11 / 12)
    assert score.validity == 100.0
    assert score.uniqueness == 75.0  # row 3 repeats row 2
    assert score.uniqueness_assessed
    assert score.consistency == pytest.approx(100.0 * 2 / 3)  # "m" is mixed
    w = TrustScoreWeights().normalized()
    assert score.overall == (  # the exact four-way blend, bit for bit
        w["completeness"] * score.completeness
        + w["validity"] * score.validity
        + w["uniqueness"] * score.uniqueness
        + w["consistency"] * score.consistency
    )
    assert score.to_dict()["dimensions"]["uniqueness"] == 75.0
    assert "uniqueness 75," in str(score)
    assert "| Uniqueness | 75.0 |" in score.to_markdown()
    by_name = {c.name: c for c in score.columns}
    assert by_name["b"].issues == ("constant column",)
    assert by_name["m"].issues == ("mixed types",)
    assert by_name["a"].issues == ()


@pytest.mark.parametrize("kind", _KINDS)
def test_clean_enterprise_arrow_nested_column(kind):
    df = _nested_frame(kind)
    result = clean_enterprise(df, verbose=False)
    assert not result.trust_before.uniqueness_assessed
    assert result.trust_after.n_rows == 3
    payload = json.loads(result.to_json())
    assert payload["trust_after"]["n_rows"] == 3
    assert payload["trust_before"]["dimensions"]["uniqueness"] is None
