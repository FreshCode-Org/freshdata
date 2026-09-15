"""Nested / unhashable cell values: profiling and cleaning must not crash.

Object columns holding Python lists or dicts make ``DataFrame.duplicated``
raise ``TypeError``; nested Arrow dtypes (list, large_list, struct, map) raise
``pyarrow.lib.ArrowNotImplementedError`` instead, because they cannot be
dictionary-encoded. Both are treated the same way: duplicate detection is
reported as impossible (``None`` in a profile, a skip note in a clean report)
rather than crashing or guessing.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import freshdata as fd


def _arrow_nested_series(kind: str, values: list) -> pd.Series:
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pd, "ArrowDtype"):
        pytest.skip("pd.ArrowDtype is not available in this pandas version")
    types = {
        "list": lambda: pa.list_(pa.string()),
        "list_equal_len": lambda: pa.list_(pa.string()),
        "large_list": lambda: pa.large_list(pa.string()),
        "struct": lambda: pa.struct([("k", pa.int64()), ("v", pa.string())]),
        "map": lambda: pa.map_(pa.string(), pa.int64()),
    }
    try:
        return pd.Series(pd.array(values, dtype=pd.ArrowDtype(types[kind]())))
    except (TypeError, ValueError, NotImplementedError) as exc:  # pragma: no cover
        pytest.skip(f"nested ArrowDtype {kind!r} unsupported here: {exc}")


# Two distinct nested payloads per kind. "list_equal_len" matters because
# pandas 1.5 turns equal-length Arrow lists into a 2-D numpy array.
_PAYLOADS = {
    "list": (["x"], ["y", "z"]),
    "list_equal_len": (["x"], ["y"]),
    "large_list": (["x"], ["y", "z"]),
    "struct": ({"k": 1, "v": "x"}, {"k": 2, "v": "y"}),
    "map": ([("k", 1)], [("j", 2)]),
}
_KINDS = sorted(_PAYLOADS)


def test_profile_arrow_list_column_repro():
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pd, "ArrowDtype"):
        pytest.skip("pd.ArrowDtype is not available in this pandas version")
    tags = pd.array([["x"], ["y", "z"]], dtype=pd.ArrowDtype(pa.list_(pa.string())))
    df = pd.DataFrame({"a": [1, 2], "tags": tags})
    p = fd.profile(df)
    assert p.n_rows == 2
    assert p.duplicate_rows is None
    by_name = {c.name: c for c in p.columns}
    assert by_name["a"].unique == 2
    assert by_name["tags"].unique is None
    assert by_name["tags"].sample_values == [["x"], ["y", "z"]]
    assert str(p)
    assert json.dumps(p.to_dict())


@pytest.mark.parametrize("kind", _KINDS)
def test_profile_arrow_nested_column(kind):
    first, second = _PAYLOADS[kind]
    df = pd.DataFrame({"a": [1, 2], "n": _arrow_nested_series(kind, [first, second])})
    snapshot = df.copy(deep=True)
    p = fd.profile(df)
    assert p.duplicate_rows is None
    assert p.missing_cells == 0
    assert [c.unique for c in p.columns] == [2, None]
    assert len(p.columns[1].sample_values) == 2
    assert str(p)
    pd.testing.assert_frame_equal(df, snapshot)  # profiling never mutates


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("nested_match", [True, False])
def test_rows_matching_or_differing_only_in_nested_column(kind, nested_match):
    # The scalar column is identical across rows, so only the nested column
    # decides whether rows are duplicates. Detection is impossible there, so
    # the profile must report None (never a misleading 0 or 1), and a clean
    # must keep every row and say why.
    first, second = _PAYLOADS[kind]
    nested = [first, first] if nested_match else [first, second]
    df = pd.DataFrame({"a": [1, 1], "n": _arrow_nested_series(kind, nested)})
    assert fd.profile(df).duplicate_rows is None
    out, report = fd.clean(df, drop_duplicates=True, return_report=True, verbose=False)
    assert len(out) == 2
    assert any(a.step == "drop_duplicates" and "unhashable" in a.description for a in report)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    ("keep", "kept_rows"),
    [("first", [0, 2]), ("last", [1, 2]), ("drop", [2]), ("aggregate", None)],
)
def test_duplicate_subset_excluding_nested_column_still_counts(kind, keep, kept_rows):
    # Rows 0 and 1 share "a" but differ in the nested column; with a subset
    # that excludes it, detection works and the nested values survive intact
    # (this previously crashed on pandas 1.5 when rows were actually removed).
    first, second = _PAYLOADS[kind]
    df = pd.DataFrame(
        {"a": [1, 1, 2], "n": _arrow_nested_series(kind, [first, second, first])}
    )
    out, report = fd.clean(
        df, drop_duplicates=True, duplicate_subset=("a",), duplicate_keep=keep,
        return_report=True, verbose=False,
    )
    assert str(out["n"].dtype) == str(df["n"].dtype)
    if kept_rows is None:  # aggregate: one row per key, first nested value
        assert out["a"].tolist() == [1, 2]
        assert out["n"].tolist() == [df["n"].iloc[0], df["n"].iloc[2]]
    else:
        assert out.index.tolist() == kept_rows
        assert out["n"].tolist() == df["n"].iloc[kept_rows].tolist()
    assert report.duplicates_removed == 3 - len(out)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("drop", [False, True])
def test_clean_arrow_nested_column(kind, drop):
    first, second = _PAYLOADS[kind]
    df = pd.DataFrame({"a": [1, 2], "n": _arrow_nested_series(kind, [first, second])})
    out, report = fd.clean(df, drop_duplicates=drop, return_report=True, verbose=False)
    assert len(out) == 2
    assert out["n"].tolist() == df["n"].tolist()
    assert any(a.step == "drop_duplicates" and "unhashable" in a.description for a in report)


@pytest.mark.parametrize(
    "values", [[["x"], ["y", "z"]], [{"k": 1}, {"k": 2}]], ids=["lists", "dicts"]
)
def test_object_column_with_lists_or_dicts(values):
    df = pd.DataFrame({"a": [1, 2], "n": values})
    p = fd.profile(df)
    assert p.duplicate_rows is None
    assert [c.unique for c in p.columns] == [2, None]
    out, report = fd.clean(df, drop_duplicates=True, return_report=True, verbose=False)
    assert len(out) == 2
    assert any(a.step == "drop_duplicates" and "unhashable" in a.description for a in report)


def test_arrow_nested_matches_object_list_behaviour():
    arrow = pd.DataFrame({"a": [1, 2], "n": _arrow_nested_series("list", [["x"], ["y", "z"]])})
    obj = pd.DataFrame({"a": [1, 2], "n": [["x"], ["y", "z"]]})
    pa_, po = fd.profile(arrow), fd.profile(obj)
    assert pa_.duplicate_rows == po.duplicate_rows
    assert [c.unique for c in pa_.columns] == [c.unique for c in po.columns]
    assert [c.sample_values for c in pa_.columns] == [c.sample_values for c in po.columns]


def test_parity_frames_without_nested_columns_unchanged():
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3, 3],
            "name": ["a", "b", "b", "c", "c"],
            "score": [1.5, 2.5, 2.5, None, None],
        }
    )
    p = fd.profile(df)
    assert p.duplicate_rows == int(df.duplicated().sum()) == 2
    assert [c.unique for c in p.columns] == [3, 3, 2]
    assert p.missing_cells == 2
    out, report = fd.clean(df, drop_duplicates=True, return_report=True, verbose=False)
    assert len(out) == 3
    assert report.duplicates_removed == 2
    assert not any("unhashable" in a.description for a in report)


def test_parity_arrow_scalar_columns_still_counted():
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pd, "ArrowDtype"):
        pytest.skip("pd.ArrowDtype is not available in this pandas version")
    df = pd.DataFrame(
        {
            "a": pd.array([1, 1, 2], dtype=pd.ArrowDtype(pa.int64())),
            "b": pd.array([1.0, 1.0, 3.0], dtype=pd.ArrowDtype(pa.float64())),
        }
    )
    p = fd.profile(df)
    assert p.duplicate_rows == 1
    assert [c.unique for c in p.columns] == [2, 2]
