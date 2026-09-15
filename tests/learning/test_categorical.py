"""``fd.learn`` on categorical columns.

Comparing two pandas Categoricals raises ``TypeError`` unless their categories
are identical, and a messy/clean pair almost never shares categories -- the
repair is what changes them. Learning must compare, align and learn such
columns by value, exactly as it does for the object-dtype equivalent. The
fixtures avoid version-specific pandas APIs so they run unchanged on pandas
1.5 (Python 3.9 lane) and pandas 2.x.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.learning import LearningProfile, learn, load_profile, save_profile
from freshdata.learning.align import align_pair
from freshdata.learning.diff import compute_diff

_RAW = [
    "Deliverd",
    "SHIPPED",
    "shipped ",
    "delivered",
    "SHIPPED",
    "Deliverd",
    "shipped ",
    "SHIPPED",
]
_CLEAN = [
    "delivered",
    "shipped",
    "shipped",
    "delivered",
    "shipped",
    "delivered",
    "shipped",
    "shipped",
]
_IDS = [f"A{i}" for i in range(len(_RAW))]


def _frames(raw: list, clean: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"A{i}" for i in range(len(raw))]
    return (
        pd.DataFrame({"order_id": ids, "status": raw}),
        pd.DataFrame({"order_id": ids, "status": clean}),
    )


def _as_object(df: pd.DataFrame) -> pd.DataFrame:
    """The object-dtype equivalent of ``df`` (categoricals decoded)."""
    return df.assign(
        **{
            str(c): df[c].astype(object)
            for c in df.columns
            if isinstance(df[c].dtype, pd.CategoricalDtype)
        }
    )


def _learned(profile: LearningProfile) -> str:
    """Everything a profile learned, as canonical JSON (timestamps excluded)."""
    audit = profile.audit().to_dict()
    payload = {
        "rules": [r.to_dict() for r in profile.rules],
        "value_maps": {c: m.to_dict() for c, m in sorted(profile.value_maps.items())},
        "examples": profile.examples.to_dict() if profile.examples is not None else None,
        "value_patterns": profile.memory.value_patterns if profile.memory is not None else None,
        "notes": audit.get("notes"),
        "demotions": audit.get("demotions"),
        "holdout_metrics": audit.get("holdout_metrics"),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _assert_parity(messy: pd.DataFrame, clean: pd.DataFrame, **kwargs) -> LearningProfile:
    profile = learn(messy, clean, **kwargs)
    reference = learn(_as_object(messy), _as_object(clean), **kwargs)
    assert _learned(profile) == _learned(reference)
    return profile


def test_repro_categoricals_with_different_categories() -> None:
    messy = pd.DataFrame({"c": pd.Categorical([" a", "B ", "a"])})
    clean = pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])})
    profile = fd.learn(messy, clean)
    assert isinstance(profile, LearningProfile)
    assert _learned(profile) == _learned(fd.learn(_as_object(messy), _as_object(clean)))


def test_diff_compares_categoricals_by_value() -> None:
    messy, clean = _frames(pd.Categorical(_RAW), pd.Categorical(_CLEAN))
    summary = compute_diff(align_pair(messy, clean, key="order_id"))
    reference = compute_diff(align_pair(_as_object(messy), _as_object(clean), key="order_id"))
    assert {col: [d.to_dict() for d in diffs] for col, diffs in summary.column_diffs.items()} == {
        col: [d.to_dict() for d in diffs] for col, diffs in reference.column_diffs.items()
    }
    pairs = {(d.raw_value, d.clean_value, d.support) for d in summary.column_diffs["status"]}
    assert pairs == {
        ("SHIPPED", "shipped", 3),
        ("Deliverd", "delivered", 2),
        ("shipped ", "shipped", 2),
    }
    # Only the values changed, never the dtype.
    assert summary.schema_diffs.dtype_changes == {}


@pytest.mark.parametrize(
    ("messy_status", "clean_status"),
    [
        pytest.param(pd.Categorical(_RAW), pd.Categorical(_CLEAN), id="both-different-categories"),
        pytest.param(
            pd.Categorical(_RAW, categories=sorted(set(_RAW) | set(_CLEAN))),
            pd.Categorical(_CLEAN, categories=sorted(set(_RAW) | set(_CLEAN))),
            id="both-same-categories",
        ),
        pytest.param(pd.Categorical(_RAW), list(_CLEAN), id="categorical-messy-object-clean"),
        pytest.param(list(_RAW), pd.Categorical(_CLEAN), id="object-messy-categorical-clean"),
        pytest.param(
            pd.Categorical(_RAW, ordered=True),
            pd.Categorical(_CLEAN, ordered=True),
            id="both-ordered",
        ),
        pytest.param(
            pd.Categorical(_RAW, ordered=True),
            pd.Categorical(_CLEAN, categories=["shipped", "delivered"], ordered=True),
            id="ordered-different-order",
        ),
        pytest.param(
            pd.Categorical(_RAW, ordered=True), pd.Categorical(_CLEAN), id="ordered-vs-unordered"
        ),
    ],
)
@pytest.mark.parametrize("key", [None, "order_id"])
def test_learning_matches_object_dtype(messy_status, clean_status, key) -> None:
    messy, clean = _frames(messy_status, clean_status)
    profile = _assert_parity(messy, clean, key=key, min_support=2)
    entries = {(e.raw_value, e.clean_value) for e in profile.value_maps["status"].entries}
    assert entries == {
        ("SHIPPED", "shipped"),
        ("shipped ", "shipped"),
        ("Deliverd", "delivered"),
    }


def test_categorical_key_column() -> None:
    messy, clean = _frames(pd.Categorical(_RAW), pd.Categorical(_CLEAN))
    messy["order_id"] = pd.Categorical(messy["order_id"])
    clean = clean.iloc[::-1].reset_index(drop=True)
    clean["order_id"] = pd.Categorical(clean["order_id"], categories=list(reversed(_IDS)))
    _assert_parity(messy, clean, key="order_id", min_support=2)


def test_missing_cells_in_categoricals() -> None:
    raw = ["x", "N/A", "y", np.nan, "x", "N/A", "y", np.nan, "x", "N/A"]
    clean = ["x", np.nan, "y", np.nan, "x", np.nan, "y", np.nan, "x", np.nan]
    messy_df, clean_df = _frames(pd.Categorical(raw), pd.Categorical(clean))
    summary = compute_diff(align_pair(messy_df, clean_df, key="order_id"))
    # Both-missing cells are equal; only the sentinel cells differ.
    (diff,) = summary.column_diffs["status"]
    assert (diff.raw_value, diff.clean_value, diff.support, diff.kind) == (
        "N/A",
        None,
        3,
        "value_to_missing",
    )
    profile = _assert_parity(messy_df, clean_df, key="order_id", min_support=2)
    assert any(r.rule == "sentinel" for r in profile.rules)


def test_imputed_categorical_cells() -> None:
    raw = ["a", np.nan, "a", "b", np.nan, "a", "a", "b", np.nan, "a"]
    clean = ["a", "a", "a", "b", "a", "a", "a", "b", "a", "a"]
    messy_df, clean_df = _frames(pd.Categorical(raw), pd.Categorical(clean))
    profile = _assert_parity(messy_df, clean_df, key="order_id", min_support=2)
    assert profile.value_maps == {}  # imputation is never a literal map


def test_sensitive_categorical_column_learns_like_object() -> None:
    raw = ["A@X.COM ", "b@@y.com", "c@z.com"] * 3 + ["d@@w.com"]
    clean = ["a@x.com", "b@y.com", "c@z.com"] * 3 + ["d@w.com"]
    messy = pd.DataFrame({"email": pd.Categorical(raw)})
    clean_df = pd.DataFrame({"email": pd.Categorical(clean)})
    _assert_parity(messy, clean_df, privacy="none", include_sensitive=True, min_support=2)
    masked = learn(messy, clean_df, min_support=2)
    reference = learn(_as_object(messy), _as_object(clean_df), min_support=2)
    assert [e.masked for e in masked.value_maps["email"].entries] == [
        e.masked for e in reference.value_maps["email"].entries
    ]


def test_categorical_profile_round_trips_and_replays_like_object(tmp_path) -> None:
    messy, clean = _frames(pd.Categorical(_RAW), pd.Categorical(_CLEAN))
    profile = learn(messy, clean, key="order_id", min_support=2)
    reference = learn(_as_object(messy), _as_object(clean), key="order_id", min_support=2)

    path = tmp_path / "categorical.fdprofile"
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded.profile_id == profile.profile_id == reference.profile_id
    assert _learned(loaded) == _learned(profile)

    batch = pd.DataFrame(
        {"order_id": ["B1", "B2", "B3"], "status": ["Deliverd", "SHIPPED", "shipped "]}
    )
    categorical_batch = batch.assign(status=pd.Categorical(batch["status"]))
    replayed = fd.clean(categorical_batch, profile=loaded, semantic_mode="auto")
    expected = fd.clean(batch, profile=reference, semantic_mode="auto")
    assert replayed["status"].astype(object).tolist() == ["delivered", "shipped", "shipped"]
    assert replayed["status"].astype(object).tolist() == expected["status"].astype(object).tolist()
