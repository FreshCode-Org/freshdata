"""Phase-2 hard protected-column guard: physical byte-identity enforcement."""

from __future__ import annotations

import random

import pandas as pd
import pytest

import freshdata as fd
from freshdata.guard import (
    ProtectedColumnError,
    hard_protected_columns,
    protected_column_set,
    snapshot_protected,
    verify_protected,
)


def _ecommerce_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cust_id": ["C001", "C002", "C003", "C003"],
            "email_addr": [" Asha@GMAIL.COM ", "ravi @ example.com",
                           "neha@@shop.in", "neha@@shop.in"],
            "age": [25, None, None, 31],
            "monthly_revenue": [" 1000 ", "2,000", "3000", "3000"],
            "status": [" Active ", "INACTIVE", "pend-ing", "pend-ing"],
        }
    )


PROTECT_CONTEXT = "Never modify revenue values."


def test_protected_column_unchanged_after_normal_cleaning():
    df = _ecommerce_df()
    out, report = fd.clean(df, context=PROTECT_CONTEXT, return_report=True, verbose=False)
    survivors = out.index
    assert out["monthly_revenue"].equals(df["monthly_revenue"].loc[survivors])
    assert str(out["monthly_revenue"].dtype) == str(df["monthly_revenue"].dtype)
    assert any(a.step == "guard" for a in report)


def test_protected_column_unchanged_with_semantic_experts():
    df = _ecommerce_df()
    context = PROTECT_CONTEXT + "\nEmails must be valid.\n" \
        "Allowed status values are active, inactive, pending."
    out, report = fd.clean(
        df, context=context, semantic_mode="auto", return_report=True, verbose=False
    )
    survivors = out.index
    assert out["monthly_revenue"].equals(df["monthly_revenue"].loc[survivors])
    # Semantic repairs did land elsewhere.
    assert "inactive" in set(out["status"])


def test_protected_column_unchanged_after_apply_plan():
    df = _ecommerce_df().drop_duplicates().reset_index(drop=True)
    context = PROTECT_CONTEXT + "\nAllowed status values are active, inactive, pending."
    plan = fd.suggest_plan(df, context=context, semantic_mode="auto", verbose=False)
    out, report = fd.apply_plan(df, plan)
    assert out["monthly_revenue"].equals(df["monthly_revenue"])
    assert report.decisions_hash


def test_mutable_false_semantic_context_is_hard_protected():
    df = pd.DataFrame({"a": [" x ", "y"], "rev": [" 1 ", " 2 "]})
    out = fd.clean(
        df,
        semantic_context={"columns": {"rev": {"mutable": False}}},
        verbose=False,
    )
    assert out["rev"].equals(df["rev"])  # not stripped, not dtype-coerced
    assert out["a"].tolist() == ["x", "y"]  # everything else still repaired


def test_deliberate_internal_mutation_raises():
    df = pd.DataFrame({"rev": [1, 2, 3]})
    snapshot = snapshot_protected(df, ("rev",))
    broken = df.copy()
    broken.loc[1, "rev"] = 99
    with pytest.raises(ProtectedColumnError, match="rev"):
        verify_protected(broken, snapshot)


def test_dropped_protected_column_raises():
    df = pd.DataFrame({"rev": [1, 2]})
    snapshot = snapshot_protected(df, ("rev",))
    with pytest.raises(ProtectedColumnError, match="dropped"):
        verify_protected(df.drop(columns=["rev"]), snapshot)


def test_dtype_change_raises():
    df = pd.DataFrame({"rev": ["1", "2"]})
    snapshot = snapshot_protected(df, ("rev",))
    changed = df.copy()
    changed["rev"] = changed["rev"].astype(int)
    with pytest.raises(ProtectedColumnError, match="dtype"):
        verify_protected(changed, snapshot)


def test_row_drops_are_not_violations():
    df = pd.DataFrame({"rev": [1, 2, 3, 4]})
    snapshot = snapshot_protected(df, ("rev",))
    verify_protected(df.iloc[[0, 2]], snapshot)  # surviving rows identical


def test_hard_protected_column_resolution():
    config = fd.CleanConfig(
        semantic_context={"columns": {"Monthly Revenue": {"mutable": False}}}
    )
    assert hard_protected_columns(config, ["monthly_revenue", "age"]) == (
        "monthly_revenue",
    )
    assert hard_protected_columns(fd.CleanConfig(), ["a"]) == ()


def test_protected_column_set_include_legacy():
    config = fd.CleanConfig(
        preserve_columns=("keep",), target_column="y", id_columns=("pk",)
    )
    assert protected_column_set(config, ["keep", "y", "pk", "other"]) == ()
    legacy = protected_column_set(
        config, ["keep", "y", "pk", "other"], include_legacy=True
    )
    assert set(legacy) == {"keep", "y", "pk"}


_SENTINEL_POOL = ["ok", " padded ", "N/A", "twenty", "$5", "yes", "no", "", "x-1"]


def test_fuzz_protected_columns_never_change():
    """Property-style fuzz: random frames, random protected columns, many configs."""
    rng = random.Random(20260703)
    for trial in range(25):
        n_rows = rng.randint(2, 12)
        n_cols = rng.randint(2, 5)
        data = {}
        for c in range(n_cols):
            kind = rng.choice(("num", "text", "mixed"))
            if kind == "num":
                data[f"c{c}"] = [
                    rng.choice([rng.uniform(-50, 50), None]) for _ in range(n_rows)
                ]
            elif kind == "text":
                data[f"c{c}"] = [rng.choice(_SENTINEL_POOL) for _ in range(n_rows)]
            else:
                data[f"c{c}"] = [
                    rng.choice([rng.randint(0, 9), "ten", " x ", None])
                    for _ in range(n_rows)
                ]
        df = pd.DataFrame(data)
        protected = rng.sample(list(df.columns), rng.randint(1, n_cols))
        semantic_context = {"columns": {c: {"mutable": False} for c in protected}}
        out = fd.clean(
            df,
            semantic_context=semantic_context,
            semantic_mode=rng.choice([None, "auto", "review"]),
            strategy=rng.choice(["conservative", "balanced", "aggressive"]),
            optimize_memory=rng.choice([True, False]),
            verbose=False,
        )
        for col in protected:
            assert col in out.columns, f"trial {trial}: {col} dropped"
            before = df[col].loc[out.index]
            after = out[col]
            assert str(before.dtype) == str(after.dtype), (
                f"trial {trial}: {col} dtype {before.dtype} -> {after.dtype}"
            )
            assert before.equals(after), f"trial {trial}: {col} values changed"


def test_guard_report_metadata_names_protected_columns():
    df = pd.DataFrame({"rev": [1, 2], "b": [" x", "y "]})
    _, report = fd.clean(
        df,
        semantic_context={"columns": {"rev": {"mutable": False}}},
        return_report=True,
        verbose=False,
    )
    guard_actions = [a for a in report if a.step == "guard"]
    assert guard_actions and guard_actions[0].metadata["protected_columns"] == ["rev"]
