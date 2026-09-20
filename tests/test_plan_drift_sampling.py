"""The frame fingerprint behind drift refusal samples the HEAD, not the frame.

``docs/repair-plans.md`` describes ``FrameSignature`` as "row count, column
names+dtypes, content sample". That wording is accurate but incomplete in a way
that matters: the sample is the first ``_SIGNATURE_SAMPLE_ROWS`` (512) rows, so
a change *after* row 512 that preserves the row count, the column names and the
dtypes does not trip ``PlanDriftError``.

That is a deliberate design choice -- the fingerprint is documented as "cheap"
and is a guard against applying a plan to the *wrong data*, not a proof the
data is unchanged. These tests pin the boundary so the documented claim is
backed by execution rather than prose, and so a future change to the sampling
strategy has to come past a failing test.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.repairplan import _SIGNATURE_SAMPLE_ROWS, compute_frame_signature


def _frame(n: int) -> pd.DataFrame:
    # A tie-free majority so the plan carries a stable semantic action.
    lower = max(1, n // 4)
    return pd.DataFrame({"country": ["USA"] * (n - lower) + ["usa"] * lower})


def _plan(df: pd.DataFrame):
    return fd.suggest_plan(df, semantic_mode="review")


def test_the_documented_sample_size_is_the_one_the_code_uses():
    assert _SIGNATURE_SAMPLE_ROWS == 512


def test_a_change_inside_the_head_sample_is_refused():
    df = _frame(1000)
    plan = _plan(df)
    drifted = df.copy()
    drifted.loc[10, "country"] = "COMPLETELY-DIFFERENT"
    with pytest.raises(fd.PlanDriftError):
        fd.apply_plan(drifted, plan)


def test_a_change_beyond_the_head_sample_is_not_detected():
    """Documented limitation, pinned deliberately -- not an endorsement."""
    df = _frame(1000)
    plan = _plan(df)
    drifted = df.copy()
    drifted.loc[900, "country"] = "COMPLETELY-DIFFERENT"
    fd.apply_plan(drifted, plan)  # no PlanDriftError


def test_the_boundary_sits_exactly_at_the_sample_size():
    df = _frame(_SIGNATURE_SAMPLE_ROWS + 10)
    base = compute_frame_signature(df)

    last_seen = df.copy()
    last_seen.loc[_SIGNATURE_SAMPLE_ROWS - 1, "country"] = "CHANGED"
    assert compute_frame_signature(last_seen).sample_hash != base.sample_hash

    first_unseen = df.copy()
    first_unseen.loc[_SIGNATURE_SAMPLE_ROWS, "country"] = "CHANGED"
    assert compute_frame_signature(first_unseen).sample_hash == base.sample_hash


def test_row_count_and_column_changes_are_still_caught_beyond_the_sample():
    """The sample is only one of three components; the other two still apply."""
    df = _frame(_SIGNATURE_SAMPLE_ROWS + 10)
    base = compute_frame_signature(df)

    assert compute_frame_signature(df.iloc[:-1]).n_rows != base.n_rows

    renamed = df.rename(columns={"country": "nation"})
    assert compute_frame_signature(renamed).columns_hash != base.columns_hash
