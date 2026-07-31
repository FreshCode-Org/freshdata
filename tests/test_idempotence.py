"""Idempotence invariant: an already-clean frame passes through ``fd.clean``
unchanged and records no data-modifying actions.

See the "does not fire when it shouldn't" testing rule in CONTRIBUTING.md;
issue #164 asks for a shared harness asserting this broadly (several
already-clean frames, not just one column type) instead of relying on each
cleaning step's own narrow non-firing test.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd


def _clean_numeric_frame() -> pd.DataFrame:
    """Plain int/float columns, already in range with no missing values."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "score": [10.1, 10.5, 9.8, 10.2, 10.0, 9.9],
        }
    )


def _clean_boolean_frame() -> pd.DataFrame:
    """A native bool column plus a tightly clustered numeric column."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "is_active": [True, False, True, False, True, False],
            "amount": [12.0, 13.5, 11.25, 14.0, 12.75, 13.25],
        }
    )


def _clean_datetime_frame() -> pd.DataFrame:
    """An already-parsed datetime64 column."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "event_date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            ),
            "amount": [100.0, 102.5, 98.0, 101.25, 99.75],
        }
    )


def _clean_text_frame() -> pd.DataFrame:
    """Low-cardinality text with no whitespace/sentinel values."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "category": ["red", "blue", "green", "blue", "red", "green"],
            "count": [3, 5, 2, 6, 4, 3],
        }
    )


CLEAN_FRAME_FACTORIES = {
    "numeric": _clean_numeric_frame,
    "boolean": _clean_boolean_frame,
    "datetime": _clean_datetime_frame,
    "text": _clean_text_frame,
}


@pytest.fixture(params=sorted(CLEAN_FRAME_FACTORIES), ids=sorted(CLEAN_FRAME_FACTORIES))
def clean_frame(request) -> pd.DataFrame:
    """One of several already-clean frames spanning different dtypes."""
    return CLEAN_FRAME_FACTORIES[request.param]()


def test_clean_frame_stays_clean(clean_frame):
    """fd.clean on an already-clean frame is a no-op: same data, zero actions."""
    snapshot = clean_frame.copy(deep=True)
    out, report = fd.clean(clean_frame, report=True)

    pd.testing.assert_frame_equal(out, snapshot)
    assert report.cells_changed == 0
    assert not any(action.count for action in report)


def test_already_clean_conftest_fixture_stays_clean(already_clean):
    """The shared conftest ``already_clean`` fixture satisfies the same invariant."""
    snapshot = already_clean.copy(deep=True)
    out, report = fd.clean(already_clean, report=True)

    pd.testing.assert_frame_equal(out, snapshot)
    assert report.cells_changed == 0


@pytest.mark.parametrize(
    "factory", CLEAN_FRAME_FACTORIES.values(), ids=CLEAN_FRAME_FACTORIES.keys()
)
def test_clean_output_is_itself_already_clean(factory):
    """Running fd.clean on its own output is also a no-op (true idempotence)."""
    df = factory()
    once = fd.clean(df)
    twice = fd.clean(once)
    pd.testing.assert_frame_equal(once, twice)


def test_idempotence_guard_would_catch_a_mutated_clean_column():
    """Local sanity check for the invariant above: introduce a stray value that
    a cleaning step genuinely repairs (surrounding whitespace, always trimmed
    by the deterministic ``strip_whitespace`` step), and confirm the guard
    actually fires instead of being vacuously true.
    """
    mutated = _clean_text_frame()
    mutated.loc[0, "category"] = "  red  "

    out, report = fd.clean(mutated, report=True)

    assert report.cells_changed > 0
    assert out.loc[0, "category"] == "red"
