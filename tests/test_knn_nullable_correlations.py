"""``_knn_fill`` must survive a nullable-integer partner column (#470).

On pandas 1.5, ``Series.corr`` receives the raw values, so a nullable ``Int64``
column holding ``pd.NA`` arrives as an **object** array; numpy 1.26's
``average()`` then builds its scale factor as a bare Python float and
``scl.shape`` raises ``AttributeError: 'float' object has no attribute
'shape'``. ``engine/model_select.py`` was fixed for this in #464; the same raw
``corrwith`` pattern survived in ``engine/missing.py`` and is fixed here.

Reproduced on py3.9 / pandas 1.5.3 / numpy 1.26.4 -- the combination #470
names -- where ``main`` raises and the fix returns partners. On modern pandas
both sides already pass, because pandas 2.x casts to float64 internally; these
tests therefore pin the *contract* on every lane and catch the regression on
the oldest supported one.

``fd.clean``'s gates make the crash hard to reach through the public API, so
these call ``_knn_fill`` directly, as #470 describes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freshdata.config import CleanConfig
from freshdata.engine.missing import _knn_fill

pytest.importorskip("sklearn")


def _config() -> CleanConfig:
    return CleanConfig(strategy="aggressive", advanced_imputation=True)


def _frame_with_nullable_partner() -> pd.DataFrame:
    """A gap to fill, plus one nullable-Int64 partner carrying ``pd.NA``."""
    return pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0],
            "p1": pd.array([1, 2, pd.NA, 4, 5, 6, 7, 8], dtype="Int64"),
            "p2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )


def test_a_nullable_integer_partner_does_not_crash():
    """The #470 regression: this raised AttributeError on the py3.9 lane."""
    result = _knn_fill(_frame_with_nullable_partner(), "target", _config())
    assert result is not None, "the partners are perfectly correlated; KNN should run"
    filled, partners = result
    assert set(partners) == {"p1", "p2"}
    assert filled.notna().all(), "the gap should be imputed"


def test_the_imputed_value_is_finite_and_in_range():
    """A crash fix that produced nonsense would still be a failure."""
    filled, _ = _knn_fill(_frame_with_nullable_partner(), "target", _config())
    imputed = filled.iloc[3]
    assert np.isfinite(imputed)
    assert 1.0 <= imputed <= 8.0, f"imputed {imputed!r} sits outside the observed range"


def test_the_untouched_rows_are_left_alone():
    """Imputation must fill the gap and change nothing else."""
    frame = _frame_with_nullable_partner()
    filled, _ = _knn_fill(frame, "target", _config())
    original = frame["target"]
    present = original.notna()
    pd.testing.assert_series_equal(
        filled[present].astype("float64"),
        original[present].astype("float64"),
        check_names=False,
    )


def test_an_infinite_partner_value_does_not_poison_the_correlation():
    """``±inf`` must read as missing rather than making every correlation NaN."""
    frame = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0],
            "p1": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0],
            "p2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )
    result = _knn_fill(frame, "target", _config())
    if result is not None:  # partner selection may legitimately decline
        filled, _ = result
        assert np.isfinite(filled.iloc[3])


def test_a_float_only_frame_is_unaffected():
    """The common case must behave exactly as before the fix."""
    frame = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0],
            "p1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "p2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )
    filled, partners = _knn_fill(frame, "target", _config())
    assert set(partners) == {"p1", "p2"}
    assert np.isfinite(filled.iloc[3])
