"""A regex rule must not fail a valid code because the column is float64.

A numeric code column with one blank cell loads from CSV as ``float64``, so
``str()`` renders ``10000266`` as ``"10000266.0"`` and the trailing ``0`` reads
as an extra digit. Every row of a perfectly valid column then fails a
digit-only pattern, and the domain trust score drops accordingly.

The repository had already settled this: ``retail/validator.py`` carried
``_integral_float_text`` for exactly the GTIN case, and its docstring described
this same CSV-blank-cell scenario. Only the GTIN checks used it -- the shared
rule engine's ``_check_regex`` did not -- so the helper now lives in
``domains.base`` and every regex rule gets the same rendering.

Both built-in regex rules were affected: GS1-008 (``[0-9]{8}``) and FIN-008
(``[A-Za-z0-9]{4,12}``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freshdata.domains import run_domain
from freshdata.domains.base import integral_float_text

VALID_BRICKS = [10000266, 10000267, 10000268, 10000269]


def _report(result):
    report = result[1] if isinstance(result, tuple) else result
    return getattr(report, "report", report)


def _retail_frame(codes) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gtin": [
                "04012345678901",
                "04012345678902",
                "04012345678903",
                "04012345678904",
            ],
            "description": ["a", "b", "c", "d"],
            "gpc_brick_code": codes,
        }
    )


def _rule(report, rule_id):
    return next((r for r in report.results if r.rule_id == rule_id), None)


def test_a_blank_cell_does_not_invalidate_every_other_code():
    """The regression: one blank made all four valid codes fail GS1-008."""
    floats = [float(c) for c in VALID_BRICKS[:3]] + [np.nan]
    report = _report(run_domain(_retail_frame(floats), "retail"))
    rule = _rule(report, "GS1-008")
    assert rule.passed, "a float64 column of valid codes must still pass"
    assert not rule.violation_rows


def test_the_float_and_integer_columns_agree():
    """Same codes, two dtypes, same verdict and same trust score."""
    as_int = _report(run_domain(_retail_frame(list(VALID_BRICKS)), "retail"))
    floats = [float(c) for c in VALID_BRICKS[:3]] + [np.nan]
    as_float = _report(run_domain(_retail_frame(floats), "retail"))
    assert _rule(as_int, "GS1-008").passed == _rule(as_float, "GS1-008").passed is True
    assert as_int.domain_trust_score == as_float.domain_trust_score


def test_genuinely_invalid_codes_are_still_caught():
    """The fix must not buy its pass rate with a false negative."""
    report = _report(run_domain(_retail_frame([1234, 5678, 91011, 121314]), "retail"))
    rule = _rule(report, "GS1-008")
    assert not rule.passed
    assert len(rule.violation_rows) == 4


def test_the_finance_regex_rule_benefits_too():
    """FIN-008 uses the same shared check, so it had the same defect."""
    frame = pd.DataFrame(
        {
            "txn_id": ["t1", "t2", "t3", "t4"],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "currency": ["USD", "USD", "USD", "USD"],
            "account_code": [10000266.0, 10000267.0, 10000268.0, np.nan],
        }
    )
    rule = _rule(_report(run_domain(frame, "finance")), "FIN-008")
    assert rule is not None, "FIN-008 should be evaluated for this frame"
    assert rule.passed, "a float64 account_code of valid codes must pass"


# -- the helper itself -------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (10000266.0, "10000266"),   # the case that mattered
        (-42.0, "-42"),
        (12.5, 12.5),               # a real decimal keeps its fraction
        (0.1, 0.1),
        ("0250", "0250"),           # text is untouched, padding survives
        (7, 7),                     # ints are already fine
        (None, None),
    ],
)
def test_only_integral_floats_are_rewritten(value, expected):
    assert integral_float_text(value) == expected


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_floats_pass_through(value):
    """``int(nan)`` would raise; these must be returned unchanged."""
    result = integral_float_text(value)
    assert isinstance(result, float)
    assert (np.isnan(result) and np.isnan(value)) or result == value
