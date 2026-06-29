"""ydata-profiling read-only baseline.

ydata-profiling (formerly pandas-profiling) *describes* a dataset; it does not
repair it. This baseline times report generation and returns the input frame
unchanged, so the harness can compare "time to a report you still have to act
on" against "time to a cleaned frame plus an explained CleanReport". Raises
``ImportError`` when the library is absent; the harness skips it.
"""

from __future__ import annotations

import pandas as pd

from . import count_authored_lines


def _require():
    try:
        from ydata_profiling import ProfileReport  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "ydata-profiling is not installed; install with "
            "`pip install ydata-profiling` to run this baseline."
        ) from exc


def run(df: pd.DataFrame) -> pd.DataFrame:
    _require()
    from ydata_profiling import ProfileReport

    report = ProfileReport(df, minimal=True, progress_bar=False)
    # Force materialisation of the description so timing reflects real work.
    report.get_description()
    return df


AUTHORED_LINES: int = count_authored_lines(run)
REPAIRS = False
