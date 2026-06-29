"""sweetviz read-only baseline.

sweetviz produces an exploratory analysis report; like ydata-profiling it does
not repair data. This baseline times ``analyze()`` and returns the frame
unchanged. Raises ``ImportError`` when sweetviz is missing; the harness skips it.
"""

from __future__ import annotations

import pandas as pd

from . import count_authored_lines


def _require():
    try:
        import sweetviz  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "sweetviz is not installed; install with `pip install sweetviz` "
            "to run this baseline."
        ) from exc


def run(df: pd.DataFrame) -> pd.DataFrame:
    _require()
    import sweetviz

    report = sweetviz.analyze(df)
    # _features is populated as analysis runs; touching it forces the work.
    _ = report._features if hasattr(report, "_features") else None
    return df


AUTHORED_LINES: int = count_authored_lines(run)
REPAIRS = False
