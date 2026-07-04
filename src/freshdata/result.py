"""User-facing result objects for standard FreshData workflows."""

from __future__ import annotations

import pandas as pd

from .report import CleanReport


class CleanResult(pd.DataFrame):
    """A cleaned ``DataFrame`` with the run report attached.

    ``CleanResult`` preserves normal DataFrame use while exposing the common
    report workflow directly on the object returned by ``fd.clean(df)``.
    Operations derived from it intentionally return plain DataFrames so the
    report does not leak into unrelated frames.
    """

    _metadata = ["_freshdata_report"]

    @property
    def _constructor(self) -> type[pd.DataFrame]:
        return pd.DataFrame

    @classmethod
    def wrap(cls, frame: pd.DataFrame, report: CleanReport) -> CleanResult:
        result = cls(frame)
        result._freshdata_report = report
        return result

    @property
    def data(self) -> pd.DataFrame:
        """Return the cleaned data as a plain pandas ``DataFrame``."""
        return pd.DataFrame(self, copy=False)

    def report(self) -> CleanReport:
        """Return the ``CleanReport`` produced by this clean run."""
        return self._freshdata_report

    def summary(self) -> str:
        """Return the same human-readable audit summary as ``result.report()``."""
        return self.report().summary()

    def visualize(self) -> str:
        """Return a self-contained HTML visualization of the clean report."""
        return self.report().to_html()
