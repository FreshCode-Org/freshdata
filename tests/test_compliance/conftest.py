"""Shared fixtures for the compliance generator tests."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small clinical/financial frame exercising PII + missingness."""
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", None, "P004"],
            "email": ["a@b.com", None, "c@d.com", "e@f.com"],
            "age": [34, None, 52, 89],
            "diagnosis": ["A10", "B20", None, "C30"],
            "revenue": [1200.0, None, 3400.0, None],
        }
    )


@pytest.fixture
def sample_report(sample_df: pd.DataFrame) -> fd.CleanReport:
    """The real CleanReport from a default clean (clean returns a tuple)."""
    _, report = fd.clean(sample_df, return_report=True)
    return report


@pytest.fixture
def make_report():
    """Factory: build a synthetic CleanReport from ``Action`` keyword dicts.

    Lets each test pin exact ``step``/``risk``/``confidence``/``count`` values
    without depending on what the engine happens to emit.
    """

    def _make(*action_kwargs: dict, **report_kwargs) -> fd.CleanReport:
        actions = [
            fd.Action(
                step=kw.get("step", "missing"),
                column=kw.get("column"),
                description=kw.get("description", "synthetic action"),
                count=kw.get("count", 0),
                rationale=kw.get("rationale", "engine decision"),
                risk=kw.get("risk", "low"),
                confidence=kw.get("confidence", 1.0),
                model_id=kw.get("model_id", ""),
            )
            for kw in action_kwargs
        ]
        return fd.CleanReport(actions=actions, **report_kwargs)

    return _make


class _TrustStub:
    def __init__(self, overall: float, grade: str = "A"):
        self.overall = overall
        self.grade = grade

    def to_dict(self) -> dict:
        return {"overall": self.overall, "grade": self.grade}


class _MaskStub:
    def __init__(self, columns: dict):
        self.columns = columns
        self.total_cells_masked = sum(1 for _ in columns)


class _ClusterStub:
    def __init__(self, column: str, n_cells_merged: int = 3, n_clusters: int = 2):
        self.column = column
        self.n_cells_merged = n_cells_merged
        self.n_clusters = n_clusters


class EnterpriseStub:
    """Duck-typed stand-in for ``EnterpriseResult`` (no enterprise deps needed)."""

    def __init__(
        self,
        clean_report: fd.CleanReport,
        overall: float = 92.0,
        masked: list[str] | None = None,
        clusters: list[str] | None = None,
    ):
        self.clean_report = clean_report
        self.trust_after = _TrustStub(overall)
        self.trust_before = _TrustStub(40.0)
        self.mask_report = _MaskStub(dict.fromkeys(masked, "sha256+salt")) if masked else None
        self.cluster_results = [_ClusterStub(c) for c in (clusters or [])]


@pytest.fixture
def enterprise_stub():
    """Factory returning an :class:`EnterpriseStub`."""
    return EnterpriseStub
