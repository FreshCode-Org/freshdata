"""freshdata.compliance — map a :class:`~freshdata.CleanReport` to named control frameworks.

>>> import freshdata as fd
>>> from freshdata.compliance import generate_compliance_report, ComplianceConfig
>>> _, report = fd.clean(df, return_report=True)
>>> bundle = generate_compliance_report(report, frameworks=["21cfr_11", "hipaa_safe_harbor"])
>>> bundle.summary()["hipaa_safe_harbor"]["passed"]          # doctest: +SKIP
True

The generators are purely additive and never mutate the input report. Each maps
freshdata's transformation actions onto a regulatory control framework and emits
a standards-grade audit artifact. Pass an optional ``dataframe=`` (to recover
per-column roles and missing ratios via :func:`freshdata.infer_roles`) and/or an
``enterprise_result=`` (to fold in the 0–100 Data Trust Score, PII-masking
events, and fuzzy-clustering lineage) when richer evidence is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._21cfr11 import generate_21cfr11
from ._adapter import ComplianceContext, NormalizedAction, build_context
from ._alcoa import generate_alcoa
from ._base import (
    GENERAL_CAVEAT,
    ComplianceBundle,
    ComplianceConfig,
    ComplianceGapError,
    FrameworkReport,
)
from ._gdpr import generate_gdpr
from ._hipaa import HIPAA_IDENTIFIERS, generate_hipaa
from ._sox404 import generate_sox404

if TYPE_CHECKING:  # annotations only — avoids an import cycle at module load
    import pandas as pd

#: Public framework keys -> their generator.
FRAMEWORK_MAP = {
    "21cfr_11": generate_21cfr11,
    "gdpr_30": generate_gdpr,
    "alcoa_plus": generate_alcoa,
    "sox_404": generate_sox404,
    "hipaa_safe_harbor": generate_hipaa,
}


def generate_compliance_report(
    report: Any,
    frameworks: list[str],
    config: ComplianceConfig | None = None,
    *,
    dataframe: pd.DataFrame | None = None,
    enterprise_result: Any = None,
) -> ComplianceBundle:
    """Generate a :class:`ComplianceBundle` for the requested ``frameworks``.

    Parameters
    ----------
    report:
        A :class:`freshdata.CleanReport`, or an enterprise result exposing
        ``clean_report`` + ``trust_after`` (its embedded report and trust/mask
        data are then used automatically).
    frameworks:
        One or more of ``FRAMEWORK_MAP`` keys.
    config:
        Optional :class:`ComplianceConfig`; sensible defaults are used otherwise.
    dataframe:
        Optional source frame used to recover per-column roles and missing
        ratios via :func:`freshdata.infer_roles`.
    enterprise_result:
        Optional enterprise result supplying the 0–100 trust score, PII-masking
        events, and clustering lineage.

    Raises
    ------
    ValueError:
        If any requested framework key is unknown.
    ComplianceGapError:
        If HIPAA gaps exist and ``config.fail_on_hipaa_gap`` is ``True``.
    """
    config = config or ComplianceConfig()
    requested = list(frameworks)
    unknown = set(requested) - set(FRAMEWORK_MAP)
    if unknown:
        raise ValueError(f"Unknown frameworks: {sorted(unknown)}. Valid: {list(FRAMEWORK_MAP)}")
    # Build the shared context once so a single session id + timestamp are
    # threaded through every framework (determinism aside from entry uuids).
    ctx = build_context(report, config, dataframe=dataframe, enterprise_result=enterprise_result)
    results = {key: FRAMEWORK_MAP[key](ctx, config) for key in requested}
    return ComplianceBundle(results)


__all__ = [
    "FRAMEWORK_MAP",
    "GENERAL_CAVEAT",
    "HIPAA_IDENTIFIERS",
    "ComplianceBundle",
    "ComplianceConfig",
    "ComplianceContext",
    "ComplianceGapError",
    "FrameworkReport",
    "NormalizedAction",
    "build_context",
    "generate_compliance_report",
    "generate_21cfr11",
    "generate_alcoa",
    "generate_gdpr",
    "generate_hipaa",
    "generate_sox404",
]
