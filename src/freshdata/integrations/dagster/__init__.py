"""Dagster integration for freshdata's trust gate.

Exposes :func:`freshdata_asset_check` — a factory that turns any Dagster asset into
an ``@asset_check`` that cleans the asset's DataFrame, scores it, and emits an
``AssetCheckResult`` (``WARN`` severity by default, ``ERROR`` when
``on_low_score="fail"``) carrying ``freshdata/*`` metadata — and
:class:`FreshDataResource`, a ``ConfigurableResource`` holding shared gate config.

Dagster is imported lazily, so ``import freshdata.integrations.dagster`` succeeds
even when Dagster is not installed; the framework is required only when you call
:func:`freshdata_asset_check` or construct :class:`FreshDataResource`. Install with
``pip install "freshdata[dagster]"``.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from .._core import OnLowScore, TrustGateResult, evaluate_trust_gate

if TYPE_CHECKING:  # annotations only
    import pandas as pd

    from freshdata import CleanConfig

__all__ = ["FreshDataResource", "freshdata_asset_check"]

_DAGSTER_HINT = (
    "The Dagster integration requires dagster. Install it with: "
    'pip install "freshdata[dagster]"'
)


def _require_dagster() -> Any:
    """Import and return the ``dagster`` module, or raise a helpful error."""
    try:
        import dagster
    except ImportError as exc:  # pragma: no cover - exercised via mocking
        raise ImportError(_DAGSTER_HINT) from exc
    return dagster


def _asset_param_name(asset: Any) -> str:
    """Return the parameter name Dagster injects the asset's loaded value under."""
    key = getattr(asset, "key", None)
    path = getattr(key, "path", None)
    if path:
        return str(path[-1])
    op = getattr(asset, "op", None)
    return str(getattr(op, "name", None) or getattr(asset, "__name__", asset))


def _meta_value(dagster: Any, value: Any) -> Any:
    """Wrap a scalar in the most specific ``MetadataValue`` (bool before int)."""
    metadata_value = dagster.MetadataValue
    if isinstance(value, bool):
        return metadata_value.bool(value)
    if isinstance(value, int):
        return metadata_value.int(value)
    if isinstance(value, float):
        return metadata_value.float(value)
    return metadata_value.text(str(value))


def _to_asset_check_result(dagster: Any, result: TrustGateResult) -> Any:
    """Translate a :class:`TrustGateResult` into a Dagster ``AssetCheckResult``."""
    severity = (
        dagster.AssetCheckSeverity.ERROR
        if result.on_low_score == "fail"
        else dagster.AssetCheckSeverity.WARN
    )
    metadata = {key: _meta_value(dagster, val) for key, val in result.as_metadata().items()}
    if result.report_dict is not None:
        metadata["freshdata/report"] = dagster.MetadataValue.json(result.report_dict)
    return dagster.AssetCheckResult(passed=result.passed, severity=severity, metadata=metadata)


def freshdata_asset_check(
    *,
    asset: Any,
    name: str = "freshdata_trust_gate",
    trust_score_threshold: float = 80.0,
    on_low_score: OnLowScore = "warn",
    clean_config: CleanConfig | None = None,
    publish_full_report: bool = False,
    system_actor: str = "freshdata",
    blocking: bool = False,
) -> Any:
    """Build a Dagster ``@asset_check`` that runs freshdata's trust gate over ``asset``.

    The returned check loads the asset's DataFrame (injected by Dagster under the
    asset's name), runs :func:`~freshdata.integrations.evaluate_trust_gate`, and
    returns an ``AssetCheckResult``. ``passed`` mirrors the gate; severity is
    ``ERROR`` when ``on_low_score="fail"`` and ``WARN`` otherwise, so a strict gate
    surfaces as an error in the Dagster UI while a soft gate stays a warning.

    Parameters
    ----------
    asset:
        The Dagster ``AssetsDefinition`` to check.
    name:
        Name for the asset check.
    trust_score_threshold, on_low_score, clean_config, publish_full_report, system_actor:
        Forwarded to :func:`~freshdata.integrations.evaluate_trust_gate`.
    blocking:
        Whether a failed check should block downstream materializations.
    """
    dagster = _require_dagster()
    param_name = _asset_param_name(asset)

    def _freshdata_trust_gate(**kwargs: Any) -> Any:
        df = kwargs[param_name]
        _, result = evaluate_trust_gate(
            df,
            clean_config=clean_config,
            trust_score_threshold=trust_score_threshold,
            on_low_score=on_low_score,
            publish_full_report=publish_full_report,
            system_actor=system_actor,
        )
        return _to_asset_check_result(dagster, result)

    # Give Dagster a single positional input named after the asset so it injects the
    # asset's loaded value.
    _freshdata_trust_gate.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    _freshdata_trust_gate.__name__ = name
    return dagster.asset_check(asset=asset, name=name, blocking=blocking)(_freshdata_trust_gate)


def _make_resource_cls() -> type:
    """Build the :class:`FreshDataResource` class (requires Dagster)."""
    dagster = _require_dagster()

    class FreshDataResource(dagster.ConfigurableResource):  # type: ignore[name-defined]
        """A Dagster resource bundling shared freshdata trust-gate configuration.

        Inject it into assets/ops to clean and gate a DataFrame with consistent
        thresholds:

        >>> @asset
        ... def cleaned(raw, freshdata: FreshDataResource):
        ...     df, result = freshdata.gate(raw)
        ...     return df
        """

        trust_score_threshold: float = 80.0
        on_low_score: str = "warn"
        publish_full_report: bool = False
        system_actor: str = "freshdata"

        def gate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TrustGateResult]:
            """Run :func:`evaluate_trust_gate` with this resource's configuration."""
            return evaluate_trust_gate(
                df,
                trust_score_threshold=self.trust_score_threshold,
                on_low_score=self.on_low_score,  # type: ignore[arg-type]
                publish_full_report=self.publish_full_report,
                system_actor=self.system_actor,
            )

    return FreshDataResource


_resource_cls: type | None = None


def __getattr__(name: str) -> Any:
    """Lazily build :class:`FreshDataResource` on first access (needs Dagster)."""
    if name == "FreshDataResource":
        global _resource_cls
        if _resource_cls is None:
            _resource_cls = _make_resource_cls()
        return _resource_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
