"""Apache Airflow integration for freshdata's trust gate.

Exposes :class:`FreshDataCleanOperator` — a ``BaseOperator`` that pulls a DataFrame
from an upstream task via XCom, cleans and gates it with
:func:`~freshdata.integrations.evaluate_trust_gate`, pushes the cleaned frame and the
gate result back to XCom, and reacts to a low score:

* ``on_low_score="fail"`` -> raises ``AirflowException`` (task fails),
* ``on_low_score="skip"`` -> raises ``AirflowSkipException`` (downstream skips),
* ``on_low_score="warn"`` -> logs a warning and continues.

Airflow is imported lazily, so ``import freshdata.integrations.airflow`` succeeds even
when Airflow is not installed; the framework is required only when you construct the
operator. Install with ``pip install "freshdata[airflow]"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._core import OnLowScore, evaluate_trust_gate

if TYPE_CHECKING:  # annotations only
    from freshdata import CleanConfig

__all__ = ["FreshDataCleanOperator"]

_AIRFLOW_HINT = (
    "The Airflow integration requires apache-airflow. Install it with: "
    'pip install "freshdata[airflow]"'
)


def _require_airflow() -> tuple[Any, Any, Any]:
    """Return ``(BaseOperator, AirflowException, AirflowSkipException)`` or raise."""
    try:
        from airflow.exceptions import AirflowException, AirflowSkipException
        from airflow.models import BaseOperator
    except ImportError as exc:  # pragma: no cover - exercised via mocking
        raise ImportError(_AIRFLOW_HINT) from exc
    return BaseOperator, AirflowException, AirflowSkipException


def _build_clean_operator() -> type:
    """Build the :class:`FreshDataCleanOperator` class (requires Airflow)."""
    base_operator, airflow_exception, airflow_skip_exception = _require_airflow()

    class FreshDataCleanOperator(base_operator):  # type: ignore[valid-type, misc]
        """Clean + trust-gate a DataFrame pulled from an upstream task's XCom."""

        def __init__(
            self,
            *,
            input_task_id: str,
            input_xcom_key: str = "return_value",
            output_xcom_key: str = "return_value",
            clean_config: CleanConfig | None = None,
            trust_score_threshold: float = 80.0,
            on_low_score: OnLowScore = "warn",
            publish_full_report: bool = False,
            system_actor: str = "freshdata",
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.input_task_id = input_task_id
            self.input_xcom_key = input_xcom_key
            self.output_xcom_key = output_xcom_key
            self.clean_config = clean_config
            self.trust_score_threshold = trust_score_threshold
            self.on_low_score = on_low_score
            self.publish_full_report = publish_full_report
            self.system_actor = system_actor

        def execute(self, context: Any) -> Any:
            """Pull, clean, gate, and push; fail/skip/warn per ``on_low_score``."""
            task_instance = context["ti"]
            df = task_instance.xcom_pull(
                task_ids=self.input_task_id, key=self.input_xcom_key
            )
            if df is None:
                raise airflow_exception(
                    f"freshdata: no DataFrame found in XCom from task "
                    f"{self.input_task_id!r} (key={self.input_xcom_key!r})."
                )

            cleaned, result = evaluate_trust_gate(
                df,
                clean_config=self.clean_config,
                trust_score_threshold=self.trust_score_threshold,
                on_low_score=self.on_low_score,
                publish_full_report=self.publish_full_report,
                system_actor=self.system_actor,
            )

            task_instance.xcom_push(key=self.output_xcom_key, value=cleaned)
            task_instance.xcom_push(key=f"{self.output_xcom_key}__gate", value=result.to_dict())

            if result.should_fail:
                raise airflow_exception(result.message)
            if result.should_skip:
                raise airflow_skip_exception(result.message)
            if not result.passed:
                self.log.warning(result.message)
            return cleaned

    return FreshDataCleanOperator


_operator_cls: type | None = None


def __getattr__(name: str) -> Any:
    """Lazily build :class:`FreshDataCleanOperator` on first access (needs Airflow)."""
    if name == "FreshDataCleanOperator":
        global _operator_cls
        if _operator_cls is None:
            _operator_cls = _build_clean_operator()
        return _operator_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
