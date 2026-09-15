"""FreshCore backend adapter.

FreshCore is FreshData's optional native cleaning-first engine. The Rust module
(``freshdata_freshcore``) owns the hot path; this adapter only materializes
Python inputs/outputs, translates :class:`CleanConfig` into a physical plan, and
maps compact native audit events back onto :class:`freshdata.CleanReport`.

If the native module is not installed, or if a config/data shape falls outside
FreshCore v1's parity boundary, the adapter delegates to the pandas reference
pipeline and records an explicit fallback event.
"""

from __future__ import annotations

import importlib
import math
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from ..._util import memory_bytes
from ...config import _DEFAULT_FACTOR
from ...report import CleanReport
from ...steps.columns import normalized_column_labels
from ...steps.duplicates import check_duplicate_ratio, report_detected_duplicates
from ...steps.strings import active_sentinels
from .._base import ExecutionEngine
from .._config import enforce_fallback_policy
from ._pandas import materialize_to_pandas

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from ...config import CleanConfig
    from .._config import EngineConfig

#: Largest integer magnitude float64 represents exactly (native numbers are f64).
_MAX_EXACT_FLOAT_INT = 2**53


class FreshCoreEngine(ExecutionEngine):
    """Optional native backend backed by the ``freshdata_freshcore`` extension."""

    name = "freshcore"

    def supports_source(self, source: Any) -> bool:
        return isinstance(source, (pd.DataFrame, str))

    def execute(
        self,
        source: Any,
        config: CleanConfig,
        engine_config: EngineConfig,
    ) -> tuple[pd.DataFrame, CleanReport]:
        started = time.perf_counter()
        module = self._load_native()
        if module is None:
            return self._fallback(
                source,
                config,
                engine_config,
                "module",
                "freshdata_freshcore is not installed",
            )

        frame = materialize_to_pandas(source)
        reason = self._unsupported_reason(frame, config)
        if reason is not None:
            return self._fallback(source, config, engine_config, "pipeline", reason)

        try:
            payload = self._payload(frame, config)
            native = module.execute_plan(payload)
        except Exception as exc:  # pragma: no cover - defensive around native boundary
            return self._fallback(
                source, config, engine_config, "native_error", f"FreshCore failed: {exc}"
            )

        if (
            not config.drop_duplicates
            and config.duplicate_ratio_action == "error"
            and native.get("duplicates_detected") is None
        ):
            # Detection-only dedup needs the native duplicate count to honour
            # the escalation. Current native modules always report it when
            # drop_duplicates is False; modules built before #323 do not, and
            # without it the error could never fire.
            return self._fallback(
                source,
                config,
                engine_config,
                "drop_duplicates",
                'duplicate_ratio_action="error" needs a duplicate-row count, '
                "which this FreshCore native module does not report",
            )

        cleaned, dtype_changes = self._frame_from_native(native, frame, config)
        report = self._report_from_native(frame, cleaned, native, started, config)
        for column, detail in dtype_changes:
            report.record_backend_difference("freshcore", "dtypes", detail, column=column)
        return cleaned, report

    @staticmethod
    def _load_native() -> Any | None:
        try:
            return importlib.import_module("freshdata_freshcore")
        except ImportError:
            return None

    def _fallback(
        self,
        source: Any,
        config: CleanConfig,
        engine_config: EngineConfig,
        step: str,
        reason: str,
    ) -> tuple[pd.DataFrame, CleanReport]:
        from ...cleaner import run_pipeline

        enforce_fallback_policy(engine_config, "freshcore", step, reason)
        frame = materialize_to_pandas(source)
        cleaned, report = run_pipeline(frame, config)
        report.backend = "pandas"
        report.record_fallback("freshcore", step, reason)
        return cleaned, report

    def _unsupported_reason(self, frame: pd.DataFrame, config: CleanConfig) -> str | None:
        if config.semantic_enabled:
            return "semantic cleaning requires the pandas in-memory path"
        if config.context is not None or config.policy is not None:
            return "context/policy protection requires the pandas in-memory path"
        if config.engine_mode is not None:
            return (
                f"strategy={config.strategy!r} runs the accuracy-first decision engine, "
                "which FreshCore v1 delegates to pandas"
            )
        if config.drop_constant_columns:
            return "drop_constant_columns is not implemented in FreshCore v1"
        if config.optimize_memory:
            return "optimize_memory is not implemented in FreshCore v1"
        # Same wording as PlanGenerator.fallback_reason() so fd.plan() and the
        # recorded fallback event agree.
        if config.impute == "missforest":
            return "missforest imputation uses scikit-learn and is evaluated by the pandas backend"
        if config.impute_strategy:
            return "impute_strategy per-column overrides are evaluated by the pandas backend"
        if config.duplicate_subset is not None:
            return "duplicate_subset is not implemented in FreshCore v1"
        if config.duplicate_keep not in ("first", "last"):
            return f"duplicate_keep={config.duplicate_keep!r} is not implemented in FreshCore v1"
        if config.outliers is not None and config.outlier_method not in ("iqr", "zscore"):
            return f"outlier_method={config.outlier_method!r} is not implemented in FreshCore v1"
        if config.outliers == "clip":
            return ("skew-aware capping fences are not implemented in FreshCore v1; "
                    "clip requires the pandas reference path")
        if config.preserve_columns or config.id_columns or config.target_column is not None:
            return "protected/id/target column semantics require the pandas reference path"
        if frame.columns.duplicated().any():
            return "duplicate input column labels require the pandas reference path"
        colliding = self._colliding_labels(frame, config)
        if colliding:
            return (
                f"column labels {self._shown(colliding)} collide once stringified: "
                "FreshCore v1 names columns by str(label), so distinct labels such as "
                "1 and '1' require the pandas reference path"
            )
        if not isinstance(frame.index, pd.RangeIndex):
            return "non-default pandas index semantics require the pandas reference path"
        non_scalar, kinds = self._non_scalar_dtype_columns(frame)
        if non_scalar:
            return (
                f"{'/'.join(kinds)} column(s) {self._shown(non_scalar)}: FreshCore v1 "
                "carries only float, bool and string columns, so datetime, timedelta, "
                "categorical, period and interval dtypes require the pandas reference path"
            )
        wide = self._wide_integer_columns(frame)
        if wide:
            return (
                f"integer column(s) {self._shown(wide)} hold values beyond ±2**53: "
                "FreshCore v1 carries numbers as float64, which cannot represent them "
                "exactly, so they require the pandas reference path"
            )
        if self._has_unsupported_object_values(frame):
            return "mixed object columns with non-string values require the pandas reference path"
        if config.outliers is not None:
            non_finite = self._non_finite_float_columns(frame)
            if non_finite:
                return (
                    f"non-finite values in numeric column(s) {self._shown(non_finite)}: "
                    "FreshCore v1 outlier fences do not exclude ±inf, so outlier "
                    "handling requires the pandas reference path"
                )
        if config.impute in ("mode", "auto"):
            bools = self._boolean_columns_to_impute(frame)
            if bools:
                return (
                    f"missing values in boolean column(s) {self._shown(bools)}: "
                    "FreshCore v1 does not impute boolean columns, so imputation "
                    "requires the pandas reference path"
                )
        return None

    @staticmethod
    def _shown(columns: Sequence[object], limit: int = 5) -> str:
        shown = ", ".join(repr(c) for c in columns[:limit])
        extra = len(columns) - limit
        return f"{shown} (+{extra} more)" if extra > 0 else shown

    @staticmethod
    def _output_labels(frame: pd.DataFrame, config: CleanConfig) -> list[object]:
        """Column labels after the (optional) column-name normalization step."""
        if config.column_names:
            return normalized_column_labels(frame.columns)
        return list(frame.columns)

    def _colliding_labels(self, frame: pd.DataFrame, config: CleanConfig) -> list[object]:
        """Distinct labels whose ``str()`` clashes before or after renaming.

        The native module keys columns by string name, so ``1`` and ``"1"``
        (or ``1`` and ``" 1 "``, which normalizes to ``"1"``) would collapse.
        """
        clashing: set[int] = set()
        for labels in (list(frame.columns), self._output_labels(frame, config)):
            names = [str(label) for label in labels]
            counts = Counter(names)
            clashing.update(i for i, name in enumerate(names) if counts[name] > 1)
        return [frame.columns[i] for i in sorted(clashing)]

    @staticmethod
    def _non_scalar_dtype_columns(frame: pd.DataFrame) -> tuple[list[object], list[str]]:
        """Columns whose dtype the native float/bool/string arrays cannot carry."""
        found: list[object] = []
        kinds: list[str] = []
        for i, col in enumerate(frame.columns):
            dtype = frame.iloc[:, i].dtype
            if isinstance(dtype, pd.CategoricalDtype):
                kind = "categorical"
            elif isinstance(dtype, pd.PeriodDtype):
                kind = "period"
            elif isinstance(dtype, pd.IntervalDtype):
                kind = "interval"
            elif is_datetime64_any_dtype(dtype):
                kind = "datetime"
            elif isinstance(dtype, np.dtype) and dtype.kind == "m":
                kind = "timedelta"
            else:
                continue
            found.append(col)
            if kind not in kinds:
                kinds.append(kind)
        return found, kinds

    @staticmethod
    def _wide_integer_columns(frame: pd.DataFrame) -> list[object]:
        """Integer columns holding a value float64 cannot represent exactly."""
        found: list[object] = []
        for i, col in enumerate(frame.columns):
            s = frame.iloc[:, i]
            if is_bool_dtype(s) or not is_integer_dtype(s):
                continue
            info = np.iinfo(getattr(s.dtype, "numpy_dtype", s.dtype))
            if int(info.min) >= -_MAX_EXACT_FLOAT_INT and int(info.max) <= _MAX_EXACT_FLOAT_INT:
                continue  # int8..int32 and their unsigned widths always fit
            non_null = s.dropna()
            if non_null.empty:
                continue
            if int(non_null.max()) > _MAX_EXACT_FLOAT_INT or (
                int(non_null.min()) < -_MAX_EXACT_FLOAT_INT
            ):
                found.append(col)
        return found

    @staticmethod
    def _non_finite_float_columns(frame: pd.DataFrame) -> list[str]:
        """Columns holding ±inf. Only float dtypes can store infinities."""
        found: list[str] = []
        for i, col in enumerate(frame.columns):
            s = frame.iloc[:, i]
            if not is_float_dtype(s):
                continue
            if np.isinf(s.to_numpy(dtype="float64", na_value=np.nan)).any():
                found.append(str(col))
        return found

    @staticmethod
    def _boolean_columns_to_impute(frame: pd.DataFrame) -> list[str]:
        """Boolean columns the pandas imputer would fill (some, not all, missing)."""
        found: list[str] = []
        for i, col in enumerate(frame.columns):
            s = frame.iloc[:, i]
            if not is_bool_dtype(s):
                continue
            n_missing = int(s.isna().sum())
            if 0 < n_missing < len(s):
                found.append(str(col))
        return found

    @staticmethod
    def _has_unsupported_object_values(frame: pd.DataFrame) -> bool:
        for col in frame.columns:
            s = frame[col]
            if not is_object_dtype(s):
                continue
            non_null = s.dropna()
            if not non_null.map(lambda v: isinstance(v, str)).all():
                return True
        return False

    def _payload(self, frame: pd.DataFrame, config: CleanConfig) -> dict[str, Any]:
        renamed = self._output_labels(frame, config)
        rename_map = [
            (str(old), str(new))
            for old, new in zip(frame.columns, renamed)
            if isinstance(old, str) and old != new
        ]
        return {
            "columns": [
                self._column_payload(str(name), frame.iloc[:, i])
                for i, name in enumerate(frame.columns)
            ],
            "config": {
                "rename_map": rename_map,
                "strip_whitespace": config.strip_whitespace,
                "normalize_sentinels": config.normalize_sentinels,
                "sentinels": sorted(active_sentinels(config)),
                "string_case": config.string_case,
                "drop_empty_columns": config.drop_empty_columns,
                "drop_empty_rows": config.drop_empty_rows,
                "drop_duplicates": config.drop_duplicates,
                "duplicate_keep": config.duplicate_keep,
                "fix_dtypes": config.fix_dtypes,
                "numeric_threshold": config.numeric_threshold,
                "preserve_leading_zeros": config.preserve_leading_zeros,
                "impute": config.impute,
                "outliers": config.outliers,
                "outlier_method": config.outlier_method,
                "outlier_factor": (
                    config.outlier_factor
                    if config.outlier_factor is not None
                    else _DEFAULT_FACTOR[config.outlier_method]
                ),
            },
        }

    @staticmethod
    def _column_payload(name: str, series: pd.Series) -> dict[str, Any]:
        if is_bool_dtype(series):
            bool_values: list[bool | None] = [
                None if pd.isna(v) else bool(v) for v in series.tolist()
            ]
            return {"name": name, "dtype": "bool", "values": bool_values}
        if is_numeric_dtype(series):
            float_values: list[float | None] = [
                None if pd.isna(v) else float(v) for v in series.tolist()
            ]
            return {"name": name, "dtype": "float", "values": float_values}
        string_values: list[str | None] = [
            None if pd.isna(v) else str(v) for v in series.tolist()
        ]
        return {"name": name, "dtype": "string", "values": string_values}

    def _frame_from_native(
        self,
        native: dict[str, Any],
        original: pd.DataFrame | None = None,
        config: CleanConfig | None = None,
    ) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
        """Rebuild the cleaned frame and list the dtype changes it could not undo.

        Native column names are ``str(label)``; with *original* and *config*
        they are mapped back to the original (possibly renamed) labels, and
        integer columns are cast back to their input dtype when possible.
        """
        sources: dict[str, tuple[object, pd.Series]] = {}
        if original is not None and config is not None:
            for i, label in enumerate(self._output_labels(original, config)):
                sources[str(label)] = (label, original.iloc[:, i])
        labels: list[object] = []
        data: dict[int, Any] = {}
        changes: list[tuple[str, str]] = []
        for position, column in enumerate(native["columns"]):
            name = column["name"]
            values = column["values"]
            dtype = column.get("dtype")
            label, source = sources.get(name, (name, None))
            labels.append(label)
            if dtype == "bool":
                data[position] = pd.Series(values, dtype="boolean")
            elif dtype == "float" and source is not None and is_integer_dtype(source):
                restored, detail = self._restore_integer(values, source.dtype)
                data[position] = restored
                if detail is not None:
                    changes.append((str(label), detail))
            else:
                data[position] = values
        cleaned = pd.DataFrame(data)
        cleaned.columns = pd.Index(labels)
        return cleaned, changes

    @staticmethod
    def _restore_integer(values: list[Any], dtype: Any) -> tuple[Any, str | None]:
        """Cast native float values back to the input integer *dtype* when exact."""
        nullable = not isinstance(dtype, np.dtype)
        info = np.iinfo(getattr(dtype, "numpy_dtype", dtype))
        ints: list[int | None] = []
        problem: str | None = None
        for v in values:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                if not nullable:
                    problem = "missing values, which a non-nullable integer dtype cannot hold"
                    break
                ints.append(None)
            elif not (math.isfinite(v) and float(v).is_integer()):
                problem = "non-integral values"
                break
            elif not int(info.min) <= int(v) <= int(info.max):
                problem = f"values outside the {dtype} range"
                break
            else:
                ints.append(int(v))
        if problem is None:
            if nullable:
                return pd.array(ints, dtype=dtype), None
            return np.array(ints, dtype=dtype), None
        detail = (
            f"FreshCore v1 returned {problem} for this {dtype} input column, "
            "so it comes back as float64"
        )
        return pd.Series(values, dtype="float64"), detail

    def _report_from_native(
        self,
        original: pd.DataFrame,
        cleaned: pd.DataFrame,
        native: dict[str, Any],
        started: float,
        config: CleanConfig,
    ) -> CleanReport:
        report = CleanReport(
            rows_before=int(native.get("rows_before", len(original))),
            rows_after=int(native.get("rows_after", len(cleaned))),
            cols_before=int(native.get("cols_before", original.shape[1])),
            cols_after=int(native.get("cols_after", cleaned.shape[1])),
            memory_before=memory_bytes(original),
            memory_after=memory_bytes(cleaned),
            missing_before=int(native.get("missing_before", original.isna().sum().sum())),
            missing_after=int(native.get("missing_after", cleaned.isna().sum().sum())),
            duration_seconds=time.perf_counter() - started,
        )
        report.backend = "freshcore"
        report.duplicates_removed = int(native.get("duplicates_removed", 0))
        report.outliers_handled = int(native.get("outliers_handled", 0))
        report.columns_dropped.extend(str(c) for c in native.get("columns_dropped", []))
        report.columns_imputed.extend(str(c) for c in native.get("columns_imputed", []))
        detection_reported = False
        for action in native.get("actions", []):
            step = str(action["step"])
            count = int(action.get("count", 0))
            if not detection_reported and step in ("impute", "outliers"):
                # The pandas step records detection before imputation/outliers.
                self._report_detected_duplicates(native, config, report)
                detection_reported = True
            high_ratio = False
            if step == "drop_duplicates" and config.drop_duplicates:
                # Raises under duplicate_ratio_action="error", like the pandas step.
                high_ratio = check_duplicate_ratio(
                    count, self._rows_entering_dedup(report), config
                )
            report.add(
                step,
                str(action["description"]),
                column=action.get("column"),
                count=count,
                risk="medium" if high_ratio else "low",
            )
            if high_ratio:
                n_before = self._rows_entering_dedup(report)
                report.add_warning(
                    f"duplicate ratio {100.0 * count / n_before:.1f}% exceeds "
                    f"duplicate_threshold ({100 * config.duplicate_threshold:.0f}%); "
                    "check for an upstream join or export problem"
                )
                report.add_recommendation(
                    "review why so many rows were duplicated before trusting downstream stats"
                )
        if not detection_reported:
            self._report_detected_duplicates(native, config, report)
        for stage, seconds in native.get("stage_timings", []):
            report.record_stage_timing("freshcore", str(stage), float(seconds))
        if any(a.step == "fix_dtypes" for a in report.actions):
            report.record_backend_difference(
                "freshcore",
                "fix_dtypes",
                "FreshCore v1 casts booleans and numeric-looking strings natively; "
                "date-like strings are profiled but left to pandas fallback when exact "
                "datetime dtype parity is required.",
            )
        return report

    @staticmethod
    def _rows_entering_dedup(report: CleanReport) -> int:
        """Rows reaching the dedup stage: the pandas step runs after empty-row removal."""
        dropped = sum(a.count for a in report.actions if a.step == "drop_empty_rows")
        return report.rows_before - dropped

    def _report_detected_duplicates(
        self, native: dict[str, Any], config: CleanConfig, report: CleanReport
    ) -> None:
        """Detection-only duplicate reporting from the native duplicate count.

        Skipped when the native module does not report ``duplicates_detected``
        (``execute`` falls back first if ``duplicate_ratio_action="error"``).
        """
        if config.drop_duplicates:
            return
        detected = native.get("duplicates_detected")
        if detected is None:
            return
        report_detected_duplicates(
            int(detected), self._rows_entering_dedup(report), config, report
        )
