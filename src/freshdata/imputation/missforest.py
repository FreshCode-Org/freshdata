"""MissForest-style iterative random-forest imputation.

The class in this module is internal and intentionally pandas-first. It keeps
scikit-learn behind FreshData's existing ``[ml]`` extra and reports every
decision through the normal :class:`~freshdata.report.CleanReport` action log.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .._util import add_column
from ..config import CleanConfig
from ..engine.context import ColumnContext
from ..report import CleanReport

_INSTALL_HINT = 'pip install "freshdata-cleaner[ml]"'
_STEP = "impute"
_TOLERANCE = 1e-3


@dataclass(frozen=True)
class _ColumnPlan:
    column: object
    context: ColumnContext
    missing_mask: pd.Series
    model_type: str


class MissForestImputer:
    """Small internal MissForest-inspired imputer for mixed pandas frames."""

    def __init__(self, config: CleanConfig, report: CleanReport) -> None:
        self.config = config
        self.report = report
        self._last_iterations = 0
        self._last_converged = False
        self._last_delta: float | None = None
        self._last_oob: dict[object, float | None] = {}

    def impute(
        self,
        df: pd.DataFrame,
        columns: list[object],
        contexts: dict[object, ColumnContext],
    ) -> pd.DataFrame:
        """Impute selected *columns* in *df*, returning the same working frame."""
        eligible: list[_ColumnPlan] = []
        for col in columns:
            if col not in df.columns:
                continue
            ctx = contexts[col]
            plan = self._eligible_plan(df, col, ctx)
            if plan is None:
                continue
            eligible.append(plan)

        if not eligible:
            return df

        forests = self._sklearn_forests()
        work = self._initial_filled_frame(df)
        previous = {plan.column: work[plan.column].copy(deep=True) for plan in eligible}
        eligible.sort(key=lambda plan: plan.context.missing_ratio)
        self._last_oob = {}
        self._last_iterations = 0
        self._last_converged = False
        self._last_delta = None

        for iteration in range(1, self.config.missforest_max_iter + 1):
            self._last_iterations = iteration
            for plan in eligible:
                self._fit_predict_column(df, work, plan, forests, iteration)

            delta = self._convergence_delta(work, previous, eligible)
            self._last_delta = delta
            previous = {plan.column: work[plan.column].copy(deep=True) for plan in eligible}
            if delta <= _TOLERANCE:
                self._last_converged = True
                break

        for plan in eligible:
            self._assign_success(df, work, plan)
        return df

    @staticmethod
    def _sklearn_forests() -> tuple[type, type]:
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        except ImportError as exc:
            raise ImportError(
                "MissForest imputation requires scikit-learn. "
                f"Install it with: {_INSTALL_HINT}"
            ) from exc
        return RandomForestRegressor, RandomForestClassifier

    def _fit_predict_column(
        self,
        df: pd.DataFrame,
        work: pd.DataFrame,
        plan: _ColumnPlan,
        forests: tuple[type, type],
        iteration: int,
    ) -> None:
        RandomForestRegressor, RandomForestClassifier = forests
        predictors = [c for c in work.columns if c != plan.column]
        if not predictors:
            self._fallback_fill(
                df,
                plan.column,
                plan.context,
                "no predictor columns available for MissForest",
            )
            return

        observed = df[plan.column].notna()
        missing = plan.missing_mask
        x_train = self._features(work.loc[observed, predictors])
        x_pred = self._features(work.loc[missing, predictors])
        if x_train.empty or x_pred.empty:
            self._fallback_fill(
                df,
                plan.column,
                plan.context,
                "no trainable rows available for MissForest",
            )
            return

        if plan.model_type == "regressor":
            model = RandomForestRegressor(**self._forest_kwargs(iteration))
            y_train = pd.to_numeric(df.loc[observed, plan.column], errors="coerce")
        else:
            model = RandomForestClassifier(**self._forest_kwargs(iteration))
            y_train = self._classification_target(df.loc[observed, plan.column], plan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model.fit(x_train, y_train)
        predicted = pd.Series(model.predict(x_pred), index=x_pred.index)
        if plan.context.role == "boolean":
            predicted = predicted.astype(bool)
        work.loc[missing, plan.column] = predicted
        self._last_oob[plan.column] = self._safe_oob(model)

    def _forest_kwargs(self, iteration: int) -> dict[str, object]:
        return {
            "n_estimators": self.config.missforest_n_estimators,
            "random_state": self.config.missforest_random_state + iteration,
            "n_jobs": 1,
            "oob_score": True,
            "bootstrap": True,
        }

    @staticmethod
    def _classification_target(s: pd.Series, plan: _ColumnPlan) -> pd.Series:
        if plan.context.role == "boolean":
            return s.astype("boolean").astype("Int8")
        return s.astype(object)

    def _eligible_plan(
        self,
        df: pd.DataFrame,
        col: object,
        ctx: ColumnContext,
    ) -> _ColumnPlan | None:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            return None
        reason = self._fallback_reason(df, col, ctx)
        if reason is not None:
            if reason in {
                "dataset has fewer rows than missforest_min_rows_for_model",
                "too few observed rows for MissForest",
                "high-cardinality categorical column is unsafe for MissForest",
            }:
                self._fallback_fill(df, col, ctx, reason)
            else:
                self._preserve(df, col, ctx, reason)
            return None
        if ctx.role == "numeric":
            model_type = "regressor"
        elif ctx.role in ("categorical", "boolean"):
            model_type = "classifier"
        else:
            self._preserve(df, col, ctx, f"role {ctx.role!r} is unsupported by MissForest")
            return None
        return _ColumnPlan(
            column=col,
            context=ctx,
            missing_mask=df[col].isna(),
            model_type=model_type,
        )

    def _fallback_reason(
        self,
        df: pd.DataFrame,
        col: object,
        ctx: ColumnContext,
    ) -> str | None:
        if ctx.role == "target":
            return "target column is never imputed"
        if ctx.role == "id":
            return "identifier column is never imputed"
        if ctx.role == "text":
            return "free-text column is never force-filled"
        if ctx.role == "datetime":
            return "datetime columns are not directly MissForest-imputed"
        if ctx.n_missing == ctx.n_rows:
            return "all values are missing"
        if (
            ctx.missing_ratio > self.config.missing_threshold_high
            and self.config.engine_mode != "aggressive"
        ):
            return "missing ratio exceeds missing_threshold_high outside aggressive mode"
        if ctx.n_rows < self.config.missforest_min_rows_for_model:
            return "dataset has fewer rows than missforest_min_rows_for_model"
        if int(df[col].notna().sum()) < self.config.missforest_min_rows_for_model:
            return "too few observed rows for MissForest"
        if ctx.role in ("categorical", "boolean") and ctx.high_cardinality:
            return "high-cardinality categorical column is unsafe for MissForest"
        if ctx.role not in ("numeric", "categorical", "boolean"):
            return f"role {ctx.role!r} is unsupported by MissForest"
        return None

    def _initial_filled_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in df.columns:
            s = df[col]
            if is_datetime64_any_dtype(s):
                numeric = pd.to_datetime(s).astype("int64").astype("float64")
                numeric[s.isna()] = np.nan
                dt_value = float(numeric.dropna().median()) if numeric.notna().any() else 0.0
                out[col] = numeric.fillna(dt_value)
            elif is_numeric_dtype(s) and not is_bool_dtype(s):
                numeric_value = s.median() if s.notna().any() else 0.0
                out[col] = s.astype("float64").fillna(numeric_value)
            elif is_bool_dtype(s):
                bool_value = _mode_value(s)
                out[col] = s.astype(object).where(
                    s.notna(),
                    bool_value if bool_value is not None else False,
                )
            else:
                fill_value = _mode_value(s)
                out[col] = s.astype(object).where(
                    s.notna(),
                    fill_value if fill_value is not None else "Missing",
                )
        return out

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(index=frame.index)
        for col in frame.columns:
            s = frame[col]
            if is_numeric_dtype(s) and not is_bool_dtype(s):
                encoded[str(col)] = pd.to_numeric(s, errors="coerce").astype("float64").fillna(0.0)
            elif is_bool_dtype(s):
                encoded[str(col)] = s.astype("boolean").astype("Int8").fillna(0).astype("float64")
            elif is_datetime64_any_dtype(s):
                numeric = pd.to_datetime(s).astype("int64").astype("float64")
                numeric[s.isna()] = 0.0
                encoded[str(col)] = numeric
            else:
                codes, _ = pd.factorize(s.astype(object), sort=True, use_na_sentinel=True)
                encoded[str(col)] = (
                    pd.Series(codes, index=s.index).replace(-1, 0).astype("float64")
                )
        return encoded

    def _convergence_delta(
        self,
        work: pd.DataFrame,
        previous: dict[object, pd.Series],
        plans: list[_ColumnPlan],
    ) -> float:
        deltas: list[float] = []
        for plan in plans:
            prev = previous[plan.column].loc[plan.missing_mask]
            cur = work[plan.column].loc[plan.missing_mask]
            if plan.model_type == "regressor":
                prev_num = pd.to_numeric(prev, errors="coerce")
                cur_num = pd.to_numeric(cur, errors="coerce")
                denom = float(np.nanstd(cur_num.to_numpy(dtype="float64"))) or 1.0
                deltas.append(float(np.nanmean(np.abs(cur_num - prev_num))) / denom)
            else:
                deltas.append(float((cur.astype(object) != prev.astype(object)).mean()))
        return max(deltas) if deltas else 0.0

    def _assign_success(self, df: pd.DataFrame, work: pd.DataFrame, plan: _ColumnPlan) -> None:
        s = df[plan.column]
        filled_values = work.loc[plan.missing_mask, plan.column]
        if isinstance(s.dtype, pd.CategoricalDtype):
            missing_categories = [v for v in pd.unique(filled_values) if v not in s.cat.categories]
            if missing_categories:
                s = s.cat.add_categories(missing_categories)
        try:
            combined = s.where(s.notna(), filled_values)
        except (TypeError, ValueError):
            combined = s.astype(object).where(s.notna(), filled_values)
        df[plan.column] = combined
        imputed = int(plan.missing_mask.sum())
        indicator_added = self._maybe_indicator(df, plan)
        model_label = (
            "random-forest regressor"
            if plan.model_type == "regressor"
            else "random-forest classifier"
        )
        confidence = self._confidence(plan)
        risk = (
            "medium"
            if plan.context.missing_ratio > self.config.missing_threshold_low
            else "low"
        )
        convergence = "stable" if self._last_converged else "max_iter"
        self.report.add(
            _STEP,
            f"imputed {imputed} missing value(s) with missforest {model_label}; "
            f"iterations={self._last_iterations}; convergence={convergence}; "
            f"confidence={confidence:.2f}; risk={risk}",
            column=str(plan.column),
            count=imputed,
            rationale="explicit MissForest imputation selected; random forests model "
            "nonlinear relationships across mixed tabular predictors",
            risk=risk,
            confidence=confidence,
            model_id=f"missforest_{plan.model_type}",
            metadata=self._metadata(
                plan,
                imputed,
                fallback_reason=None,
                indicator_added=indicator_added,
            ),
        )
        self.report.columns_imputed.append(str(plan.column))

    def _fallback_fill(
        self,
        df: pd.DataFrame,
        col: object,
        ctx: ColumnContext,
        reason: str,
    ) -> None:
        s = df[col]
        value = s.median() if ctx.role == "numeric" and s.notna().any() else _mode_value(s)
        if value is None or pd.isna(value):
            value = "Missing" if ctx.role in ("categorical", "boolean") else None
        if value is None or pd.isna(value):
            self._preserve(df, col, ctx, reason)
            return
        if isinstance(s.dtype, pd.CategoricalDtype) and value not in s.cat.categories:
            s = s.cat.add_categories([value])
        try:
            filled = s.fillna(value)
        except (TypeError, ValueError):
            filled = s.astype(object).fillna(value)
        df[col] = filled
        imputed = ctx.n_missing
        confidence = 0.65 if ctx.n_rows >= self.config.missforest_min_rows_for_model else 0.55
        self.report.add(
            _STEP,
            f"missforest fallback filled {imputed} missing value(s) with safe simple imputation",
            column=str(col),
            count=imputed,
            rationale=f"MissForest not used: {reason}",
            risk="medium",
            confidence=confidence,
            model_id="missforest_fallback",
            metadata=self._metadata(
                _ColumnPlan(col, ctx, df[col].isna(), ctx.role),
                imputed,
                fallback_reason=reason,
                indicator_added=False,
            ),
        )
        self.report.columns_imputed.append(str(col))

    def _preserve(self, df: pd.DataFrame, col: object, ctx: ColumnContext, reason: str) -> None:
        self.report.add(
            _STEP,
            f"missforest skipped; preserved {ctx.n_missing} missing value(s)",
            column=str(col),
            count=0,
            rationale=f"MissForest not used: {reason}",
            risk="high" if ctx.role == "target" else "medium",
            confidence=0.9,
            model_id="missforest_fallback",
            status="skipped",
            human_review=(
                ctx.role in ("target", "id")
                or ctx.missing_ratio > self.config.missing_threshold_high
            ),
            metadata={
                "missing_count_before": ctx.n_missing,
                "imputed_count": 0,
                "selected_model_type": None,
                "fallback_reason": reason,
                "iterations": 0,
                "converged": False,
                "convergence_delta": None,
                "oob_score": None,
                "indicator_added": False,
            },
        )
        self.report.columns_preserved.append(str(col))

    def _maybe_indicator(self, df: pd.DataFrame, plan: _ColumnPlan) -> bool:
        wanted = self.config.missforest_add_indicators is True or (
            self.config.missforest_add_indicators == "auto"
            and self.config.missing_indicators is not False
            and plan.context.informative_missing
        )
        if not wanted:
            return False
        name = f"{plan.column}_was_missing"
        if name in df.columns:
            return False
        # The column is already imputed here, so the indicator must come from
        # the pre-fill mask captured in the plan, never from df[col].isna().
        add_column(df, name, plan.missing_mask.astype(bool))
        return True

    def _confidence(self, plan: _ColumnPlan) -> float:
        oob = self._last_oob.get(plan.column)
        if oob is not None and np.isfinite(oob):
            return float(max(0.5, min(0.95, 0.5 + 0.45 * max(0.0, oob))))
        base = 0.82 if plan.model_type == "regressor" else 0.78
        if plan.context.missing_ratio > self.config.missing_threshold_medium:
            base -= 0.12
        return base

    def _metadata(
        self,
        plan: _ColumnPlan,
        imputed: int,
        *,
        fallback_reason: str | None,
        indicator_added: bool,
    ) -> dict[str, Any]:
        selected = None
        if plan.model_type == "regressor":
            selected = "regressor"
        elif plan.model_type == "classifier":
            selected = "classifier"
        return {
            "missing_count_before": plan.context.n_missing,
            "imputed_count": imputed,
            "selected_model_type": selected,
            "fallback_reason": fallback_reason,
            "iterations": self._last_iterations if fallback_reason is None else 0,
            "converged": self._last_converged if fallback_reason is None else False,
            "convergence_delta": self._last_delta if fallback_reason is None else None,
            "oob_score": self._last_oob.get(plan.column) if fallback_reason is None else None,
            "indicator_added": indicator_added,
        }

    @staticmethod
    def _safe_oob(model: object) -> float | None:
        value = getattr(model, "oob_score_", None)
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if np.isfinite(out) else None


def _mode_value(s: pd.Series) -> Any | None:
    try:
        modes = s.mode(dropna=True)
        if len(modes):
            return modes.iloc[0]
    except TypeError:
        pass
    try:
        counts = s.value_counts(dropna=True)
    except TypeError:
        return None
    return counts.index[0] if len(counts) else None
