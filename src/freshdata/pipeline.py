"""Explicit, fluent cleaning pipelines: ``fd.pipeline()``.

The one-call ``fd.clean(df)`` decides for you; a :class:`Pipeline` does only
what you chain, in the pipeline's fixed execution order, with nothing implied::

    import freshdata as fd

    cleaned, report = (
        fd.pipeline()
        .normalize_columns()
        .normalize_missing()
        .validate_types()
        .impute(strategy="median", columns=["age"])
        .deduplicate(subset=["email"])
        .run(df, return_report=True)
    )

Every step maps 1:1 onto a :class:`~freshdata.CleanConfig` field, so the
backend fallback matrix, reports, and audit trail are identical to
``fd.clean`` — the builder adds no execution machinery of its own. Instances
are immutable (each method returns a new pipeline), serializable
(:meth:`to_json` / :meth:`from_json`), and comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

PIPELINE_SCHEMA_VERSION = "freshdata-pipeline-v1"

#: step name -> allowed parameter names (validated on construction and load).
_STEP_PARAMS: dict[str, frozenset[str]] = {
    "normalize_columns": frozenset(),
    "strip_whitespace": frozenset(),
    "normalize_missing": frozenset({"extra_sentinels"}),
    "drop_empty": frozenset({"rows", "columns"}),
    "validate_types": frozenset(),
    "deduplicate": frozenset({"subset", "keep"}),
    "impute": frozenset({"strategy", "columns"}),
    "outliers": frozenset({"method", "action"}),
    "drop_constant_columns": frozenset(),
    "optimize_memory": frozenset(),
}


@dataclass(frozen=True)
class Pipeline:
    """An immutable, ordered chain of explicit cleaning steps."""

    steps: tuple[tuple[str, dict[str, Any]], ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for name, params in self.steps:
            if name not in _STEP_PARAMS:
                raise ValueError(
                    f"unknown pipeline step {name!r}; expected one of "
                    f"{sorted(_STEP_PARAMS)}"
                )
            unknown = set(params) - _STEP_PARAMS[name]
            if unknown:
                raise ValueError(f"step {name!r} got unknown parameter(s) {sorted(unknown)}")
            if name in seen:
                raise ValueError(
                    f"step {name!r} appears twice; a pipeline runs each step at most once"
                )
            seen.add(name)

    # -- chaining ------------------------------------------------------------

    def _with(self, name: str, **params: Any) -> Pipeline:
        return Pipeline(steps=(*self.steps, (name, params)))

    def normalize_columns(self) -> Pipeline:
        """snake_case column names, deduplicate collisions."""
        return self._with("normalize_columns")

    def strip_whitespace(self) -> Pipeline:
        """Trim surrounding whitespace in text cells."""
        return self._with("strip_whitespace")

    def normalize_missing(self, *, extra_sentinels: tuple[str, ...] = ()) -> Pipeline:
        """Turn sentinel strings ("N/A", "null", "-", …) into real missing values."""
        return self._with("normalize_missing", extra_sentinels=tuple(extra_sentinels))

    def drop_empty(self, *, rows: bool = True, columns: bool = True) -> Pipeline:
        """Remove all-missing rows and/or columns."""
        return self._with("drop_empty", rows=rows, columns=columns)

    def validate_types(self) -> Pipeline:
        """Repair dtypes: numeric/datetime/boolean text becomes typed columns."""
        return self._with("validate_types")

    def deduplicate(
        self, *, subset: list[str] | tuple[str, ...] | None = None, keep: str = "first"
    ) -> Pipeline:
        """Resolve duplicate rows (whole-row, or compared on *subset*)."""
        return self._with(
            "deduplicate", subset=tuple(subset) if subset is not None else None, keep=keep
        )

    def impute(
        self,
        *,
        strategy: str,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> Pipeline:
        """Fill missing values (globally, or only in *columns*)."""
        return self._with(
            "impute",
            strategy=strategy,
            columns=tuple(columns) if columns is not None else None,
        )

    def outliers(self, *, method: str = "iqr", action: str = "flag") -> Pipeline:
        """Detect outliers in numeric columns; ``action`` is ``"flag"`` or ``"clip"``."""
        return self._with("outliers", method=method, action=action)

    def drop_constant_columns(self) -> Pipeline:
        """Remove columns holding a single constant value."""
        return self._with("drop_constant_columns")

    def optimize_memory(self) -> Pipeline:
        """Downcast dtypes to reduce memory (pandas output only)."""
        return self._with("optimize_memory")

    # -- compilation ----------------------------------------------------------

    def compile(self) -> Any:
        """Compile to the :class:`~freshdata.CleanConfig` that executes this
        pipeline. Only chained steps are enabled — nothing is implied."""
        from .config import CleanConfig  # noqa: PLC0415 - keep module import light

        if not self.steps:
            raise ValueError("empty pipeline: chain at least one step before run()")
        cfg: dict[str, Any] = {
            "strategy": "conservative",
            "column_names": False,
            "strip_whitespace": False,
            "normalize_sentinels": False,
            "drop_empty_rows": False,
            "drop_empty_columns": False,
            "fix_dtypes": False,
            "drop_duplicates": False,
            "verbose": False,
        }
        for name, params in self.steps:
            if name == "normalize_columns":
                cfg["column_names"] = True
            elif name == "strip_whitespace":
                cfg["strip_whitespace"] = True
            elif name == "normalize_missing":
                cfg["normalize_sentinels"] = True
                if params.get("extra_sentinels"):
                    cfg["extra_sentinels"] = tuple(params["extra_sentinels"])
            elif name == "drop_empty":
                cfg["drop_empty_rows"] = bool(params.get("rows", True))
                cfg["drop_empty_columns"] = bool(params.get("columns", True))
            elif name == "validate_types":
                cfg["fix_dtypes"] = True
            elif name == "deduplicate":
                cfg["drop_duplicates"] = True
                if params.get("subset") is not None:
                    cfg["duplicate_subset"] = tuple(params["subset"])
                cfg["duplicate_keep"] = params.get("keep", "first")
            elif name == "impute":
                if params.get("columns") is not None:
                    cfg["impute_strategy"] = dict.fromkeys(params["columns"], params["strategy"])
                else:
                    cfg["impute"] = params["strategy"]
            elif name == "outliers":
                cfg["outliers"] = params.get("action", "flag")
                cfg["outlier_method"] = params.get("method", "iqr")
            elif name == "drop_constant_columns":
                cfg["drop_constant_columns"] = True
            elif name == "optimize_memory":
                cfg["optimize_memory"] = True
        return CleanConfig(**cfg)

    def run(
        self,
        df: Any,
        *,
        engine: str = "pandas",
        output_format: str = "pandas",
        fallback_policy: str | None = None,
        engine_config: Any = None,
        return_report: bool = False,
    ) -> Any:
        """Execute the pipeline via :func:`freshdata.clean` (same reports,
        same backends, same fallback matrix)."""
        from .api import clean  # noqa: PLC0415 - avoid import cycle

        return clean(
            df,
            config=self.compile(),
            engine=engine,
            output_format=output_format,
            fallback_policy=fallback_policy,
            engine_config=engine_config,
            return_report=return_report,
        )

    # -- introspection / serialization ----------------------------------------

    def describe(self) -> str:
        """One line per step, in execution order."""
        if not self.steps:
            return "pipeline: (empty)"
        lines = ["pipeline:"]
        for name, params in self.steps:
            shown = {k: v for k, v in params.items() if v not in (None, (), False)}
            suffix = f" {shown}" if shown else ""
            lines.append(f"  - {name}{suffix}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "steps": [
                {
                    "step": name,
                    "params": {
                        k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()
                    },
                }
                for name, params in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Pipeline:
        schema = d.get("schema_version", PIPELINE_SCHEMA_VERSION)
        if schema != PIPELINE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported pipeline schema_version {schema!r}; "
                f"this freshdata reads {PIPELINE_SCHEMA_VERSION!r}"
            )
        steps = tuple(
            (
                s["step"],
                {
                    k: (tuple(v) if isinstance(v, list) else v)
                    for k, v in s.get("params", {}).items()
                },
            )
            for s in d.get("steps", ())
        )
        return cls(steps=steps)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Pipeline:
        return cls.from_dict(json.loads(text))

    def __str__(self) -> str:
        return self.describe()


def pipeline() -> Pipeline:
    """Start an empty :class:`Pipeline`; chain steps and call ``.run(df)``."""
    return Pipeline()
