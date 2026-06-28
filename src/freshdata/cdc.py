"""CDC / event-time quality gate (``fd.cdc_profile``).

Change-data-capture and event streams fail in ways that are *not* missing values:
records arrive **stale**, **late** (past the watermark), **out of order**, with
**duplicate CDC keys**, **invalid operation codes**, a **missing event time**, or
as a **replay-risk** batch. Treating those as nulls hides them. ``cdc_profile`` is
a read-only profiler that classifies these freshness/ordering defects separately
from completeness defects and reports enterprise trust penalties for freshness,
ordering, and CDC integrity.

>>> import freshdata as fd
>>> rep = fd.cdc_profile(df, event_time="event_ts", key="entity_id",
...                      lateness="5m", operation_col="op")
>>> print(rep.summary())
>>> rep.to_frame()        # one row per defect class
>>> rep.passed            # False if any error-level defect

It never mutates the input and never imputes — it explains. Pair it with
``fd.clean`` for value-level repair; the two concerns stay distinct on purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from .streaming._timeseries import to_timedelta

__all__ = ["CDCDefect", "CDCReport", "cdc_profile"]

#: Default CDC operation codes: Debezium-style (c/r/u/d) plus I/U/D conventions.
DEFAULT_OPERATIONS: tuple[str, ...] = ("I", "U", "D", "c", "r", "u", "d")

DefectKind = Literal[
    "stale",
    "late",
    "out_of_order",
    "duplicate_key",
    "invalid_operation",
    "missing_event_time",
    "replay_risk",
]
Level = Literal["info", "warning", "error"]


@dataclass
class CDCDefect:
    """One class of CDC / event-time defect, with a count and offending sample."""

    kind: DefectKind
    level: Level
    n_rows: int
    rationale: str
    sample_keys: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "level": self.level,
            "n_rows": self.n_rows,
            "rationale": self.rationale,
            "sample_keys": list(self.sample_keys),
            "details": dict(self.details),
        }


@dataclass
class CDCReport:
    """Outcome of :func:`cdc_profile` — freshness/ordering/CDC defect classes."""

    event_time: str
    key: str | None
    n_rows: int
    freshness_seconds: float | None
    defects: list[CDCDefect] = field(default_factory=list)
    trust_penalties: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when no error-level defect was found."""
        return not any(d.level == "error" for d in self.defects)

    @property
    def n_errors(self) -> int:
        return sum(1 for d in self.defects if d.level == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for d in self.defects if d.level == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "key": self.key,
            "n_rows": self.n_rows,
            "passed": self.passed,
            "freshness_seconds": self.freshness_seconds,
            "trust_penalties": dict(self.trust_penalties),
            "defects": [d.to_dict() for d in self.defects],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, sort_keys=True)

    def to_frame(self) -> pd.DataFrame:
        """One row per defect class — sortable/exportable."""
        cols = ["kind", "level", "n_rows", "rationale", "sample_keys"]
        rows = [
            {
                "kind": d.kind,
                "level": d.level,
                "n_rows": d.n_rows,
                "rationale": d.rationale,
                "sample_keys": list(d.sample_keys),
            }
            for d in self.defects
        ]
        return pd.DataFrame(rows, columns=cols)

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        head = (
            f"cdc profile on {self.event_time!r}"
            + (f" by {self.key!r}" if self.key else "")
            + f": {verdict} ({self.n_errors} error(s), {self.n_warnings} warning(s))"
        )
        lines = [head]
        if self.freshness_seconds is not None:
            lines.append(f"  freshness: {self.freshness_seconds:.0f}s behind reference")
        if self.trust_penalties:
            pretty = ", ".join(f"{k}={v:.2f}" for k, v in self.trust_penalties.items())
            lines.append(f"  trust penalties: {pretty}")
        for d in self.defects:
            marker = "✗" if d.level == "error" else "!"
            lines.append(f"  {marker} [{d.kind}] {d.n_rows} row(s): {d.rationale}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _sample_keys(
    mask: pd.Series, df: pd.DataFrame, key: str | None, limit: int = 5
) -> tuple[str, ...]:
    """Up to ``limit`` offending key values (or row labels when no key)."""
    if not bool(mask.any()):
        return ()
    if key is not None:
        vals = df.loc[mask, key].dropna().unique()[:limit]
    else:
        vals = df.index[mask][:limit]
    return tuple(str(v) for v in vals)


def _running_max(ts: pd.Series, key_series: pd.Series | None) -> pd.Series:
    """Per-key (or global) running max of event time, in arrival order."""
    if key_series is not None:
        return ts.groupby(key_series, sort=False).cummax()
    return ts.cummax()


def _ordering_defects(
    df: pd.DataFrame,
    ts: pd.Series,
    key: str | None,
    lateness_td: pd.Timedelta,
    defects: list[CDCDefect],
) -> dict[str, int]:
    """Detect out-of-order and late (past-watermark) records."""
    key_series = df[key] if key is not None else None
    running = _running_max(ts, key_series)
    behind = ts < running
    lag = (running - ts).where(behind)
    late_mask = behind & (lag > lateness_td)
    ooo_mask = behind & ~late_mask  # out of order but within the lateness tolerance

    n_ooo = int(ooo_mask.sum())
    n_late = int(late_mask.sum())
    if n_ooo:
        defects.append(
            CDCDefect(
                kind="out_of_order",
                level="warning",
                n_rows=n_ooo,
                rationale="event time earlier than a prior record (within lateness tolerance)",
                sample_keys=_sample_keys(ooo_mask, df, key),
            )
        )
    if n_late:
        defects.append(
            CDCDefect(
                kind="late",
                level="error",
                n_rows=n_late,
                rationale=f"event time past the watermark by more than {lateness_td}",
                sample_keys=_sample_keys(late_mask, df, key),
                details={"lateness": str(lateness_td)},
            )
        )
    return {"out_of_order": n_ooo, "late": n_late}


def _freshness(
    ts: pd.Series,
    now: pd.Timestamp | None,
    stale_after_td: pd.Timedelta | None,
    defects: list[CDCDefect],
) -> float | None:
    """Compute freshness vs. a reference 'now' and flag a stale batch."""
    max_ts = ts.max()
    if pd.isna(max_ts):
        return None
    reference = now if now is not None else pd.Timestamp.now(tz=max_ts.tz)
    freshness_seconds = float((reference - max_ts).total_seconds())
    if stale_after_td is not None and freshness_seconds > stale_after_td.total_seconds():
        defects.append(
            CDCDefect(
                kind="stale",
                level="warning",
                n_rows=0,
                rationale=(
                    f"newest event is {freshness_seconds:.0f}s old, "
                    f"older than stale_after={stale_after_td}"
                ),
                details={"freshness_seconds": freshness_seconds},
            )
        )
    return freshness_seconds


def cdc_profile(
    df: pd.DataFrame,
    *,
    event_time: str,
    key: str | None = None,
    watermark: object | None = None,
    lateness: object = "0s",
    sequence: str | None = None,
    operation_col: str | None = None,
    valid_operations: tuple[str, ...] = DEFAULT_OPERATIONS,
    now: object | None = None,
    stale_after: object | None = None,
    replay_threshold: float = 0.2,
) -> CDCReport:
    """Classify CDC / event-time defects without mutating *df*.

    Parameters
    ----------
    df:
        The change/event frame, in arrival order.
    event_time:
        Name of the event-timestamp column. Parsed with ``errors="coerce"``;
        unparseable/empty values become ``missing_event_time`` defects.
    key:
        Optional CDC key column (entity id). Enables per-key ordering and
        duplicate-key detection.
    watermark:
        Optional explicit low watermark (timestamp). Records strictly before it
        are reported as ``late``. When omitted a *running* per-key watermark is
        used together with ``lateness``.
    lateness:
        Allowed lateness past the running watermark (``"5m"``, seconds, or a
        ``Timedelta``). Out-of-order records beyond this are ``late`` (error);
        within it they are ``out_of_order`` (warning).
    sequence:
        Optional monotonic sequence/LSN column; included in duplicate detection.
    operation_col:
        Optional CDC operation column; values outside ``valid_operations`` (or
        null) are ``invalid_operation`` defects.
    now, stale_after:
        Reference 'now' and the max age before the batch is ``stale``. ``now``
        defaults to the current UTC time; pass it explicitly for determinism.
    replay_threshold:
        Fraction of (duplicate + late) rows above which a ``replay_risk`` batch
        warning is raised.

    Returns
    -------
    CDCReport
        With ``.passed`` (no error defects), ``.summary()``, ``.to_frame()``,
        ``.to_dict()``, ``.to_json()`` and ``.trust_penalties`` (freshness,
        ordering, cdc — each 0..1, higher is worse).
    """
    for col in (event_time, key, sequence, operation_col):
        if col is not None and col not in df.columns:
            raise KeyError(f"column {col!r} not found in frame")

    n_rows = len(df)
    lateness_td = to_timedelta(lateness) or pd.Timedelta(0)
    stale_after_td = to_timedelta(stale_after)
    now_ts = pd.Timestamp(now) if now is not None else None

    raw = df[event_time]
    ts = raw if pd.api.types.is_datetime64_any_dtype(raw) else pd.to_datetime(raw, errors="coerce")

    defects: list[CDCDefect] = []

    # 1) missing event time — a completeness-adjacent but freshness-critical defect.
    missing_mask = ts.isna()
    n_missing = int(missing_mask.sum())
    if n_missing:
        defects.append(
            CDCDefect(
                kind="missing_event_time",
                level="error",
                n_rows=n_missing,
                rationale=f"{event_time!r} is null or unparseable",
                sample_keys=_sample_keys(missing_mask, df, key),
            )
        )

    # 2) ordering: out-of-order + late (relative to watermark / running watermark).
    if watermark is not None:
        wm = pd.Timestamp(watermark)
        late_mask = ts < wm
        n_late = int(late_mask.sum())
        counts = {"out_of_order": 0, "late": n_late}
        if n_late:
            defects.append(
                CDCDefect(
                    kind="late",
                    level="error",
                    n_rows=n_late,
                    rationale=f"event time before the watermark {wm}",
                    sample_keys=_sample_keys(late_mask, df, key),
                    details={"watermark": str(wm)},
                )
            )
    else:
        counts = _ordering_defects(df, ts, key, lateness_td, defects)

    # 3) duplicate CDC keys (replayed/duplicated changes).
    n_dup = 0
    if key is not None:
        subset = [key, event_time] + ([sequence] if sequence else [])
        dup_mask = df.duplicated(subset=subset, keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup:
            defects.append(
                CDCDefect(
                    kind="duplicate_key",
                    level="error",
                    n_rows=n_dup,
                    rationale=f"duplicate change rows for {subset}",
                    sample_keys=_sample_keys(dup_mask, df, key),
                )
            )

    # 4) invalid operation codes.
    n_badop = 0
    if operation_col is not None:
        allowed = set(valid_operations)
        op = df[operation_col]
        bad_mask = op.isna() | ~op.isin(allowed)
        n_badop = int(bad_mask.sum())
        if n_badop:
            defects.append(
                CDCDefect(
                    kind="invalid_operation",
                    level="error",
                    n_rows=n_badop,
                    rationale=f"operation outside {sorted(allowed)}",
                    sample_keys=_sample_keys(bad_mask, df, key),
                    details={"valid_operations": sorted(allowed)},
                )
            )

    # 5) freshness / stale batch.
    freshness_seconds = _freshness(ts, now_ts, stale_after_td, defects)

    # 6) replay-risk batch heuristic.
    n_late = counts.get("late", 0)
    if n_rows and n_dup and (n_dup + n_late) / n_rows >= replay_threshold:
        defects.append(
            CDCDefect(
                kind="replay_risk",
                level="warning",
                n_rows=n_dup + n_late,
                rationale="high share of duplicate/late rows suggests a replayed batch",
                details={"duplicate_rows": n_dup, "late_rows": n_late},
            )
        )

    penalties = _trust_penalties(
        n_rows=n_rows,
        n_ooo=counts.get("out_of_order", 0),
        n_late=n_late,
        n_badop=n_badop,
        n_dup=n_dup,
        n_missing=n_missing,
        freshness_seconds=freshness_seconds,
        stale_after_td=stale_after_td,
    )

    return CDCReport(
        event_time=event_time,
        key=key,
        n_rows=n_rows,
        freshness_seconds=freshness_seconds,
        defects=defects,
        trust_penalties=penalties,
    )


def _trust_penalties(
    *,
    n_rows: int,
    n_ooo: int,
    n_late: int,
    n_badop: int,
    n_dup: int,
    n_missing: int,
    freshness_seconds: float | None,
    stale_after_td: pd.Timedelta | None,
) -> dict[str, float]:
    """Freshness / ordering / cdc penalties in ``[0, 1]`` (higher is worse)."""
    denom = max(n_rows, 1)
    ordering = min(1.0, (n_ooo + n_late) / denom)
    cdc = min(1.0, (n_badop + n_dup + n_missing) / denom)
    if freshness_seconds is not None and stale_after_td is not None:
        freshness = min(1.0, max(0.0, freshness_seconds / stale_after_td.total_seconds()))
    else:
        freshness = 0.0
    return {"freshness": round(freshness, 4), "ordering": round(ordering, 4), "cdc": round(cdc, 4)}
