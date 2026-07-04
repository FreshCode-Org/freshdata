"""CleanBench T4 — learned-profile replay benchmark (Phase 4).

Ten fixture families, each a (train messy, train clean) pair plus a held-out
test batch corrupted by the *same* family. A profile is learned from the
train pair and replayed on the test batch; the gates compare against a
no-profile baseline:

* mean cell-repair F1 lift on the replayable families >= +15 points,
* false-modification rate with the profile <= without it,
* protected-column violations == 0,
* zero raw sensitive literals inside the serialized profile under
  privacy="mask".

The corruptions are deliberately ones the no-profile pipeline cannot fully
solve on its own (domain vocabularies, nonstandard sentinels, ambiguous
day-first dates, region-less phone formats), so the lift measures what the
*profile* adds — not what the deterministic experts already do.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import freshdata as fd
from freshdata.learning import learn

from .metrics import (
    cell_repair_f1,
    false_modification_rate,
    protected_column_violation_rate,
)

__all__ = [
    "T4Fixture",
    "T4Result",
    "make_t4_fixtures",
    "privacy_leak_count",
    "profile_drift_block_rate",
    "run_t4",
    "run_t4_fixture",
]

_TRAIN_ROWS = 60
_TEST_ROWS = 30


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class T4Fixture:
    """One corruption family: a train pair and a same-family test batch."""

    name: str
    train_messy: pd.DataFrame
    train_clean: pd.DataFrame
    test_messy: pd.DataFrame
    test_truth: pd.DataFrame
    key: str = "id"
    context: str = "id is a unique identifier. Never modify id."
    protected_columns: tuple[str, ...] = ("id",)
    #: Raw sensitive strings from the train pair that must never appear in
    #: the serialized profile under privacy="mask".
    sensitive_literals: tuple[str, ...] = ()
    #: Families like imputation/unexplained replay nothing by design and are
    #: excluded from the F1-lift gate (they still run the safety gates).
    expect_lift: bool = True
    learn_kwargs: dict[str, Any] = field(default_factory=dict)


def _ids(n: int, prefix: str) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(n)]


def _cycle(values: list, n: int) -> list:
    return [values[i % len(values)] for i in range(n)]


def _pair_frames(
    n: int,
    prefix: str,
    column: str,
    messy_values: list,
    clean_values: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligned (messy, clean) frames: one corrupted column plus stable ballast."""
    ids = _ids(n, prefix)
    ballast = [round(10.0 + (i % 7) * 1.5, 2) for i in range(n)]
    messy = pd.DataFrame({"id": ids, column: _cycle(messy_values, n), "ballast": ballast})
    clean = pd.DataFrame({"id": ids, column: _cycle(clean_values, n), "ballast": ballast})
    return messy, clean


def _vocab_fixture(
    name: str,
    column: str,
    mapping: dict[str, object],
) -> T4Fixture:
    """A fixture whose corruption is a domain vocabulary (raw -> clean)."""
    raws = list(mapping)
    cleans = [mapping[r] for r in raws]
    train_messy, train_clean = _pair_frames(_TRAIN_ROWS, "T", column, raws, cleans)
    test_messy, test_truth = _pair_frames(_TEST_ROWS, "H", column, raws, cleans)
    return T4Fixture(
        name=name,
        train_messy=train_messy,
        train_clean=train_clean,
        test_messy=test_messy,
        test_truth=test_truth,
    )


def _make_category_map_fixture() -> T4Fixture:
    # Department codes: an internal vocabulary no generic cleaner knows.
    return _vocab_fixture(
        "category_map_departments",
        "department",
        {
            "H.R.": "human_resources",
            "HR ": "human_resources",
            "Fin": "finance",
            "FIN.": "finance",
            "Eng": "engineering",
            "ENGG": "engineering",
        },
    )


def _make_status_typos_fixture() -> T4Fixture:
    return _vocab_fixture(
        "status_reference_typos",
        "status",
        {
            "Deliverd": "delivered",
            "delivred": "delivered",
            "SHIPPED": "shipped",
            "shiped": "shipped",
            "In Transit": "in_transit",
            "IN-TRANSIT": "in_transit",
        },
    )


def _make_condition_vocab_fixture() -> T4Fixture:
    return _vocab_fixture(
        "condition_abbreviations",
        "condition",
        {
            "NIB": "new_in_box",
            "N.I.B.": "new_in_box",
            "LN": "like_new",
            "L.N.": "like_new",
            "FR": "for_repair",
        },
    )


def _make_city_typos_fixture() -> T4Fixture:
    return _vocab_fixture(
        "city_reference_typos",
        "city",
        {
            "Bengalru": "Bengaluru",
            "Bangalore": "Bengaluru",
            "Mumbay": "Mumbai",
            "Bombay": "Mumbai",
            "Dehli": "Delhi",
            "New Dehli": "Delhi",
        },
    )


def _make_sentinel_fixture() -> T4Fixture:
    # Nonstandard sentinels the default sentinel set does not contain.
    raws = ["MISSING_VAL", "-999", "n.v.", "42", "77", "13"]
    cleans = [np.nan, np.nan, np.nan, 42.0, 77.0, 13.0]
    train_messy, train_clean = _pair_frames(_TRAIN_ROWS, "T", "reading", raws, cleans)
    test_messy, test_truth = _pair_frames(_TEST_ROWS, "H", "reading", raws, cleans)
    return T4Fixture(
        name="nonstandard_sentinels",
        train_messy=train_messy,
        train_clean=train_clean,
        test_messy=test_messy,
        test_truth=test_truth,
    )


def _make_dayfirst_fixture() -> T4Fixture:
    # Ambiguous day-first dates: without the learned dayfirst delta the
    # baseline parses them month-first and gets *wrong* timestamps.
    raw_dates = ["05/02/2024", "07/03/2024", "09/04/2024", "11/05/2024", "01/06/2024"]
    truth_dates = pd.to_datetime(raw_dates, dayfirst=True)
    train_messy, train_clean = _pair_frames(
        _TRAIN_ROWS, "T", "event_date", raw_dates, list(truth_dates)
    )
    test_messy, test_truth = _pair_frames(
        _TEST_ROWS, "H", "event_date", raw_dates, list(truth_dates)
    )
    return T4Fixture(
        name="dayfirst_dates",
        train_messy=train_messy,
        train_clean=train_clean,
        test_messy=test_messy,
        test_truth=test_truth,
    )


def _make_phone_fixture() -> T4Fixture:
    locals_ = ["98765 43210", "91234-56789", "099887 76655", "9012345678", "98111 22333"]
    canonical = [
        "+919876543210",
        "+919123456789",
        "+919988776655",
        "+919012345678",
        "+919811122333",
    ]
    train_messy, train_clean = _pair_frames(_TRAIN_ROWS, "T", "phone", locals_, canonical)
    test_messy, test_truth = _pair_frames(_TEST_ROWS, "H", "phone", locals_, canonical)
    return T4Fixture(
        name="phone_in_formats",
        train_messy=train_messy,
        train_clean=train_clean,
        test_messy=test_messy,
        test_truth=test_truth,
        sensitive_literals=tuple(locals_) + tuple(canonical),
    )


def _make_email_fixture() -> T4Fixture:
    doubled = [
        "asha@@gmail.com",
        "bob@@yahoo.com",
        "carol@@hotmail.com",
        "dev@@site.com",
        "eve@@example.com",
    ]
    fixed = [d.replace("@@", "@") for d in doubled]
    train_messy, train_clean = _pair_frames(_TRAIN_ROWS, "T", "email", doubled, fixed)
    test_messy, test_truth = _pair_frames(_TEST_ROWS, "H", "email", doubled, fixed)
    return T4Fixture(
        name="email_double_at",
        train_messy=train_messy,
        train_clean=train_clean,
        test_messy=test_messy,
        test_truth=test_truth,
        sensitive_literals=tuple(doubled) + tuple(fixed),
        # The deterministic email expert already repairs doubled @s; the
        # profile must not *hurt*, but the lift comes from other families.
        expect_lift=False,
    )


def _make_imputation_fixture() -> T4Fixture:
    # The clean side median-fills missing readings. The profile must learn at
    # most an imputation *strategy* hint — never literal fill values — so
    # replay leaves the new batch's NaNs alone rather than stamping the
    # train median into them.
    n = _TRAIN_ROWS
    ids = _ids(n, "T")
    values = [float(20 + (i % 9)) for i in range(n)]
    messy_vals: list[object] = list(values)
    for i in range(0, n, 6):
        messy_vals[i] = np.nan
    median = float(np.nanmedian(np.asarray(messy_vals, dtype=float)))
    clean_vals = [median if pd.isna(v) else v for v in messy_vals]
    train_messy = pd.DataFrame({"id": ids, "reading": messy_vals})
    train_clean = pd.DataFrame({"id": ids, "reading": clean_vals})

    test_ids = _ids(_TEST_ROWS, "H")
    test_vals: list[object] = [float(30 + (i % 5)) for i in range(_TEST_ROWS)]
    for i in range(0, _TEST_ROWS, 5):
        test_vals[i] = np.nan
    test_messy = pd.DataFrame({"id": test_ids, "reading": test_vals})
    # Truth keeps the NaNs: a literal replay of the *train* median would be
    # a false modification, which is exactly what the gate checks.
    test_truth = test_messy.copy()
    return T4Fixture(
        name="imputation_no_literal_replay",
        train_messy=train_messy,
        train_clean=train_clean,
        test_messy=test_messy,
        test_truth=test_truth,
        expect_lift=False,
    )


def _make_unexplained_fixture() -> T4Fixture:
    # One-off free-text typos, each with support 1: nothing meets
    # min_support, so the profile stores examples only and replays nothing.
    n = _TRAIN_ROWS
    ids = _ids(n, "T")
    notes = [f"customer note number {i} with unique typo xq{i}z" for i in range(n)]
    clean_notes = [t.replace(f"xq{i}z", "fixed") for i, t in enumerate(notes)]
    train_messy = pd.DataFrame({"id": ids, "note": notes})
    train_clean = pd.DataFrame({"id": ids, "note": clean_notes})
    test_ids = _ids(_TEST_ROWS, "H")
    test_notes = [f"new note {i} with typo aa{i}bb" for i in range(_TEST_ROWS)]
    test_messy = pd.DataFrame({"id": test_ids, "note": test_notes})
    test_truth = test_messy.copy()
    return T4Fixture(
        name="unexplained_examples_only",
        train_messy=train_messy,
        train_clean=train_clean,
        test_messy=test_messy,
        test_truth=test_truth,
        expect_lift=False,
    )


def make_t4_fixtures() -> list[T4Fixture]:
    """The ten T4 corruption families, deterministic (no RNG needed)."""
    return [
        _make_category_map_fixture(),
        _make_status_typos_fixture(),
        _make_condition_vocab_fixture(),
        _make_city_typos_fixture(),
        _make_sentinel_fixture(),
        _make_dayfirst_fixture(),
        _make_phone_fixture(),
        _make_email_fixture(),
        _make_imputation_fixture(),
        _make_unexplained_fixture(),
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def privacy_leak_count(profile_bytes: bytes, sensitive_literals: tuple[str, ...]) -> int:
    """Count sensitive literals readable anywhere in a serialized profile.

    Scans both the raw archive bytes and every decompressed member, so zip
    compression cannot hide a leak.
    """
    blobs = [profile_bytes]
    try:
        with zipfile.ZipFile(io.BytesIO(profile_bytes)) as zf:
            blobs.extend(zf.read(name) for name in zf.namelist())
    except zipfile.BadZipFile:  # pragma: no cover - defensive
        pass
    leaks = 0
    for literal in sensitive_literals:
        needle = literal.encode("utf-8")
        if any(needle in blob for blob in blobs):
            leaks += 1
    return leaks


def profile_drift_block_rate(reports: list) -> float:
    """Share of runs whose profile replay was blocked by the drift gate."""
    if not reports:
        return 0.0
    blocked = sum(
        1
        for r in reports
        if getattr(r, "profile_replay", None) is not None and not r.profile_replay.get("ok")
    )
    return blocked / len(reports)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class T4Result:
    name: str
    expect_lift: bool
    baseline_f1: float
    profile_f1: float
    lift_f1: float
    baseline_fmr: float
    profile_fmr: float
    protected_violation_rate: float
    privacy_leaks: int
    replay_ok: bool
    profile_actions: int


def _serialize(profile) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profile.fdprofile"
        profile.save(path)
        return path.read_bytes()


def run_t4_fixture(fixture: T4Fixture) -> T4Result:
    """Learn from the fixture's train pair, replay on its test batch."""
    profile = learn(
        fixture.train_messy,
        fixture.train_clean,
        context=fixture.context,
        key=fixture.key,
        dataset_id=f"t4-{fixture.name}",
        **fixture.learn_kwargs,
    )
    leaks = privacy_leak_count(_serialize(profile), fixture.sensitive_literals)

    # Shape-preserving options: metrics need identically-shaped frames, so no
    # flag columns, no row drops, no index resets.
    clean_kwargs: dict[str, Any] = {
        "semantic_mode": "auto",
        "verbose": False,
        "drop_duplicates": False,
        "reset_index": False,
        "outlier_action": None,
        "return_report": True,
    }
    baseline, _ = fd.clean(fixture.test_messy, **clean_kwargs)
    repaired, report = fd.clean(fixture.test_messy, profile=profile, **clean_kwargs)

    truth, corrupted = fixture.test_truth, fixture.test_messy
    baseline_f1 = cell_repair_f1(truth, corrupted, baseline)
    profile_f1 = cell_repair_f1(truth, corrupted, repaired)
    profile_actions = sum(
        1 for a in report.actions if a.metadata and a.metadata.get("profile_influenced")
    )
    return T4Result(
        name=fixture.name,
        expect_lift=fixture.expect_lift,
        baseline_f1=baseline_f1,
        profile_f1=profile_f1,
        lift_f1=profile_f1 - baseline_f1,
        baseline_fmr=false_modification_rate(truth, corrupted, baseline),
        profile_fmr=false_modification_rate(truth, corrupted, repaired),
        protected_violation_rate=protected_column_violation_rate(
            corrupted, repaired, list(fixture.protected_columns)
        ),
        privacy_leaks=leaks,
        replay_ok=bool(report.profile_replay is not None and report.profile_replay.get("ok")),
        profile_actions=profile_actions,
    )


def run_t4(fixtures: list[T4Fixture] | None = None) -> list[T4Result]:
    return [run_t4_fixture(f) for f in fixtures or make_t4_fixtures()]


if __name__ == "__main__":  # pragma: no cover - manual benchmark entry point
    results = run_t4()
    lifted = [r for r in results if r.expect_lift]
    print(f"{'fixture':32} {'base F1':>8} {'prof F1':>8} {'lift':>7} {'FMR b/p':>12} {'leaks':>5}")
    for r in results:
        print(
            f"{r.name:32} {r.baseline_f1:8.3f} {r.profile_f1:8.3f} "
            f"{r.lift_f1:+7.3f} {r.baseline_fmr:5.3f}/{r.profile_fmr:5.3f} "
            f"{r.privacy_leaks:5d}"
        )
    mean_lift = sum(r.lift_f1 for r in lifted) / len(lifted)
    print(f"\nmean F1 lift (replayable families): {mean_lift:+.3f}")
