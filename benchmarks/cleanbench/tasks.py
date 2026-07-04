"""CleanBench task directories: the on-disk exchange format.

Every track task materializes as::

    benchmarks/cleanbench/tasks/<task_id>/
        messy.csv     (or messy.parquet)
        clean.csv     (or clean.parquet)
        context.txt   (may be empty)
        policy.json   (compiled policy snapshot, may be null)
        meta.json     (track, seed, protected columns, clean kwargs, expectations)

Tasks are generated from the in-code fixtures — deterministic, reviewable,
and small. T5 stores only its generator parameters in ``meta.json``; its
frames are regenerated at load time so no 50k-row CSV lives in git.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

TASKS_ROOT = Path(__file__).resolve().parent / "tasks"


@dataclass
class Task:
    task_id: str
    track: str
    truth: pd.DataFrame | None
    corrupted: pd.DataFrame | None
    context: str
    clean_kwargs: dict[str, Any]
    meta: dict[str, Any]


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    try:
        target = path.with_suffix(".parquet")
        frame.to_parquet(target, index=False)
    except (ImportError, ValueError):
        target = path.with_suffix(".csv")
        frame.to_csv(target, index=False)
    return target.name


def _read_frame(directory: Path, stem: str) -> pd.DataFrame | None:
    parquet = directory / f"{stem}.parquet"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    csv = directory / f"{stem}.csv"
    if csv.is_file():
        return pd.read_csv(csv, keep_default_na=False, na_values=[""], dtype=str)
    return None


def write_task(
    root: Path,
    task_id: str,
    *,
    track: str,
    truth: pd.DataFrame | None,
    corrupted: pd.DataFrame | None,
    context: str,
    clean_kwargs: dict[str, Any],
    meta: dict[str, Any],
) -> Path:
    directory = root / task_id
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    if truth is not None:
        files["clean"] = _write_frame(directory / "clean", truth)
    if corrupted is not None:
        files["messy"] = _write_frame(directory / "messy", corrupted)
    (directory / "context.txt").write_text(context, encoding="utf-8")

    policy_snapshot = None
    if context.strip() and corrupted is not None:
        try:
            import freshdata as fd  # noqa: PLC0415

            policy_snapshot = fd.compile_context(context, df=corrupted).to_dict()
        except Exception:  # pragma: no cover - snapshot is best-effort metadata
            policy_snapshot = None
    (directory / "policy.json").write_text(
        json.dumps(policy_snapshot, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")

    payload = dict(meta, track=track, task_id=task_id,
                   clean_kwargs=clean_kwargs, files=files)
    (directory / "meta.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return directory


def load_task(directory: Path | str) -> Task:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    context = (directory / "context.txt").read_text(encoding="utf-8")
    truth = _read_frame(directory, "clean")
    corrupted = _read_frame(directory, "messy")
    return Task(
        task_id=str(meta["task_id"]),
        track=str(meta["track"]),
        truth=truth,
        corrupted=corrupted,
        context=context,
        clean_kwargs=dict(meta.get("clean_kwargs", {})),
        meta=meta,
    )


def build_all(root: Path | str = TASKS_ROOT) -> list[Path]:
    """(Re)generate every track's task directory from the in-code fixtures."""
    from . import fixtures  # noqa: PLC0415

    root = Path(root)
    written: list[Path] = []

    truth, corrupted, kwargs = fixtures.make_t1_representation_fixture()
    written.append(write_task(
        root, "t1_representation", track="T1", truth=truth, corrupted=corrupted,
        context="", clean_kwargs=kwargs,
        meta={"seed": 30, "protected_columns": []},
    ))

    truth, corrupted, kwargs = fixtures.make_t2_semantic_fixture()
    context = kwargs.pop("context")
    written.append(write_task(
        root, "t2_semantic_values", track="T2", truth=truth, corrupted=corrupted,
        context=context, clean_kwargs=kwargs,
        meta={"seed": 0, "protected_columns": ["monthly_revenue"]},
    ))

    truth, corrupted, kwargs = fixtures.make_t3_context_fixture()
    context = kwargs.pop("context")
    written.append(write_task(
        root, "t3_context_compliance", track="T3", truth=truth, corrupted=corrupted,
        context=context, clean_kwargs=kwargs,
        meta={
            "seed": 10,
            "protected_columns": ["monthly_revenue"],
            "expected_constraints": [
                ["cust_id", "unique"], ["email_addr", "valid_format"],
                ["mobile", "locale_format"], ["status", "allowed_values"],
                ["age", "impute_missing"], ["monthly_revenue", "protected"],
            ],
            "expected_params": {
                "email_addr|valid_format": {"format": "email"},
                "status|allowed_values": {"values": ["active", "inactive", "pending"]},
            },
        },
    ))

    (pair_messy, pair_clean, batch_truth, batch_corrupted, _drifted, kwargs,
     _sm, _sc) = fixtures.make_t4_profile_fixture()
    directory = write_task(
        root, "t4_profile_replay", track="T4", truth=batch_truth,
        corrupted=batch_corrupted, context="", clean_kwargs=kwargs,
        meta={"seed": 40, "protected_columns": [],
              "note": "pair frames stored alongside as pair_messy/pair_clean"},
    )
    _write_frame(directory / "pair_messy", pair_messy)
    _write_frame(directory / "pair_clean", pair_clean)
    written.append(directory)

    written.append(write_task(
        root, "t5_scale", track="T5", truth=None, corrupted=None,
        context="", clean_kwargs={"verbose": False, "drop_duplicates": False,
                                  "reset_index": False},
        meta={"seed": 50, "protected_columns": [],
              "generator": {"name": "make_t5_scale_fixture", "target_rows": 50_000}},
    ))
    return written
