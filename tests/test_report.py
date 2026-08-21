import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jsonschema import Draft202012Validator, validate

import freshdata as fd


def test_shape_and_memory_bookkeeping(messy):
    _, report = fd.clean(messy, return_report=True, drop_duplicates=True)
    assert report.rows_before == 5
    assert report.cols_before == 6
    assert report.rows_after == 4
    assert report.cols_after == 5
    assert report.memory_before > 0
    assert report.memory_after > 0
    assert report.duration_seconds >= 0


def test_report_is_iterable_and_sized(messy):
    _, report = fd.clean(messy, return_report=True)
    assert len(report) == len(list(report))
    assert all(isinstance(a, fd.Action) for a in report)
    assert report.cells_changed > 0


def test_to_dict_is_json_serializable(messy):
    _, report = fd.clean(messy, return_report=True)
    payload_dict = report.to_dict()
    payload = json.dumps(payload_dict)
    assert "drop_duplicates" in payload
    assert len(payload_dict["actions"]) == len(report)


def test_to_json_serializes_real_report(messy):
    _, report = fd.clean(messy, return_report=True)

    payload = json.loads(report.to_json(indent=2, sort_keys=True))

    assert payload == report.to_dict()
    assert len(payload["actions"]) == len(report)


def test_write_json_round_trip(messy, tmp_path: Path):
    _, report = fd.clean(messy, return_report=True)
    path = tmp_path / "audit" / "report.json"
    path.parent.mkdir()

    result = report.write_json(path, indent=2)

    assert result is None
    assert json.loads(path.read_text(encoding="utf-8")) == report.to_dict()
    assert path.read_bytes().endswith(b"\n")


def test_json_exports_serialize_nested_audit_values(messy, tmp_path: Path):
    extracted_at = datetime(2026, 8, 20, 9, 30, tzinfo=timezone.utc)
    _, report = fd.clean(
        messy,
        source_provenance={"AGE": {"extracted_at": extracted_at}},
        return_report=True,
    )
    path = tmp_path / "report.json"

    payload = json.loads(report.to_json())
    report.write_json(path)

    assert report.to_dict()["source_provenance"]["AGE"]["extracted_at"] is extracted_at
    assert payload["source_provenance"]["AGE"]["extracted_at"] == str(extracted_at)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_json_schema_validates_representative_report():
    report = fd.CleanReport(
        actions=[
            fd.Action(
                step="missing",
                column="age",
                description="Filled one missing value",
                count=1,
                rationale="Median is robust to the observed skew",
                risk="medium",
                confidence=0.9,
                status="approved",
                reversible=True,
                memory_influenced=True,
                human_review=True,
                metadata={"fill_value": 42.0},
            )
        ],
        rows_before=4,
        rows_after=4,
        cols_before=2,
        cols_after=2,
        memory_before=256,
        memory_after=256,
        duration_seconds=0.02,
        missing_before=1,
        missing_after=0,
        columns_imputed=["age"],
        coerced_cells={"age": {"row-1": "unknown"}},
        domain="customers",
        domain_trust_score=0.98,
        domain_findings=[{"status": "passed"}],
        domain_repairs=[{"status": "applied"}],
        streaming={"batch_id": "batch-7"},
        backend="pandas",
        requested_backend="auto",
        peak_memory=1024,
        rows_materialized=4,
        materialized=False,
        fallback_events=[
            {
                "backend": "polars",
                "fallback_step": "missing",
                "fallback_reason": "unsupported dtype",
            }
        ],
        backend_differences=[
            {
                "backend": "polars",
                "step": "outliers",
                "column": "age",
                "detail": "nearest-rank quantile",
            }
        ],
        stage_timings=[{"backend": "pandas", "stage": "missing", "seconds": 0.01}],
        source_provenance={"age": {"source_file": "customers.csv"}},
        contract_violations={"passed": True},
        decisions_hash="a" * 64,
        profile_replay={"profile_id": "customers-v1"},
    )
    schema = report.to_json_schema()

    Draft202012Validator.check_schema(schema)
    validate(instance=report.to_dict(), schema=schema)
    validate(instance=json.loads(report.to_json()), schema=schema)


def test_json_schema_rejects_invalid_action_risk():
    payload = fd.CleanReport(
        actions=[fd.Action(step="missing", column="age", description="Filled", risk="urgent")]
    ).to_dict()

    validator = Draft202012Validator(fd.CleanReport.to_json_schema())

    assert "urgent" in str(next(validator.iter_errors(payload)))


def test_to_frame(messy):
    _, report = fd.clean(messy, return_report=True)
    frame = report.to_frame()
    assert list(frame.columns) == ["step", "column", "description", "count",
                                   "rationale", "risk", "confidence", "model_id",
                                   "status", "reversible", "memory_influenced",
                                   "human_review"]
    assert len(frame) == len(report)


def test_summary_mentions_key_facts(messy):
    _, report = fd.clean(messy, return_report=True, drop_duplicates=True)
    text = report.summary()
    assert "rows:" in text and "5 -> 4" in text
    assert "[fix_dtypes]" in text
    assert str(report) == text


def test_summary_includes_domain_and_contract_context():
    report = fd.CleanReport(
        rows_before=3,
        rows_after=3,
        cols_before=2,
        cols_after=2,
        domain="crm",
        domain_trust_score=0.75,
        domain_findings=[
            {"status": "violated", "severity": "error"},
            {"status": "violated", "severity": "warning"},
            {"status": "passed", "severity": "error"},
        ],
        domain_repairs=[{"status": "applied"}, {"status": "skipped"}],
        contract_violations={
            "passed": False,
            "baseline_name": "customer_contract",
            "baseline_version": "2",
            "n_errors": 1,
            "n_warnings": 1,
            "findings": [
                {
                    "status": "failed",
                    "check_id": "not_null",
                    "column": "email",
                    "message": "email is required",
                },
                {
                    "status": "warning",
                    "check_id": "range",
                    "column": None,
                    "message": "unexpected drift",
                },
                {
                    "status": "passed",
                    "check_id": "ignored",
                    "message": "not rendered",
                },
            ],
        },
        decisions_hash="abc123",
    )

    payload = report.to_dict()
    text = report.summary()

    assert payload["contract_violations"]["baseline_name"] == "customer_contract"
    assert payload["decisions_hash"] == "abc123"
    assert "domain:  crm" in text
    assert "1 error(s), 1 warning(s), 1 repair(s) applied" in text
    assert "contract 'customer_contract' v2: FAIL" in text
    assert "[not_null] `email`: email is required" in text
    assert "[range]: unexpected drift" in text


def test_revert_restores_values_and_skips_missing_columns():
    report = fd.CleanReport()
    report.undo_log = {
        "entries": [
            {"action_id": "a1", "column": "age", "index": ["r1", "absent"], "value": 41},
            {"action_id": "a2", "column": "missing_column", "index": ["r1"], "value": "x"},
        ],
        "column_dtypes": {"age": "int64"},
    }
    df = pd.DataFrame({"age": [0, 2], "name": ["Ann", "Bo"]}, index=["r1", "r2"])

    restored = report.revert(df)

    assert restored.loc["r1", "age"] == 41
    assert restored.loc["r2", "age"] == 2
    assert str(restored["age"].dtype) == "int64"
    assert "missing_column" not in restored.columns


def test_action_str_format():
    action = fd.Action(step="impute", column="age", description="filled 2", count=2)
    assert str(action) == "[impute] 'age': filled 2"
    table_level = fd.Action(step="drop_empty_rows", column=None, description="dropped 1",
                            count=1)
    assert str(table_level) == "[drop_empty_rows] dropped 1"


def test_bool_reflects_whether_anything_changed(messy, already_clean):
    _, dirty_report = fd.clean(messy, return_report=True)
    _, clean_report = fd.clean(already_clean, return_report=True)
    assert dirty_report
    assert not clean_report


def test_repr_is_compact(messy):
    _, report = fd.clean(messy, return_report=True)
    assert repr(report).startswith("<CleanReport:")
