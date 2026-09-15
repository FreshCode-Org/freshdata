"""dbt transform: ``on_low_score="fail"`` raises (#343); unique audit files (#344)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy import event

from freshdata.integrations import TrustGateError
from freshdata.integrations.dbt import FreshDataDbtTransform, gate_manifest


@pytest.fixture
def warehouse(tmp_path, sample_df):
    conn = f"sqlite:///{tmp_path / 'wh.db'}"
    engine = sa.create_engine(conn)
    with engine.begin() as connection:
        sample_df.to_sql("orders", connection, index=False)
        sample_df.to_sql("customers", connection, index=False)
    engine.dispose()
    return conn


@pytest.fixture
def schemas(tmp_path, monkeypatch):
    """A sqlite warehouse exposing ``staging``, ``marts`` and ``archive`` via ATTACH.

    ``staging.orders`` (3 rows), ``marts.orders`` (4 rows) and ``archive.ORDERS``
    (5 rows) differ in row count, so an audit file's ``row_count_in`` shows which
    model wrote it. The main database holds ``customers`` and a 2-row ``orders``.
    """
    frames = {
        "staging": ("orders", pd.DataFrame({"a": [1, None, None], "b": [None, None, "z"]})),
        "marts": ("orders", pd.DataFrame({"a": [1, 2, 3, 4]})),
        "archive": ("ORDERS", pd.DataFrame({"a": [1, 2, 3, 4, 5]})),
    }
    paths = {}
    for schema, (table, frame) in frames.items():
        paths[schema] = tmp_path / f"{schema}.db"
        # A stdlib connection: pandas 1.5's case-sensitivity check for a
        # non-lowercase table name does not work on a SQLAlchemy 2 Connection.
        with closing(sqlite3.connect(paths[schema])) as connection:
            frame.to_sql(table, connection, index=False)
            connection.commit()
    main = f"sqlite:///{tmp_path / 'main.db'}"
    engine = sa.create_engine(main)
    with engine.begin() as connection:
        pd.DataFrame({"id": [1, 2]}).to_sql("customers", connection, index=False)
        pd.DataFrame({"a": [5, 6]}).to_sql("orders", connection, index=False)
    engine.dispose()

    original = sa.create_engine

    def create_engine(url, **kwargs):  # noqa: ANN001, ANN003, ANN202
        eng = original(url, **kwargs)

        @event.listens_for(eng, "connect")
        def _attach(dbapi_connection, _record):  # noqa: ANN001, ANN202
            for schema, path in paths.items():
                dbapi_connection.execute(f"ATTACH DATABASE '{path}' AS {schema}")

        return eng

    monkeypatch.setattr(sa, "create_engine", create_engine)
    return main


def _write_manifest(tmp_path, nodes):
    path = tmp_path / "manifest.json"
    manifest = {
        "nodes": {
            unique_id: {"resource_type": "model", **node} for unique_id, node in nodes.items()
        }
    }
    path.write_text(json.dumps(manifest))
    return path


def _audit_files(directory):
    return sorted(p.name for p in directory.iterdir())


# --------------------------------------------------------------------------- #
# #343: on_low_score="fail"                                                    #
# --------------------------------------------------------------------------- #
def test_on_low_score_fail_raises_and_still_writes_audit(warehouse, tmp_path):
    out = tmp_path / "audit"
    with pytest.raises(TrustGateError, match="trust gate failed"):
        FreshDataDbtTransform(
            model_name="orders",
            conn_str=warehouse,
            trust_score_threshold=999.0,
            on_low_score="fail",
            output_dir=str(out),
        ).run()
    audit = json.loads((out / "orders_audit.json").read_text())
    assert audit["passed"] is False
    assert audit["on_low_score"] == "fail"


def test_run_raise_on_fail_false_returns_failing_result(warehouse):
    result = FreshDataDbtTransform(
        model_name="orders",
        conn_str=warehouse,
        trust_score_threshold=999.0,
        on_low_score="fail",
        fail_on_low_score=True,
    ).run(raise_on_fail=False)
    assert result.passed is False
    assert result.should_fail is True


@pytest.mark.parametrize("on_low_score", ["warn", "skip"])
def test_non_fail_policies_do_not_raise(warehouse, on_low_score):
    result = FreshDataDbtTransform(
        model_name="orders",
        conn_str=warehouse,
        trust_score_threshold=999.0,
        on_low_score=on_low_score,
    ).run()
    assert result.passed is False
    assert result.should_fail is False


@pytest.mark.parametrize("kwargs", [{"on_low_score": "fail"}, {"fail_on_low_score": True}])
def test_passing_gate_never_raises(warehouse, kwargs):
    result = FreshDataDbtTransform(
        model_name="orders", conn_str=warehouse, trust_score_threshold=0.0, **kwargs
    ).run()
    assert result.passed is True


def test_gate_manifest_fail_policy_records_failure_and_continues(warehouse, tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {
            "model.proj.orders": {"name": "orders", "schema": None, "alias": "orders"},
            "model.proj.customers": {
                "name": "customers",
                "schema": None,
                "alias": "customers",
            },
        },
    )
    summary = gate_manifest(
        str(manifest),
        conn_str=warehouse,
        trust_score_threshold=999.0,
        on_low_score="fail",
    )
    assert summary["models_processed"] == 2
    assert summary["failed_models"] == 2
    assert summary["all_passed"] is False
    assert [m["model"] for m in summary["models"]] == ["orders", "customers"]
    for model in summary["models"]:
        assert "error" not in model
        assert model["passed"] is False


# --------------------------------------------------------------------------- #
# #344: unique audit files for same-alias models                               #
# --------------------------------------------------------------------------- #
def test_same_alias_in_different_schemas_writes_distinct_audits(schemas, tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {
            "model.proj.staging_orders": {
                "name": "staging_orders",
                "schema": "staging",
                "alias": "orders",
            },
            "model.proj.marts_orders": {
                "name": "marts_orders",
                "schema": "marts",
                "alias": "orders",
            },
            "model.proj.customers": {"name": "customers", "schema": None},
        },
    )
    out = tmp_path / "audit"
    summary = gate_manifest(
        str(manifest), conn_str=schemas, output_dir=str(out), trust_score_threshold=0.0
    )
    assert summary["models_processed"] == 3
    assert all("error" not in m for m in summary["models"])
    assert _audit_files(out) == [
        "customers_audit.json",
        "marts.orders_audit.json",
        "staging.orders_audit.json",
    ]
    staging = json.loads((out / "staging.orders_audit.json").read_text())
    marts = json.loads((out / "marts.orders_audit.json").read_text())
    assert staging["row_count_in"] == 3
    assert marts["row_count_in"] == 4


def test_non_colliding_models_keep_alias_audit_names(warehouse, tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {
            "model.proj.orders": {"name": "orders", "schema": None, "alias": "orders"},
            "model.proj.customers": {"name": "customers", "schema": None},
        },
    )
    out = tmp_path / "audit"
    summary = gate_manifest(
        str(manifest), conn_str=warehouse, output_dir=str(out), trust_score_threshold=0.0
    )
    assert summary["all_passed"] is True
    assert _audit_files(out) == ["customers_audit.json", "orders_audit.json"]


def test_colliding_alias_without_schema_uses_unique_id(schemas, tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {
            "model.proj.main_orders": {"name": "main_orders", "schema": None, "alias": "orders"},
            "model.proj.archive_orders": {
                "name": "archive_orders",
                "schema": "archive",
                "alias": "ORDERS",
            },
            "model.proj.customers": {"name": "customers", "schema": None},
        },
    )
    out = tmp_path / "audit"
    summary = gate_manifest(
        str(manifest), conn_str=schemas, output_dir=str(out), trust_score_threshold=0.0
    )
    assert summary["models_processed"] == 3
    assert all("error" not in m for m in summary["models"])
    # "orders" and "ORDERS" collide case-insensitively: the schema-less model falls
    # back to its unique_id, the other gets "<schema>.<alias>". "customers" is unique.
    assert _audit_files(out) == [
        "archive.ORDERS_audit.json",
        "customers_audit.json",
        "model.proj.main_orders_audit.json",
    ]
    main = json.loads((out / "model.proj.main_orders_audit.json").read_text())
    archive = json.loads((out / "archive.ORDERS_audit.json").read_text())
    assert main["row_count_in"] == 2
    assert archive["row_count_in"] == 5


def test_unsafe_schema_in_colliding_alias_is_rejected(schemas, tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {
            "model.proj.evil_orders": {
                "name": "evil_orders",
                "schema": "../evil",
                "alias": "orders",
            },
            "model.proj.marts_orders": {
                "name": "marts_orders",
                "schema": "marts",
                "alias": "orders",
            },
        },
    )
    out = tmp_path / "audit"
    summary = gate_manifest(
        str(manifest), conn_str=schemas, output_dir=str(out), trust_score_threshold=0.0
    )
    evil, marts = summary["models"]
    assert "safe dbt model name" in evil["error"]
    assert "error" not in marts
    assert summary["failed_models"] == 1
    assert _audit_files(out) == ["marts.orders_audit.json"]
    assert not list(tmp_path.glob("evil*"))


@pytest.mark.parametrize("audit_name", ["", ".", "..", "../orders", "a/b", "a\\b", "/abs"])
def test_transform_rejects_unsafe_audit_name(audit_name):
    with pytest.raises(ValueError, match="safe dbt model name"):
        FreshDataDbtTransform(model_name="orders", audit_name=audit_name)


def test_transform_audit_name_overrides_file_name(warehouse, tmp_path):
    FreshDataDbtTransform(
        model_name="orders",
        conn_str=warehouse,
        output_dir=str(tmp_path / "audit"),
        trust_score_threshold=0.0,
        audit_name="main.orders",
    ).run()
    assert _audit_files(tmp_path / "audit") == ["main.orders_audit.json"]
