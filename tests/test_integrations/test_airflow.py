"""Tests for the Airflow integration (using a fake ``airflow`` module)."""

from __future__ import annotations

import pytest


def test_operator_passes_and_pushes(fake_airflow, make_ti, sample_df):
    from freshdata.integrations.airflow import FreshDataCleanOperator

    op = FreshDataCleanOperator(
        task_id="gate", input_task_id="extract", trust_score_threshold=0.0
    )
    ti = make_ti(sample_df)
    out = op.execute({"ti": ti})
    assert out is not None
    assert "return_value" in ti.pushed
    assert "return_value__gate" in ti.pushed
    assert ti.pushed["return_value__gate"]["passed"] is True


def test_operator_fail_raises(fake_airflow, make_ti, sample_df):
    from freshdata.integrations.airflow import FreshDataCleanOperator

    op = FreshDataCleanOperator(
        task_id="gate",
        input_task_id="extract",
        trust_score_threshold=999.0,
        on_low_score="fail",
    )
    with pytest.raises(fake_airflow.AirflowException):
        op.execute({"ti": make_ti(sample_df)})


def test_operator_skip_raises(fake_airflow, make_ti, sample_df):
    from freshdata.integrations.airflow import FreshDataCleanOperator

    op = FreshDataCleanOperator(
        task_id="gate",
        input_task_id="extract",
        trust_score_threshold=999.0,
        on_low_score="skip",
    )
    with pytest.raises(fake_airflow.AirflowSkipException):
        op.execute({"ti": make_ti(sample_df)})


def test_operator_missing_df_raises(fake_airflow, make_ti):
    from freshdata.integrations.airflow import FreshDataCleanOperator

    op = FreshDataCleanOperator(task_id="gate", input_task_id="extract")
    with pytest.raises(fake_airflow.AirflowException):
        op.execute({"ti": make_ti(None)})


def test_operator_warn_does_not_raise(fake_airflow, make_ti, sample_df):
    from freshdata.integrations.airflow import FreshDataCleanOperator

    op = FreshDataCleanOperator(
        task_id="gate",
        input_task_id="extract",
        trust_score_threshold=999.0,
        on_low_score="warn",
    )
    out = op.execute({"ti": make_ti(sample_df)})
    assert out is not None  # warn returns the cleaned frame
