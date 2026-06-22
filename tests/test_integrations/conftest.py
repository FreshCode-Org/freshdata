"""Fixtures for the integrations tests.

The orchestrator adapters import their framework lazily (inside functions), so we can
exercise them without installing dagster/airflow by injecting lightweight fakes into
``sys.modules``. Each fake fixture also resets the adapter module's cached lazy class
so it is rebuilt against the fake.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small frame with a missing value, a duplicate row, and an outlier."""
    return pd.DataFrame(
        {
            "a": [1, 2, 2, None, 1000],
            "b": ["x", "y", "y", "z", "x"],
        }
    )


# --------------------------------------------------------------------------- #
# Fake Dagster                                                                 #
# --------------------------------------------------------------------------- #
class _FakeMetadataValue:
    @staticmethod
    def float(v):  # noqa: ANN001, ANN205
        return ("float", v)

    @staticmethod
    def int(v):  # noqa: ANN001, ANN205
        return ("int", v)

    @staticmethod
    def bool(v):  # noqa: ANN001, ANN205
        return ("bool", v)

    @staticmethod
    def text(v):  # noqa: ANN001, ANN205
        return ("text", v)

    @staticmethod
    def json(v):  # noqa: ANN001, ANN205
        return ("json", v)


class _FakeSeverity:
    WARN = "WARN"
    ERROR = "ERROR"


class _FakeAssetCheckResult:
    def __init__(self, *, passed, severity, metadata):  # noqa: ANN001, ANN204
        self.passed = passed
        self.severity = severity
        self.metadata = metadata


class _FakeConfigurableResource:
    def __init__(self, **kwargs):  # noqa: ANN003, ANN204
        for key, value in kwargs.items():
            setattr(self, key, value)


def _fake_asset_check(*, asset, name, blocking=False):  # noqa: ANN001, ANN202
    def _decorator(fn):  # noqa: ANN001, ANN202
        fn.dagster_asset = asset
        fn.dagster_name = name
        fn.dagster_blocking = blocking
        return fn

    return _decorator


@pytest.fixture
def fake_dagster(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake ``dagster`` module and reset the adapter's cached class."""
    module = types.ModuleType("dagster")
    module.MetadataValue = _FakeMetadataValue
    module.AssetCheckSeverity = _FakeSeverity
    module.AssetCheckResult = _FakeAssetCheckResult
    module.ConfigurableResource = _FakeConfigurableResource
    module.asset_check = _fake_asset_check
    monkeypatch.setitem(sys.modules, "dagster", module)

    import freshdata.integrations.dagster as adapter

    monkeypatch.setattr(adapter, "_resource_cls", None, raising=False)
    return module


class _FakeAsset:
    """Stand-in for a Dagster AssetsDefinition exposing ``.key.path``."""

    def __init__(self, name: str) -> None:
        self.key = types.SimpleNamespace(path=[name])


@pytest.fixture
def fake_asset() -> _FakeAsset:
    return _FakeAsset("orders")


# --------------------------------------------------------------------------- #
# Fake Airflow                                                                 #
# --------------------------------------------------------------------------- #
class _FakeBaseOperator:
    def __init__(self, **kwargs):  # noqa: ANN003, ANN204
        # Mimic Airflow's BaseOperator accepting task_id etc.; expose a logger.
        self.task_id = kwargs.get("task_id")
        import logging

        self.log = logging.getLogger("airflow.task")


class _FakeAirflowException(Exception):
    pass


class _FakeAirflowSkipException(Exception):
    pass


@pytest.fixture
def fake_airflow(monkeypatch: pytest.MonkeyPatch):
    """Inject fake ``airflow`` modules and reset the adapter's cached class."""
    airflow = types.ModuleType("airflow")
    models = types.ModuleType("airflow.models")
    exceptions = types.ModuleType("airflow.exceptions")
    models.BaseOperator = _FakeBaseOperator
    exceptions.AirflowException = _FakeAirflowException
    exceptions.AirflowSkipException = _FakeAirflowSkipException
    monkeypatch.setitem(sys.modules, "airflow", airflow)
    monkeypatch.setitem(sys.modules, "airflow.models", models)
    monkeypatch.setitem(sys.modules, "airflow.exceptions", exceptions)

    import freshdata.integrations.airflow as adapter

    monkeypatch.setattr(adapter, "_operator_cls", None, raising=False)
    return types.SimpleNamespace(
        AirflowException=_FakeAirflowException,
        AirflowSkipException=_FakeAirflowSkipException,
    )


class _FakeTI:
    """A minimal Airflow TaskInstance: an in-memory XCom store."""

    def __init__(self, pull_value):  # noqa: ANN001, ANN204
        self._pull_value = pull_value
        self.pushed: dict[str, object] = {}

    def xcom_pull(self, task_ids=None, key="return_value"):  # noqa: ANN001, ANN202
        return self._pull_value

    def xcom_push(self, key, value):  # noqa: ANN001, ANN202
        self.pushed[key] = value


@pytest.fixture
def make_ti():
    """Return a factory building a fake Airflow TaskInstance with a pull value."""
    return _FakeTI
