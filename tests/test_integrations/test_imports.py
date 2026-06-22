"""Import-guard tests: every subpackage imports without its optional dependency."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from freshdata import integrations


def test_core_exports():
    assert {
        "evaluate_trust_gate",
        "TrustGateResult",
        "TrustGateError",
        "OnLowScore",
    }.issubset(integrations.__all__)


@pytest.mark.parametrize(
    "module",
    [
        "freshdata.integrations.dagster",
        "freshdata.integrations.airflow",
        "freshdata.integrations.dbt",
        "freshdata.integrations.dbt.cli",
    ],
)
def test_subpackage_imports_without_optional_dep(module):
    assert importlib.import_module(module) is not None


def test_adapter_all_lists():
    import freshdata.integrations.airflow as airflow_mod
    import freshdata.integrations.dagster as dagster_mod
    import freshdata.integrations.dbt as dbt_mod

    assert dagster_mod.__all__ == ["FreshDataResource", "freshdata_asset_check"]
    assert airflow_mod.__all__ == ["FreshDataCleanOperator"]
    assert set(dbt_mod.__all__) == {"FreshDataDbtTransform", "gate_manifest"}


def test_macro_file_ships_with_package():
    import freshdata.integrations.dbt as dbt_mod

    macro = Path(dbt_mod.__file__).parent / "macros" / "freshdata_trust_gate.sql"
    assert macro.exists()
    assert "freshdata_trust_gate" in macro.read_text()


def test_unknown_attribute_raises():
    import freshdata.integrations.airflow as airflow_mod
    import freshdata.integrations.dagster as dagster_mod

    with pytest.raises(AttributeError):
        _ = dagster_mod.does_not_exist
    with pytest.raises(AttributeError):
        _ = airflow_mod.does_not_exist


def test_lazy_class_requires_dep_when_absent():
    if "dagster" in sys.modules:
        pytest.skip("dagster is installed")
    import freshdata.integrations.dagster as dagster_mod

    with pytest.raises(ImportError):
        _ = dagster_mod.FreshDataResource
