"""Tests for the domain plugin registry."""

from __future__ import annotations

import pytest

from freshdata.domains import (
    ColumnMapping,
    DomainValidator,
    RepairLog,
    ValidationReport,
    available,
    get_validator,
    register,
)
from freshdata.domains import registry as registry_mod
from freshdata.domains.finance import FinanceValidator
from freshdata.domains.registry import UnknownDomainError


class _Dummy(DomainValidator):
    domain_name = "dummy"
    version = "9.9.9"
    schema_version = "test"

    def __init__(self, *, column_map=None):
        self.column_map = column_map

    def detect_columns(self, df):
        return ColumnMapping()

    def validate(self, df):
        return ValidationReport(self.domain_name, self.version, self.schema_version)

    def repair(self, df, report):
        return df, RepairLog()

    def describe(self):
        return {"domain": self.domain_name}


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate runtime registrations between tests."""
    saved = dict(registry_mod._REGISTERED)
    yield
    registry_mod._REGISTERED.clear()
    registry_mod._REGISTERED.update(saved)


def test_finance_is_available():
    assert "finance" in available()
    assert isinstance(get_validator("finance"), FinanceValidator)


def test_unknown_domain_lists_available():
    with pytest.raises(UnknownDomainError) as exc:
        get_validator("does_not_exist")
    assert exc.value.name == "does_not_exist"
    assert "finance" in exc.value.available


def test_runtime_registration():
    register("dummy", _Dummy)
    assert "dummy" in available()
    assert isinstance(get_validator("dummy"), _Dummy)


def test_register_rejects_non_validator():
    with pytest.raises(TypeError):
        register("bad", dict)


def test_builtin_takes_precedence_over_registration():
    register("finance", _Dummy)            # must not shadow the built-in
    assert isinstance(get_validator("finance"), FinanceValidator)


def test_entry_point_discovery(monkeypatch):
    monkeypatch.setattr(registry_mod, "_entry_point_classes", lambda: {"extpack": _Dummy})
    assert "extpack" in available()
    assert isinstance(get_validator("extpack"), _Dummy)


def test_column_map_forwarded():
    register("dummy", _Dummy)
    v = get_validator("dummy", column_map={"a": "b"})
    assert v.column_map == {"a": "b"}


def test_broken_entry_point_is_skipped(monkeypatch):
    class _BadEP:
        name = "broken"

        def load(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(registry_mod, "entry_points", lambda **_: [_BadEP()])
    # A plugin that fails to load is swallowed, not propagated.
    assert registry_mod._entry_point_classes() == {}


def test_invalid_entry_point_class_is_skipped(monkeypatch):
    class _InvalidEP:
        name = "invalid"

        def load(self):
            return dict

    monkeypatch.setattr(registry_mod, "entry_points", lambda **_: [_InvalidEP()])
    assert registry_mod._entry_point_classes() == {}
