"""Fixture preflight behavior for expectation helpers."""

from __future__ import annotations

import pytest
from _pytest.outcomes import Failed

from expectations import load_fixture, load_online_fixture


def test_load_fixture_fails_when_local_fixture_missing():
    with pytest.raises(Failed, match="fixture definitely_missing_fixture not found"):
        load_fixture("definitely_missing_fixture")


def test_load_online_fixture_strict_mode_fails_when_cache_missing(monkeypatch):
    monkeypatch.setenv("FRESHDATA_STRICT_ONLINE_FIXTURES", "1")
    with pytest.raises(Failed, match="online cache missing"):
        load_online_fixture("definitely_missing_online_fixture")
