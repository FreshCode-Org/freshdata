"""Plugin isolation: malformed plugins/proposals, duplicate names, late network opt-in.

Regression tests for #297 (plugin failures escaping isolation), #298 (duplicate
names replaced silently) and #299 (``FRESHDATA_ALLOW_NETWORK_PLUGINS`` set after
registration never activating the plugin).
"""

from __future__ import annotations

import dataclasses
import importlib.metadata as md
import logging

import pandas as pd
import pytest

import freshdata as fd
from freshdata.plugins import (
    active_backend_names,
    active_experts,
    clear_plugins,
    get_active_backend,
    registered_plugins,
)
from freshdata.semantic.scoring import make_proposal

_ENV = "FRESHDATA_ALLOW_NETWORK_PLUGINS"


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    clear_plugins()
    yield
    clear_plugins()


def _prop(column: str, raw: str = "x", proposed: str = "y"):
    return make_proposal(
        column=column,
        raw_value=raw,
        proposed_value=proposed,
        issue_type="email_format",
        expert="p",
        base_confidence=0.99,
        evidence=(),
        count=1,
        rationale="r",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"a": ["x", "q", "y", "z"]})


def _plugin_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "freshdata.plugins" and r.levelno >= logging.WARNING
    ]


def _install_fake_entry_points(monkeypatch, factories):
    class _EP:
        def __init__(self, name, factory):
            self.name = name
            self._factory = factory

        def load(self):
            return self._factory

    def fake_entry_points(*, group=None, **_kw):
        return [_EP(n, f) for n, f in factories.get(group, [])]

    monkeypatch.setattr(md, "entry_points", fake_entry_points)


# --------------------------------------------------------------------------- #
# #297 - failures degrade safely
# --------------------------------------------------------------------------- #


class WrongColumnExpert:
    name = "wrong_column"
    issue_type = "email_format"

    def applies(self, info):
        return True

    def propose(self, series, info):
        return [_prop("no_such_column")]


class BadProvenanceExpert(WrongColumnExpert):
    name = "bad_provenance"

    def propose(self, series, info):
        return [dataclasses.replace(_prop(info.name), provenance="oops")]


class RaisingExpert(WrongColumnExpert):
    name = "raising"

    def propose(self, series, info):
        raise RuntimeError("plugin bug")


class TestProposalIsolation:
    def test_wrong_column_proposal_is_dropped_and_clean_completes(self, caplog):
        fd.register_expert(WrongColumnExpert())
        df = _frame()
        with caplog.at_level(logging.WARNING, logger="freshdata.plugins"):
            out = fd.clean(df, semantic_mode="auto", verbose=False)
        assert list(out["a"]) == list(df["a"])
        messages = [m for m in _plugin_warnings(caplog) if "not given" in m]
        assert len(messages) == 1  # logged once per plugin, not per column
        assert "wrong_column" in messages[0]

    def test_wrong_column_logged_once_across_columns(self, caplog):
        fd.register_expert(WrongColumnExpert())
        df = pd.DataFrame({"a": ["x", "q", "y", "z"], "b": ["x", "q", "y", "z"]})
        with caplog.at_level(logging.WARNING, logger="freshdata.plugins"):
            fd.clean(df, semantic_mode="auto", verbose=False)
            fd.clean(df, semantic_mode="auto", verbose=False)
        assert len([m for m in _plugin_warnings(caplog) if "not given" in m]) == 1

    def test_expert_proposal_for_its_own_column_is_kept(self):
        fd.register_expert(WrongColumnExpert())
        (expert,) = active_experts()

        class _Info:
            name = "a"

        expert._record.obj.propose = lambda s, i: [_prop(i.name)]
        kept = expert.propose(pd.Series(["x"], name="a"), _Info())
        assert [p.column for p in kept] == ["a"]
        assert kept[0].backend == "plugin:wrong_column"
        assert kept[0].provenance["plugin"] == "wrong_column"

    def test_malformed_provenance_is_dropped_and_clean_completes(self, caplog):
        fd.register_expert(BadProvenanceExpert())
        df = _frame()
        with caplog.at_level(logging.WARNING, logger="freshdata.plugins"):
            out = fd.clean(df, semantic_mode="auto", verbose=False)
        assert list(out["a"]) == list(df["a"])
        assert any(
            "malformed proposal" in m and "bad_provenance" in m for m in _plugin_warnings(caplog)
        )

    def test_raising_propose_is_still_isolated(self, caplog):
        fd.register_expert(RaisingExpert())
        df = _frame()
        with caplog.at_level(logging.WARNING, logger="freshdata.plugins"):
            out = fd.clean(df, semantic_mode="auto", verbose=False)
        assert list(out["a"]) == list(df["a"])
        assert any("propose() failed" in m for m in _plugin_warnings(caplog))

    def test_backend_proposal_for_unknown_column_is_dropped(self, caplog):
        class OffFrameBackend:
            name = "offframe"
            max_risk = "high"

            def propose(self, df, ctx, budget):
                return [_prop("no_such_column")]

        fd.register_backend(OffFrameBackend())
        df = _frame()
        with caplog.at_level(logging.WARNING, logger="freshdata.plugins"):
            out = fd.clean(
                df, semantic_mode="auto", semantic_backends=("offframe",), verbose=False
            )
        assert list(out["a"]) == list(df["a"])
        assert any("not given" in m and "offframe" in m for m in _plugin_warnings(caplog))


class TestMalformedMetadata:
    def test_unhashable_max_risk_falls_back_to_default(self):
        class Unhashable:
            name = "unhashable"
            max_risk = ["low"]

            def applies(self, info):
                return True

            def propose(self, series, info):
                return []

        fd.register_expert(Unhashable())
        (rec,) = registered_plugins("expert")
        assert rec["max_risk"] == "high"
        assert rec["active"] is True

    def test_malformed_entry_point_is_skipped_and_rest_register(self, monkeypatch, caplog):
        class Bad:
            name = "bad"
            max_risk = "low"

            @property
            def uses_network(self):
                raise RuntimeError("broken metadata")

            def applies(self, info):
                return True

            def propose(self, series, info):
                return []

        class Good:
            name = "good"
            max_risk = "low"

            def applies(self, info):
                return True

            def propose(self, series, info):
                return []

        _install_fake_entry_points(
            monkeypatch, {"freshdata.experts": [("bad", Bad), ("good", Good)]}
        )
        with caplog.at_level(logging.WARNING, logger="freshdata.plugins"):
            first = [p["name"] for p in registered_plugins()]
        assert first == ["good"]
        assert [p["name"] for p in registered_plugins()] == ["good"]
        assert any("'bad'" in m and "failed to load" in m for m in _plugin_warnings(caplog))

    def test_issue_repro_unhashable_entry_point_metadata(self, monkeypatch):
        class Bad:
            name = "bad"
            max_risk = ["low"]

            def applies(self, i):
                return True

            def propose(self, s, i):
                return []

        class Good(Bad):
            name = "good"
            max_risk = "low"

        _install_fake_entry_points(
            monkeypatch, {"freshdata.experts": [("bad", Bad), ("good", Good)]}
        )
        names = [p["name"] for p in registered_plugins()]
        assert "good" in names


# --------------------------------------------------------------------------- #
# #298 - duplicate names warn
# --------------------------------------------------------------------------- #


class _DupA:
    name = "dup"

    def applies(self, i):
        return True

    def propose(self, s, i):
        return []


class _DupB(_DupA):
    pass


class TestDuplicateNames:
    def test_duplicate_name_warns_and_last_wins(self, caplog):
        second = _DupB()
        with caplog.at_level(logging.WARNING, logger="freshdata.plugins"):
            fd.register_expert(_DupA())
            assert _plugin_warnings(caplog) == []
            fd.register_expert(second)
        messages = _plugin_warnings(caplog)
        assert len(messages) == 1
        assert "'dup'" in messages[0]
        assert "_DupA" in messages[0] and "_DupB" in messages[0]
        (expert,) = active_experts()
        assert expert._record.obj is second

    def test_reregistering_same_object_does_not_warn(self, caplog):
        plugin = _DupA()
        with caplog.at_level(logging.WARNING, logger="freshdata.plugins"):
            fd.register_expert(plugin)
            fd.register_expert(plugin)
        assert _plugin_warnings(caplog) == []

    def test_same_name_in_different_kinds_does_not_warn(self, caplog):
        class Validator:
            name = "dup"

            def validate(self, df, policy, ctx):
                return []

        with caplog.at_level(logging.WARNING, logger="freshdata.plugins"):
            fd.register_expert(_DupA())
            fd.register_validator(Validator())
        assert _plugin_warnings(caplog) == []


# --------------------------------------------------------------------------- #
# #299 - network opt-in honoured at use time
# --------------------------------------------------------------------------- #


class _NetExpert:
    name = "net"
    uses_network = True
    max_risk = "low"

    def __init__(self):
        self.calls: list[str] = []

    def applies(self, info):
        self.calls.append(info.name)
        return False

    def propose(self, series, info):
        return []


class _NetBackend:
    name = "netbackend"
    uses_network = True
    max_risk = "low"

    def propose(self, df, ctx, budget):
        return []


class TestLateNetworkOptIn:
    def test_env_set_after_registration_activates_plugin(self, monkeypatch):
        plugin = _NetExpert()
        fd.register_expert(plugin)
        (rec,) = registered_plugins("expert")
        assert rec["active"] is False
        assert _ENV in rec["inactive_reason"]

        monkeypatch.setenv(_ENV, "1")
        fd.clean(_frame(), semantic_mode="auto", verbose=False)
        (rec,) = registered_plugins("expert")
        assert rec["active"] is True
        assert rec["inactive_reason"] is None
        assert "a" in plugin.calls

    def test_network_plugin_stays_inactive_without_env(self):
        plugin = _NetExpert()
        fd.register_expert(plugin)
        fd.clean(_frame(), semantic_mode="auto", verbose=False)
        assert active_experts() == ()
        (rec,) = registered_plugins("expert")
        assert rec["active"] is False
        assert plugin.calls == []

    @pytest.mark.parametrize("value", ["", "0", "true", "yes"])
    def test_only_env_value_one_opts_in(self, monkeypatch, value):
        fd.register_expert(_NetExpert())
        monkeypatch.setenv(_ENV, value)
        assert active_experts() == ()

    def test_unsetting_env_deactivates_env_gated_plugin(self, monkeypatch):
        monkeypatch.setenv(_ENV, "1")
        fd.register_expert(_NetExpert())
        assert len(active_experts()) == 1
        monkeypatch.delenv(_ENV)
        assert active_experts() == ()
        (rec,) = registered_plugins("expert")
        assert rec["active"] is False

    def test_explicit_allow_network_is_not_env_dependent(self, monkeypatch):
        fd.register_expert(_NetExpert(), allow_network=True)
        assert len(active_experts()) == 1
        monkeypatch.setenv(_ENV, "0")
        assert len(active_experts()) == 1

    def test_backend_accessors_honour_late_env(self, monkeypatch):
        fd.register_backend(_NetBackend())
        assert get_active_backend("netbackend") is None
        assert active_backend_names() == ()
        monkeypatch.setenv(_ENV, "1")
        assert get_active_backend("netbackend") is not None
        assert active_backend_names() == ("netbackend",)

    def test_env_does_not_override_missing_dependency(self, monkeypatch):
        class NetNeedsThing(_NetExpert):
            name = "netneeds"
            requires = ("a_package_that_does_not_exist_xyz",)

        monkeypatch.setenv(_ENV, "1")
        fd.register_expert(NetNeedsThing())
        assert active_experts() == ()
        (rec,) = registered_plugins("expert")
        assert "missing dependency" in rec["inactive_reason"]

    def test_entry_point_network_plugin_honours_late_env(self, monkeypatch):
        _install_fake_entry_points(monkeypatch, {"freshdata.experts": [("net", _NetExpert)]})
        assert active_experts() == ()
        monkeypatch.setenv(_ENV, "1")
        assert len(active_experts()) == 1
