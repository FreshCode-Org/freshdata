"""Plugin system: registration, discovery, safety guarantees, contract tests.

Every test clears the registry in teardown so plugins never leak between tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata.findings import QualityFinding
from freshdata.plugins import (
    active_experts,
    clear_plugins,
    known_backend_names,
    registered_plugins,
)
from freshdata.semantic.scoring import make_proposal
from freshdata.semantic.types import SemanticEvidence


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_plugins()
    yield
    clear_plugins()


# --------------------------------------------------------------------------- #
# sample plugins
# --------------------------------------------------------------------------- #


class TitleCaseExpert:
    name = "titlecase"
    issue_type = "category_synonym"
    semantic_types = ("categorical",)
    max_risk = "medium"
    uses_network = False
    requires = ()

    def applies(self, info):
        return info.role == "categorical" and not info.identifier_like

    def propose(self, series, info):
        out = []
        for raw, count in series.value_counts(dropna=True).items():
            if isinstance(raw, str) and raw != raw.title():
                out.append(make_proposal(
                    column=info.name, raw_value=raw, proposed_value=raw.title(),
                    issue_type=self.issue_type, expert=self.name, base_confidence=0.95,
                    evidence=(SemanticEvidence("pattern", "not title case", 0.0),),
                    count=int(count), rationale=f"title-case {raw!r}", info=info,
                ))
        return out


class PrefixBackend:
    name = "prefixbackend"
    semantic_types = ("categorical",)
    max_risk = "medium"
    uses_network = False
    requires = ()

    def warm_up(self):
        pass

    def propose(self, df, ctx, budget):
        out = []
        for col in df.columns:
            info = ctx.columns.get(col)
            if info is None:
                continue
            for raw in df[col].dropna().unique():
                if isinstance(raw, str) and raw.startswith("lbl_"):
                    out.append(make_proposal(
                        column=col, raw_value=raw, proposed_value=raw[4:],
                        issue_type="category_synonym", expert=self.name, base_confidence=0.95,
                        evidence=(SemanticEvidence("pattern", "strip lbl_", 0.0),),
                        count=1, info=info, rationale="strip lbl_ prefix",
                    ))
        return out


class RowCountValidator:
    name = "rowcount"
    max_risk = "low"
    uses_network = False
    requires = ()

    def validate(self, df, policy, ctx):
        if len(df) < 5:
            return [QualityFinding.create(
                severity="warning", step="plugin", rule_name="plugin.rowcount",
                message=f"only {len(df)} rows", expected_condition=">=5 rows")]
        return []


# --------------------------------------------------------------------------- #
# registration + contract tests
# --------------------------------------------------------------------------- #


class TestRegistration:
    def test_register_and_introspect_expert(self):
        fd.register_expert(TitleCaseExpert())
        names = [p["name"] for p in registered_plugins("expert")]
        assert names == ["titlecase"]
        assert registered_plugins("expert")[0]["active"] is True

    def test_register_backend_appears_in_known_names(self):
        fd.register_backend(PrefixBackend())
        assert "prefixbackend" in known_backend_names()

    def test_clear_plugins_removes_all(self):
        fd.register_expert(TitleCaseExpert())
        fd.register_validator(RowCountValidator())
        clear_plugins()
        assert registered_plugins() == []


class TestContracts:
    def test_expert_contract_passes(self):
        fd.testing.expert_contract(TitleCaseExpert())

    def test_backend_contract_passes(self):
        fd.testing.semantic_backend_contract(PrefixBackend())

    def test_validator_contract_passes(self):
        fd.testing.validator_contract(RowCountValidator())

    def test_expert_contract_rejects_bad_max_risk(self):
        class BadRisk(TitleCaseExpert):
            max_risk = "catastrophic"

        with pytest.raises(AssertionError, match="max_risk"):
            fd.testing.expert_contract(BadRisk())

    def test_expert_contract_rejects_mutating_expert(self):
        class Mutator(TitleCaseExpert):
            def propose(self, series, info):
                series.iloc[0] = "MUTATED"  # illegal
                return []

        with pytest.raises(AssertionError, match="must not mutate"):
            fd.testing.expert_contract(Mutator())

    def test_expert_contract_rejects_non_proposal_output(self):
        class BadOutput(TitleCaseExpert):
            def propose(self, series, info):
                return ["not a proposal"]

        with pytest.raises(AssertionError, match="SemanticProposal"):
            fd.testing.expert_contract(BadOutput())

    def test_validator_contract_rejects_mutating_validator(self):
        class Mutator(RowCountValidator):
            def validate(self, df, policy, ctx):
                df["value"] = 0  # illegal
                return []

        with pytest.raises(AssertionError, match="must not mutate"):
            fd.testing.validator_contract(Mutator())


# --------------------------------------------------------------------------- #
# end-to-end through the pipeline
# --------------------------------------------------------------------------- #


class TestExpertPipeline:
    def test_expert_proposal_flows_with_provenance(self):
        fd.register_expert(TitleCaseExpert())
        df = pd.DataFrame({"city": ["new york", "boston", "chicago", "denver"]})
        _out, report = fd.clean(df, semantic_mode="assist", return_report=True)
        plugin_actions = [a for a in report if a.step == "semantic"
                          and a.metadata.get("plugin") == "titlecase"]
        assert plugin_actions
        assert all(a.model_id.endswith("plugin:titlecase") for a in plugin_actions)

    def test_plugin_cannot_change_protected_column(self):
        class IdMangler:
            name = "idmangler"
            issue_type = "spelled_number"
            semantic_types = ()
            max_risk = "low"
            uses_network = False
            requires = ()

            def applies(self, info):
                return True

            def propose(self, series, info):
                return [make_proposal(
                    column=info.name, raw_value=v, proposed_value="ZZZ",
                    issue_type="spelled_number", expert=self.name, base_confidence=0.99,
                    evidence=(SemanticEvidence("x", "y", 0.0),), count=1, info=info,
                    rationale="mangle") for v in series.dropna().unique()]

        fd.register_expert(IdMangler())
        df = pd.DataFrame({"customer_id": ["X1", "X2", "X3"], "note": ["a", "b", "c"]})
        out = fd.clean(df, semantic_mode="auto", id_columns=("customer_id",))
        assert list(out["customer_id"]) == ["X1", "X2", "X3"]

    def test_over_risk_proposals_are_dropped(self):
        class Rogue:
            name = "rogue"
            issue_type = "unsafe_ambiguous"
            semantic_types = ()
            max_risk = "low"
            uses_network = False
            requires = ()

            def applies(self, info):
                return True

            def propose(self, series, info):
                return [make_proposal(
                    column=info.name, raw_value=v, proposed_value="HACKED",
                    issue_type="unsafe_ambiguous", expert=self.name, base_confidence=0.99,
                    evidence=(SemanticEvidence("x", "y", 0.0),), count=1, info=info,
                    rationale="rogue") for v in series.dropna().unique()]

        fd.register_expert(Rogue())
        df = pd.DataFrame({"city": ["a", "b", "c"]})
        out = fd.clean(df, semantic_mode="auto")
        assert list(out["city"]) == ["a", "b", "c"]

    def test_failing_plugin_degrades_safely(self):
        class Boom:
            name = "boom"
            issue_type = "x"
            semantic_types = ()
            max_risk = "low"
            uses_network = False
            requires = ()

            def applies(self, info):
                raise RuntimeError("boom")

            def propose(self, series, info):
                raise RuntimeError("boom")

        fd.register_expert(Boom())
        df = pd.DataFrame({"city": ["a", "b", "c"]})
        out = fd.clean(df, semantic_mode="auto")  # must not raise
        assert list(out["city"]) == ["a", "b", "c"]


class TestBackendPipeline:
    def test_backend_proposal_applies_and_is_attributed(self):
        fd.register_backend(PrefixBackend())
        df = pd.DataFrame({"tag": ["lbl_red", "lbl_blue", "lbl_green", "plain"]})
        out, report = fd.clean(
            df, semantic_mode="auto", semantic_backends=("deterministic", "prefixbackend"),
            drop_duplicates=False, return_report=True)
        assert set(out["tag"]) == {"red", "blue", "green", "plain"}
        assert any(a.metadata.get("plugin") == "prefixbackend"
                   for a in report if a.step == "semantic")

    def test_unknown_backend_name_records_skip_not_error(self):
        df = pd.DataFrame({"tag": ["a", "b"]})
        _out, report = fd.clean(
            df, semantic_mode="auto", semantic_backends=("deterministic", "nope"),
            return_report=True)
        assert any("nope" in e["fallback_reason"] for e in report.fallback_events)

    def test_unknown_backend_name_raises_in_strict(self):
        df = pd.DataFrame({"tag": ["a", "b"]})
        with pytest.raises(Exception, match="nope"):
            fd.clean(df, semantic_mode="auto", strict=True,
                     semantic_backends=("deterministic", "nope"))


class TestValidatorPipeline:
    def test_validator_findings_appended(self):
        fd.register_validator(RowCountValidator())
        df = pd.DataFrame({"tag": ["a", "b", "a"]})
        findings = fd.validate(df, context="tag is unique.")
        assert any(f.rule_name == "plugin.rowcount" for f in findings)

    def test_failing_validator_degrades_safely(self):
        class BoomValidator:
            name = "boomv"
            max_risk = "low"
            uses_network = False
            requires = ()

            def validate(self, df, policy, ctx):
                raise RuntimeError("boom")

        fd.register_validator(BoomValidator())
        df = pd.DataFrame({"tag": ["a", "b", "a"]})
        findings = fd.validate(df, context="tag is unique.")  # must not raise
        assert any(f.rule_name == "context.unique" for f in findings)


# --------------------------------------------------------------------------- #
# network gating
# --------------------------------------------------------------------------- #


class TestNetworkGating:
    def _net_expert(self):
        class NetExpert:
            name = "net"
            issue_type = "x"
            semantic_types = ()
            max_risk = "low"
            uses_network = True
            requires = ()

            def applies(self, info):
                return False

            def propose(self, series, info):
                return []

        return NetExpert()

    def test_network_plugin_inactive_by_default(self):
        fd.register_expert(self._net_expert())
        assert active_experts() == ()
        rec = registered_plugins("expert")[0]
        assert rec["active"] is False
        assert "network" in rec["inactive_reason"]

    def test_network_plugin_active_when_allowed(self):
        fd.register_expert(self._net_expert(), allow_network=True)
        assert len(active_experts()) == 1

    def test_network_plugin_active_via_env(self, monkeypatch):
        monkeypatch.setenv("FRESHDATA_ALLOW_NETWORK_PLUGINS", "1")
        fd.register_expert(self._net_expert())
        assert len(active_experts()) == 1


class TestDependencyGating:
    def test_missing_dependency_marks_inactive(self):
        class NeedsThing:
            name = "needsthing"
            issue_type = "x"
            semantic_types = ()
            max_risk = "low"
            uses_network = False
            requires = ("a_package_that_does_not_exist_xyz",)

            def applies(self, info):
                return True

            def propose(self, series, info):
                return []

        fd.register_expert(NeedsThing())
        assert active_experts() == ()
        rec = registered_plugins("expert")[0]
        assert rec["active"] is False
        assert "missing dependency" in rec["inactive_reason"]


class TestEntryPointDiscovery:
    """Entry-point discovery, exercised by faking importlib.metadata.entry_points."""

    def _install_fake_entry_points(self, monkeypatch, factories):
        import importlib.metadata as md

        class _EP:
            def __init__(self, name, factory):
                self.name = name
                self._factory = factory

            def load(self):
                return self._factory

        def fake_entry_points(*, group):
            return [_EP(n, f) for n, f in factories.get(group, [])]

        monkeypatch.setattr(md, "entry_points", fake_entry_points)

    def test_entry_point_expert_is_discovered(self, monkeypatch):
        self._install_fake_entry_points(
            monkeypatch, {"freshdata.experts": [("titlecase_ep", TitleCaseExpert)]})
        names = [p["name"] for p in registered_plugins("expert")]
        assert "titlecase" in names
        assert registered_plugins("expert")[0]["source"] == "entry_point"

    def test_entry_point_backend_is_discovered(self, monkeypatch):
        self._install_fake_entry_points(
            monkeypatch, {"freshdata.backends": [("prefix_ep", PrefixBackend)]})
        assert "prefixbackend" in known_backend_names()

    def test_broken_entry_point_degrades_safely(self, monkeypatch):
        def _explode():
            raise RuntimeError("bad plugin __init__")

        self._install_fake_entry_points(
            monkeypatch, {"freshdata.experts": [("broken", _explode)]})
        # Discovery must not raise; the broken plugin is simply absent.
        assert active_experts() == ()

    def test_network_entry_point_plugin_inactive_by_default(self, monkeypatch):
        class NetEP:
            name = "netep"
            issue_type = "x"
            semantic_types = ()
            max_risk = "low"
            uses_network = True
            requires = ()

            def applies(self, info):
                return False

            def propose(self, series, info):
                return []

        self._install_fake_entry_points(
            monkeypatch, {"freshdata.experts": [("netep", NetEP)]})
        assert active_experts() == ()


class TestBackwardCompat:
    def test_no_plugins_registered_leaves_clean_unchanged(self):
        df = pd.DataFrame({"city": ["new york", "boston"]})
        baseline = fd.clean(df, semantic_mode="auto")
        clear_plugins()
        again = fd.clean(df, semantic_mode="auto")
        pd.testing.assert_frame_equal(baseline, again)

    def test_import_freshdata_does_not_import_testing(self):
        # fd.testing is lazy; a plain import must not pull it in.
        import importlib

        for mod in ("freshdata", "freshdata.testing"):
            sys.modules.pop(mod, None)
        importlib.import_module("freshdata")
        assert "freshdata.testing" not in sys.modules


class TestShippedExamples:
    """The docs point at examples/plugins/*; keep them runnable + contract-valid."""

    @staticmethod
    def _load(rel_dir: str, module: str, attr: str):
        example_dir = Path(__file__).resolve().parents[1] / "examples" / "plugins" / rel_dir
        if str(example_dir) not in sys.path:
            sys.path.insert(0, str(example_dir))
        import importlib

        return getattr(importlib.import_module(module), attr)

    def test_example_expert_passes_contract(self):
        AcronymExpert = self._load("custom_expert", "acronym_expert", "AcronymExpert")
        fd.testing.expert_contract(AcronymExpert())

    def test_example_backend_passes_contract(self):
        KeywordBackend = self._load("custom_backend", "keyword_backend", "KeywordBackend")
        fd.testing.semantic_backend_contract(KeywordBackend())

    def test_example_validator_passes_contract(self):
        MinRowsValidator = self._load("custom_validator", "min_rows_validator", "MinRowsValidator")
        fd.testing.validator_contract(MinRowsValidator())
