"""Replay wiring: ``profile=`` on clean/clean_csv/Cleaner/suggest_plan.

Covers the drift gate, option folding precedence, the profile backend's
path through the policy gate, protected-column safety, and the
``profile=None`` backward-compatibility guarantee.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.learning import LearningProfile, learn, save_profile
from freshdata.learning.replay import (
    ProfileReplayGate,
    annotate_profile_report,
    check_profile_drift,
    fold_profile_options,
    resolve_profile,
)
from freshdata.learning.types import ValueMap, ValueMapEntry
from freshdata.memory import learn_cleaning_memory
from freshdata.models import runtime as model_runtime
from freshdata.models.stub import StubEncoder
from freshdata.semantic.backends.base import Budget
from freshdata.semantic.backends.profile import ProfileBackend
from freshdata.semantic.context import build_semantic_context


def _profile_actions(report):
    return [a for a in report.actions if a.metadata and a.metadata.get("profile_influenced")]


class TestCleanWithProfile:
    def test_value_map_replays_on_new_batch(self, orders_profile, new_batch):
        cleaned, report = fd.clean(
            new_batch, profile=orders_profile, semantic_mode="auto", return_report=True
        )
        assert list(cleaned["status"]) == ["delivered", "shipped", "shipped"]
        assert report.profile_replay is not None
        assert report.profile_replay["ok"] is True
        assert report.profile_replay["profile_id"] == orders_profile.profile_id
        assert _profile_actions(report)

    def test_profile_path_accepted(self, orders_profile, new_batch, tmp_path):
        path = tmp_path / "orders.fdprofile"
        save_profile(orders_profile, path)
        cleaned, report = fd.clean(
            new_batch, profile=str(path), semantic_mode="auto", return_report=True
        )
        assert list(cleaned["status"]) == ["delivered", "shipped", "shipped"]
        assert report.profile_replay["profile_id"] == orders_profile.profile_id

    def test_learned_dayfirst_folds_into_config(self, orders_profile, new_batch):
        cleaned = fd.clean(new_batch, profile=orders_profile, semantic_mode="auto")
        parsed = pd.to_datetime(cleaned["order_date"])
        assert parsed.iloc[0] == pd.Timestamp("2024-02-05")

    def test_action_metadata_carries_provenance(self, orders_profile, new_batch):
        _, report = fd.clean(
            new_batch, profile=orders_profile, semantic_mode="auto", return_report=True
        )
        action = _profile_actions(report)[0]
        assert action.metadata["profile_id"] == orders_profile.profile_id
        assert action.metadata["support"] >= 2
        assert action.metadata["learned_precision"] > 0
        assert action.metadata["transform_family"]
        assert action.model_id.endswith(":profile")
        assert action.memory_influenced is False

    def test_no_duplicate_warnings(self, orders_profile, new_batch):
        _, report = fd.clean(
            new_batch, profile=orders_profile, semantic_mode="auto", return_report=True
        )
        assert len(report.warnings) == len(set(report.warnings))

    def test_protected_column_never_modified(self, orders_profile, new_batch):
        cleaned = fd.clean(new_batch, profile=orders_profile, semantic_mode="auto")
        assert list(cleaned["order_id"]) == ["B1", "B2", "B3"]

    def test_policy_protection_beats_profile(self, orders_profile, new_batch):
        cleaned = fd.clean(
            new_batch,
            profile=orders_profile,
            semantic_mode="auto",
            context="Never modify status.",
        )
        # Byte-identity protection: even whitespace stripping is blocked.
        assert list(cleaned["status"]) == ["Deliverd", "SHIPPED", "shipped "]

    def test_user_option_beats_learned_config_delta(self, orders_profile, new_batch):
        cleaned = fd.clean(
            new_batch,
            profile=orders_profile,
            semantic_mode="auto",
            dayfirst=False,
        )
        parsed = pd.to_datetime(cleaned["order_date"])
        # 05/02/2024 without dayfirst is May 2, not Feb 5.
        assert parsed.iloc[0] == pd.Timestamp("2024-05-02")

    def test_invalid_profile_arg_raises(self, new_batch):
        with pytest.raises(TypeError, match="profile="):
            fd.clean(new_batch, profile=42)


class TestBackwardCompat:
    def test_profile_none_is_identity(self, new_batch):
        with_none, rep_none = fd.clean(
            new_batch, profile=None, semantic_mode="auto", return_report=True
        )
        without, rep_plain = fd.clean(new_batch, semantic_mode="auto", return_report=True)
        pd.testing.assert_frame_equal(with_none, without)
        assert rep_none.profile_replay is None
        assert rep_plain.profile_replay is None

    def test_report_to_dict_omits_profile_replay_when_absent(self, new_batch):
        _, report = fd.clean(new_batch, return_report=True)
        assert "profile_replay" not in report.to_dict()


class TestDriftGate:
    def test_severe_drift_blocks_replay(self, orders_profile):
        drifted = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        _, report = fd.clean(drifted, profile=orders_profile, return_report=True)
        assert report.profile_replay["ok"] is False
        assert report.profile_replay["severity"] == "severe"
        assert any("not replayed" in w for w in report.warnings)

    def test_mild_drift_partial_replay(self, orders_profile, new_batch):
        partial = new_batch.drop(columns=["amount", "phone"])
        cleaned, report = fd.clean(
            partial, profile=orders_profile, semantic_mode="auto", return_report=True
        )
        assert report.profile_replay["ok"] is True
        assert report.profile_replay["severity"] == "mild"
        assert list(cleaned["status"]) == ["delivered", "shipped", "shipped"]
        assert any("partially replayed" in w for w in report.warnings)

    def test_check_profile_drift_direct(self, orders_profile, new_batch):
        gate = check_profile_drift(new_batch, orders_profile)
        assert gate.ok and gate.severity == "none"
        assert set(gate.compatible_columns) >= {"status", "order_date"}

    def test_dtype_flip_marks_column_incompatible(self, orders_profile, new_batch):
        flipped = new_batch.copy()
        flipped["status"] = [1, 2, 3]
        gate = check_profile_drift(flipped, orders_profile)
        assert gate.ok
        assert gate.severity == "mild"
        assert "status" not in gate.compatible_columns


class TestFolding:
    def test_fold_respects_existing_options(self, orders_profile):
        gate = ProfileReplayGate(
            ok=True,
            severity="none",
            overlap=1.0,
            compatible_columns=("status", "order_date", "email", "phone"),
        )
        options = fold_profile_options(orders_profile, {"dayfirst": False}, gate)
        assert options["dayfirst"] is False

    def test_fold_blocked_gate_is_noop(self, orders_profile):
        gate = ProfileReplayGate(ok=False, severity="severe", overlap=0.0)
        options = fold_profile_options(orders_profile, {"a": 1}, gate)
        assert options == {"a": 1}

    def test_resolve_profile_type_error(self):
        with pytest.raises(TypeError):
            resolve_profile(3.14)

    def test_resolve_profile_identity(self, orders_profile):
        assert resolve_profile(orders_profile) is orders_profile


class TestAnnotate:
    def test_idempotent_per_profile(self, orders_profile):
        _, report = fd.clean(pd.DataFrame({"status": ["x"]}), return_report=True)
        gate = ProfileReplayGate(ok=True, severity="none", overlap=1.0)
        annotate_profile_report(report, orders_profile, gate)
        n_warnings = len(report.warnings)
        annotate_profile_report(report, orders_profile, gate)
        assert len(report.warnings) == n_warnings
        assert report.profile_replay["profile_id"] == orders_profile.profile_id


class TestCleanerAndPlan:
    def test_cleaner_constructor_profile(self, orders_profile, new_batch):
        cleaner = fd.Cleaner(semantic_mode="auto", profile=orders_profile)
        cleaned, report = cleaner.clean(new_batch, report=True)
        assert list(cleaned["status"]) == ["delivered", "shipped", "shipped"]
        assert report.profile_replay["ok"] is True

    def test_cleaner_call_profile_overrides(self, orders_profile, new_batch):
        cleaner = fd.Cleaner(semantic_mode="auto")
        cleaned, report = cleaner.clean(new_batch, report=True, profile=orders_profile)
        assert list(cleaned["status"]) == ["delivered", "shipped", "shipped"]
        assert report.profile_replay["ok"] is True

    def test_cleaner_without_profile_unchanged(self, new_batch):
        cleaner = fd.Cleaner(semantic_mode="auto")
        _, report = cleaner.clean(new_batch, report=True)
        assert report.profile_replay is None

    def test_suggest_plan_folds_profile(self, orders_profile, new_batch):
        plan = fd.suggest_plan(new_batch, profile=orders_profile, semantic_mode="auto")
        assert plan.config.dayfirst is True
        columns = (plan.config.semantic_context or {}).get("columns", {})
        assert columns.get("email", {}).get("semantic_type") == "email"
        assert columns.get("phone", {}).get("semantic_type") == "phone"

    def test_suggest_plan_user_option_wins(self, orders_profile, new_batch):
        plan = fd.suggest_plan(
            new_batch, profile=orders_profile, semantic_mode="auto", dayfirst=False
        )
        assert plan.config.dayfirst is False


class TestEmbeddedMemory:
    def test_profile_memory_used_when_no_memory_kwarg(self, orders_profile):
        assert orders_profile.memory is not None

    def test_user_memory_wins(self, orders_profile, new_batch):
        other = learn_cleaning_memory(new_batch, [], "other-dataset")
        _, report = fd.clean(
            new_batch,
            profile=orders_profile,
            memory=other,
            semantic_mode="auto",
            return_report=True,
        )
        # Replay still works via the profile backend even when the user's
        # own memory (which knows nothing) is supplied.
        assert report.profile_replay["ok"] is True


class TestProfileBackendUnit:
    def test_skips_masked_entries(self, orders_profile, new_batch):
        # Email literals are masked under privacy="mask": the profile backend
        # must never emit literal email repairs.
        _, report = fd.clean(
            new_batch, profile=orders_profile, semantic_mode="auto", return_report=True
        )
        email_profile_actions = [a for a in _profile_actions(report) if a.column == "email"]
        assert email_profile_actions == []

    def test_backend_requires_learning_profile(self):
        backend = ProfileBackend.__new__(ProfileBackend)
        assert backend is not None
        assert isinstance(LearningProfile, type)


def _retrieval_pair(n_boilerplate: int = 19) -> tuple[pd.DataFrame, pd.DataFrame]:
    # High clean-side cardinality (> classify.py's allowed-vocabulary
    # threshold of 20) keeps these free-text edits out of category_map /
    # allowed_value_map / reference_normalize, so they land as
    # "unexplained" examples — exactly what retrieval is meant to serve.
    boilerplate = [f"log entry number {i:02d} filed" for i in range(n_boilerplate)]
    messy_notes = [
        *boilerplate,
        "customer requested refund todays",
        "shipment delayed due to weathr",
    ]
    clean_notes = [
        *boilerplate,
        "customer requested refund today",
        "shipment delayed due to weather",
    ]
    ids = list(range(len(messy_notes)))
    messy = pd.DataFrame({"id": ids, "notes": messy_notes})
    clean = pd.DataFrame({"id": ids, "notes": clean_notes})
    return messy, clean


class TestExampleRetrieval:
    def setup_method(self) -> None:
        model_runtime.set_encoder_factory(lambda model_id: StubEncoder())

    def teardown_method(self) -> None:
        model_runtime.set_encoder_factory(None)

    def _profile(self) -> LearningProfile:
        messy, clean = _retrieval_pair()
        # "notes" reads as a free_text semantic type, which learn() treats as
        # sensitive by default (masked, unvectorized); disable that here since
        # this class is exercising retrieval mechanics, not privacy masking.
        return learn(messy, clean, key="id", min_support=1, privacy="none", include_sensitive=True)

    def test_retrieves_similar_unexplained_examples(self) -> None:
        profile = self._profile()
        bank = profile.examples
        assert bank is not None and bank.vectors_path
        assert profile.vectors is not None
        assert len(bank.examples) == len(profile.vectors)

        new_batch = pd.DataFrame(
            {
                "id": [100, 101],
                "notes": [
                    "customer requested refund todayss",
                    "shipment delayed due to wethr",
                ],
            }
        )
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        proposals = backend._retrieval_proposals(new_batch, ctx, Budget())
        by_raw = {p.raw_value: p for p in proposals}
        assert (
            by_raw["customer requested refund todayss"].proposed_value
            == "customer requested refund today"
        )
        assert (
            by_raw["shipment delayed due to wethr"].proposed_value
            == "shipment delayed due to weather"
        )
        proposal = by_raw["customer requested refund todayss"]
        assert proposal.expert == "profile:example_retrieval"
        assert proposal.backend == "profile"
        assert proposal.provenance["retrieval_similarity"] >= 0.9
        assert proposal.provenance["profile_influenced"] is True
        assert proposal.provenance["learned_precision"] is None

    def test_no_match_below_similarity_threshold(self) -> None:
        profile = self._profile()
        new_batch = pd.DataFrame({"id": [200], "notes": ["totally unrelated sentence here"]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_skips_exact_raw_value_match(self) -> None:
        profile = self._profile()
        # Identical to a stored training raw value: not a "similar" example.
        new_batch = pd.DataFrame({"id": [201], "notes": ["customer requested refund todays"]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_no_bank_returns_empty(self) -> None:
        profile = self._profile()
        profile.examples = None
        new_batch = pd.DataFrame({"id": [1], "notes": ["x"]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_no_vectors_returns_empty(self) -> None:
        profile = self._profile()
        profile.vectors = None
        new_batch = pd.DataFrame({"id": [1], "notes": ["x"]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_missing_encoder_degrades_to_no_proposals(self) -> None:
        profile = self._profile()
        model_runtime.set_encoder_factory(None)
        new_batch = pd.DataFrame({"id": [100], "notes": ["customer requested refund todayss"]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_protected_column_is_skipped(self) -> None:
        profile = self._profile()
        new_batch = pd.DataFrame({"id": [100], "notes": ["customer requested refund todayss"]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        info = ctx.columns["notes"]
        ctx.columns["notes"] = dataclasses.replace(info, mutable=False)
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_high_cardinality_column_is_skipped(self) -> None:
        profile = self._profile()
        wide_notes = [f"distinct entry number {i:03d}" for i in range(250)]
        new_batch = pd.DataFrame({"id": list(range(250)), "notes": wide_notes})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_column_absent_from_frame_is_skipped(self) -> None:
        profile = self._profile()
        new_batch = pd.DataFrame({"id": [1], "other": ["x"]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_candidate_vector_count_mismatch_returns_empty(self) -> None:
        profile = self._profile()
        profile.vectors = profile.vectors[:-1]
        new_batch = pd.DataFrame({"id": [100], "notes": ["customer requested refund todayss"]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_encoder_factory_returning_none_degrades_to_no_proposals(self) -> None:
        profile = self._profile()
        model_runtime.set_encoder_factory(lambda model_id: None)
        new_batch = pd.DataFrame({"id": [100], "notes": ["customer requested refund todayss"]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_no_string_candidates_in_new_column_values_is_skipped(self) -> None:
        profile = self._profile()
        # The "notes" column exists but every value is missing, so there are
        # no candidate strings to embed and compare.
        new_batch = pd.DataFrame({"id": [100], "notes": [None]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []

    def test_encode_texts_failure_returns_proposals_so_far(self) -> None:
        profile = self._profile()

        class _RaisingEncoder:
            def encode_texts(self, texts: object) -> object:
                raise RuntimeError("boom")

        model_runtime.set_encoder_factory(lambda model_id: _RaisingEncoder())
        new_batch = pd.DataFrame({"id": [100], "notes": ["customer requested refund todayss"]})
        ctx = build_semantic_context(new_batch, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._retrieval_proposals(new_batch, ctx, Budget()) == []


class TestValueMapProposalEdgeCases:
    def test_column_not_in_frame_is_skipped(self) -> None:
        value_map = ValueMap(
            column="missing_col",
            entries=[ValueMapEntry("a", "b", 5, 0.99, "category_map")],
            min_precision=0.9,
            min_support=1,
            capped=False,
            masked=False,
        )
        profile = SimpleNamespace(profile_id="test-profile", value_maps={"missing_col": value_map})
        df = pd.DataFrame({"other": [1, 2]})
        ctx = build_semantic_context(df, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._value_map_proposals(df, ctx, Budget()) == []

    def test_all_masked_entries_leave_lookup_empty(self) -> None:
        value_map = ValueMap(
            column="status",
            entries=[ValueMapEntry("a", "b", 5, 0.99, "category_map", masked=True)],
            min_precision=0.9,
            min_support=1,
            capped=False,
            masked=True,
        )
        profile = SimpleNamespace(profile_id="test-profile", value_maps={"status": value_map})
        df = pd.DataFrame({"status": ["a", "a"]})
        ctx = build_semantic_context(df, CleanConfig())
        backend = ProfileBackend(profile)
        assert backend._value_map_proposals(df, ctx, Budget()) == []
