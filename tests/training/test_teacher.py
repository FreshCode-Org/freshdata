"""Teacher harness: schemas, cache stability, PII masking, compliance, review."""

from __future__ import annotations

import json

import pytest
from training.teacher import compliance
from training.teacher.cache import CacheKey, TeacherCache
from training.teacher.clients import StubTeacherClient
from training.teacher.review import DISAGREEMENT_THRESHOLD, SAMPLE_RATE, ReviewBatch
from training.teacher.schemas import SchemaError, validate_batch, validate_payload
from training.teacher.tasks import TASKS, build_prompt, mask_pii, run_task


class TestSchemas:
    def test_valid_column_role_label_passes(self):
        payload = {
            "column_name": "email", "masked_samples": ["a***@b.com"],
            "semantic_type": "email", "confidence": 0.9,
        }
        assert validate_payload("ColumnRoleLabel", payload)["confidence"] == 0.9

    def test_missing_required_field_fails(self):
        with pytest.raises(SchemaError, match="missing required field"):
            validate_payload("ColumnRoleLabel", {"column_name": "email"})

    def test_unknown_field_fails(self):
        payload = {
            "column_name": "email", "masked_samples": [], "semantic_type": "email",
            "confidence": 0.9, "extra_field": 1,
        }
        with pytest.raises(SchemaError, match="unknown keys"):
            validate_payload("ColumnRoleLabel", payload)

    def test_wrong_type_fails(self):
        payload = {
            "column_name": "email", "masked_samples": "not-a-list",
            "semantic_type": "email", "confidence": 0.9,
        }
        with pytest.raises(SchemaError):
            validate_payload("ColumnRoleLabel", payload)

    def test_confidence_out_of_range_fails(self):
        payload = {
            "column_name": "email", "masked_samples": [], "semantic_type": "email",
            "confidence": 1.5,
        }
        with pytest.raises(SchemaError, match="confidence"):
            validate_payload("ColumnRoleLabel", payload)

    def test_ambiguity_judgment_verdict_enum(self):
        payload = {"raw_value": "x", "candidates": ["a", "b"], "verdict": "not_a_verdict",
                   "rationale": "r"}
        with pytest.raises(SchemaError, match="verdict"):
            validate_payload("AmbiguityJudgment", payload)

    def test_unknown_schema_name_fails(self):
        with pytest.raises(SchemaError, match="unknown teacher schema"):
            validate_payload("NotASchema", {})

    def test_validate_batch(self):
        payloads = [{"column_name": "x", "masked_samples": [], "semantic_type": "email",
                    "confidence": 0.5}]
        assert validate_batch("ColumnRoleLabel", payloads) == payloads


class TestCacheKeyStability:
    def test_same_inputs_same_digest(self):
        key_a = CacheKey("stub", "m1", "t1", "i1", "s1")
        key_b = CacheKey("stub", "m1", "t1", "i1", "s1")
        assert key_a.digest() == key_b.digest()

    def test_different_input_changes_digest(self):
        key_a = CacheKey("stub", "m1", "t1", "i1", "s1")
        key_b = CacheKey("stub", "m1", "t1", "i2", "s1")
        assert key_a.digest() != key_b.digest()

    def test_put_then_get_roundtrips(self, tmp_path):
        cache = TeacherCache(tmp_path)
        key = CacheKey("stub", "m1", "t1", "i1", "s1")
        cache.put(key, prompt="p", response="r", payloads=[{"a": 1}],
                 terms_snapshot_id="snap-1")
        entry = cache.get(key)
        assert entry is not None
        assert entry["payloads"] == [{"a": 1}]
        assert entry["prompt"] == "p"
        assert entry["response"] == "r"
        assert entry["terms_snapshot_id"] == "snap-1"
        assert "created_at" in entry

    def test_miss_returns_none(self, tmp_path):
        cache = TeacherCache(tmp_path)
        assert cache.get(CacheKey("stub", "m", "t", "i", "s")) is None


class TestPIIMasking:
    def test_email_local_masked(self):
        assert mask_pii("asha.voskette@example.com").startswith("a***@")

    def test_digits_masked(self):
        masked = mask_pii("+919876543210")
        assert "9876543210" not in masked
        assert set(masked) <= set("+9 ")

    def test_long_alnum_token_hashed(self):
        masked = mask_pii("SUPERSECRETTOKEN1234567890abc")
        assert "SUPERSECRETTOKEN" not in masked
        assert masked.startswith("tok_")

    def test_masking_is_deterministic(self):
        assert mask_pii("token1234567890ABCDEFxyz") == mask_pii("token1234567890ABCDEFxyz")


class TestNoFullRows:
    def test_task_item_with_too_many_fields_rejected(self):
        client = StubTeacherClient({"ColumnRoleLabel": []})
        big_item = {f"f{i}": i for i in range(10)}
        with pytest.raises(ValueError, match="full row"):
            run_task("column_role_labeling", [big_item], client=client,
                    cache=TeacherCache.__new__(TeacherCache))

    def test_too_many_masked_samples_rejected(self, tmp_path):
        client = StubTeacherClient({"ColumnRoleLabel": []})
        cache = TeacherCache(tmp_path)
        item = {"column_name": "x", "masked_samples": list(range(11))}
        with pytest.raises(ValueError, match="10 masked sample"):
            run_task("column_role_labeling", [item], client=client, cache=cache)


class TestDegradeSafely:
    def test_failed_client_returns_empty_batch(self, tmp_path):
        cache = TeacherCache(tmp_path)
        client = StubTeacherClient(fail=True)
        out = run_task("column_role_labeling", [{"column_name": "x", "masked_samples": []}],
                       client=client, cache=cache)
        assert out == []

    def test_invalid_json_response_degrades(self, tmp_path):
        cache = TeacherCache(tmp_path)

        class BadClient(StubTeacherClient):
            def complete(self, prompt, *, schema):
                return "not json"

        out = run_task("column_role_labeling", [{"column_name": "x", "masked_samples": []}],
                       client=BadClient(), cache=cache)
        assert out == []

    def test_cache_hit_avoids_second_call(self, tmp_path):
        cache = TeacherCache(tmp_path)
        canned = {"ColumnRoleLabel": [{"column_name": "x", "masked_samples": [],
                                       "semantic_type": "email", "confidence": 0.9}]}
        client = StubTeacherClient(canned)
        item = {"column_name": "x", "masked_samples": []}
        first = run_task("column_role_labeling", [item], client=client, cache=cache)
        assert len(client.calls) == 1
        second = run_task("column_role_labeling", [item], client=client, cache=cache)
        assert len(client.calls) == 1  # no second network call
        assert first == second


class TestPromptAudit:
    def test_prompt_contains_items_and_cache_stores_audit(self, tmp_path):
        cache = TeacherCache(tmp_path)
        canned = {"ColumnRoleLabel": [{"column_name": "x", "masked_samples": [],
                                       "semantic_type": "email", "confidence": 0.9}]}
        client = StubTeacherClient(canned)
        item = {"column_name": "x", "masked_samples": ["a***@b.com"]}
        run_task("column_role_labeling", [item], client=client, cache=cache)
        [entry] = cache.entries()
        assert "a***@b.com" in entry["prompt"]
        assert entry["provider"] == "stub"
        assert entry["model"] == "stub-teacher-0"

    def test_build_prompt_is_deterministic(self):
        spec = TASKS["column_role_labeling"]
        prompt_a, sha_a = build_prompt(spec, [{"column_name": "x", "masked_samples": []}])
        prompt_b, sha_b = build_prompt(spec, [{"column_name": "x", "masked_samples": []}])
        assert prompt_a == prompt_b
        assert sha_a == sha_b


class TestCompliance:
    def test_stub_provider_is_approved(self):
        entry = compliance.require_approved("stub", "column_role_labeling")
        assert entry["status"] == "approved"

    def test_unknown_provider_blocked(self):
        with pytest.raises(compliance.ComplianceError, match="no terms snapshot"):
            compliance.require_approved("totally-unknown-provider", "column_role_labeling")

    def test_disallowed_use_blocked(self, tmp_path):
        ledger_path = tmp_path / "ledger.json"
        compliance.record_check(
            provider="acme", terms_url="https://acme.example/terms",
            terms_snapshot_id="snap1", reviewer="JWD",
            allowed_uses=["red_teaming"], training_use_allowed=True,
            redistribution_allowed=False, ledger_path=ledger_path,
        )
        with pytest.raises(compliance.ComplianceError, match="not approved"):
            compliance.require_approved("acme", "column_role_labeling", ledger_path=ledger_path)

    def test_missing_reviewer_blocked(self, tmp_path):
        ledger_path = tmp_path / "ledger.json"
        ledger_path.write_text(json.dumps({"ledger_version": "1", "providers": [{
            "provider": "acme2", "terms_url": "https://x", "terms_snapshot_id": "s1",
            "date_checked": "2026-01-01T00:00:00+00:00", "reviewer": "",
            "allowed_uses": ["column_role_labeling"], "training_use_allowed": True,
            "redistribution_allowed": True, "status": "approved",
        }]}), encoding="utf-8")
        with pytest.raises(compliance.ComplianceError, match="reviewer is missing"):
            compliance.require_approved("acme2", "column_role_labeling", ledger_path=ledger_path)

    def test_training_use_disallowed_blocked(self, tmp_path):
        ledger_path = tmp_path / "ledger.json"
        compliance.record_check(
            provider="acme3", terms_url="https://x", terms_snapshot_id="s1",
            reviewer="JWD", allowed_uses=["column_role_labeling"],
            training_use_allowed=False, redistribution_allowed=True,
            ledger_path=ledger_path,
        )
        with pytest.raises(compliance.ComplianceError, match="training use"):
            compliance.require_approved("acme3", "column_role_labeling", ledger_path=ledger_path)

    def test_stale_snapshot_blocked(self, tmp_path):
        ledger_path = tmp_path / "ledger.json"
        ledger_path.write_text(json.dumps({"ledger_version": "1", "providers": [{
            "provider": "acme4", "terms_url": "https://x", "terms_snapshot_id": "s1",
            "date_checked": "2020-01-01T00:00:00+00:00", "reviewer": "JWD",
            "allowed_uses": ["column_role_labeling"], "training_use_allowed": True,
            "redistribution_allowed": True, "status": "approved",
        }]}), encoding="utf-8")
        with pytest.raises(compliance.ComplianceError, match="stale"):
            compliance.require_approved("acme4", "column_role_labeling", ledger_path=ledger_path)

    def test_run_task_blocked_for_unapproved_provider(self, tmp_path):
        cache = TeacherCache(tmp_path)

        class UnknownProviderClient(StubTeacherClient):
            provider = "nope"

        with pytest.raises(compliance.ComplianceError):
            run_task("column_role_labeling", [{"column_name": "x", "masked_samples": []}],
                     client=UnknownProviderClient({}), cache=cache)


class TestHumanReview:
    def test_sample_rate_is_at_least_five_percent(self):
        batch = ReviewBatch("t", [{"i": i} for i in range(100)])
        ids = batch.sample_for_review(seed=1)
        assert len(ids) >= int(100 * SAMPLE_RATE)

    def test_disagreement_rate_computed(self):
        batch = ReviewBatch("t", [{"i": i} for i in range(20)])
        ids = batch.sample_for_review(seed=1, rate=0.5)
        for i, item_id in enumerate(ids):
            batch.record_review(item_id, reviewer="JWD", agrees=(i != 0),
                                disagreement_reason="wrong" if i == 0 else "")
        assert 0 < batch.disagreement_rate() < 1

    def test_disagreement_above_threshold_requires_full_review(self):
        batch = ReviewBatch("t", [{"i": i} for i in range(20)])
        ids = batch.sample_for_review(seed=1, rate=0.5)
        for item_id in ids:
            batch.record_review(item_id, reviewer="JWD", agrees=False, disagreement_reason="bad")
        assert batch.disagreement_rate() > DISAGREEMENT_THRESHOLD
        assert batch.requires_full_review() is True

    def test_low_disagreement_does_not_require_full_review(self):
        batch = ReviewBatch("t", [{"i": i} for i in range(20)])
        ids = batch.sample_for_review(seed=1, rate=0.5)
        for item_id in ids:
            batch.record_review(item_id, reviewer="JWD", agrees=True)
        assert batch.requires_full_review() is False

    def test_release_gating_requires_full_coverage(self):
        batch = ReviewBatch("t", [{"i": i} for i in range(5)])
        for i in range(4):
            batch.record_review(batch.item_id(i), reviewer="JWD", agrees=True)
        assert batch.usable_for_release_gating() is False
        batch.record_review(batch.item_id(4), reviewer="JWD", agrees=True)
        assert batch.usable_for_release_gating() is True

    def test_disagreement_without_reason_rejected(self):
        batch = ReviewBatch("t", [{"i": 0}])
        with pytest.raises(ValueError, match="reason"):
            batch.record_review("item_00000", reviewer="JWD", agrees=False)

    def test_missing_reviewer_rejected(self):
        batch = ReviewBatch("t", [{"i": 0}])
        with pytest.raises(ValueError, match="reviewer"):
            batch.record_review("item_00000", reviewer="", agrees=True)

    def test_export_summary_json(self, tmp_path):
        batch = ReviewBatch("t", [{"i": 0}])
        batch.record_review("item_00000", reviewer="JWD", agrees=True)
        path = batch.export_summary(tmp_path / "summary.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["n_reviewed"] == 1
        assert data["reviewers"] == ["JWD"]
