"""Release-gating label store: 100% human-verification enforcement."""

from __future__ import annotations

import pytest
from training.eval.human_verified import (
    VerificationError,
    check_all_verified,
    load_verified,
    save_verified,
    verify_labels,
)


def test_verify_labels_stamps_reviewer_and_timestamp():
    labels = [{"a": 1}, {"a": 2}]
    stamped = verify_labels(labels, reviewer="JWD")
    assert all(row["human_verified"] is True for row in stamped)
    assert all(row["reviewer"] == "JWD" for row in stamped)
    assert all("reviewed_at" in row for row in stamped)


def test_verify_labels_requires_reviewer():
    with pytest.raises(VerificationError):
        verify_labels([{"a": 1}], reviewer="")


def test_check_all_verified_passes_when_fully_verified():
    labels = verify_labels([{"a": 1}], reviewer="JWD")
    check_all_verified(labels)  # no raise


def test_check_all_verified_fails_on_partial_verification():
    labels = verify_labels([{"a": 1}, {"a": 2}], reviewer="JWD")
    labels[1]["human_verified"] = False
    with pytest.raises(VerificationError, match="not human-verified"):
        check_all_verified(labels)


def test_check_all_verified_fails_on_empty_set():
    with pytest.raises(VerificationError, match="empty"):
        check_all_verified([])


def test_check_all_verified_fails_without_reviewer_field():
    with pytest.raises(VerificationError):
        check_all_verified([{"human_verified": True}])


def test_save_then_load_roundtrip(tmp_path):
    labels = verify_labels([{"a": 1}, {"a": 2}], reviewer="JWD")
    path = tmp_path / "eval.jsonl"
    save_verified(path, labels)
    loaded = load_verified(path)
    assert loaded == labels


def test_save_verified_rejects_unverified(tmp_path):
    with pytest.raises(VerificationError):
        save_verified(tmp_path / "eval.jsonl", [{"a": 1}])


def test_committed_role_eval_is_fully_verified():
    from training.distill.train_role_head import HUMAN_EVAL_PATH

    labels = load_verified(HUMAN_EVAL_PATH)
    assert len(labels) > 0
