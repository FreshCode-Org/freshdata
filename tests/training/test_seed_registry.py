"""Seed registry validation and synthetic-data generator tests."""

from __future__ import annotations

import copy

import pandas as pd
from training.datasets.validators import check_licenses, validate_source
from training.seed.synthetic import (
    make_context_sentences,
    make_customers,
    make_transactions,
    seed_tables,
)

VALID_ENTRY = {
    "source_id": "ok_source",
    "name": "OK source",
    "url": "https://example.invalid/data",
    "license": "CC0-1.0",
    "license_text_path": "licenses/cc0-1.0.txt",
    "attribution": "Nobody in particular",
    "allowed_for_training": True,
    "pii_risk": "none",
}


def _entry(**overrides):
    entry = copy.deepcopy(VALID_ENTRY)
    entry.update(overrides)
    return entry


class TestRegistryValidation:
    def test_committed_registry_passes(self):
        assert check_licenses() == []

    def test_valid_license_passes(self, tmp_path):
        (tmp_path / "licenses").mkdir()
        (tmp_path / "licenses" / "cc0-1.0.txt").write_text("x", encoding="utf-8")
        assert validate_source(_entry(), tmp_path) == []

    def test_missing_license_fails(self, tmp_path):
        entry = _entry()
        del entry["license"]
        problems = validate_source(entry, tmp_path)
        assert any("license" in p for p in problems)

    def test_missing_attribution_fails(self, tmp_path):
        entry = _entry(attribution="")
        problems = validate_source(entry, tmp_path)
        assert any("attribution" in p for p in problems)

    def test_unclear_training_permission_fails(self, tmp_path):
        entry = _entry(allowed_for_training=False)
        problems = validate_source(entry, tmp_path)
        assert any("allowed_for_training" in p for p in problems)

    def test_non_boolean_training_permission_fails(self, tmp_path):
        entry = _entry(allowed_for_training="yes")
        problems = validate_source(entry, tmp_path)
        assert any("explicit boolean" in p for p in problems)

    def test_unresolved_pii_risk_fails(self, tmp_path):
        (tmp_path / "licenses").mkdir()
        (tmp_path / "licenses" / "cc0-1.0.txt").write_text("x", encoding="utf-8")
        entry = _entry(pii_risk="review_required")
        problems = validate_source(entry, tmp_path)
        assert any("unresolved PII risk" in p for p in problems)

    def test_reviewed_pii_risk_passes(self, tmp_path):
        (tmp_path / "licenses").mkdir()
        (tmp_path / "licenses" / "cc0-1.0.txt").write_text("x", encoding="utf-8")
        entry = _entry(pii_risk="review_required",
                       legal_review={"approved": True, "reviewer": "counsel"})
        assert validate_source(entry, tmp_path) == []

    def test_share_alike_without_review_fails(self, tmp_path):
        entry = _entry(license="CC-BY-SA-4.0")
        problems = validate_source(entry, tmp_path)
        assert any("legal_review" in p for p in problems)

    def test_share_alike_with_review_passes(self, tmp_path):
        (tmp_path / "licenses").mkdir()
        (tmp_path / "licenses" / "cc0-1.0.txt").write_text("x", encoding="utf-8")
        entry = _entry(license="CC-BY-SA-4.0",
                       legal_review={"approved": True, "reviewer": "counsel"})
        assert validate_source(entry, tmp_path) == []

    def test_synthetic_source_passes(self, tmp_path):
        (tmp_path / "licenses").mkdir()
        (tmp_path / "licenses" / "cc0-1.0.txt").write_text("x", encoding="utf-8")
        entry = _entry(pii_risk="synthetic")
        assert validate_source(entry, tmp_path) == []

    def test_unclear_commercial_license_fails(self, tmp_path):
        entry = _entry(license="Some-Weird-License")
        problems = validate_source(entry, tmp_path)
        assert any("unclear commercial-use status" in p for p in problems)


class TestSyntheticData:
    def test_customers_deterministic(self):
        a = make_customers(30, seed=5)
        b = make_customers(30, seed=5)
        assert a.equals(b)

    def test_transactions_deterministic(self):
        a = make_transactions(30, seed=5)
        b = make_transactions(30, seed=5)
        assert a.equals(b)

    def test_different_seeds_differ(self):
        a = make_customers(30, seed=1)
        b = make_customers(30, seed=2)
        assert not a.equals(b)

    def test_customers_schema(self):
        df = make_customers(10)
        for column in ("cust_id", "full_name", "email", "phone", "address",
                       "city", "state", "country", "postal_code", "status"):
            assert column in df.columns

    def test_customers_marked_synthetic(self):
        df = make_customers(10)
        assert df.attrs.get("synthetic") is True
        assert (df["synthetic"] == True).all()  # noqa: E712

    def test_transactions_marked_synthetic(self):
        df = make_transactions(10)
        assert df.attrs.get("synthetic") is True

    def test_no_real_domains_in_emails(self):
        df = make_customers(50)
        assert not df["email"].str.contains("gmail.com|yahoo.com|hotmail.com").any()

    def test_context_sentences_have_required_fields(self):
        sentences = make_context_sentences()
        assert sentences
        for row in sentences:
            assert {"sentence", "intent", "slots", "author", "synthetic"} <= set(row)
            assert row["synthetic"] is True

    def test_seed_tables_returns_all_tables(self):
        tables = seed_tables(seed=0)
        assert set(tables) == {"customers", "transactions"}
        assert all(isinstance(v, pd.DataFrame) for v in tables.values())
