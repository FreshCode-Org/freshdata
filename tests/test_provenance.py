"""Tests for provenance-aware cleaning (F5, ``source_provenance=``)."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.provenance import normalize_provenance


def _doc_frame() -> pd.DataFrame:
    # 'amount' is OCR'd text with currency junk + a sentinel -> will be coerced.
    return pd.DataFrame(
        {
            "amount": ["$1,200", "$3,400", "N/A"],
            "vendor": ["Acme", "Globex", "Initech"],  # clean, untouched
        }
    )


def _prov(amount_conf: float = 0.55) -> dict:
    return {
        "amount": {"parser_confidence": amount_conf, "source_file": "invoice.pdf",
                   "page": 3, "region": "tbl-1"},
        "vendor": {"parser_confidence": 0.98, "source_file": "invoice.pdf", "page": 1},
    }


def test_low_confidence_coercion_warns():
    cleaned, rep = fd.clean(_doc_frame(), source_provenance=_prov(), return_report=True)
    assert rep.source_provenance is not None
    assert rep.source_provenance["amount"]["low_confidence_repair"] is True
    assert rep.source_provenance["vendor"]["low_confidence_repair"] is False
    assert any("amount" in w and "low-confidence" in w for w in rep.warnings)
    assert any("amount" in r for r in rep.recommendations)


def test_high_confidence_coercion_not_flagged():
    # amount confidence above threshold -> even though coerced, no warning
    cleaned, rep = fd.clean(_doc_frame(), source_provenance=_prov(amount_conf=0.95),
                            return_report=True)
    assert rep.source_provenance["amount"]["modified"] is True
    assert rep.source_provenance["amount"]["low_confidence_repair"] is False
    assert not any("low-confidence" in w for w in rep.warnings)


def test_threshold_is_configurable():
    cleaned, rep = fd.clean(_doc_frame(), source_provenance=_prov(amount_conf=0.8),
                            provenance_confidence_threshold=0.9, return_report=True)
    assert rep.source_provenance["amount"]["low_confidence_repair"] is True


def test_provenance_carried_into_to_dict():
    _, rep = fd.clean(_doc_frame(), source_provenance=_prov(), return_report=True)
    assert "source_provenance" in rep.to_dict()
    assert rep.to_dict()["source_provenance"]["amount"]["page"] == 3


def test_requires_return_report():
    with pytest.raises(ValueError, match="return_report=True"):
        fd.clean(_doc_frame(), source_provenance=_prov())


def test_normalize_validates_confidence_range():
    with pytest.raises(ValueError, match="parser_confidence"):
        normalize_provenance({"a": {"parser_confidence": 1.5}}, ["a"])


def test_normalize_requires_dict_metadata():
    with pytest.raises(TypeError):
        normalize_provenance({"a": 0.5}, ["a"])


def test_clean_enterprise_threads_provenance():
    res = fd.clean_enterprise(_doc_frame(), source_provenance=_prov())
    assert res.clean_report.source_provenance is not None
    assert res.clean_report.source_provenance["amount"]["low_confidence_repair"] is True


def test_does_not_break_without_provenance():
    cleaned, rep = fd.clean(_doc_frame(), return_report=True)
    assert rep.source_provenance is None
