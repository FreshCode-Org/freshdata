"""Tests for the centralized reference-data layer (freshdata.domains.reference)."""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.domains.reference import (
    ReferenceSet,
    available_references,
    load_reference,
)


def test_available_includes_bundled_sets():
    names = available_references()
    assert {"iso3166", "iso4217", "uom_codes", "ucum_common", "un_locode_sample"} <= set(names)


def test_load_flat_code_list():
    cur = load_reference("iso4217")
    assert cur.contains("USD")
    assert not cur.contains("ZZZ")
    assert cur.mapping is None
    assert cur.meta.get("version")  # _meta carried through


def test_load_mapping_shaped_set():
    iso = load_reference("iso3166")
    assert iso.contains("US")
    assert iso.mapping is not None
    assert iso.mapping["US"]["alpha-3"] == "USA"
    assert "US" in iso.codes  # mapping keys are the codes


def test_every_bundled_set_has_meta_with_disclaimer():
    for name in available_references():
        ref = load_reference(name)
        assert "disclaimer" in ref.meta or name in {"iso3166", "iso4217", "uom_codes"}, name
        assert ref.meta, name


def test_load_reference_is_cached():
    assert load_reference("iso4217") is load_reference("iso4217")


def test_unknown_reference_raises_keyerror():
    with pytest.raises(KeyError, match="unknown reference set"):
        load_reference("definitely_not_a_set")


def test_exact_normalizer_is_case_sensitive():
    ucum = load_reference("ucum_common", normalizer="exact")
    assert ucum.contains("Cel")
    assert not ucum.contains("cel")  # UCUM is case-sensitive


def test_casefold_normalizer_is_case_insensitive():
    cur = load_reference("iso4217", normalizer="casefold")
    assert cur.contains("usd")
    assert cur.contains("USD")


def test_invalid_mask_skips_missing_and_flags_unknown():
    cur = load_reference("iso4217")
    s = pd.Series(["USD", "EUR", None, "ZZZ"])
    mask = cur.invalid_mask(s)
    assert mask.tolist() == [False, False, False, True]


def test_coerce_maps_synonyms_to_canonical():
    # A synthetic set with a coerce table exercises normalization + synonyms.
    ref = ReferenceSet(
        name="demo",
        codes=frozenset({"good", "bad"}),
        coerce_map={"ok": "good", "g": "good"},
        meta={"version": "test"},
        normalizer="casefold",
    )
    out = ref.coerce(pd.Series(["OK", "G", "bad", "weird", None]))
    assert out.tolist()[:4] == ["good", "good", "bad", "weird"]
    assert pd.isna(out.tolist()[4])


def test_bad_normalizer_rejected():
    with pytest.raises(ValueError, match="unknown normalizer"):
        ReferenceSet("x", frozenset(), {}, {}, normalizer="rot13")
