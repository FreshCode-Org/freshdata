"""Corruption engine: label validity, determinism, composability, traps."""

from __future__ import annotations

import pandas as pd
import pytest
from training.corruptors import CORRUPTOR_REGISTRY, FAMILIES, compose, get_corruptor
from training.corruptors.base import TRANSFORM_FAMILIES, apply_corruptor
from training.datasets.validators import validate_labels
from training.seed.synthetic import make_customers, make_transactions


def _frame():
    customers = make_customers(60, seed=3)
    transactions = make_transactions(60, seed=4)
    return customers, transactions


def test_registry_has_roughly_forty_corruptors():
    assert len(CORRUPTOR_REGISTRY) >= 38


def test_every_family_is_recognized():
    for family in FAMILIES:
        assert family in TRANSFORM_FAMILIES


@pytest.mark.parametrize("name", sorted(CORRUPTOR_REGISTRY))
def test_corruptor_emits_valid_labels(name):
    customers, transactions = _frame()
    corruptor = get_corruptor(name)
    frame = transactions if name in (
        "vendor_code_swap",
    ) or "monthly_revenue" in str(corruptor.params) else customers
    columns = None
    if corruptor.kind == "cell":
        # Pick a plausible column for value-shaped corruptors so at least
        # some cells have a chance to mutate; header/row corruptors ignore
        # the columns argument.
        candidates = {
            "email_at_whitespace": ["email"], "email_double_at": ["email"],
            "email_casing": ["email"], "email_punct_noise": ["email"],
            "phone_in_spacing": ["phone"], "phone_in_zero_prefix": ["phone"],
            "phone_in_plus91_format": ["phone"], "phone_hyphenation": ["phone"],
            "phone_unsafe_mutation": ["phone"],
            "allowed_value_case": ["status"], "allowed_value_whitespace": ["status"],
            "allowed_value_separator": ["status"], "edit_distance_typo": ["status"],
            "ambiguous_category": ["status"], "country_code_ambiguity": ["country"],
            "short_code_ambiguity": ["country"], "close_category_pair": ["country"],
            "date_format_shuffle": ["signup_date"], "date_dayfirst_ambiguity": ["signup_date"],
            "relative_date_phrase": ["signup_date"], "month_name_abbreviation": ["signup_date"],
            "invalid_date_phrase": ["signup_date"],
            "boolean_synonym_replacement": ["newsletter_opt_in"],
            "vendor_code_swap": ["status"],
        }
        columns = candidates.get(name)
    frame_out, labels = apply_corruptor(frame, corruptor, columns, seed=42, share=0.9)
    assert isinstance(frame_out, pd.DataFrame)
    problems = validate_labels([label.to_dict() for label in labels])
    assert not problems, problems


@pytest.mark.parametrize("name", sorted(CORRUPTOR_REGISTRY))
def test_corruptor_is_deterministic(name):
    customers, _ = _frame()
    corruptor = get_corruptor(name)
    a, labels_a = apply_corruptor(customers, corruptor, None, seed=7, share=0.7)
    b, labels_b = apply_corruptor(customers, corruptor, None, seed=7, share=0.7)
    assert a.equals(b)
    assert [label.to_dict() for label in labels_a] == [label.to_dict() for label in labels_b]


def test_input_frame_never_mutated():
    customers, _ = _frame()
    before = customers.copy(deep=True)
    apply_corruptor(customers, get_corruptor("casing_change"), ["city"], seed=1, share=1.0)
    assert customers.equals(before)


def test_protected_column_trap_labels_mutation_as_failure():
    customers, _ = _frame()
    _, labels = apply_corruptor(
        customers, get_corruptor("protected_column_trap"), seed=3,
        params={"column": "national_id_like", "n": 5},
    )
    assert labels
    for label in labels:
        assert label.protected is True
        assert label.should_repair is False
        assert label.should_auto_apply is False


def test_target_column_trap_never_auto_apply():
    _, transactions = _frame()
    _, labels = apply_corruptor(
        transactions, get_corruptor("target_column_trap"), seed=3,
        params={"column": "monthly_revenue", "n": 3},
    )
    assert labels
    assert all(not label.should_auto_apply and not label.should_repair for label in labels)


@pytest.mark.parametrize("name", [
    "ambiguous_category", "country_code_ambiguity", "short_code_ambiguity",
    "close_category_pair", "date_dayfirst_ambiguity", "relative_date_phrase",
    "phone_unsafe_mutation",
])
def test_ambiguous_corruptors_are_never_auto_apply(name):
    corruptor = get_corruptor(name)
    assert corruptor.ambiguous is True
    assert corruptor.should_auto_apply is False


def test_construction_rejects_ambiguous_auto_apply():
    from training.corruptors.base import Corruptor

    with pytest.raises(ValueError, match="ambiguous"):
        Corruptor(name="bad", family="reference_value", fn=lambda v, r, p: v,
                  ambiguous=True, should_auto_apply=True)


def test_construction_rejects_protected_auto_apply():
    from training.corruptors.base import Corruptor

    with pytest.raises(ValueError, match="protected"):
        Corruptor(name="bad2", family="trap", fn=lambda v, r, p: v,
                  protected=True, should_auto_apply=True)


def test_composition_preserves_ground_truth_no_double_corruption():
    customers, _ = _frame()
    steps = [
        (get_corruptor("whitespace_insertion"), ["full_name"], 0.9),
        (get_corruptor("casing_change"), ["full_name"], 0.9),
    ]
    _, labels = compose(customers, steps, seed=11)
    # Every (row, column) pair is labeled by at most one step.
    seen = [(label.row, label.column) for label in labels]
    assert len(seen) == len(set(seen))
    # Every label's clean_value equals the frame's original value at that cell.
    for label in labels:
        if label.row is not None and label.column is not None:
            original = str(customers.iloc[label.row][label.column])
            if label.clean_value is not None:
                assert str(label.clean_value) == original


def test_sentinel_style_labels_target_missing_not_original_value():
    customers, _ = _frame()
    _, labels = apply_corruptor(
        customers, get_corruptor("sentinel_injection"), ["full_name"], seed=1, share=1.0)
    assert labels
    assert all(label.clean_value is None for label in labels)


def test_duplicate_row_injection_appends_labeled_rows():
    customers, _ = _frame()
    out, labels = apply_corruptor(
        customers, get_corruptor("duplicate_row_injection"), seed=2,
        params={"n_duplicates": 3})
    assert len(out) == len(customers) + 3
    assert len(labels) == 3
    assert all(label.transform_family == "row_structure" for label in labels)
