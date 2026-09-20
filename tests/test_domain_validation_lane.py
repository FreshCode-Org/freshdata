"""Domain-validation lane: cross-pack behaviour of ``freshdata.domains``.

This file is deliberately *cross-cutting*: the per-pack suites under
``tests/domains/`` check that each pack flags the right rows, while these tests
pin the properties that only show up when the eight packs are compared with one
another, or when a pack is handed input it was never meant to see.

What each section guards
------------------------
1.  ``TestRuleGroundTruth`` — valid/invalid cases taken from each pack's own
    ``rules.yaml`` (severities, layers, repair strategies), not invented.
2.  ``TestBoundaryValues`` — every declared numeric/length bound is *inclusive*:
    exact boundary passes, one step outside fails.
3.  ``TestWrongDomainInput`` — an 8x8 matrix: no pack ever silently validates
    another pack's data.
4.  ``TestIncompleteSchema`` — a missing column is reported, never treated as a
    pass, and every dependent rule is ``skipped`` rather than ``passed``.
5.  ``TestColumnMapOverrides`` — what a wrong ``column_map`` does (and what it
    silently does *not* do).
6.  ``TestTrustScore`` — the exact arithmetic of ``_trust_score`` and a
    demonstration that scores are **not** comparable across domains.
7.  ``TestReferenceSnapshotIsNotExhaustive`` — "absent from the bundled snapshot"
    versus "definitively invalid".
8.  ``TestVersioningSurfaced`` — rule/schema versions reach ``ValidationReport``.
9.  ``TestRunDomainNeverMutatesInput`` — verified with a SHA-256 digest over
    columns, index, dtypes and values, not ``assert_frame_equal``.

Findings pinned here (see the PR body for the write-up). Where current behaviour
looks wrong the test asserts *what the code does today* and says so, so the
behaviour cannot drift unnoticed and a deliberate fix has one obvious place to
update:

* ``test_regex_check_reads_an_integral_float_column_correctly`` — was an S2
  finding: the shared ``_check_regex`` stringified a float64 column as
  ``"10000266.0"``, so FIN-008 and GS1-008 raised false findings for the
  ordinary "CSV column with one blank cell" case. **Fixed** by moving the
  retail pack's ``_integral_float_text`` into ``domains.base`` as
  ``integral_float_text`` and applying it to every regex rule; this test now
  asserts the repaired behaviour.
* ``test_balanced_tolerance_never_admits_a_one_cent_imbalance`` — S3: FIN-006
  declares ``tolerance: 0.01`` but compares raw binary floats, so a one-cent
  rounding difference is always above tolerance.
* ``test_absent_from_a_disclaimed_subset_is_reported_as_error`` — S2: packs
  disagree on whether "not in our documented subset" is an error or a warning.
* ``test_trust_score_is_not_comparable_across_domains`` — S3 / spec gap.
* ``test_missing_required_field_sentinel_collides_with_a_row_label`` — S3: the
  in-band ``MISSING_REQUIRED_FIELD`` row-label sentinel.
* ``test_override_naming_an_absent_column_is_silently_ignored`` and friends — S3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from freshdata.domains import DomainError, Rule, available, get_validator, run_domain
from freshdata.domains.base import (
    MISSING_REQUIRED_FIELD,
    SEVERITY_WEIGHT,
    ConfigDrivenValidator,
)
from freshdata.domains.reference import available_references, load_reference

DOMAINS_DIR = Path(get_validator("finance").rules_path).resolve().parent.parent

#: Extra constructor kwargs the multi-schema packs need to pick a sub-schema.
PACK_KWARGS: dict[str, dict[str, str]] = {
    "transport": {"gtfs_file": "stops"},
    "healthcare": {"fhir_resource": "Patient"},
    "media": {"media_type": "content"},
}


# --------------------------------------------------------------------- fixtures


def _good_frames() -> dict[str, pd.DataFrame]:
    """One all-valid frame per pack (trust == 1.0 against its own pack)."""
    return {
        "finance": pd.DataFrame({
            "transaction_id": ["T1", "T1", "T2", "T2"],
            "date": ["2024-01-15", "2024-01-15", "2024-02-01", "2024-02-01"],
            "account_code": ["AC1001", "AC2002", "AC1001", "AC3003"],
            "debit": [100.0, 0.0, 50.0, 0.0],
            "credit": [0.0, 100.0, 0.0, 50.0],
            "currency": ["USD", "USD", "EUR", "EUR"],
            "description": ["sale", "sale", "refund", "refund"],
            "entity_id": ["E1", "E1", "E2", "E2"],
        }),
        "retail": pd.DataFrame({
            "gtin": ["00012345678905", "5901234123457"],
            "gln": ["1234567890128", "1234567890128"],
            "product_description": ["Organic Whole Milk 1L", "Pasta Fusilli 500g"],
            "net_content": [1.0, 0.5],
            "net_content_uom": ["LTR", "KGM"],
            "country_of_origin": ["DE", "IT"],
            "gpc_brick_code": ["10000266", "10001336"],
        }),
        "transport": pd.DataFrame({
            "stop_id": ["S001", "S002"],
            "stop_name": ["Central", "Airport"],
            "stop_lat": [51.5074, -33.8688],
            "stop_lon": [-0.1278, 151.2093],
        }),
        "healthcare": pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "birth_date": ["1985-06-15", "1972-11-30"],
            "gender": ["male", "female"],
            "deceased": [False, False],
            "address_country": ["US", "GB"],
        }),
        "education": pd.DataFrame({
            "student_unique_id": ["STU001", "STU002"],
            "school_id": ["SCH100", "SCH100"],
            "school_year": [2024, 2024],
            "grade_level": ["Eighth grade", "Twelfth grade"],
            "enrollment_date": ["2023-09-05", "2023-09-05"],
        }),
        "agriculture": pd.DataFrame({
            "field_id": ["F001", "F002"],
            "operation_id": ["OP001", "OP002"],
            "operation_type": ["Planting", "Harvesting"],
            "operation_date": ["2024-04-15", "2024-10-20"],
            "area": [50.0, 50.0],
            "area_unit": ["HAR", "HAR"],
            "soil_ph": [6.8, 7.2],
            "soil_om_pct": [3.2, 4.1],
            "season_year": [2024, 2024],
        }),
        "media": pd.DataFrame({
            "eidr_id": ["10.5240/7791-8534-2C23-9030-8610-5"],
            "title": ["Inception"],
            "content_type": ["Movie"],
            "release_date": ["2010-07-16"],
            "country_of_origin": ["US"],
            "language": ["en"],
            "runtime_seconds": [8820],
        }),
        "energy": pd.DataFrame({
            "timestamp": ["2024-03-15T00:00:00", "2024-03-15T00:00:01"],
            "asset_id": ["RTU-1", "RTU-2"],
            "register_address": [1, 100],
            "function_code": [3, 4],
            "value": [50.1, 50.2],
            "quality": ["good", "good"],
            "unit": ["V", "kW"],
        }),
    }


@pytest.fixture
def good_frames() -> dict[str, pd.DataFrame]:
    return _good_frames()


def _finance_ledger(**overrides: object) -> pd.DataFrame:
    """A balanced 4-row ledger; keyword arguments replace whole columns."""
    df = _good_frames()["finance"]
    for column, values in overrides.items():
        df[column] = values
    return df


def _result(outcome, rule_id: str):
    """The single :class:`RuleResult` for *rule_id* (KeyError-loud if absent)."""
    matches = [r for r in outcome.report.results if r.rule_id == rule_id]
    assert len(matches) == 1, f"expected exactly one {rule_id} result, got {len(matches)}"
    return matches[0]


def _digest(df: pd.DataFrame) -> str:
    """Content digest over columns, index, dtypes and every cell value.

    ``assert_frame_equal`` tolerates dtype and index-type differences by default;
    this does not, which is what "never mutates the input" has to mean.
    """
    hasher = hashlib.sha256()
    hasher.update(repr(list(df.columns)).encode())
    hasher.update(repr(list(df.index)).encode())
    hasher.update(repr([str(dtype) for dtype in df.dtypes]).encode())
    hasher.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    for column in df.columns:
        hasher.update(repr(df[column].tolist()).encode())
    return hasher.hexdigest()


# ------------------------------------------------- 1. rules.yaml as ground truth


class TestRuleGroundTruth:
    """Assertions derived from each pack's own rule file, not from assumption."""

    def test_every_pack_validates_its_own_good_frame_at_full_trust(self, good_frames):
        scored = {}
        for name, df in good_frames.items():
            _, outcome = run_domain(df, name, **PACK_KWARGS.get(name, {}))
            scored[name] = (outcome.trust_score, outcome.report.passed,
                            [r.rule_id for r in outcome.report.results if r.violated])
        assert scored == {name: (1.0, True, []) for name in good_frames}

    def test_declared_rule_metadata_matches_what_the_engine_reports(self, good_frames):
        """A rule's reported layer/severity/check/repair is its rules.yaml entry."""
        for name, df in good_frames.items():
            validator = get_validator(name, **PACK_KWARGS.get(name, {}))
            validator.detect_columns(df)  # activates the sub-schema where relevant
            declared = {
                rule.id: (rule.layer, rule.severity, rule.check, rule.repair)
                for rule in validator.rules
            }
            _, outcome = run_domain(df, name, **PACK_KWARGS.get(name, {}))
            reported = {
                r.rule_id: (r.layer, r.severity, r.check, r.repair)
                for r in outcome.report.results
            }
            assert reported == declared, f"{name}: engine/rules.yaml metadata drift"

    def test_finance_invalid_cases_map_to_the_rule_that_declares_them(self):
        """One row per FIN-* failure mode, each derived from finance/rules.yaml."""
        df = pd.DataFrame({
            # T3 (row 2) is balanced on its own; T2 (row 3) is a lone 5.00 debit.
            "transaction_id": ["T1", "T1", "T3", "T2", None],
            # row 2: not ISO-8601 (FIN-003); row 3: impossible date (FIN-003)
            "date": ["2024-01-15", "2024-01-15", "15/04/2024", "2024-13-40", "2024-01-15"],
            "account_code": ["AC1001", "AC2002", "AC1", "AC3003", "AC4004"],
            "debit": [100.0, 0.0, 10.005, 5.0, 0.0],
            "credit": [0.0, 100.0, 10.005, 0.0, 0.0],
            "currency": ["USD", "USD", "XYZ", "EUR", "EUR"],
            "description": ["a", "b", "c", "d", "e"],
            "entity_id": ["E1", "E1", "E2", "E2", "E3"],
        })
        _, outcome = run_domain(df, "finance")
        assert _result(outcome, "FIN-001").violation_rows == [4]      # null txn id
        assert _result(outcome, "FIN-003").violation_rows == [2, 3]   # format + impossible
        assert _result(outcome, "FIN-004").violation_rows == [2]      # >2 decimals
        assert _result(outcome, "FIN-005").violation_rows == [2]      # XYZ not ISO 4217
        assert _result(outcome, "FIN-006").violation_rows == [3]      # T2 debit 5, credit 0
        assert _result(outcome, "FIN-007").violation_rows == [2]      # both sides non-zero
        assert _result(outcome, "FIN-008").violation_rows == [2]      # "AC1" is 3 chars
        assert _result(outcome, "FIN-009").status == "passed"         # no future dates

    def test_repair_strategies_are_honoured_exactly_as_declared(self):
        """``flag_only`` logs but never writes; ``coerce`` writes only what it can."""
        df = _finance_ledger(
            date=["15/04/2024", "2024-01-15", "2024-02-01", "2024-02-01"],
            currency=["XYZ", "USD", "EUR", "EUR"],
        )
        repaired, outcome = run_domain(df, "finance")
        # FIN-003 declares repair: coerce -> an unambiguous D/M/Y date is rewritten.
        assert repaired["date"].tolist()[0] == "2024-04-15"
        # FIN-005 declares repair: flag_only -> the bad currency survives untouched.
        assert repaired["currency"].tolist()[0] == "XYZ"
        flagged = [a for a in outcome.repairs if a.rule_id == "FIN-005"]
        assert [a.status for a in flagged] == ["flagged"]
        assert [(a.from_value, a.to_value) for a in flagged] == [(None, None)]

    def test_identifier_columns_are_never_imputed_or_dropped(self, good_frames):
        """``id_fields`` is the packs' no-silent-repair promise; hold every pack to it."""
        for name, df in good_frames.items():
            validator = get_validator(name, **PACK_KWARGS.get(name, {}))
            mapping = validator.detect_columns(df)
            id_columns = [mapping.actual(f) for f in validator.id_fields
                          if mapping.is_mapped(f)]
            if not id_columns:
                continue
            blanked = df.copy()
            for column in id_columns:
                blanked.loc[blanked.index[0], column] = None
            repaired, _ = run_domain(blanked, name, **PACK_KWARGS.get(name, {}))
            assert len(repaired) == len(blanked), f"{name}: a row was dropped"
            for column in id_columns:
                assert pd.isna(repaired.loc[repaired.index[0], column]), (
                    f"{name}: id column {column!r} was imputed"
                )


# ------------------------------------------------------------ 2. boundary values


class TestBoundaryValues:
    """Every declared bound is inclusive: on it passes, one step outside fails."""

    def test_transport_latitude_and_longitude_bounds_are_inclusive(self):
        df = pd.DataFrame({
            "stop_id": [f"S{i}" for i in range(8)],
            "stop_lat": [-90.0, 90.0, -90.000001, 90.000001, 0.0, 0.0, 0.0, 0.0],
            "stop_lon": [0.0, 0.0, 0.0, 0.0, -180.0, 180.0, -180.000001, 180.000001],
        })
        _, outcome = run_domain(df, "transport", gtfs_file="stops")
        assert _result(outcome, "GTFS-S002").violation_rows == [2, 3]
        assert _result(outcome, "GTFS-S003").violation_rows == [6, 7]

    def test_energy_modbus_register_address_bounds_are_inclusive(self):
        df = pd.DataFrame({
            "timestamp": [f"2024-03-15T00:00:0{i}" for i in range(4)],
            "asset_id": list("ABCD"),
            "register_address": [0, 65535, -1, 65536],
        })
        _, outcome = run_domain(df, "energy")
        assert _result(outcome, "ENG-005").violation_rows == [2, 3]

    def test_agriculture_soil_ph_has_two_nested_inclusive_bands(self):
        """AG-014 is the physical band [0, 14]; AG-015 the agronomic [3.5, 9.5]."""
        df = pd.DataFrame({
            "field_id": [f"F{i}" for i in range(8)],
            "soil_ph": [0.0, 14.0, -0.000001, 14.000001, 3.5, 9.5, 3.499999, 9.500001],
        })
        _, outcome = run_domain(df, "agriculture")
        assert _result(outcome, "AG-014").violation_rows == [2, 3]
        assert _result(outcome, "AG-014").severity == "error"
        # 0.0 and 14.0 are physically possible but outside the agronomic band, so
        # the warning-severity rule flags them too; that overlap is by design.
        assert _result(outcome, "AG-015").violation_rows == [0, 1, 2, 3, 6, 7]
        assert _result(outcome, "AG-015").severity == "warning"

    def test_retail_description_length_bound_is_inclusive(self):
        df = pd.DataFrame({
            "gtin": ["00012345678905"] * 3,
            "product_description": ["X" * 199, "X" * 200, "X" * 201],
        })
        _, outcome = run_domain(df, "retail")
        assert _result(outcome, "GS1-007").violation_rows == [2]

    def test_media_runtime_seconds_bound_is_inclusive(self):
        df = pd.DataFrame({
            "eidr_id": ["10.5240/7791-8534-2C23-9030-8610-5"] * 3,
            "content_type": ["Movie"] * 3,
            "runtime_seconds": [86399, 86400, 86401],
        })
        _, outcome = run_domain(df, "media", media_type="content")
        assert _result(outcome, "MD-C008").violation_rows == [2]

    def test_finance_amount_scale_boundary_is_two_decimal_places(self):
        df = _finance_ledger(
            transaction_id=["T1", "T2", "T3", "T4"],
            debit=[100.0, 100.99, 100.005, 0.0],
            credit=[100.0, 100.99, 100.005, 0.0],
        )
        _, outcome = run_domain(df, "finance")
        assert _result(outcome, "FIN-004").violation_rows == [2]

    def test_balanced_tolerance_never_admits_a_one_cent_imbalance(self):
        """FINDING (S3): FIN-006 declares ``tolerance: 0.01`` but compares raw floats.

        ``100.00 - 99.99`` is ``0.010000000000005116`` in binary floating point, so
        ``abs(diff) > 0.01`` is true and a one-cent rounding difference — precisely
        what a 0.01 tolerance exists to absorb — is reported as an error. This pins
        today's behaviour; a fix would round the difference to the rule's scale
        (or compare against ``tolerance + eps``) before the comparison.
        """
        def imbalance(debit: float, credit: float) -> str:
            df = pd.DataFrame({
                "transaction_id": ["T1", "T1"],
                "date": ["2024-01-15", "2024-01-15"],
                "debit": [debit, 0.0],
                "credit": [0.0, credit],
                "currency": ["USD", "USD"],
            })
            _, outcome = run_domain(df, "finance")
            return _result(outcome, "FIN-006").status

        assert imbalance(100.0, 100.0) == "passed"
        assert abs(100.0 - 99.99) > 0.01           # the float fact behind the finding
        assert imbalance(100.0, 99.99) == "violated"
        assert imbalance(1.0, 0.99) == "violated"
        assert imbalance(100.0, 99.98) == "violated"


# ------------------------------------------------------- 3. wrong-domain input


class TestWrongDomainInput:
    """Feeding pack B's data to pack A must never look like a clean validation."""

    def test_no_pack_silently_validates_another_packs_data(self, good_frames):
        """The full 8x8 matrix: only the diagonal may pass."""
        names = sorted(good_frames)
        assert names == available()
        matrix: dict[tuple[str, str], tuple[float, bool]] = {}
        for source in names:
            for pack in names:
                _, outcome = run_domain(good_frames[source], pack,
                                        **PACK_KWARGS.get(pack, {}))
                matrix[(source, pack)] = (outcome.trust_score, outcome.report.passed)
        expected = {
            (source, pack): (1.0, True) if source == pack else (0.0, False)
            for source in names for pack in names
        }
        assert matrix == expected

    def test_a_shared_column_name_alone_never_produces_a_pass(self, good_frames):
        """Retail's ``product_description`` alias does match finance's ``description``.

        The alias match is real and GS1-007 genuinely passes on it — but the frame
        still fails overall, because ``gtin`` is absent and GS1-001 fires. Partial
        alias luck must never add up to a validation.
        """
        _, outcome = run_domain(good_frames["finance"], "retail")
        assert outcome.report.mapping.mapped == {"product_description": "description"}
        assert _result(outcome, "GS1-007").status == "passed"
        assert _result(outcome, "GS1-001").message == f"{MISSING_REQUIRED_FIELD}: gtin"
        assert outcome.report.passed is False
        assert outcome.trust_score == 0.0

    @pytest.mark.parametrize("pack,error_name", [
        ("healthcare", "AmbiguousFHIRResourceError"),
        ("media", "AmbiguousMediaTypeError"),
    ])
    def test_auto_detecting_packs_raise_rather_than_guess(self, good_frames, pack,
                                                          error_name):
        for source, df in good_frames.items():
            if source == pack:
                continue
            with pytest.raises(Exception) as excinfo:  # noqa: B017,PT011 - typed below
                run_domain(df, pack)
            assert type(excinfo.value).__name__ == error_name, (
                f"{pack} on {source} data raised {type(excinfo.value).__name__}"
            )

    def test_transport_refuses_a_frame_with_no_file_selector(self, good_frames):
        with pytest.raises(DomainError, match="requires gtfs_file"):
            run_domain(good_frames["finance"], "transport")


# ------------------------------- 4. incomplete schemas / misleading column names


class TestIncompleteSchema:
    """A column that is not there must never be reported as validated."""

    def test_no_rule_reports_passed_for_a_field_that_is_absent(self, good_frames):
        """The central no-silent-validation invariant, swept over every pack.

        Drop each mapped column in turn from each pack's own good frame; no rule
        naming that canonical field may come back ``passed``.
        """
        offenders = []
        for name, df in good_frames.items():
            kwargs = PACK_KWARGS.get(name, {})
            mapping = get_validator(name, **kwargs).detect_columns(df)
            for canonical, actual in mapping.mapped.items():
                _, outcome = run_domain(df.drop(columns=[actual]), name, **kwargs)
                offenders += [
                    (name, canonical, r.rule_id)
                    for r in outcome.report.results
                    if canonical in r.fields and r.status == "passed"
                ]
        assert offenders == []

    def test_missing_required_column_is_a_table_level_finding_not_a_row_finding(self):
        df = _good_frames()["finance"].drop(columns=["currency"])
        _, outcome = run_domain(df, "finance")
        presence = _result(outcome, "FIN-002")
        assert presence.status == "violated"
        assert presence.violation_rows == [MISSING_REQUIRED_FIELD]
        assert presence.message == f"{MISSING_REQUIRED_FIELD}: currency"
        # The reference rule on the same field is skipped, never passed.
        reference = _result(outcome, "FIN-005")
        assert reference.status == "skipped"
        assert reference.message == "skipped: currency not present"
        assert outcome.report.mapping.unmapped_required == ["currency"]
        assert outcome.report.passed is False

    def test_mapping_log_records_how_every_field_was_resolved(self):
        """Nothing is guessed silently: each canonical field gets a log entry."""
        df = pd.DataFrame({
            "transaction_id": ["T1"],       # exact
            "DATE": ["2024-01-15"],         # case-insensitive
            "dr": [1.0],                    # regex alias
            "cr": [1.0],                    # regex alias
            "currency": ["USD"],            # exact
        })
        mapping = get_validator("finance").detect_columns(df)
        methods = {entry["canonical"]: entry["method"] for entry in mapping.log}
        assert methods == {
            "transaction_id": "exact",
            "date": "case_insensitive",
            "account_code": "missing",
            "debit": "regex",
            "credit": "regex",
            "currency": "exact",
            "description": "missing",
            "entity_id": "missing",
        }
        assert len(mapping.log) == len(get_validator("finance").canonical_fields)

    def test_a_misleading_column_name_is_matched_but_always_disclosed(self):
        """``salary_credit`` must not be grabbed; a bare ``cr`` is, and is logged."""
        validator = get_validator("finance")
        wide = pd.DataFrame({"credit_limit": [1], "salary_credit": [2], "cr": [3]})
        mapping = validator.detect_columns(wide)
        # Aliases are anchored with re.fullmatch, so neither affix form matches.
        assert mapping.actual("credit") == "cr"
        assert {"credit_limit", "salary_credit"}.isdisjoint(set(mapping.mapped.values()))
        assert {"canonical": "credit", "actual": "cr", "method": "regex"} in mapping.log


# ------------------------------------------------------- 5. column_map overrides


class TestColumnMapOverrides:
    """``column_map`` is ``{actual_column: canonical_field}`` and is not validated."""

    def test_a_correct_override_maps_every_field_and_is_logged_as_an_override(self):
        df = pd.DataFrame({
            "txn": ["T1", "T1"], "d": ["2024-01-15", "2024-01-15"],
            "amount_out": [100.0, 0.0], "amount_in": [0.0, 100.0], "ccy": ["USD", "USD"],
        })
        column_map = {"txn": "transaction_id", "d": "date", "amount_out": "debit",
                      "amount_in": "credit", "ccy": "currency"}
        _, outcome = run_domain(df, "finance", column_map=column_map)
        assert outcome.report.mapping.mapped == {
            "transaction_id": "txn", "date": "d", "debit": "amount_out",
            "credit": "amount_in", "currency": "ccy",
        }
        overridden = {e["canonical"] for e in outcome.report.mapping.log
                      if e["method"] == "override"}
        assert overridden == set(column_map.values())
        assert outcome.trust_score == 1.0

    def test_override_naming_an_absent_column_is_silently_ignored(self):
        """FINDING (S3): a typo'd source column is dropped without any diagnostic.

        Detection simply falls through to the normal exact/alias search, so the
        user silently gets a *different* column than the one they asked for, and
        the mapping log reports the fallback method as if no override was given.
        """
        df = pd.DataFrame({"debit": [1.0], "amount_out": [2.0]})
        mapping = get_validator("finance", column_map={"amount_owt": "debit"}) \
            .detect_columns(df)
        assert mapping.actual("debit") == "debit"       # not "amount_out"
        assert {"canonical": "debit", "actual": "debit", "method": "exact"} in mapping.log
        assert not any(e["method"] == "override" for e in mapping.log)

    def test_override_naming_an_unknown_canonical_field_is_silently_ignored(self):
        """FINDING (S3): ``currency_code`` is not a finance canonical field.

        Nothing raises and nothing is logged; the override is simply never
        consulted, because detection only iterates the pack's own field names.
        """
        df = pd.DataFrame({
            "transaction_id": ["T1"], "date": ["2024-01-15"],
            "debit": [0.0], "credit": [0.0], "ccy": ["USD"],
        })
        mapping = get_validator("finance", column_map={"ccy": "currency_code"}) \
            .detect_columns(df)
        assert mapping.actual("currency") == "ccy"      # via the alias, not the override
        assert {"canonical": "currency", "actual": "ccy", "method": "regex"} in mapping.log
        assert "currency_code" not in {e["canonical"] for e in mapping.log}

    def test_two_columns_mapped_to_one_canonical_field_keeps_only_the_last(self):
        """FINDING (S3): the override dict is inverted, so earlier keys are lost."""
        df = pd.DataFrame({
            "txn": ["T1"], "d": ["2024-01-15"],
            "amount_out": [1.0], "amount_in": [2.0], "ccy": ["USD"],
        })
        column_map = {"txn": "transaction_id", "d": "date", "ccy": "currency",
                      "amount_out": "debit", "amount_in": "debit"}
        mapping = get_validator("finance", column_map=column_map).detect_columns(df)
        assert mapping.actual("debit") == "amount_in"   # last key wins
        assert mapping.unmapped_required == ["credit"]  # ...and credit is now missing

    def test_a_semantically_swapped_override_validates_cleanly_by_design(self):
        """An override is the user's explicit instruction; the pack cannot second-guess it.

        Swapping debit and credit keeps the ledger structurally valid, so trust is
        1.0. The audit guard is that the mapping log names both columns and marks
        the method ``override``, so the swap is visible in the report.
        """
        df = pd.DataFrame({
            "txn": ["T1", "T1"], "d": ["2024-01-15", "2024-01-15"],
            "amount_out": [100.0, 0.0], "amount_in": [0.0, 100.0], "ccy": ["USD", "USD"],
        })
        swapped = {"txn": "transaction_id", "d": "date", "amount_out": "credit",
                   "amount_in": "debit", "ccy": "currency"}
        _, outcome = run_domain(df, "finance", column_map=swapped)
        assert outcome.trust_score == 1.0
        assert outcome.report.mapping.mapped["debit"] == "amount_in"
        assert outcome.report.mapping.mapped["credit"] == "amount_out"
        assert outcome.report.to_dict()["mapping"]["mapped"]["debit"] == "amount_in"

    def test_an_override_onto_the_wrong_kind_of_column_fails_loudly(self):
        """Pointing ``date`` at an amount column trips the format/business layers."""
        df = pd.DataFrame({
            "txn": ["T1", "T1"], "d": ["2024-01-15", "2024-01-15"],
            "amount_out": [100.0, 0.0], "amount_in": [0.0, 100.0], "ccy": ["USD", "USD"],
        })
        crossed = {"txn": "transaction_id", "d": "debit", "amount_out": "date",
                   "amount_in": "credit", "ccy": "currency"}
        _, outcome = run_domain(df, "finance", column_map=crossed)
        violated = sorted(r.rule_id for r in outcome.report.results if r.violated)
        assert violated == ["FIN-003", "FIN-004", "FIN-006"]
        assert outcome.report.passed is False


# ---------------------------------------------------------------- 6. trust score


class TestTrustScore:
    """What ``ConfigDrivenValidator._trust_score`` actually measures.

    Established empirically and from the source, not assumed::

        score = clamp01(1 - SUM over violated rules of
                            weight(severity) * min(violated_rows / n_rows, 1))

    with ``weight = {error: 1.0, warning: 0.25, info: 0.05}``. Skipped and passed
    rules contribute nothing; a table-level finding (a missing required column)
    contributes the full weight regardless of row count.
    """

    def test_weights_are_exactly_one_quarter_row_fraction_per_severity(self):
        """One bad row in four, at each severity, against the documented weights."""
        assert SEVERITY_WEIGHT == {"error": 1.0, "warning": 0.25, "info": 0.05}

        _, error_only = run_domain(
            _finance_ledger(currency=["ZZZ", "USD", "EUR", "EUR"]), "finance")
        assert error_only.report.severity_counts == {"info": 0, "warning": 0, "error": 1}
        assert error_only.trust_score == 0.75            # 1 - 1.00 * (1/4)

        _, warning_only = run_domain(pd.DataFrame({
            "field_id": ["F1", "F2", "F3", "F4"],
            "soil_ph": [6.5, 6.8, 7.0, 3.0],             # inside [0,14], outside [3.5,9.5]
        }), "agriculture")
        assert warning_only.report.severity_counts == {"info": 0, "warning": 1, "error": 0}
        assert warning_only.trust_score == 0.9375        # 1 - 0.25 * (1/4)

        _, info_only = run_domain(pd.DataFrame({
            "timestamp": [f"2024-03-15T00:00:0{i}" for i in range(4)],
            "asset_id": list("ABCD"),
            "quality": ["good", "good", "good", "bad"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }), "energy")
        assert info_only.report.severity_counts == {"info": 1, "warning": 0, "error": 0}
        assert info_only.trust_score == 0.9875           # 1 - 0.05 * (1/4)

    def test_penalties_from_several_rules_add_up(self):
        _, outcome = run_domain(pd.DataFrame({
            "field_id": ["F1", "F2", "F3", "F4"],
            "soil_ph": [6.5, 6.8, 7.0, 20.0],            # trips AG-014 and AG-015
        }), "agriculture")
        assert [r.rule_id for r in outcome.report.results if r.violated] == \
            ["AG-014", "AG-015"]
        assert outcome.trust_score == 0.6875             # 1 - (1.0 + 0.25) * (1/4)

    def test_the_row_fraction_dilutes_a_single_bad_row_in_a_large_frame(self):
        big = pd.concat([_good_frames()["finance"]] * 250, ignore_index=True)
        big.loc[0, "currency"] = "ZZZ"
        _, outcome = run_domain(big, "finance")
        assert len(big) == 1000
        assert outcome.trust_score == 0.999              # 1 - 1.00 * (1/1000)

    def test_a_missing_column_costs_the_full_weight_and_silences_its_other_rules(self):
        """Missing fields and rule violations are *not* commensurate.

        A missing required column is charged a row fraction of 1.0 no matter how
        large the frame is, and simultaneously removes every other rule on that
        field from the score (they are skipped). The score's response to a missing
        column is therefore a step, not something proportional to lost data.
        """
        big = pd.concat([_good_frames()["finance"]] * 250, ignore_index=True)
        _, outcome = run_domain(big.drop(columns=["currency"]), "finance")
        assert len(big) == 1000
        assert _result(outcome, "FIN-002").n_violations == 1     # one table-level finding
        assert _result(outcome, "FIN-005").status == "skipped"   # contributes nothing
        assert outcome.trust_score == 0.0                        # not 1 - 1/1000

    def test_skipped_rules_never_affect_the_score(self, good_frames):
        """The one trust-score property CONTRIBUTING_DOMAINS.md actually states."""
        _, outcome = run_domain(good_frames["transport"], "transport", gtfs_file="stops")
        statuses = [r.status for r in outcome.report.results]
        assert statuses.count("skipped") == 10      # routes/trips/stop_times rules
        assert statuses.count("passed") == 4
        assert outcome.trust_score == 1.0

    def test_the_score_saturates_at_zero_and_stops_discriminating(self):
        """Below the floor the score cannot rank anything, even within one pack."""
        _, one_rule = run_domain(_finance_ledger(currency=["ZZZ"] * 4), "finance")
        _, many_rules = run_domain(_finance_ledger(
            currency=["ZZZ"] * 4, date=["nope"] * 4,
            debit=[-1.0] * 4, credit=[-2.0] * 4,
        ), "finance")
        assert len([r for r in one_rule.report.results if r.violated]) == 1
        assert len([r for r in many_rules.report.results if r.violated]) == 5
        assert one_rule.trust_score == many_rules.trust_score == 0.0

    def test_trust_score_is_not_comparable_across_domains(self, good_frames):
        """FINDING (S3 / SPECIFICATION GAP): the score is a within-pack measure only.

        Nothing in the repo states how ``domain_trust_score`` should be read.
        CONTRIBUTING_DOMAINS.md says only that "skipped rules never affect the
        trust score". Three measured properties show the number cannot be compared
        between packs, so the missing decision is explicit: is this an absolute,
        cross-pack data-quality index, or a within-pack/within-schema indicator?

        (a) the penalty is an unnormalised sum over rules, so denser packs fall
            faster: media can accrue 21.00 of penalty, finance only 6.75;
        (b) packs cover the same defect with different numbers of rules, so one
            bad cell in four rows costs 0.25 in finance but 0.3125 in agriculture;
        (c) packs disagree on the severity of the same *kind* of check (see
            ``TestReferenceSnapshotIsNotExhaustive``).
        """
        # (a) the dynamic range differs per pack.
        ranges = {}
        for name, df in good_frames.items():
            validator = get_validator(name, **PACK_KWARGS.get(name, {}))
            validator.detect_columns(df)
            ranges[name] = round(sum(SEVERITY_WEIGHT[r.severity] for r in validator.rules), 2)
        assert ranges == {
            "agriculture": 15.25, "education": 12.75, "energy": 7.30, "finance": 6.75,
            "healthcare": 8.30, "media": 21.00, "retail": 7.50, "transport": 14.00,
        }

        # (b) the same defect shape scores differently in different packs.
        _, finance = run_domain(
            _finance_ledger(currency=["ZZZ", "USD", "EUR", "EUR"]), "finance")
        _, agriculture = run_domain(pd.DataFrame({
            "field_id": ["F1", "F2", "F3", "F4"],
            "soil_ph": [20.0, 6.8, 7.0, 6.5],
        }), "agriculture")
        education = _good_frames()["education"]
        education = pd.concat([education] * 2, ignore_index=True)
        education.loc[0, "grade_level"] = "Not A Grade"
        _, education_outcome = run_domain(education, "education")
        # One invalid coded value in one of four rows, three times over:
        assert (finance.trust_score, agriculture.trust_score,
                education_outcome.trust_score) == (0.75, 0.6875, 0.75)

    def test_missing_required_field_sentinel_collides_with_a_row_label(self):
        """FINDING (S3): ``MISSING_REQUIRED_FIELD`` is an in-band row-label sentinel.

        ``_trust_score`` decides "table-level finding" by testing whether any
        ``violation_rows`` entry equals the literal string ``MISSING_REQUIRED_FIELD``.
        A frame whose index carries that label therefore has an ordinary one-row
        violation charged as a whole-table one: 0.75 becomes 0.0. Nothing is
        corrupted and no built-in pack uses ``repair: reject``, so the blast radius
        today is the score alone — but the sentinel belongs on ``RuleResult`` as a
        flag, not in the row-label channel.
        """
        df = _good_frames()["finance"]
        df.index = [MISSING_REQUIRED_FIELD, "r1", "r2", "r3"]
        df.loc[MISSING_REQUIRED_FIELD, "currency"] = "ZZZ"
        _, outcome = run_domain(df, "finance")
        result = _result(outcome, "FIN-005")
        assert result.status == "violated"
        assert result.violation_rows == [MISSING_REQUIRED_FIELD]   # a real row label
        assert result.n_violations == 1
        assert outcome.trust_score == 0.0                          # would be 0.75

        # Control: the same violation on any other label scores as expected.
        control = _good_frames()["finance"]
        control.index = ["r0", "r1", "r2", "r3"]
        control.loc["r0", "currency"] = "ZZZ"
        _, control_outcome = run_domain(control, "finance")
        assert control_outcome.trust_score == 0.75


# ----------------------------------- 7. reference snapshots are not exhaustive


class TestReferenceSnapshotIsNotExhaustive:
    """``domains/reference.py``: the bundled sets are "documented common subsets"."""

    def test_every_bundled_set_declares_its_provenance(self):
        for name in available_references():
            meta = load_reference(name).meta
            assert meta.get("version"), f"{name}: no version in _meta"
            assert meta.get("source"), f"{name}: no source in _meta"

    def test_healthcare_treats_absence_from_a_subset_as_a_warning(self):
        """The reference behaviour: subset-gated rules are warnings, not errors.

        ``icd10_common`` is a 32-code sample; HC-C005 is named "ICD-10 code is in
        the documented common set" and is warning severity, so an unlisted but
        perfectly real ICD-10-CM code does not fail the frame.
        """
        assert "NOT the full" in load_reference("ucum_common").meta["disclaimer"]
        df = pd.DataFrame({
            "condition_id": ["C1", "C2"],
            "patient_id": ["P1", "P2"],
            "clinical_status": ["active", "active"],
            "code_system": ["http://hl7.org/fhir/sid/icd-10-cm"] * 2,
            "code_value": ["E11.9", "S72.001A"],   # second is real but not bundled
            "onset_date": ["2024-01-01", "2024-01-02"],
        })
        _, outcome = run_domain(df, "healthcare", fhir_resource="Condition")
        result = _result(outcome, "HC-C005")
        assert (result.severity, result.violation_rows) == ("warning", [1])
        assert outcome.report.passed is True
        assert outcome.trust_score == 0.875           # 1 - 0.25 * (1/2)

    def test_absent_from_a_disclaimed_subset_is_reported_as_error(self):
        """FINDING (S2): three packs equate "not in our subset" with "invalid".

        ``bundled/uom_codes.json`` says in its own ``_meta``: "Curated subset of
        commonly used unit-of-measure codes for retail net content; not the
        exhaustive UN/CEFACT list" — 50 of roughly 1800 Rec-20 codes. GS1-006 is
        nevertheless named "net_content_uom is a valid UN/CEFACT code" and is
        error severity, so a real Rec-20 code outside the subset sets
        ``report.passed = False`` and drives trust to 0. ``quality_codes.json``
        ("NOT an exhaustive or vendor-specific quality model") and the curated
        agriculture unit sets behave the same way.

        This pins the current dispositions so the divergence from healthcare is
        explicit; the decision this needs is whether subset-gated reference rules
        should be warnings pack-wide, or whether these sets should be completed.
        """
        uom = load_reference("uom_codes")
        assert len(uom.codes) == 50
        assert "not the exhaustive UN/CEFACT list" in uom.meta["note"]

        retail = pd.DataFrame({
            "gtin": ["00012345678905"] * 3,
            "product_description": ["Crude oil", "Vitamin sachet", "Bulk grain"],
            "net_content": [1.0, 50.0, 1.0],
            # BLL (barrel), MC (microgram) and KTN (kilotonne) are UN/CEFACT Rec-20
            # codes that this curated 50-code subset does not carry.
            "net_content_uom": ["BLL", "MC", "KTN"],
        })
        _, outcome = run_domain(retail, "retail")
        result = _result(outcome, "GS1-006")
        assert (result.severity, result.violation_rows) == ("error", [0, 1, 2])
        assert outcome.report.passed is False
        assert outcome.trust_score == 0.0

        energy = pd.DataFrame({
            "timestamp": ["2024-03-15T00:00:00", "2024-03-15T00:00:01"],
            "asset_id": ["RTU-1", "RTU-2"],
            "quality": ["good", "substituted"],   # a vendor sub-status
            "value": [1.0, 2.0],
        })
        _, energy_outcome = run_domain(energy, "energy")
        assert _result(energy_outcome, "ENG-008").severity == "error"
        assert energy_outcome.report.passed is False

    def test_a_value_is_never_rewritten_merely_for_being_outside_a_subset(self):
        """The safety half of the contract does hold, across every ``coerce`` rule.

        AG-007, AG-010 and ENG-008 all declare ``repair: coerce``. When the value
        is simply unknown to the bundled set the repair is logged ``unresolvable``
        and the cell is left exactly as it was — the snapshot never silently
        overwrites data it does not recognise.
        """
        agriculture = pd.DataFrame({
            "field_id": ["F1", "F2"], "area": [1.0, 2.0], "area_unit": ["hectares", "ANN"],
        })
        repaired, outcome = run_domain(agriculture, "agriculture")
        # "hectares" is a declared synonym -> coerced; "ANN" is unknown -> untouched.
        assert repaired["area_unit"].tolist() == ["HAR", "ANN"]
        statuses = {(a.rule_id, a.from_value): a.status for a in outcome.repairs
                    if a.rule_id == "AG-007"}
        assert statuses == {("AG-007", "hectares"): "applied", ("AG-007", "ANN"): "unresolvable"}

        energy = pd.DataFrame({
            "timestamp": ["2024-03-15T00:00:00", "2024-03-15T00:00:01"],
            "asset_id": ["RTU-1", "RTU-2"], "quality": ["g", "substituted"],
            "value": [1.0, 2.0],
        })
        energy_out, _ = run_domain(energy, "energy")
        assert energy_out["quality"].tolist() == ["good", "substituted"]

    def test_pack_reference_sources_reach_the_audit_trail(self, good_frames):
        """Every reference set a pack consults is describable, with its ``_meta``."""
        for name in good_frames:
            validator = get_validator(name, **PACK_KWARGS.get(name, {}))
            sources = validator.reference_sources()
            if not sources:
                continue
            for source in sources:
                assert source.get("name"), f"{name}: unnamed reference source"
                assert source.get("version") or source.get("retrieved_date"), (
                    f"{name}/{source['name']}: reference source carries no version"
                )
            json.dumps(sources)   # audit trail must stay JSON-serialisable


# ----------------------------------------------------------------- 8. versioning


class TestVersioningSurfaced:
    """Rule and schema versions must reach the report, not just the class."""

    def test_report_carries_pack_version_and_schema_version(self, good_frames):
        surfaced = {}
        for name, df in good_frames.items():
            _, outcome = run_domain(df, name, **PACK_KWARGS.get(name, {}))
            report = outcome.report
            surfaced[name] = (report.version, report.schema_version)
            payload = report.to_dict()
            assert payload["version"] == report.version
            assert payload["schema_version"] == report.schema_version
            assert f"v{report.version}" in report.summary()
            assert f"schema {report.schema_version}" in report.summary()
        assert surfaced == {
            "agriculture": ("0.1.0", "adapt-2024"),
            "education": ("0.1.0", "ed-fi-2024"),
            "energy": ("0.1.0", "energy/2025.06"),
            "finance": ("0.1.0", "2024-01"),
            "healthcare": ("0.1.0", "fhir-r4-patient"),
            "media": ("0.1.0", "eidr-ddex-2024"),
            "retail": ("0.1.0", "2024-01"),
            "transport": ("0.1.0", "gtfs-2024"),
        }

    def test_sub_schemas_that_swap_rule_files_get_their_own_schema_version(self):
        """Healthcare per-resource and finance ledger/tick each version separately."""
        healthcare = {
            resource: get_validator("healthcare", fhir_resource=resource).schema_version
            for resource in ("Patient", "Observation", "Encounter", "Condition",
                             "MedicationRequest")
        }
        assert healthcare == {
            "Patient": "fhir-r4-patient",
            "Observation": "fhir-r4-observation",
            "Encounter": "fhir-r4-encounter",
            "Condition": "fhir-r4-condition",
            "MedicationRequest": "fhir-r4-medicationrequest",
        }
        assert get_validator("finance").schema_version == "2024-01"
        assert get_validator("finance", finance_mode="tick").schema_version == \
            "finance-tick/2025.06"

    def test_media_sub_schema_is_not_distinguished_by_schema_version(self):
        """FINDING (S4): ``media_type`` changes the effective rule set but not the version.

        Unlike healthcare and finance, the media pack keeps one ``schema_version``
        for both sub-schemas and filters the shared 24 rules by ``media_type`` at
        run time. ``describe()`` does report ``media_type``; ``ValidationReport``
        does not, so a stored report identifies the sub-schema only indirectly,
        through the MD-C* / MD-R* rule-id prefix.
        """
        content = get_validator("media", media_type="content")
        release = get_validator("media", media_type="release")
        assert content.schema_version == release.schema_version == "eidr-ddex-2024"
        assert content.describe()["media_type"] == "content"
        assert release.describe()["media_type"] == "release"
        assert not hasattr(content.validate(_good_frames()["media"]), "media_type")

    def test_describe_is_uniform_across_packs_except_unresolved_healthcare(self):
        """FINDING (S3): ``describe()`` has no stable shape before resource detection.

        Every other pack answers ``describe()`` with the same eight keys. An
        un-activated healthcare validator answers with a five-key placeholder, so
        a caller building an audit trail from ``describe()["schema_version"]``
        raises ``KeyError`` for healthcare and only for healthcare.
        """
        common = {"domain", "version", "schema_version", "canonical_fields",
                  "required_fields", "id_fields", "rules", "reference_sources"}
        for name in available():
            if name == "healthcare":
                continue
            assert common <= set(get_validator(name).describe()), name

        unresolved = get_validator("healthcare").describe()
        assert set(unresolved) == {"domain", "version", "fhir_resource",
                                   "supported_resources", "note"}
        assert "schema_version" not in unresolved
        # Once the resource is known the shape matches every other pack.
        assert common <= set(get_validator("healthcare",
                                           fhir_resource="Patient").describe())

    def test_every_rule_file_declares_a_version_and_schema_version(self):
        """The versions the engine reports are the ones written in the rule files."""
        yaml = pytest.importorskip("yaml")
        rule_files = sorted(DOMAINS_DIR.glob("*/rules*.yaml")) + \
            sorted(DOMAINS_DIR.glob("*/rules/*.yaml"))
        assert len(rule_files) >= 10
        for path in rule_files:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert data.get("version"), f"{path.name}: no version"
            assert data.get("schema_version"), f"{path.name}: no schema_version"
            assert data.get("rules"), f"{path.name}: no rules"


# -------------------------------------------------------------- 9. non-mutation


def _dirty_frames() -> dict[str, tuple[pd.DataFrame, dict[str, str]]]:
    """One frame per pack that actually trips repairs (coerce, flag, fill)."""
    return {
        "finance": (pd.DataFrame({
            "transaction_id": ["T1", "T1", "T2"],
            "date": ["15/04/2024", "2024-01-15", "2024-13-99"],
            "account_code": ["1000", "2000", "X"],
            "debit": [100.0, 0.0, 5.005], "credit": [0.0, 100.0, 0.0],
            "currency": ["USD", "ZZZ", "USD"],
        }), {}),
        "retail": (pd.DataFrame({
            "gtin": ["0-001-2345-6789", "1234567", None],
            "product_description": ["a", "b", "X" * 201],
            "net_content": [1.0, None, 2.0],
            "net_content_uom": ["LTR", "ZZZ", None],
            "country_of_origin": ["DE", "ZZ", "US"],
            "gpc_brick_code": ["10000266", "123", None],
        }), {}),
        "agriculture": (pd.DataFrame({
            "field_id": ["F1", None], "operation_type": ["Planting", "Flying"],
            "operation_date": ["2024-04-15", "not-a-date"], "area": [50.0, None],
            "area_unit": ["hectares", "ZZZ"], "soil_ph": [6.8, 15.0],
            "soil_om_pct": [3.2, 25.0], "season_year": [2024, 2024],
        }), {}),
        "energy": (pd.DataFrame({
            "timestamp": ["2024-03-15T00:00:00", "2099-01-01T00:00:00"],
            "asset_id": ["A", None], "register_address": [1, 70000],
            "function_code": [3, 99], "value": [1.0, 2.0],
            "quality": ["g", "weird"], "unit": ["V", "ZZZ"],
        }), {}),
        "healthcare": (pd.DataFrame({
            "patient_id": ["P1", None], "birth_date": ["1985-06-15", "2090-01-01"],
            "gender": ["MALE", "apache"], "deceased": [False, False],
            "address_country": ["US", "ZZ"],
        }), {"fhir_resource": "Patient"}),
        "education": (pd.DataFrame({
            "student_unique_id": ["S1", None], "school_id": ["SCH1", "SCH2"],
            "school_year": [2024, 1850], "grade_level": ["Eighth grade", "Nope"],
            "enrollment_date": ["2023-09-05", "bad"],
        }), {}),
        "media": (pd.DataFrame({
            "eidr_id": ["10.5240/XXXX-XXXX", None], "content_type": ["Movie", "Season"],
            "release_date": ["not-a-date", "2022"], "country_of_origin": ["ZZ", "US"],
            "language": ["xx", "en"], "runtime_seconds": [8820, -3],
        }), {"media_type": "content"}),
        "transport": (pd.DataFrame({
            "stop_id": ["S1", "S1", None], "stop_lat": [200.0, 51.5, 40.7],
            "stop_lon": [-0.1, -0.1, -74.0],
        }), {"gtfs_file": "stops"}),
    }


class TestRunDomainNeverMutatesInput:
    """``run_domain`` validates, then repairs a copy. Verified by digest."""

    @pytest.mark.parametrize("pack", sorted(_dirty_frames()))
    def test_input_frame_is_untouched_on_a_unique_index(self, pack):
        df, kwargs = _dirty_frames()[pack]
        before = _digest(df)
        repaired, outcome = run_domain(df, pack, **kwargs)
        assert _digest(df) == before
        assert repaired is not df
        assert len(outcome.repairs) > 0, f"{pack}: frame triggered no repair at all"

    @pytest.mark.parametrize("pack", sorted(_dirty_frames()))
    def test_input_frame_is_untouched_on_a_non_unique_index(self, pack):
        """``run_domain`` re-indexes internally for duplicate labels; the input must
        not see that RangeIndex, and the output must carry the original labels back."""
        df, kwargs = _dirty_frames()[pack]
        df = df.copy()
        df.index = ["dup"] * len(df)
        before = _digest(df)
        repaired, _ = run_domain(df, pack, **kwargs)
        assert _digest(df) == before
        assert list(df.index) == ["dup"] * len(df)
        assert list(repaired.index) == ["dup"] * len(repaired)

    def test_validate_alone_never_mutates_and_repair_writes_only_its_copy(self):
        df, _ = _dirty_frames()["finance"]
        validator = get_validator("finance")
        before = _digest(df)
        report = validator.validate(df)
        assert _digest(df) == before
        repaired, log = validator.repair(df, report)
        assert _digest(df) == before
        assert repaired["date"].tolist()[0] == "2024-04-15"   # the copy did change
        assert len(log.applied) == 1

    def test_reported_row_labels_are_translated_back_but_stay_ambiguous(self):
        """Row labels come back in the caller's own index space, duplicates and all.

        OBSERVATION (S4): because ``run_domain`` re-indexes a non-unique frame to a
        RangeIndex and then maps findings back to the *labels*, a duplicated label
        identifies a group of rows rather than the row that was flagged. Here rows
        0 and 2 are flagged and come back as ``["a", "b"]``, but ``df.loc["a"]``
        covers rows 0 and 1. Positional reporting would be unambiguous; the current
        contract is label-based, which is what this pins.
        """
        df, _ = _dirty_frames()["finance"]
        df = df.copy()
        df.index = ["a", "a", "b"]
        _, outcome = run_domain(df, "finance")
        flagged = _result(outcome, "FIN-003").violation_rows
        assert flagged == ["a", "b"]                       # positions 0 and 2
        assert set(flagged) <= set(df.index)               # always valid labels
        assert len(df.loc[flagged]) == 3                   # ...but 2 labels select 3 rows

        # With a unique index the mapping is exact, which is the normal path.
        unique = df.copy()
        unique.index = ["a", "b", "c"]
        _, exact = run_domain(unique, "finance")
        rows = _result(exact, "FIN-003").violation_rows
        assert rows == ["a", "c"]
        assert unique.loc[rows, "date"].tolist() == ["15/04/2024", "2024-13-99"]


# ------------------------------------------- cross-cutting engine-level findings


class TestSharedCheckEngine:
    """Defects that live in ``ConfigDrivenValidator``, so they hit several packs."""

    def test_regex_check_reads_an_integral_float_column_correctly(self):
        """A float64 code column must not raise false findings.

        This test was written to pin the defect. A numeric code column loaded
        from CSV becomes float64 as soon as one cell is blank, and
        ``Series.astype("string")`` then rendered ``10000266`` as
        ``"10000266.0"``, which no digit-only pattern can match. Both regex
        rules were affected: GS1-008 (``[0-9]{8}``) and FIN-008
        (``[A-Za-z0-9]{4,12}``).

        The repo had already settled the intended behaviour for GTIN in
        ``retail/validator.py::_integral_float_text``, whose docstring describes
        the same CSV-blank-cell scenario; the shared engine simply did not apply
        it. The helper now lives in ``domains.base`` and every regex rule uses
        it, so the assertions below are the repaired behaviour.
        """
        as_object = pd.DataFrame({
            "gtin": ["00012345678905", "00012345678905"],
            "product_description": ["a", "b"],
            "gpc_brick_code": ["10000266", "10001336"],
        })
        _, clean = run_domain(as_object, "retail")
        assert _result(clean, "GS1-008").status == "passed"

        with_blank_cell = pd.DataFrame({
            "gtin": ["00012345678905", "00012345678905"],
            "product_description": ["a", "b"],
            "gpc_brick_code": [10000266, None],     # identical code, float64 column
        })
        assert with_blank_cell["gpc_brick_code"].dtype == "float64"
        _, floaty = run_domain(with_blank_cell, "retail")
        assert _result(floaty, "GS1-008").violation_rows == []
        assert floaty.trust_score == clean.trust_score, (
            "the same codes must score the same whatever the column dtype"
        )

        # The same engine path, the same false positive, in the finance pack.
        ledger = pd.DataFrame({
            "transaction_id": ["T1", "T2"], "date": ["2024-01-15", "2024-01-15"],
            "debit": [0.0, 0.0], "credit": [0.0, 0.0], "currency": ["USD", "USD"],
            "account_code": [1000, None],
        })
        assert ledger["account_code"].dtype == "float64"
        _, finance = run_domain(ledger, "finance")
        assert _result(finance, "FIN-008").violation_rows == []
        # ...and it does not happen when the column stays an integer.
        ledger["account_code"] = [1000, 2000]
        _, integral = run_domain(ledger, "finance")
        assert _result(integral, "FIN-008").status == "passed"

    def test_unknown_check_and_unknown_custom_func_raise_domain_error(self):
        class Broken(ConfigDrivenValidator):
            domain_name = "broken"
            canonical_fields = ("a",)

        validator = Broken()
        mapping = validator.detect_columns(pd.DataFrame({"a": [1]}))
        bogus = Rule(id="X-1", name="x", layer="format", severity="error",
                     fields=("a",), check="nonsuch")
        with pytest.raises(DomainError, match="unknown check 'nonsuch'"):
            validator._dispatch_check(pd.DataFrame({"a": [1]}), mapping, bogus)
        missing_fn = Rule(id="X-2", name="x", layer="format", severity="error",
                          fields=("a",), check="custom", params={"func": "nope"})
        with pytest.raises(DomainError, match="unknown custom check 'nope'"):
            validator._dispatch_check(pd.DataFrame({"a": [1]}), mapping, missing_fn)

    def test_rule_ids_are_unique_within_and_across_every_pack(self, good_frames):
        seen: dict[str, str] = {}
        for name in good_frames:
            validator = get_validator(name, **PACK_KWARGS.get(name, {}))
            validator.detect_columns(good_frames[name])
            ids = [rule.id for rule in validator.rules]
            assert len(ids) == len(set(ids)), f"{name}: duplicate rule id"
            for rule_id in ids:
                assert rule_id not in seen, f"{rule_id} used by {seen[rule_id]} and {name}"
                seen[rule_id] = name
