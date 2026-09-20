"""Boundary tests for the confidence -> risk mapping and the audit feature record.

Mutation testing of ``semantic/scoring.py`` returned a **35% kill rate**: the
risk-tier thresholds could be moved (``0.70`` -> ``0.75``, ``0.85`` -> ``0.90``)
and the comparisons loosened (``>=`` -> ``>``) without a single test noticing.

That matters because ``risk_for`` decides what gets applied. Under
``semantic_mode="auto"`` the policy gate applies a proposal only when
``risk != "high"``, so moving a boundary silently changes which repairs run
without review. These tests pin each boundary at the value, just below it and
just above it.

They also record the shape of the mapping, which was previously only implicit:

    unsafe_ambiguous          -> always high, whatever the confidence
    confidence < 0.70         -> high
    higher-risk issue types   -> medium at >= 0.85, else high
    everything else           -> low at >= 0.90, else medium

With these tests the module's mutation kill rate over **all 58 mutation
sites** goes from 53.4% to 98.3% (31 -> 57 killed). The single survivor is
proven equivalent in the last test in this file. Harness and logs live
outside the repo, under ``freshdata-program-state/``.
"""

from __future__ import annotations

import bisect
import json

import pytest

from freshdata.semantic import scoring
from freshdata.semantic.scoring import (
    _HIGHER_RISK_ISSUES,
    _IsotonicTable,
    calibrate_proposals,
    calibration_features,
    confidence_from_evidence,
    features_hash,
    make_proposal,
    risk_for,
)
from freshdata.semantic.types import (
    SemanticColumnInfo,
    SemanticContext,
    SemanticEvidence,
)

ORDINARY = "spelled_number"
HIGHER = "category_synonym"

_CTX = SemanticContext(
    dataset=None,
    columns={},
    auto_threshold=0.95,
    review_threshold=0.70,
    max_distinct_values=50,
    sample_size=1000,
    privacy_policy="strict",
    mode="review",
    id_columns=frozenset(),
    preserve_columns=frozenset(),
    target_column=None,
)


def test_unsafe_ambiguous_is_always_high_whatever_the_confidence():
    for confidence in (0.0, 0.5, 0.7, 0.9, 0.999, 1.0):
        assert risk_for("unsafe_ambiguous", confidence) == "high"


# -- the 0.70 floor ---------------------------------------------------------


@pytest.mark.parametrize("issue", [ORDINARY, HIGHER])
def test_below_the_review_floor_is_high(issue):
    """Kills the `0.70 -> 0.75` and `< -> <=` mutants."""
    assert risk_for(issue, 0.69) == "high"
    assert risk_for(issue, 0.6999) == "high"


def test_exactly_at_the_review_floor_is_not_high_for_an_ordinary_issue():
    """0.70 is *inside* the allowed band: `<` must not become `<=`."""
    assert risk_for(ORDINARY, 0.70) == "medium"


# -- the 0.85 boundary for higher-risk issue types --------------------------


def test_higher_risk_issue_at_its_boundary_is_medium():
    """Kills `>= 0.85 -> > 0.85` and `0.85 -> 0.90`."""
    assert risk_for(HIGHER, 0.85) == "medium"


def test_higher_risk_issue_just_below_its_boundary_is_high():
    assert risk_for(HIGHER, 0.8499) == "high"


def test_a_higher_risk_issue_never_reaches_low():
    """Even at the ceiling it stays medium -- that is the point of the set."""
    assert risk_for(HIGHER, 0.999) == "medium"


@pytest.mark.parametrize("issue", sorted(_HIGHER_RISK_ISSUES - {"unsafe_ambiguous"}))
def test_every_declared_higher_risk_type_follows_the_same_rule(issue):
    assert risk_for(issue, 0.85) == "medium"
    assert risk_for(issue, 0.8499) == "high"
    assert risk_for(issue, 0.999) == "medium"


# -- the 0.90 boundary for ordinary issue types -----------------------------


def test_an_ordinary_issue_at_its_boundary_is_low():
    """Kills `>= 0.90 -> > 0.90`."""
    assert risk_for(ORDINARY, 0.90) == "low"


def test_an_ordinary_issue_just_below_its_boundary_is_medium():
    assert risk_for(ORDINARY, 0.8999) == "medium"


def test_the_ordinary_band_between_the_two_thresholds_is_medium():
    for confidence in (0.70, 0.80, 0.85, 0.8999):
        assert risk_for(ORDINARY, confidence) == "medium"


# -- the confidence clamp ---------------------------------------------------


def test_the_floor_is_zero_not_a_positive_number():
    """Kills `max(0.0, ...) -> max(0.05, ...)`.

    A floor above zero would quietly lift every hopeless proposal off the
    bottom of the scale.
    """
    ev = (SemanticEvidence("t", "d", -5.0),)
    assert confidence_from_evidence(0.5, ev) == 0.0


def test_the_ceiling_keeps_semantic_repairs_below_deterministic_certainty():
    """Deterministic representation repairs report 1.0; semantic must not."""
    ev = (SemanticEvidence("t", "d", 5.0),)
    assert confidence_from_evidence(0.9, ev) == 0.999


def test_evidence_weights_are_summed_onto_the_base():
    ev = (SemanticEvidence("a", "d", 0.02), SemanticEvidence("b", "d", -0.01))
    assert confidence_from_evidence(0.90, ev) == pytest.approx(0.91)


def test_an_empty_evidence_tuple_leaves_the_base_untouched():
    assert confidence_from_evidence(0.83, ()) == pytest.approx(0.83)


# ---------------------------------------------------------------------------
# The audit feature record
#
# ``calibration_features`` is hashed into ``ActionConfidence.features_hash`` so
# that, per its docstring, "a report consumer can verify two actions were
# scored from identical evidence". A silently wrong feature breaks exactly that
# promise: two actions scored from *different* evidence would hash the same.
#
# Mutation testing left seven survivors in this function -- every extracted
# constant (the evidence-string split indices, the role-confidence pair, the
# coverage rounding, the memory-support increment) could be changed without a
# test noticing. The tests below pin each one.
# ---------------------------------------------------------------------------


def _col(**kw):
    """A ``SemanticColumnInfo`` with the non-defaulted fields filled in."""
    fields = {
        "name": "amount",
        "role": "numeric",
        "n_nonnull": 100,
        "nunique": 12,
        "high_cardinality": False,
        "preserve": False,
        "free_text": False,
        "numeric_like": True,
        "boolean_like": False,
        "money_like": False,
        "unit_like": False,
        "identifier_like": False,
    }
    fields.update(kw)
    return SemanticColumnInfo(**fields)


def _prop(*, evidence=(), count=10, confidence=0.9, backend="deterministic"):
    return make_proposal(
        backend=backend,
        column="amount",
        raw_value="x",
        proposed_value="y",
        issue_type=ORDINARY,
        expert="test",
        base_confidence=confidence,
        evidence=evidence,
        count=count,
        rationale="because",
    )


def _features(proposal, info):
    return calibration_features(proposal, _CTX, info)


def test_confidence_is_recorded_to_four_decimal_places():
    # 0.912345 + 0.0 stays 0.912345 before rounding; the stored value is the
    # rounded one, and the number of places is part of the report contract.
    proposal = _prop(confidence=0.9123456)
    assert proposal.confidence == 0.9123
    assert str(proposal.confidence) == "0.9123"


def test_semantic_type_confidence_is_read_from_the_evidence_detail():
    ev = (
        SemanticEvidence(
            kind="semantic_type",
            detail="matched currency with confidence=0.83 over 40 values",
        ),
    )
    features = _features(_prop(evidence=ev), _col())
    assert features["semantic_type_confidence"] == 0.83


def test_semantic_type_confidence_reads_the_first_marker_and_its_next_token():
    # Three candidate numbers, each reachable by a different off-by-one:
    # the token after the FIRST marker (0.42) is the feature, not the token
    # after that (0.99) and not the one after a second marker (0.11).
    ev = (
        SemanticEvidence(
            kind="semantic_type",
            detail="confidence=0.42 0.99 confidence=0.11",
        ),
    )
    features = _features(_prop(evidence=ev), _col())
    assert features["semantic_type_confidence"] == 0.42


def test_semantic_type_confidence_is_none_without_the_marker():
    ev = (SemanticEvidence(kind="semantic_type", detail="matched currency"),)
    assert _features(_prop(evidence=ev), _col())["semantic_type_confidence"] is None


def test_margin_is_read_from_the_token_following_its_marker():
    ev = (SemanticEvidence(kind="embedding", detail="top match margin=0.27 vs 0.10"),)
    assert _features(_prop(evidence=ev), _col())["margin_to_second_candidate"] == 0.27


def test_a_malformed_confidence_marker_degrades_to_none_not_to_a_wrong_number():
    ev = (SemanticEvidence(kind="semantic_type", detail="confidence=high"),)
    assert _features(_prop(evidence=ev), _col())["semantic_type_confidence"] is None


def test_role_confidence_is_one_when_a_role_is_known_and_zero_when_it_is_not():
    assert _features(_prop(), _col(role="numeric"))["role_confidence"] == 1.0
    assert _features(_prop(), _col(role=""))["role_confidence"] == 0.0
    assert _features(_prop(), None)["role_confidence"] == 0.0


def test_coverage_is_the_proposal_share_of_non_null_values_to_six_places():
    # 7 / 3000 = 0.00233333... -- a rounding place shallower than six would
    # collapse distinguishable coverages onto the same feature value.
    features = _features(_prop(count=7), _col(n_nonnull=3000))
    assert features["coverage"] == 0.002333

    coarser = _features(_prop(count=8), _col(n_nonnull=3000))
    assert coarser["coverage"] == 0.002667
    assert coarser["coverage"] != features["coverage"]


def test_coverage_is_none_when_there_are_no_non_null_values():
    assert _features(_prop(), _col(n_nonnull=0))["coverage"] is None
    assert _features(_prop(), None)["coverage"] is None


def test_memory_support_counts_one_per_replay_evidence_item():
    def n(count):
        ev = tuple(
            SemanticEvidence(kind="memory_replay", detail=f"seen {i}") for i in range(count)
        )
        return _features(_prop(evidence=ev), _col())["memory_support_count"]

    assert n(0) == 0
    assert n(1) == 1
    assert n(3) == 3


def test_memory_support_ignores_evidence_of_other_kinds():
    ev = (
        SemanticEvidence(kind="memory_replay", detail="seen before"),
        SemanticEvidence(kind="pattern", detail="looks numeric"),
        SemanticEvidence(kind="column_role", detail="numeric column"),
    )
    assert _features(_prop(evidence=ev), _col())["memory_support_count"] == 1


def test_two_proposals_scored_from_different_evidence_do_not_hash_alike():
    """The property the feature record exists to provide."""
    weak = (SemanticEvidence(kind="semantic_type", detail="confidence=0.20"),)
    strong = (SemanticEvidence(kind="semantic_type", detail="confidence=0.95"),)
    info = _col()
    assert features_hash(_features(_prop(evidence=weak), info)) != features_hash(
        _features(_prop(evidence=strong), info)
    )


def test_the_same_evidence_hashes_identically_across_calls():
    ev = (SemanticEvidence(kind="semantic_type", detail="confidence=0.55"),)
    info = _col()
    first = features_hash(_features(_prop(evidence=ev), info))
    second = features_hash(_features(_prop(evidence=ev), info))
    assert first == second


def test_a_confidence_marker_in_the_wrong_evidence_kind_is_ignored():
    # The guard is `kind == "semantic_type" AND "confidence=" in detail`. A
    # pattern note that happens to contain the marker must not be mistaken for
    # the semantic-type score.
    ev = (SemanticEvidence(kind="pattern", detail="matched with confidence=0.99"),)
    assert _features(_prop(evidence=ev), _col())["semantic_type_confidence"] is None


def test_a_margin_marker_in_the_wrong_evidence_kind_is_ignored():
    ev = (SemanticEvidence(kind="pattern", detail="margin=0.99 between patterns"),)
    assert _features(_prop(evidence=ev), _col())["margin_to_second_candidate"] is None


# ---------------------------------------------------------------------------
# The isotonic calibration table
#
# Calibration ships as identity -- without an installed table the version
# string is literally "uncalibrated" -- so none of this code runs on a default
# install and none of it was covered. It is still shipped code that decides
# confidences for anyone who installs a table, and the confidence it produces
# is what the policy gate reads. These tests install a table explicitly.
# ---------------------------------------------------------------------------


def _table(curves, version="calib-test-1"):
    """Build a table through ``from_json`` so the curve is a *valid* one."""
    return _IsotonicTable.from_json(json.dumps({"version": version, "tables": curves}))


def _install(monkeypatch, table):
    monkeypatch.setattr(scoring, "_load_calibration_table", lambda: table)


class _Report:
    def __init__(self):
        self.warnings = []

    def add_warning(self, message):
        self.warnings.append(message)


def _calibrated(monkeypatch, table, *, backend="deterministic", confidence=0.5, report=None):
    _install(monkeypatch, table)
    proposal = _prop(confidence=confidence, backend=backend)
    out = calibrate_proposals([proposal], None, _CTX, report)
    return out[0]


def test_a_value_at_the_bottom_of_the_grid_clamps_instead_of_interpolating():
    # A non-decreasing curve may repeat its first x -- ``from_json`` accepts
    # it. At exactly that x the clamp must win; interpolating instead would
    # step to the *second* knot and return a different y.
    table = _table({"e": {"*": {"x": [0.2, 0.2, 0.6], "y": [0.1, 0.3, 0.5]}}})
    assert table.apply("e", "any", 0.2) == 0.1
    assert table.apply("e", "any", 0.05) == 0.1


def test_a_value_at_the_top_of_the_grid_clamps_instead_of_running_off_the_end():
    table = _table({"e": {"*": {"x": [0.2, 0.6], "y": [0.1, 0.5]}}})
    assert table.apply("e", "any", 0.6) == 0.5
    assert table.apply("e", "any", 0.95) == 0.5


def test_a_value_between_knots_is_linearly_interpolated():
    table = _table({"e": {"*": {"x": [0.0, 1.0], "y": [0.0, 0.5]}}})
    assert table.apply("e", "any", 0.4) == pytest.approx(0.2)


def test_a_missing_curve_passes_the_raw_score_through():
    table = _table({"e": {"category_synonym": {"x": [0.0, 1.0], "y": [0.0, 0.5]}}})
    assert table.curve("other_backend", "category_synonym") is None
    assert table.apply("other_backend", "category_synonym", 0.42) == 0.42


def test_a_curve_must_be_non_decreasing_in_both_axes():
    with pytest.raises(ValueError, match="non-decreasing"):
        _table({"e": {"*": {"x": [0.0, 1.0], "y": [0.5, 0.1]}}})
    with pytest.raises(ValueError, match="non-decreasing"):
        _table({"e": {"*": {"x": [1.0, 0.0], "y": [0.1, 0.5]}}})


def test_a_curve_needs_at_least_two_equal_length_points():
    with pytest.raises(ValueError, match="at least two points"):
        _table({"e": {"*": {"x": [0.5], "y": [0.5]}}})
    with pytest.raises(ValueError, match="equal-length"):
        _table({"e": {"*": {"x": [0.0, 1.0], "y": [0.5]}}})


def test_the_calibrated_point_is_recorded_to_four_decimal_places(monkeypatch):
    table = _table({"deterministic": {"*": {"x": [0.0, 0.7], "y": [0.0, 0.1]}}})
    out = _calibrated(monkeypatch, table, confidence=0.5)
    assert out.confidence == 0.0714  # 0.5/0.7 * 0.1 = 0.0714285...


def test_a_calibrated_point_is_never_pushed_up_off_the_floor(monkeypatch):
    table = _table({"deterministic": {"*": {"x": [0.0, 1.0], "y": [0.0, 0.0]}}})
    out = _calibrated(monkeypatch, table, confidence=0.5)
    assert out.confidence == 0.0


def test_a_calibrated_point_above_one_is_clamped_to_one(monkeypatch):
    # Nothing bounds a curve's y values at 1.0, so the clamp is the only thing
    # keeping confidence inside [0, 1] -- and the gate compares it to 0.95.
    table = _table({"deterministic": {"*": {"x": [0.0, 1.0], "y": [1.2, 1.5]}}})
    out = _calibrated(monkeypatch, table, confidence=0.5)
    assert out.confidence == 1.0


def test_an_installed_table_without_a_matching_curve_stays_uncalibrated(monkeypatch):
    """A table being *present* is not the same as a curve being *found*."""
    table = _table({"deterministic": {"*": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}})
    out = _calibrated(monkeypatch, table, backend="embedding", confidence=0.5)
    assert out.calibration.calibration_version == "uncalibrated"
    assert out.calibration.raw == 0.5


def test_the_missing_table_warning_is_emitted_once_not_once_per_proposal(monkeypatch):
    _install(monkeypatch, None)
    report = _Report()
    proposals = [
        _prop(confidence=0.5, backend="embedding"),
        _prop(confidence=0.6, backend="embedding"),
        _prop(confidence=0.7, backend="embedding"),
    ]
    calibrate_proposals(proposals, None, _CTX, report)
    assert len(report.warnings) == 1
    assert "calibration_version=uncalibrated" in report.warnings[0]


def test_only_the_embedding_backend_warns_about_a_missing_table(monkeypatch):
    # A deterministic proposal is returned untouched; any other non-embedding
    # backend goes uncalibrated but must not raise the model-score warning,
    # which is specifically about raw *model* scores.
    _install(monkeypatch, None)
    report = _Report()
    calibrate_proposals([_prop(confidence=0.5, backend="memory")], None, _CTX, report)
    assert report.warnings == []


def test_a_deterministic_proposal_without_a_curve_is_returned_byte_identical(monkeypatch):
    _install(monkeypatch, None)
    proposal = _prop(confidence=0.5, backend="deterministic")
    (out,) = calibrate_proposals([proposal], None, _CTX, None)
    assert out is proposal


def test_the_zero_span_guard_in_the_interpolator_is_unreachable():
    """Proof that one surviving mutant is equivalent, not a missing test.

    ``_IsotonicTable.apply`` ends with::

        frac = (raw - xs[lo]) / span if span else 0.0

    Mutating that ``0.0`` survives every test, and no test can kill it,
    because ``span`` is never zero there:

    * the line is reached only when ``xs[0] < raw < xs[-1]`` -- both clamps
      above it have already returned;
    * ``hi = bisect_right(xs, raw)`` is the first index with ``xs[hi] > raw``,
      so ``xs[hi] > raw`` *strictly*;
    * ``lo = hi - 1`` and ``xs[lo] <= raw``.

    Therefore ``span = xs[hi] - xs[lo] > 0`` for every curve, including the
    repeated-x curves ``from_json`` accepts. The guard is dead code; the
    mutant is equivalent. Asserted below over the awkward curves rather than
    argued only in prose.
    """
    curves = [
        [0.0, 0.5, 1.0],
        [0.2, 0.2, 0.6],  # repeated first knot
        [0.0, 0.5, 0.5, 1.0],  # repeated interior knot
        [0.0, 0.5, 0.5, 0.5, 1.0],  # thrice-repeated interior knot
        [0.0, 1.0, 1.0],  # repeated last knot
    ]
    for xs in curves:
        for raw in [x + d for x in xs for d in (-0.01, 0.0, 0.01)]:
            if not xs[0] < raw < xs[-1]:
                continue  # clamped before the interpolator is reached
            hi = bisect.bisect_right(xs, raw)
            lo = hi - 1
            assert xs[hi] - xs[lo] > 0, (xs, raw)


def test_the_feature_hash_does_not_depend_on_key_insertion_order():
    """The hash must be canonical, not merely deterministic.

    ``features_hash`` exists so a report consumer can compare two actions'
    evidence. ``json.dumps(..., sort_keys=True)`` is what makes the digest a
    function of the feature *content*; without it the digest also encodes the
    order the dict literal happens to be written in, so reordering that
    literal would silently invalidate every previously published hash while
    still looking perfectly deterministic within any one build.

    Mutation testing caught this: flipping ``sort_keys`` to ``False`` survived
    every test, because each one only ever compared hashes built in one order.
    """
    a = {"raw_score": 0.9, "backend": "deterministic", "risk": "low"}
    b = {"risk": "low", "backend": "deterministic", "raw_score": 0.9}
    assert list(a) != list(b)  # genuinely different insertion order
    assert features_hash(a) == features_hash(b)
