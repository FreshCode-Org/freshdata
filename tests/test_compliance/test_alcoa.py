"""ALCOA+ attestation generator."""

from __future__ import annotations

from freshdata.compliance import ComplianceConfig, generate_compliance_report

_PRINCIPLES = {
    "Attributable",
    "Legible",
    "Contemporaneous",
    "Original",
    "Accurate",
    "Complete",
    "Consistent",
    "Enduring",
    "Available",
}


def _alcoa(report, **kw):
    return generate_compliance_report(report, ["alcoa_plus"], **kw)["alcoa_plus"]


def test_all_nine_principles_present(sample_report):
    alcoa = _alcoa(sample_report)
    assert set(alcoa.data["principles"]) == _PRINCIPLES


def test_legible_fails_on_empty_rationale(make_report):
    report = make_report({"step": "fix_dtypes", "column": "x", "count": 2, "rationale": ""})
    alcoa = _alcoa(report)
    assert alcoa.data["principles"]["Legible"]["passed"] is False
    assert any("Legible" in w for w in alcoa.warnings)


def test_mean_confidence_is_computed(make_report):
    report = make_report(
        {"step": "missing", "column": "a", "count": 0, "confidence": 0.8},
        {"step": "missing", "column": "b", "count": 0, "confidence": 0.6},
    )
    alcoa = _alcoa(report)
    assert alcoa.data["principles"]["Accurate"]["evidence"]["mean_confidence"] == 0.7


def test_passed_true_on_engine_report(sample_report):
    # Engine missing-decisions carry rationale and there are no hard failures.
    alcoa = _alcoa(sample_report)
    assert alcoa.passed is True


def test_attributable_errors_without_actor(make_report):
    report = make_report({"step": "missing", "column": "a", "count": 0})
    alcoa = _alcoa(report, config=ComplianceConfig(system_actor=""))
    assert alcoa.passed is False
    assert alcoa.errors
    assert alcoa.data["principles"]["Attributable"]["passed"] is False
