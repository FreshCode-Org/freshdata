---
title: Compliance reports
description: >-
  Map a freshdata CleanReport onto regulatory control frameworks — 21 CFR Part 11,
  GDPR, ALCOA+, SOX-404, and HIPAA Safe Harbor — and emit standards-grade audit
  artifacts.
keywords: compliance report python, 21 cfr part 11 audit trail, gdpr article 30, hipaa safe harbor, sox 404 data controls, alcoa+
---

# Compliance reports

The `freshdata.compliance` subpackage turns a [`CleanReport`](cleaning-engine.md)
into a regulatory audit artifact. Each *generator* maps the transformations
freshdata applied — imputations, outlier handling, normalisations, deduplication,
PII masking — onto a named control framework and emits a structured report you can
attach to a data-governance workflow.

The generators are **purely additive**: they never mutate the input report or
DataFrame. They read what freshdata already recorded and re-express it against a
framework's controls.

!!! warning "Report generation, not certification"
    A compliance artifact **supports** a compliance workflow; it does not
    **constitute** a legal determination of compliance. Every report carries a
    verbatim caveat ([`GENERAL_CAVEAT`](#caveat)) to that effect, and review by a
    qualified compliance professional is required.

## Quickstart

```python
import freshdata as fd
from freshdata.compliance import generate_compliance_report, ComplianceConfig

cleaned, report = fd.clean(df, return_report=True)

bundle = generate_compliance_report(
    report,
    frameworks=["21cfr_11", "hipaa_safe_harbor"],
    config=ComplianceConfig(operator_id="svc-1", retention_days=2555),
    dataframe=df,            # optional: recovers column roles + missing ratios
)

bundle.summary()
# {'21cfr_11': {'passed': True, 'warnings': [], 'errors': []},
#  'hipaa_safe_harbor': {'passed': True, 'warnings': [...], 'errors': []}}

bundle["hipaa_safe_harbor"].passed     # -> True / False
print(bundle.to_json())                # full audit artifact as JSON
```

`generate_compliance_report` also accepts an enterprise result directly:
`generate_compliance_report(enterprise_result, frameworks=[...])` (or
`report` exposing `clean_report` + `trust_after`) folds in the embedded report
and trust/mask data automatically.

## Frameworks

Pass one or more of these keys in `frameworks=`:

| Key | Framework | What it documents |
| --- | --- | --- |
| `21cfr_11` | 21 CFR §11.10(e) | A tamper-evident audit trail: one entry per action, each marked non-obscuring (a pre-image was retained or the change cannot hide prior data). |
| `gdpr_30` | GDPR Article 30 + Article 17 | Record of processing activities plus erasure (right-to-be-forgotten) evidence. |
| `alcoa_plus` | ALCOA+ | Data-integrity attributes (Attributable, Legible, Contemporaneous, Original, Accurate, +). |
| `sox_404` | SOX-404 transformation control | Internal-control evidence over data transformations, gated on the Data Trust Score. |
| `hipaa_safe_harbor` | HIPAA Safe Harbor | Coverage of the 18 Safe Harbor identifiers, flagging any that remain unmasked. |

Unknown keys raise `ValueError` listing the valid keys.

## Configuration

`ComplianceConfig` carries the caller-supplied context; every field has a sensible
default, so `ComplianceConfig()` is valid. The most useful knobs:

| Field | Default | Purpose |
| --- | --- | --- |
| `operator_id` | `None` (`"system"`) | Operator recorded on each 21 CFR audit entry. |
| `retention_days` | `2555` (7 years) | Retention period stamped on audit entries. |
| `masked_columns` | `[]` | Columns known to be PII-masked (report-only path; merged with any enterprise mask report). |
| `trust_score` | `None` | 0–100 trust score for the SOX / 21 CFR gates when no enterprise result is supplied. |
| `fail_on_hipaa_gap` | `False` | Raise [`ComplianceGapError`](#errors) instead of warning when Safe Harbor gaps remain. |
| `strict_cfr_normalization` | `False` | See below. |
| `controller_name` / `controller_contact` / `processing_purpose` / `legal_basis` / `data_subject_categories` | — | GDPR Article 30 record fields. |

### `strict_cfr_normalization`

Controls how the 21 CFR audit classifies *normalising rewrites* (whitespace
trims, sentinel canonicalisation):

- **`False` (default)** — these lossless normalisations are treated as
  **non-obscuring**: trimming `"  Ann "` to `"Ann"` cannot hide prior information,
  so the gate stays green. Only genuine value rewrites with no retained pre-image
  (outlier capping, fuzzy clustering) fail the gate.
- **`True`** — any rewrite that did not retain a pre-image, *including*
  normalisation, is treated as **obscuring**: the entry's
  `original_value_class` becomes `"not_captured"`, `non_obscuring_guarantee`
  becomes `False`, a warning is emitted, and the gate fails. Use this when your SOP
  requires an independently recorded pre-image for every value change.

## Richer evidence

Two optional, keyword-only arguments add evidence when available:

- **`dataframe=`** — the source frame. Recovers per-column roles and missing
  ratios via [`freshdata.infer_roles`](api-reference.md), sharpening the HIPAA and
  ALCOA reports.
- **`enterprise_result=`** — an [enterprise](feature-overview.md) result supplying
  the 0–100 Data Trust Score, PII-masking events, and fuzzy-clustering lineage.

## Output

`generate_compliance_report` returns a `ComplianceBundle` — a mapping of framework
key to `FrameworkReport`:

| On `ComplianceBundle` | Returns |
| --- | --- |
| `bundle.summary()` | `{key: {"passed", "warnings", "errors"}}` for a quick gate check. |
| `bundle[key]` | The `FrameworkReport` for one framework. |
| `key in bundle`, `len(bundle)`, `list(bundle)` | Membership, count, and the framework keys. |
| `bundle.to_dict()` / `bundle.to_json(indent=2)` | The full artifact as a dict / JSON string. |
| `bundle.to_frame()` | All reports flattened into a single `pandas.DataFrame`. |

Each `FrameworkReport` exposes `framework_key`, `framework_name`, `passed` (bool),
`warnings`, `errors`, and `data` (the framework-specific payload, e.g. the 21 CFR
`audit_entries` or the HIPAA identifier coverage), plus its own `to_dict()`,
`to_json()`, and `to_frame()`.

## Errors {#errors}

- `ValueError` — an unknown framework key was requested.
- `ComplianceGapError` — HIPAA Safe Harbor gaps remain **and**
  `config.fail_on_hipaa_gap=True`. Otherwise gaps surface as warnings on the
  report and `passed` is `False`.

## Caveat {#caveat}

Every report embeds `GENERAL_CAVEAT` (importable from `freshdata.compliance`)
unless a framework defines its own verbatim caveat. It states plainly that the
artifact documents freshdata's transformations and their mapping to the named
control framework, but is not a certified compliance system and does not
constitute a legal determination — professional review is required.
