---
title: Format parsers, reference data & finance tick mode
description: >-
  Parse HL7 v2, GPX, SDMX, and EDIFACT into DataFrames with fd.parse_domain; validate
  market tick data with finance_mode="tick"; and load bundled reference code sets
  (ISO-4217, UCUM, UN/LOCODE) through the centralized reference layer.
keywords: HL7 v2 parser, GPX parser, SDMX parser, EDIFACT parser, market tick data validation, BCBS 239, SOX, ISO 4217, UCUM, UN/LOCODE, reference data
---

# Format parsers, reference data & finance tick mode

This page covers three additions that turn freshdata from a DataFrame cleaner into a
front-to-back pipeline: **format parsers** (raw message → DataFrame), the **reference
layer** (one cached way to load code sets), and **finance tick mode** (market/trade-data
validation).

## Format parsers

A parser performs **structural** parsing only — it reads a wire/file format and returns
one or more pandas DataFrames plus an auditable `ParseResult`. Cleaning and
domain-validation are a separate step (`fd.clean`), so parsing and rules stay decoupled.

| Format | `format=` | Frames | Suggested domain |
|---|---|---|---|
| HL7 v2 ER7 | `hl7v2` | `patient`, `encounter`, `observation` | `healthcare` |
| GPX | `gpx` | `waypoints`, `route_points`, `track_points` | `transport` |
| SDMX-ML | `sdmx` | `observations` | — (audit-only) |
| UN/EDIFACT | `edifact` | `segments` | — |

```python
import freshdata as fd

result = fd.parse_domain(hl7_message, format="hl7v2")
result.frames["observation"]          # one row per OBX
result.warnings                       # audit notes for anything skipped
result.to_dict()                      # JSON-friendly summary

# Parse a file, then clean a chosen frame with a domain in one call:
patients = fd.clean_domain_file(
    "admit.hl7", format="hl7v2", domain="healthcare",
    frame="patient", fhir_resource="Patient",
)
```

`fd.parse_domain` accepts a **path, raw text/bytes, or a file-like object**. Malformed
input is recorded in `ParseResult.warnings` rather than raising, so a partial message is
still usable.

!!! note "Honest scope"
    Parsers are structural readers for the common parts of each format — HL7 MSH/PID/PV1/OBX,
    GPX waypoints/routes/tracks, SDMX generic & structure-specific data, EDIFACT
    segments/elements with `UNA` delimiters and the release character. They are **not**
    full conformance engines. SDMX is **audit-only**: an unrecognized layout yields a
    warning and an empty frame. HL7 observation code systems are mapped to URIs
    (LOINC `http://loinc.org`, SNOMED `http://snomed.info/sct`, ICD-10).

Third-party parsers can register through the `freshdata.parsers` entry-point group, or at
runtime with `freshdata.parsers.register("myfmt", MyParser)`.

## Reference-data layer

`freshdata.domains.reference` gives every pack and parser one cached, normalizer-aware way
to load a bundled reference code set, instead of each caller re-implementing case folding
and synonym coercion.

```python
from freshdata.domains.reference import load_reference, available_references

available_references()                      # ['iso3166', 'iso4217', 'ucum_common', ...]
cur = load_reference("iso4217", normalizer="upper")
cur.contains("usd")                          # True (normalized)
cur.invalid_mask(series)                     # boolean mask of unknown codes
cur.coerce(series)                           # map synonyms/case to canonical
```

Bundled sets include ISO-4217 (currencies), ISO-3166 (countries), UN/CEFACT units,
**UCUM** (case-sensitive — load with `normalizer="exact"`), and a **UN/LOCODE** sample.
Every set carries a `_meta` block (version / source / **disclaimer**): these are
*documented common subsets*, not exhaustive code systems.

## Finance tick mode

Pass `finance_mode="tick"` to validate market **tick / trade data** instead of the default
double-entry ledger model:

```python
out, report = fd.clean(ticks_df, domain="finance", finance_mode="tick", return_report=True)
```

The tick model is one row per `(symbol, timestamp, price, size)` tick, with optional
`exchange`, `bid`, `ask`, and `currency`. Rules (FIN-T-001…010) cover:

- ISO-8601, non-future `timestamp`; `symbol` present (an ID — never imputed);
- `price` and `size` strictly positive;
- `currency` ∈ ISO-4217 (validated through the **reference layer**, case-insensitively);
- `bid <= ask` (no crossed quotes);
- duplicate-tick detection on `(symbol, timestamp, price, size)`;
- **BCBS-239 / SOX-style completeness**: the `currency`/`exchange` control dimensions
  needed to roll up control totals are flagged when missing (audit-only, info severity).

Symbol and exchange are IDs and are never imputed; bad ticks are flagged for audit, never
silently dropped. The default `finance_mode="ledger"` is unchanged.
