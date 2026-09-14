# Contributing a domain validator pack

A *domain pack* teaches `fd.clean(df, domain="…")` how to validate and repair a
specific kind of tabular data against versioned, reviewable rules. Built-in packs
live in `src/freshdata/domains/<name>/`; third-party packs ship as separate
PyPI distributions and register through an entry point. Both implement the same
interface.

The built-in packs are `finance`, `retail` (GS1), `transport` (GTFS), `healthcare`
(FHIR/US Core), `education` (Ed-Fi), `agriculture` (ADAPT), and `media` (EIDR/DDEX).
`finance` is the simplest end-to-end reference; the four newest packs add the advanced
patterns documented under [Advanced patterns](#advanced-patterns) below.

## The interface

Every pack implements `freshdata.domains.DomainValidator`:

```python
class DomainValidator(ABC):
    domain_name: str        # "finance"
    version: str            # "0.1.0"
    schema_version: str     # "2024-01"

    def detect_columns(self, df) -> ColumnMapping: ...
    def validate(self, df) -> ValidationReport: ...
    def repair(self, df, report) -> tuple[DataFrame, RepairLog]: ...
    def describe(self) -> dict: ...
```

Most packs subclass **`ConfigDrivenValidator`**, which implements the layered
engine for you — you supply a `rules.yaml`, declare canonical fields, and add any
custom checks. The finance pack (`src/freshdata/domains/finance/`) is the
reference example.

## Anatomy of a pack

```
mypack/
  __init__.py        # exports your Validator class
  validator.py       # subclass ConfigDrivenValidator; register custom checks
  rules.yaml         # the rules (see below)
  reference/*.json   # bundled code sets, each with a _meta block
```

### `validator.py`

```python
from freshdata.domains import ConfigDrivenValidator

class MyValidator(ConfigDrivenValidator):
    domain_name = "mydomain"
    version = "0.1.0"
    schema_version = "2024-01"

    canonical_fields = ("id", "amount", "code")
    required_fields = ("id", "amount")
    id_fields = ("id",)                      # never repaired
    aliases = {"id": (r"my_?id", r"identifier")}   # regex, case-insensitive
    rules_path = str(Path(__file__).parent / "rules.yaml")

    def register_extensions(self):
        self.register_check("my_check", self._my_check)        # params.func
        self.register_repair("my_fix", self._my_fix)           # repair_params.func

    def load_reference_values(self, name):                     # for reference checks
        ...
```

### `rules.yaml`

Each rule carries `id`, `name`, `layer` (`schema|format|reference|business|
semantic`), `severity` (`error|warning|info`), `field`/`fields`, `check`
(`not_null|required|regex|enum|reference|range|custom`), optional `params`, and
an optional `repair` (`fill_default|coerce|flag_only|reject|none`) with
`repair_params`. Layers always run in order; a rule whose target column is
absent is skipped (the schema layer reports it as `MISSING_REQUIRED_FIELD`).

### Reference data

Bundle code sets as static JSON with a `_meta` block so they stay auditable:

```json
{ "_meta": {"source": "ISO 4217", "retrieved_date": "2024-01-15", "version": "2024-01"},
  "codes": ["USD", "EUR", "..."] }
```

## Rules of the road

- **Validation never mutates data.** All changes happen in `repair`, and every
  attempt is logged to the `RepairLog` with from/to values, the rule id, and a
  status (`applied`/`flagged`/`unresolvable`).
- **Identifier columns are never repaired** — list them in `id_fields`.
- **Never guess column mappings silently** — detection logs how each match was made.
- **No network calls or LLM calls** in a pack.

## Registering a third-party pack

Expose your validator through the `freshdata.domains` entry-point group in your
package's `pyproject.toml`:

```toml
[project.entry-points."freshdata.domains"]
mydomain = "mypack.validator:MyValidator"
```

Once installed, `fd.clean(df, domain="mydomain")` finds it automatically.
Built-in names take precedence, so you cannot shadow `finance` (etc.).

## Advanced patterns

These conventions were established by the healthcare/education/agriculture/media packs.
Reuse them rather than reinventing them.

### Shared custom checks (`freshdata/domains/_common.py`)

Generic, config-driven checks (ISO date/datetime, future-date, numeric/positivity,
`both_present`, `requires_field`, `requires_when_value`, `at_least_one`, `ge_date`, …)
live in `_common.py`. Register the ones you need by name in `register_extensions` and
reference them from YAML via `params.func`; pass field relationships through `params`
(e.g. `params: {func: requires_field, requires: other_field}`). `to_datetime_safe`
parses mixed ISO date/datetime shapes correctly across pandas versions — always use it
instead of `pd.to_datetime` so date columns aren't spuriously coerced to `NaT`.

### Sub-schema routing (one pack, several schemas)

`healthcare` (FHIR `Patient`/`Observation`/`Encounter`) and `media`
(`content`/`release`) select a sub-schema from a constructor kwarg
(`fhir_resource=` / `media_type=`), promoted to top-level `fd.clean(...)` kwargs
alongside `gtfs_file`. The pattern, modeled on `transport`:

- Override `__init__(self, *, column_map=None, <selector>=None, **_kwargs)` and call
  `super().__init__(column_map=column_map)`. Accept `**_kwargs` so unrelated selectors
  are ignored.
- When the selector is omitted, **auto-detect** it from the column signature in
  `detect_columns` (use distinctive, non-shared columns; never the shared ones). If the
  signature is indeterminate, raise a pack-specific `Ambiguous…Error` (subclass
  `DomainError`) listing the candidates. An unsupported explicit value raises an
  `Unsupported…Error`.
- Healthcare uses one rules file per resource (`rules/<resource>.yaml`) and swaps
  `rules_path`/`canonical_fields`/… on activation. Media keeps a single `rules.yaml`,
  tags every rule with `params.media_type`, and overrides `_run_rule` to **skip** rules
  for the inactive sub-schema (returning a `skipped` `RuleResult`, exactly like
  `transport`'s per-file `gtfs_file` skip). Skipped rules never affect the trust score.

### PHI redaction (`PHI_FIELDS` + `audit_include_phi`)

Packs handling sensitive identifiers set a `PHI_FIELDS` class attribute and accept
`audit_include_phi=False` in `__init__`. Override `repair` to call
`redact_phi_actions(df, log, report.mapping, self.PHI_FIELDS, self._audit_include_phi)`
after `super().repair(...)`: it enriches flagged PHI-column findings with the offending
value (so the audit is useful) and then masks `from`/`to` as `[PHI]` unless the caller
opts in with `audit_include_phi=True`.

### One rule = one severity; companion info rules

A rule carries a single severity, so when a field is "valid but worth noting" *and*
"sometimes invalid", split it. Healthcare `HC-P003` (format, **error**) rejects malformed
birth dates while the companion `HC-P003I` (format, **info**) surfaces FHIR partial dates
(`YYYY`/`YYYY-MM`) that are retained, never coerced. Suffix the companion id (`…I`) and
document it.

### Curated vs. authoritative reference sets

Reference checks against a **complete, authoritative** set (ISO codes, FHIR value sets,
unit codes) are `error` severity. Checks against a **curated, non-exhaustive** set —
LOINC, SNOMED, FAO crop codes — are `warning` severity, because an unrecognized code may
simply be absent from the bundled subset, not invalid. Say so in the JSON `_meta.note`,
and include any required attribution (e.g. LOINC © Regenstrief Institute). Unit-coercion
maps (informal spelling → canonical code) live in the reference JSON under a `coerce`
key so the repair stays config-driven.

### Tested pure check-digit functions

Identifier check digits are pure, separately unit-tested functions: see
`eidr_check_char` / `is_valid_eidr` (ISO 7064 hybrid MOD 37,36) and `is_valid_icpn`
(GS1 mod-10 for UPC/EAN) in `media/validator.py`, anchored by a published known-answer
plus round-trip and tamper tests.
