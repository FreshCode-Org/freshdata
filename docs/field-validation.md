# Text cleaning & context-aware field validation

Two standalone, dependency-light layers for handling scraped or third-party
feeds where the same string can be valid in one column and wrong in another:

- **`fd.clean_text`** — a configurable, field-aware text-cleaning pipeline
  that never destroys information;
- **`fd.validate_fields`** — per-cell validation that judges each value *in
  the context of its field*, with a configurable, non-destructive
  remediation policy.

Both work on plain pandas frames and are independent of the main
`fd.clean` engine (which now also *warns* when a mostly-numeric column is
kept as text because of a few unparseable values, naming the exact rows).

## Text cleaning

```python
import pandas as pd
import freshdata as fd

df = pd.DataFrame({
    "note": ["  <p>Great&nbsp;product…</p>  ", "fine"],
    "amount": [" 1,200.50 ", "99.99"],
})

cleaned, report = fd.clean_text(
    df,
    config=fd.TextCleanConfig(strip_html=True),
    field_types={"amount": "currency_amount"},   # guards aggressive ops
)

print(cleaned.loc[0, "note"])     # Great product...
print(cleaned.loc[0, "amount"])   # 1,200.50  (only whitespace trimmed)
print(report.summary())
```

Key properties:

- **Non-destructive** — the input frame is never modified; every changed
  cell is logged with `original`, `cleaned` and the ordered transform list.
- **Field-aware** — lossy operations (case folding, punctuation or HTML
  stripping) are automatically withheld from structural types (amounts,
  identifiers, emails, URLs, dates, tickers) and from entity names.
- **Deterministic and lossless by default** — HTML/URL removal, case
  folding and punctuation removal are opt-in via `TextCleanConfig`.

Scalar use: `fd.clean_text_value("  x  ")` returns a `CleanedText` with
`.original`, `.cleaned`, `.transforms`.

## Context-aware validation: the "apple" example

A scraped feed inserts the string `"apple"` into different columns. The
correct verdict depends entirely on the field:

```python
import pandas as pd
import freshdata as fd

df = pd.DataFrame([{
    "transaction_amount": "apple",        # invalid: monetary field
    "company_name": "Apple",              # valid: real-word names are fine
    "stock_ticker": "apple",              # invalid format; AAPL would pass
    "transaction_description": "Payment to Apple",  # valid free text
}])

schema = {
    "transaction_amount": fd.FieldSpec(semantic_type="currency_amount"),
    "company_name": fd.FieldSpec(semantic_type="company_name"),
    "stock_ticker": fd.FieldSpec(semantic_type="ticker",
                                 suggest={"apple": "AAPL"}),
    "transaction_description": fd.FieldSpec(semantic_type="free_text"),
}

report = fd.validate_fields(df, schema)
print(report.summary())
for issue in report.issues:
    print(issue.column, issue.classification, "->", issue.action,
          "| suggestion:", issue.suggestion)
```

Output (abridged):

```
transaction_amount  semantic_mismatch -> quarantine  | suggestion: None
stock_ticker        domain_mismatch   -> manual_review | suggestion: AAPL
```

Rules of the road:

- values are **never silently converted** — `"apple"` does not become `0`
  or `NaN`;
- suggestions come **only from a trusted mapping or callable you supply**
  (`FieldSpec.suggest`); nothing is invented;
- reference verification (e.g. a ticker universe) is injected via
  `FieldSpec.reference` — never hard-coded;
- a column *without* a spec is still protected: when ≥80% of its values
  share one shape, nonconforming cells are reported as `semantic_mismatch`
  with the exact row, reason and confidence.

## Failure classes and remediation policy

Distinct problems stay distinct — there is no generic "outlier" bucket:

| classification             | default action        | severity |
|----------------------------|-----------------------|----------|
| `parse_failure`            | `quarantine`          | error    |
| `semantic_mismatch`        | `quarantine`          | error    |
| `domain_mismatch`          | `manual_review`       | error    |
| `schema_violation`         | `reject`              | error    |
| `statistical_outlier`      | `accept_with_warning` | warning  |
| `categorical_rare`         | `accept_with_warning` | warning  |
| `cross_field_inconsistency`| `manual_review`       | warning  |

Rare or extreme values are **warned about, never auto-rejected**: an
unusually large transaction is still a transaction. Every action is
configurable per class:

```python
policy = fd.RemediationPolicy(semantic_mismatch="replace_with_null")
report = fd.validate_fields(df, schema, policy=policy)
result = fd.apply_field_policy(df, report)
result.accepted      # cleaned copy (replacements applied, audited)
result.quarantined   # rows held back for review
result.rejected      # rows dropped from the accepted set (still returned!)
result.needs_review  # rows awaiting a human decision
result.audit         # one record per issue: original, action, reason, rule
```

`apply_field_policy` never mutates the input frame, and every
`replace_with_null` keeps the original value in the audit trail.

Context-dependent missing codes are per-field: `"N/A"` can be a null marker
in one column and a legitimate code in another
(`FieldSpec(null_markers=frozenset({""}), allowed_values={"N/A", ...})`).

Cross-column rules are plain callables:

```python
def settle_after_trade(row):
    if str(row["settlement_date"]) < str(row["trade_date"]):
        return "settlement_date precedes trade_date"
    return None

fd.validate_fields(df, schema, cross_rules=[settle_after_trade])
```

## Reporting

`report.to_findings()` exports issues as standard
[`QualityFinding`](api-reference.md) records (step `"fieldcheck"`), so they
flow into the existing exporters (dbt tests, Great Expectations suites,
exception tables, lineage). `report.to_frame()` gives a flat DataFrame;
`report.normalized_cells` is the text-normalization audit.

## Performance

Validation runs a vectorized pre-screen per column and only pays the
per-cell explanation path for suspect cells, so mostly-clean feeds validate
at bulk speed. Reproduce the comparison against regex and pandas-coercion
baselines with:

```bash
python benchmarks/bench_fieldcheck.py 100000
```

## Known limitations

- Semantic-type coverage is deliberately narrow and deterministic (numbers,
  dates, emails, URLs, phones, tickers, identifiers, entity names, free
  text); embedding-based typing lives in the optional
  [semantic layer](semantic-models.md).
- Column-consensus checks need a dominant shape (≥80%) and at least 3
  values; genuinely mixed columns are left alone by design.
- `FieldSpec.reference` given as a callable disables the vectorized
  pre-screen for that column (every value is checked individually).
- Date validation accepts any format `pandas.to_datetime` can parse;
  day-first/locale ambiguity is the cleaning engine's concern
  (`fd.clean(dayfirst=...)`).
