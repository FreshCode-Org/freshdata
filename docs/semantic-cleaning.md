# Semantic cleaning layer

FreshData's core repairs *representation* (whitespace, sentinels, dtypes, duplicates) and
runs a statistical engine for missing values and outliers. The **Semantic Cleaning Layer**
adds a complementary stage that repairs *meaning*: values that are syntactically fine but
semantically wrong.

Examples it handles, deterministically and offline:

| Issue type        | Example            | Becomes  | Where                       |
|-------------------|--------------------|----------|-----------------------------|
| `spelled_number`  | `"twenty"`         | `20`     | numeric-like columns        |
| `boolean_synonym` | `"yes"`, `"y"`     | `True`   | boolean-like columns        |
| `currency_string` | `"$1,200.50"`      | `1200.5` | money-like columns          |
| `unit_suffix`     | `"10 kg"`          | `10`     | unit-consistent columns     |
| `category_synonym`| `"M"`, `"Male "`   | `"male"` | low-cardinality categoricals|
| `date_phrase`     | `"2026-07-01"`, `"today"` | `Timestamp` | date-like columns    |

## How it works

Every change flows through a fixed, auditable pipeline:

1. **profile** semantic issues (on distinct values only — never row-by-row);
2. **generate** repair proposals via deterministic experts;
3. **score** each proposal (explainable confidence + risk);
4. **validate** against policy, config, and column protection;
5. **auto-apply** only when safe;
6. otherwise **record a suggestion** for review;
7. **preserve every decision** in the `CleanReport`.

> An LLM never mutates the DataFrame. Everything is fully deterministic and offline (no
> network, no API keys). Retrieval-backed [semantic memory replay](#semantic-memory-replay)
> is implemented as a local, server-free extension; the interfaces still leave a clean
> extension point for an optional future private-LLM candidate generator.

## Enabling it

The layer is **off by default**. Opt in with `semantic_mode`:

| Mode       | Behavior                                                                 |
|------------|--------------------------------------------------------------------------|
| `None`/`"off"` | disabled (default)                                                  |
| `"assist"` | detect and report proposals only; never mutates                          |
| `"review"` | apply zero-risk deterministic repairs; suggest the rest                  |
| `"auto"`   | apply proposals with confidence ≥ `semantic_auto_threshold`, non-high risk, that pass policy |

```python
import freshdata as fd

# assist: suggestions only, no mutation
cleaned, report = fd.clean(df, semantic_mode="assist", return_report=True)
```

```python
# auto: safe repairs applied, with column hints
cleaned, report = fd.clean(
    df,
    semantic_mode="auto",
    semantic_context={
        "columns": {
            "age": {"semantic_type": "number"},
            "customer_id": {"semantic_type": "identifier", "mutable": False},
        }
    },
    return_report=True,
)
```

### Date phrase repair

`DatePhraseExpert` normalizes obvious date phrases and strings in date-like columns
(matched by role, `semantic_type`, a date-like column name, or a strong share of
parseable values): ISO/slash-ordered dates (`2026-07-01`, `2026/07/01`), month-name dates
(`July 1 2026`, `1 July 2026`), and numeric dates where day/month order is unambiguous.

`"today"`/`"yesterday"`/`"tomorrow"` are **only** resolved when you supply an explicit
`reference_date` — the real wall-clock date is never used silently:

```python
cleaned, report = fd.clean(
    df,
    semantic_mode="auto",
    semantic_context={
        "reference_date": "2026-07-01",
        "columns": {"signup_date": {"semantic_type": "date"}},
    },
    return_report=True,
)
```

Numeric dates where **both** day and month are `<= 12` (`"01/02/2026"`) are ambiguous and
are never auto-applied — they are recorded as high-risk, `human_review=True` suggestions
unless you resolve the convention explicitly with `dayfirst`:

```python
semantic_context={"columns": {"event_date": {"semantic_type": "date", "dayfirst": True}}}
```

Applied date repairs convert the column to pandas `datetime64` dtype only when *every*
value in the column resolved successfully — a column with any leftover unconverted
string stays as-is, so the conversion never silently drops information.

### Semantic memory replay

Accepted semantic repairs can be learned into a `CleaningMemory`
(`fd.learn_cleaning_memory`) and replayed as candidate proposals on similar future data:

```python
cleaned, report = fd.clean(df, semantic_mode="auto", return_report=True)

memory = fd.learn_cleaning_memory(df, decisions=report, dataset_id="crm_signups")

next_cleaned, next_report = fd.clean(
    next_df,
    semantic_mode="auto",
    memory=memory,
    return_report=True,
)
```

- Memory is local and server-free — it stores learned repairs in
  `memory.value_patterns["semantic_repairs"]`, no different from the rest of
  `CleaningMemory`'s JSON/SQLite storage.
- Memory is **evidence, not authority**: every retrieved repair still passes through the
  same policy gate as a deterministic proposal, so it can never override
  `target_column`/`id_columns`/`preserve_columns` protection, the confidence floor, or the
  ambiguous-date rules above.
- Replayed repairs are fully audited: `memory_influenced=True` and
  `model_id="semantic:<issue_type>:memory"` mark which decisions came from memory, and
  `Action.metadata` carries the raw/proposed value and evidence either way.
- Retrieval matches on the exact normalized value first, falling back to a lightweight,
  no-dependency similarity check (`difflib`) for minor value drift; low-similarity or
  conflicting repairs are never auto-applied.

## Configuration

| `CleanConfig` field            | Default            | Meaning                                       |
|--------------------------------|--------------------|-----------------------------------------------|
| `semantic_mode`                | `None`             | `None`/`off`/`assist`/`review`/`auto`         |
| `semantic_auto_threshold`      | `0.95`             | min confidence to auto-apply                  |
| `semantic_review_threshold`    | `0.70`             | below this, never apply (skip)                |
| `semantic_max_distinct_values` | `500`              | skip very high-cardinality columns            |
| `semantic_sample_size`         | `10_000`           | rows sampled when profiling distinct values   |
| `semantic_backends`            | `("deterministic",)` | candidate backends (extension point)        |
| `semantic_context`             | `None`             | per-column hints (semantic_type/unit/allowed_values/mutable/dayfirst) plus a top-level `reference_date` for relative date phrases |
| `semantic_privacy_policy`      | `"local_only"`     | privacy posture for future external inference |
| `semantic_budget`              | `None`             | budget hints (extension point)                |

## Safety guarantees

- **ID, target, and `preserve_columns` are protected.** Identifier-like columns are vetoed
  unless `semantic_context` marks them `mutable`.
- **Ambiguous repairs are suggestions, not silent mutations** (`status="suggested"`,
  `human_review=True`).
- **Codes are never mangled** — zero-padded numbers (`"007"`), mixed alphanumerics
  (`"105A"`), and punctuated handles (`"A-100"`, `"D@vid"`) are left alone.
- **Deterministic and repeatable** — the same input always produces the same report.

## Auditing

Each decision appears in the report with `step="semantic"`:

```python
for a in report:
    if a.step == "semantic":
        print(a.status, a.column, a.confidence, a.risk, a.model_id, a.description)
        print(a.memory_influenced, a.metadata)  # metadata carries raw/proposed value + evidence
```

Preview proposal counts before running, via `suggest_plan`:

```python
plan = fd.suggest_plan(df, semantic_mode="assist")
plan.to_frame()[["column", "semantic_proposals"]]
```

## Backends

Semantic cleaning runs on the in-memory pandas path. When you select a native engine
(Polars, DuckDB, Spark), FreshData routes the clean through pandas and records the fallback
in `report.fallback_events` rather than silently skipping the semantic stage.
