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

## How it works

Every change flows through a fixed, auditable pipeline:

1. **profile** semantic issues (on distinct values only — never row-by-row);
2. **generate** repair proposals via deterministic experts;
3. **score** each proposal (explainable confidence + risk);
4. **validate** against policy, config, and column protection;
5. **auto-apply** only when safe;
6. otherwise **record a suggestion** for review;
7. **preserve every decision** in the `CleanReport`.

> An LLM never mutates the DataFrame. The first version is fully deterministic and offline
> (no network, no API keys). The interfaces leave clean extension points for future
> retrieval-backed memory and optional private LLM candidate generation.

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

## Configuration

| `CleanConfig` field            | Default            | Meaning                                       |
|--------------------------------|--------------------|-----------------------------------------------|
| `semantic_mode`                | `None`             | `None`/`off`/`assist`/`review`/`auto`         |
| `semantic_auto_threshold`      | `0.95`             | min confidence to auto-apply                  |
| `semantic_review_threshold`    | `0.70`             | below this, never apply (skip)                |
| `semantic_max_distinct_values` | `500`              | skip very high-cardinality columns            |
| `semantic_sample_size`         | `10_000`           | rows sampled when profiling distinct values   |
| `semantic_backends`            | `("deterministic",)` | candidate backends (extension point)        |
| `semantic_context`             | `None`             | per-column hints (semantic_type/unit/allowed_values/mutable) |
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
