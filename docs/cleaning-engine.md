---
title: The cleaning engine
description: >-
  How freshdata's two-layer cleaning engine works — representation repair plus a
  per-column decision engine for missing values, outliers, and duplicates with
  explainable rationale, risk, and confidence.
keywords: data cleaning engine, missing value handling, outlier detection pandas, duplicate removal, automated feature cleaning
---

# The cleaning engine

`freshdata` cleans in **two layers**: deterministic representation repair, then a
role-aware decision engine. Every action is recorded with a rationale, a risk
level, and a confidence score.

## Layer 1 — representation repair (always on)

| order | step | what it does |
|---|---|---|
| 1 | `column_names` | snake_case names, deduplicate collisions (`"a", "a"` → `"a", "a_2"`) |
| 2 | `strip_whitespace` | trim surrounding whitespace in text cells (internal spacing kept) |
| 3 | `normalize_sentinels` | `"N/A"`, `"null"`, `"-"`, `""`, `"#REF!"`, … → missing |
| 4 | `drop_empty_columns` / `drop_empty_rows` | remove all-missing columns and rows |
| 5 | `fix_dtypes` | text → numeric (`"$1,234.56"` works) / datetime / boolean, validated |
| 6 | `drop_duplicates` | resolve duplicate rows (`first` / `last` / `drop` / `aggregate`) |

## Layer 2 — the decision engine

With `strategy="balanced"` (the default), the engine infers each column's role —
**id**, **target/label**, **datetime**, **free text**, **categorical**,
**numeric** — and applies explicit threshold rules.

### Missing values

| missing ratio | numeric | categorical | datetime |
|---|---|---|---|
| ≤ 5% (low) | mean if ~normal & no outliers, else median | mode if clear majority, else `"Unknown"` | ffill/bfill if time-ordered |
| 5–30% (medium) | median (KNN only in aggressive mode) | mode if dominant, else `"Missing"` | ffill/bfill if time-ordered |
| > 30% (high) | **preserved** + warning (balanced); dropped in aggressive unless informative | same | same |

!!! note "Role gates run first"
    **Targets are never modified**, **IDs are never imputed**, and **free text is
    never force-filled** — those columns are preserved with the reason written
    into the report, so a remaining `NaN` is never silent. A `<col>_was_missing`
    indicator column is added when missingness itself correlates with other
    features. On frames under 30 rows, the engine preserves and recommends manual
    review instead of guessing on noisy ratios.

### Outliers

Detection methods: IQR fences (default), z-score, `"auto"` (z-score for ~normal
columns, IQR for skewed), or `"isolation_forest"` (scikit-learn, ≥ 100 rows,
falls back to IQR).

The default `outlier_action="auto"` is context-aware: it **flags** (adds a
boolean `<col>_outlier` column) under **balanced** mode and **caps**
(winsorizes to the fences) under **aggressive** mode, and flags heavy-tailed
columns (>15% outlying) rather than rewriting real data. Setting an explicit
`"cap"`, `"remove"`, or `"flag"` is a directive applied to every eligible
numeric column — heavy-tailed columns too, with a warning. Outliers in
ID/target columns, `preserve_columns`, and domain-sensitive columns (AQI,
pollutants, fraud / risk names) are always preserved — there the extremes
usually *are* the signal.

### Duplicates

Exact duplicates are removed by default (count and percentage reported).
Time-indexed frames never lose rows unless `allow_timeseries_duplicates=True`.
A duplicate ratio above `duplicate_threshold` (10%) raises a quality warning.
With `duplicate_subset`, `duplicate_keep="aggregate"` collapses each group
(numeric mean, first non-missing otherwise).

## Tuning

```python
fd.clean(
    df,
    strategy="balanced",             # "aggressive" | "conservative"
    missing_threshold_low=0.05,      # band edges for the missing-value rules
    missing_threshold_medium=0.30,
    missing_threshold_high=0.60,
    duplicate_threshold=0.10,
    outlier_method="iqr",            # "zscore" | "auto" | "isolation_forest"
    outlier_action="auto",           # context-aware; "cap" | "remove" | "flag" | None
    target_column="churn",
    preserve_columns=("notes",),
    id_columns=("ref",),
    return_report=True,
)
```

Explicit choices always override the engine. Every option lives on one frozen
dataclass — [`CleanConfig`](api-reference.md) — and unknown names fail
immediately with a "did you mean" suggestion.

## What freshdata will **not** do

- Touch a target/label column, impute an identifier, or force-fill free text.
- Remove outliers blindly — fraud/anomaly-style columns keep their extremes.
- Guess fuzzy entity resolution in `clean()` — that is opt-in via the enterprise
  layer's clustering.
- Parse ambiguous European decimal commas (`"1.234,56"`) — too risky to guess.
- Mutate your DataFrame (unless you pass `preserve_original=False`).
