---
title: Decision-preserving workflow
description: >-
  Reuse cleaning decisions across runs, compare to a baseline, track quality
  debt, suggest fuzzy join keys, lint text encoding, and brief stakeholders.
keywords: cleaning memory, data drift, data quality debt, fuzzy join, encoding lint
---

# Decision-preserving workflow

freshdata is the **explainable cleaning layer**: clean once, explain always,
remember next time. These features turn a one-off clean into a reusable,
auditable workflow from notebook exploration to production governance.

## Cleaning memory

Record human-approved decisions and replay them on similar future data.

```python
import freshdata as fd

cleaned, report = fd.clean(df, return_report=True)

memory = fd.learn_cleaning_memory(df, decisions=report, dataset_id="crm_contacts")
memory.to_json("crm_memory.json")        # or "crm_memory.sqlite" — no server

# next week, on new data of the same shape:
memory = fd.load_cleaning_memory("crm_memory.json")
cleaned, report = fd.clean(df_next, memory=memory, return_report=True)
```

Memory stores the dataset signature, column roles, accepted/rejected decisions,
thresholds, value-pattern decisions and stakeholder exceptions, with timestamp
and version metadata. On replay freshdata checks the new data still matches what
it learned; **if it drifted too much, replay is blocked and the report explains
why** rather than applying stale decisions. Replayed actions are marked
`status="approved"` / `memory_influenced=True` so the audit trail is honest.
`memory.diff(other)` and `memory.summary()` round out the API.

## Compare to a baseline (drift)

Dead-simple temporal quality comparison.

```python
diff = fd.compare_to_baseline(
    current_df,
    baseline=last_week_df,        # a raw DataFrame, or a prebuilt DatasetBaseline
    key="customer_id",
    event_time="updated_at",
)
diff.what_likely_matters()        # business-language highlights
diff.show()                       # interactive drift report
```

Reports schema diff (added / removed / dtype-changed columns), completeness and
duplicate deltas, category churn, distribution shift, and — with `key=` —
key-level record counts (added / removed / changed). `.to_frame()`, `.to_dict()`
and `.to_html()` are all available.

## Quality-debt ledger

A middle ground between hard failure and ignoring issues.

```python
cleaned, gate = fd.evaluate_quality_debt(
    df,
    debt_policy="warn_then_fail",     # "warn" | "fail" | "warn_then_fail"
    ledger="quality_debt.sqlite",
)
print(gate.status)                    # "pass" | "warn" | "fail"
```

Scores nine debt dimensions (missingness, duplicates, schema drift, type
instability, outlier spikes, PII risk, category churn, failed repairs, human-
review backlog), persists the history to SQLite, and **escalates warn→fail when
an issue repeats or worsens** across runs.

## Dirty-join assistant

Reviewable fuzzy joins for messy keys — never a silent low-confidence join.

```python
matches = fd.suggest_join_keys(
    left_df, right_df,
    on=["company_name", "address"],
    exact_within=["country"],         # blocking key: must match exactly
)
matches.to_frame()                    # candidates with confidence + per-field scores
```

Suggests exact keys, ranks fuzzy candidates with confidence and per-field
explanations, groups candidates by block, and flags ambiguous matches in a
separate section for human review.

## Text & encoding lint

```python
lint = fd.lint_text_encoding(
    df, columns=["name", "address", "city"], locale_hints=["en_IN", "ar_AE"])
lint.to_frame()
```

Detects mixed scripts, mojibake-like artifacts, NFC/NFD inconsistency, RTL/LTR
risk, locale-ambiguous dates/numbers, replacement characters and stray
control/zero-width whitespace. **Diagnostic only** — each issue carries a
severity, examples, a suggested repair, and whether that repair is safe to
automate. Nothing is modified.

## Stakeholder-safe summaries

```python
summary = fd.stakeholder_summary(report, audience="business", format="markdown")
print(summary.to_markdown())          # or summary.to_html()
```

> Customer email completeness fell from 98.7% to 92.1%. Five columns changed
> meaningfully. Phone numbers were normalized automatically. Suspicious revenue
> outliers were preserved for review.
