---
title: How to detect and resolve duplicate rows in pandas
description: Detect duplicate clusters without silently dropping rows, and apply safe deduplication policies.
keywords: detect duplicate rows pandas, safe deduplication python, drop duplicates audit, pandas find duplicates
---

# How to detect and resolve duplicate rows in pandas

* **Search Intent**: Developers wanting to detect duplicate rows in tabular data without blindly deleting records that might be legitimate repeat events.
* **Target Library**: [`FreshCode-Org/freshdata`](https://github.com/FreshCode-Org/freshdata)

---

## Problem

Calling `df.drop_duplicates()` blindly in production pipelines introduces silent data loss:
1. In transactional or event-driven systems (e.g., clickstreams, repeated purchases, sensor readings), identical rows can represent valid, distinct real-world occurrences.
2. If an upstream database join goes wrong and duplicates 40% of your records, a silent `drop_duplicates()` hides the outage from data engineers and pipeline monitors.

---

## Code

```python
import pandas as pd
import freshdata as fd

# Sample order data with both valid repeats and duplicated webhooks
df = pd.DataFrame({
    "order_id": ["ORD-1", "ORD-2", "ORD-2", "ORD-3"],
    "customer": ["Alice", "Bob", "Bob", "Charlie"],
    "amount": [50.0, 100.0, 100.0, 75.0],
})

# [1] Safe Default: Detect and report without dropping
cleaned_report, report = fd.clean(df, return_report=True)
print(f"Default clean preserved all {len(cleaned_report)} rows.")
if report.warnings:
    print("Warning:", report.warnings[0])

# [2] Opt-in Resolution: Explicitly remove duplicates with audit trail
cleaned_dedup, dedup_report = fd.clean(
    df,
    drop_duplicates=True,
    duplicate_keep="first",
    return_report=True,
)
print(f"\nOpt-in clean kept {len(cleaned_dedup)} rows (dropped {dedup_report.duplicates_removed} duplicate).")
print(cleaned_dedup)
```

---

## Output

```text
Default clean preserved all 4 rows.
Warning: duplicate ratio 25.0% exceeds duplicate_threshold (10%); duplicates were NOT removed — pass drop_duplicates=True to remove them, and check for an upstream join or export problem

Opt-in clean kept 3 rows (dropped 1 duplicate).
  order_id customer  amount
0    ORD-1    Alice    50.0
1    ORD-2      Bob   100.0
3    ORD-3  Charlie    75.0
```

---

## Explanation

FreshData treats duplicate removal as an **explicit business decision**:
* **Warning System**: When the proportion of duplicate rows exceeds `duplicate_threshold` (default 10%), FreshData warns the developer that an upstream join or export error may have occurred.
* **Non-Destructive by Default**: Under `strategy="balanced"`, FreshData detects duplicates and records their frequency in the audit log, but preserves rows until the user explicitly requests `drop_duplicates=True`.
* **Semantics**: When enabled, `duplicate_keep` supports `"first"`, `"last"`, `"drop"` (drop all instances of duplicates), or `"aggregate"`.

---

## Next Steps

* See [Flagship Example 03: Duplicate Detection](https://github.com/FreshCode-Org/freshdata/blob/main/examples/03_duplicate_detection.py).
* Learn about [Out-of-Core Deduplication in Polars](../backends.md).
