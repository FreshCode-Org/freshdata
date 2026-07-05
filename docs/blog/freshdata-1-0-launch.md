---
title: "Announcing freshdata 1.0 — data cleaning that shows its work"
description: >-
  freshdata 1.0 is here: an accuracy-first, explainable DataFrame cleaner for
  pandas that never guesses silently, scales out-of-core, and takes cleaning
  rules in plain English.
keywords: freshdata 1.0, data cleaning python, pandas, explainable data cleaning, launch
author: Johnny Wilson Dougherty
date: 2026-07-05
---

# Announcing freshdata 1.0 — data cleaning that shows its work

*2026-07-05 · Johnny Wilson Dougherty*

Most data-cleaning code is a pile of `fillna`, `dropna`, and `astype` calls that
nobody can explain three weeks later. `freshdata` started as a reaction to that:
a cleaner that makes a *defensible* decision for every column and tells you
exactly what it did and **why**. Today it hits **1.0**.

```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("export.csv")
cleaned, report = fd.clean(df, return_report=True)
print(report.summary())
```

That `report` is the whole point. Every decision carries a rationale, a risk
level, and a confidence score. If a `NaN` survives the clean, the report tells
you why it was safer to leave it.

## What 1.0 actually gives you

**A decision engine, not a `fillna` wrapper.** freshdata profiles every column —
missing ratio, dtype, skew, cardinality, inferred role — and picks the right
action per column. It will not impute an identifier, overwrite a target/label,
force-fill free text, or drop outliers blindly.

**Rules in plain English.** You can hand freshdata the domain knowledge that
lives in your head:

```python
cleaned = fd.clean(df, context="Never modify revenue. customer_id is unique.")
```

The context compiler is deterministic and offline — no LLM, no network. It
understands a documented set of sentence patterns (uniqueness, protection,
formats, allowed values, ranges, dedup keys). Anything it doesn't understand is
surfaced as *unparsed*, never guessed at. Protected columns get a hard
byte-identity guarantee: freshdata verifies they came out untouched.

**Scale when you need it, honesty about when you don't.** The accuracy-first
engine is pandas, but the safe representation steps run natively on Polars,
DuckDB, or Spark, and `StreamingCleaner` cleans an unbounded stream in flat
memory. When a step genuinely needs the pandas engine, the native backends fall
back — and the report says so. As of 1.0 the streaming path honors context
policies too: the policy is compiled once against the first batch and protects
those columns across the entire stream.

**A trust score you can gate on.** Every clean produces a 0–100 trust score
that decreases monotonically as data gets dirtier — so you can fail a pipeline
(`fail_under_trust=...`) instead of shipping garbage downstream.

**Explainable output for ML.** Typed, leakage-aware frames that drop straight
into scikit-learn or XGBoost, with the decision log attached.

## The part most launches skip: what it *doesn't* do

freshdata's one rule is that it never claims more than it does, so the
[Honest limitations](../limitations.md) page ships as a first-class document.
A couple of things worth calling out on launch day:

- **Calibration is honest, not magical.** Out of the box, freshdata's confidence
  calibration reaches an expected calibration error around 0.038 on CleanBench —
  good enough to clear our 0.05 target, but not the stricter 0.03 tier. That
  stricter tier needs a trained `calib-v1` artifact that isn't published yet, and
  we say so in the docs and the benchmark output rather than hiding the gap.
- **Ambiguity is never resolved silently.** A malformed email, a phone number
  with the wrong digit count, an ambiguous date — these become suggestions or
  flags, not quiet edits. `strict=True` turns ambiguity into a hard error.

The [CleanBench](../benchmarks.md) results are committed to the repo and
reproducible with a single command. If a number looks too good, reproduce it.

## Get started

```bash
pip install freshdata-cleaner
```

Then read the [Quickstart](../quickstart.md), skim the
[Honest limitations](../limitations.md), and clean something messy. If freshdata
makes a call you disagree with, the report will tell you exactly which decision
to override — and [issues and PRs](https://github.com/FreshCode-Org/freshdata)
are very welcome.

Thanks to everyone who filed bugs, argued about defaults, and pushed on the
"show your work" principle that got us here.

— Johnny
