# FreshData Benchmark & Proof System

This directory contains the reproducible benchmark harness, synthetic enterprise fixtures, and baseline comparison runners for **FreshData 2.0**.

FreshData's benchmark system is designed around one core principle: **reproducible evidence without cherry-picking**. Every metric is machine-readable, seed-controlled, and executed through public API calls without internal patching.

---

## 1. The Nine Standardized Metrics

The harness measures nine standardized metrics across all runs:

| # | Metric | Definition | Target / Invariant |
|---|---|---|:---:|
| 1 | **Wall-clock latency** | $p_{50}$ and $p_{95}$ seconds for `fd.clean(..., return_report=True)` across 5 timed repeats (I/O excluded). | Sub-second for $\le$100k rows |
| 2 | **Peak memory** | Peak and delta RAM consumption in MB measured via `tracemalloc`. | $<2\times$ raw data size |
| 3 | **Repair fidelity** | Ratio of correctly repaired defective cells vs. ground-truth gold standard. | $\ge 95\%$ on in-scope defects |
| 4 | **False-repair rate** | Percentage of protected cells (primary keys, targets, locked columns) altered by mistake. | **Strictly 0.0%** (invariant) |
| 5 | **Preservation rate** | Percentage of protected or ambiguous cells preserved identical in/out. | **Strictly 100.0%** (invariant) |
| 6 | **Authored-code reduction** | Lines of code (LOC) required for custom cleaning scripts vs. one-call `fd.clean()`. | $>80\%$ reduction |
| 7 | **Diagnosis speed** | Wall-clock latency to generate `report.summary()`, `to_frame()`, and `to_json()`. | $<15\text{ ms}$ |
| 8 | **Trust-score monotonicity** | Strict monotonic decrease in trust score as injected corruption rate rises ($0\% \to 60\%$). | Monotonic ($r \ge 0.98$) |
| 9 | **Audit completeness** | Verification that $100\%$ of transformed cells have a corresponding `Action` record with rationale, risk, and confidence. | **Strictly 100.0%** |

---

## 2. Benchmark Datasets & Scenarios

Fixtures are generated with fixed seeds (`generate(n_rows, seed=42)`) so runs on different hardware produce identical datasets.

| Fixture | Columns | Injected Corruptions | Realistic Industry Scenario |
|---|:---:|---|---|
| **`crm`** | 40 | Sentinels (`"N/A"`, `"-"`), mixed dates, whitespace, dirty currency, invalid emails, negative ages | Customer master data extracted from legacy Salesforce/HubSpot exports |
| **`finance`** | 35 | Precision strings, formula injection tokens, account ID gaps, currency signs, duplicate ledger rows | General ledger extracts, transaction tables, and banking feeds |
| **`event_log`** | 20 | Out-of-order timestamps, sensor outliers, duplicate telemetry IDs, null device tokens | IoT telemetry streams, CDC clickstreams, and application audit logs |
| **`wide_schema`** | 120 | Sparse missingness, high cardinality categoricals, mixed float/int columns | Machine learning feature stores and clinical study electronic data capture |

### Scale Scenarios

* **Small (CI Fast Lane)**: $10,000$ rows ($40$ columns) — runs in $<0.5\text{ s}$ per fixture.
* **Medium**: $100,000$ rows — standard developer workstation baseline.
* **Large**: $1,000,000$ rows — out-of-core and memory pressure verification.
* **Stress Scale**: $10,000,000$ to $25,000,000$ rows — Polars and DuckDB out-of-core scalability evaluation.

---

## 3. Safety Invariants: What FreshData Refuses to Change

Automatic cleaning can introduce silent corruption if it guesses blindly. FreshData enforces strict safety boundaries:

```text
       Input Value          Feature Role          FreshData Action         Rationale & Invariant
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
"C-98214" -> None          Customer ID        Preserve Missing (NaN)      Never fabricate primary keys.
"target_churn" -> None     ML Target Label    Preserve Missing + Warn     Never impute ground truth labels.
"2023-99-99"               Date string        Coerce to NaT + Review      Never invent arbitrary timestamps.
age = -14                  Numeric Age        Flag in 'age_outlier'       Never silently drop rows.
"USD 500" vs "$500"        Ambiguous currency Preserve with Review        Never guess ambiguous exchange rates.
```

If confidence in an action drops below configured thresholds or involves an identifier, FreshData **refuses the transformation**, reports the cell in `report.coerced_cells`, and suggests human review in `report.recommendations`.

---

## 4. How to Reproduce Locally

### Quick Run (10k rows)

```bash
# 1. Install dependencies
pip install -e ".[dev,bench,ml]" jsonschema

# 2. Run the standard benchmark suite
python benchmarks/bench.py run

# 3. Generate Markdown and JSON summary
python benchmarks/bench.py report
```

### Run a Single Fixture & Metric

```bash
# Measure wall-clock latency for 100,000 CRM rows
python benchmarks/bench.py single --fixture crm --size 100000 --metric time

# Measure peak memory on financial ledger data
python benchmarks/bench.py single --fixture finance --size 100000 --metric memory
```

### Compare Against Baseline Tools

```bash
# Compare FreshData vs pandas on 50,000 rows
python benchmarks/bench.py compare --fixture crm --size 50000
```

---

## 5. Environment & Hardware Reference

Measurements committed to this repository were gathered under the following controlled environment:

* **Hardware**: Apple M2 Max / 12-core CPU / 32 GB unified memory
* **OS**: macOS Sonoma 14.5 (Darwin 23.5.0)
* **Python**: 3.11.9 (64-bit)
* **Key Dependencies**: `pandas==2.2.2`, `numpy==1.26.4`, `polars==0.20.31`, `duckdb==1.0.0`
* **FreshData Version**: `2.0.0`

### Summary of Measured Results

| Fixture | Rows | FreshData Time ($p_{50}$) | Peak Memory | False Repair Rate | Repair Fidelity |
|---|:---:|:---:|:---:|:---:|:---:|
| `crm` | 10,000 | 0.082s | 14.2 MB | **0.0%** | 98.4% |
| `crm` | 100,000 | 0.741s | 86.5 MB | **0.0%** | 98.6% |
| `finance` | 10,000 | 0.071s | 12.8 MB | **0.0%** | 99.1% |
| `finance` | 100,000 | 0.680s | 74.3 MB | **0.0%** | 99.0% |
| `event_log` | 100,000 | 0.490s | 51.2 MB | **0.0%** | 97.9% |

---

## 6. Documented Limitations & Failure Cases

Honesty is central to FreshData's engineering principles. We do not claim general superiority:

1. **Lone Primitive Speed**: A single pandas `df["col"].fillna(0)` is faster than `fd.clean(df)` because FreshData performs profiling, sentinel detection, role inference, and generates an audit log. Do not use FreshData if you only need a single hardcoded primitive on an already-clean column.
2. **25M+ Row In-Memory Limit**: At 25M rows, materializing a full pandas DataFrame in RAM can cause memory pressure. For datasets $\ge$10M rows, use the native handle path: `fd.clean(df, engine="polars")` or `output_format="duckdb"`.
3. **Ambiguous Natural Language**: Free-form unstructured text columns (e.g. customer support transcripts) are preserved as text. FreshData cleans delimiters, encoding artifacts, and whitespace, but does not invent text or rewrite sentences without explicit language plugins.
