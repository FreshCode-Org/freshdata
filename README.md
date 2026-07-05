<div align="center">

# freshdata

### The explainable cleaning layer for pandas — decision-preserving data hygiene.

*One call turns a messy CSV, Excel, or SQL export into analysis- and ML-ready data — and tells you exactly what it changed and **why**.*

**Clean once, explain always, remember next time.** freshdata profiles, repairs,
logs, scores, explains, and *remembers* data-quality decisions, then makes them
reusable in notebooks, streaming runs, and orchestrated pipelines.

[![PyPI Version](https://img.shields.io/pypi/v/freshdata.svg)](https://pypi.org/project/freshdata/)
[![Python Versions](https://img.shields.io/pypi/pyversions/freshdata.svg)](https://pypi.org/project/freshdata/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml/badge.svg)](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml)
[![Docs](https://github.com/FreshCode-Org/freshdata/actions/workflows/docs.yml/badge.svg)](https://freshcode-org.github.io/freshdata/)
[![Downloads](https://img.shields.io/pypi/dm/freshdata.svg)](https://pypi.org/project/freshdata/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FreshCode-Org/freshdata/badges/coverage.json)](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)

[**Documentation**](https://freshcode-org.github.io/freshdata/) ·
[**Quickstart**](https://freshcode-org.github.io/freshdata/quickstart/) ·
[**API Reference**](https://freshcode-org.github.io/freshdata/api-reference/) ·
[**Examples**](examples/) ·
[**Changelog**](CHANGELOG.md)

</div>

![Terminal demo: run `fd.clean(df)` and inspect `report.summary()`](docs/assets/demo/freshdata-terminal-demo.gif)

*Before/after at a glance: the terminal starts with a messy `messy_export.csv`, runs `fd.clean(df)`, and then prints `report.summary()` showing the cleaning actions and before/after shape.*

---

`freshdata` is an **automated data-cleaning library for Python** that does real,
intelligent preprocessing of real-world tabular data. It is **not** a `fillna`
wrapper: a rule-based decision engine profiles every column (missing ratio,
dtype, skewness, cardinality, inferred role) and chooses the right action per
column — then logs a rationale, a risk level, and a confidence score for each
decision so nothing happens silently.

```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("export.csv")

result = fd.clean(df)                               # one line
cleaned = result.data                               # plain pandas DataFrame
print(result.summary())                             # full audit trail
result.visualize()                                  # self-contained HTML

cleaned, report = fd.clean(df, return_report=True)  # legacy tuple still works
```

```text
freshdata clean report
  rows:    525 -> 500 (-25)
  columns: 7 -> 6 (-1)
  missing: 421 -> 0 cell(s)
  memory:  100.8 KB -> 89.2 KB
  time:    0.017s
  engine:  25 duplicate row(s) removed; 20 outlier(s) flagged; imputed: age, segment
  actions (7):
    - [fix_dtypes] 'mostly_gone': converted to Int64
    - [drop_duplicates] dropped 25 duplicate row(s) (4.8% of rows, keep='first')
    - [missing] 'age': filled 12 missing value(s) with median (39.6846)
    - [missing] 'segment': filled 90 missing value(s) with sentinel "Missing" ('Missing')
    - [missing] 'mostly_gone': preserved 300 missing value(s)
    - [outliers] 'amount': flagged 15 outlier(s), 3.0% of values (method=iqr, factor=1.5) in new column 'amount_outlier'
    - [outliers] 'age': flagged 5 outlier(s), 1.0% of values (method=iqr, factor=1.5) in new column 'age_outlier'
  review (1):
    ? column 'mostly_gone' preserved at 60.0% missing in balanced mode
```

## ✨ Key features

- **Automated DataFrame cleaning in one call** — `fd.clean(df)` handles missing
  values, outliers, duplicates, dtype repair, and messy column names.
- **Per-column decision engine** — infers each column's role (id, target,
  datetime, free text, categorical, numeric) and applies explicit, documented
  threshold rules instead of one blunt global strategy.
- **Explainable by design** — every decision carries a rationale, risk level,
  and confidence score. If a `NaN` survives, the report says exactly why.
- **Safe defaults** — never imputes an identifier, never modifies a target/label
  column, never force-fills free text, never removes outliers blindly.
- **AI-ready preprocessing** — produces clean, typed, leakage-aware frames ready
  for scikit-learn, XGBoost, or any ML pipeline.
- **Data profiling** — `fd.profile(df)` gives read-only data-quality insight
  using the same inference code as `clean`, so previews are faithful.
- **pandas-first, Polars-optional** — pandas + NumPy core; pass a Polars frame
  and get a Polars frame back when the optional adapter is installed.
- **Enterprise layer** — opt-in fuzzy clustering, PII masking, semantic
  validation, a 0–100 Data Trust Score, OpenLineage metadata, and a batch CLI.
- **Drift, privacy & entity resolution** — persisted schema/stat baselines with
  contract + KS/PSI drift monitoring, Presidio-style PII detection with reversible
  tokenization / format-preserving anonymization / k-anonymity, and DuckDB-backed
  probabilistic entity resolution (see the enterprise section below).
- **Typed, tested, fast** — fully type-hinted (`py.typed`), 1,200+ tests with a
  93% coverage gate enforced in CI (`--cov-fail-under=93`), vectorized
  pandas/NumPy throughout.

## 🤔 Why FreshData exists

Most data-cleaning code is hand-written, one-off, and silent. People reach for
`df.dropna()` or `df.fillna(0)` and quietly corrupt their analysis — imputing an
ID, leaking a target, or deleting the very outliers that *were* the signal.
General-purpose tools don't fix this:

- **pandas** gives you primitives, not decisions — you still write every rule.
- **profiling tools** (sweetviz, ydata-profiling) *describe* data but don't
  clean it.
- **validation tools** (Great Expectations) *check* data but don't repair it.

`freshdata` fills the gap: an opinionated engine that **makes the right cleaning
decision per column and explains it**, so you get reproducible, auditable,
ML-ready data without writing — or trusting — yet another bespoke script.

## 📦 Installation

```bash
pip install freshdata                 # pandas + numpy, reporting, standard HTML visualization
pip install "freshdata[ml]"           # + scikit-learn (KNN imputation, IsolationForest)
pip install "freshdata[domains]"      # + PyYAML (finance, GS1, and GTFS packs)
pip install "freshdata[enterprise]"   # + polars, pyarrow, requests, pyyaml (enterprise layer + CLI)
pip install "freshdata[privacy]"      # + Presidio NER & pyffx (stronger PII detection / crypto FPE)
pip install "freshdata[entity-resolution]"  # + duckdb (probabilistic linkage at scale)
pip install "freshdata[all]"          # everything, including cleanlab
```

Requires **Python ≥ 3.9** and **pandas ≥ 1.5**. Verify the install:

```bash
python -c "import freshdata as fd; print(fd.__version__)"
```

## 🚀 Quickstart

```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("messy_export.csv")

# Clean with sensible, explainable defaults
result = fd.clean(df)
cleaned = result.data
report = result.report()

print(report.summary())        # human-readable audit trail
report.to_frame()              # decisions as a DataFrame
report.to_dict()               # JSON-friendly for logging / dashboards
result.visualize()             # self-contained HTML action timeline + audit ledger
report.show()                  # inline in notebooks, or writes a standalone .html file
```

Interactive output, decision memory, drift, debt, joins, encoding, and
stakeholder summaries (see [Interactive output](docs/interactive.md) and the
[Decision-preserving workflow](docs/decision-workflow.md) guides):

```python
fd.profile(df).show()                                   # inline quality cockpit
fd.suggest_plan(df).show()                              # per-column decision cards

# Remember reviewed decisions and replay them on next week's data
memory = fd.learn_cleaning_memory(df, decisions=report, dataset_id="crm")
cleaned, report = fd.clean(df_next, memory=memory, return_report=True)

diff  = fd.compare_to_baseline(df, baseline=last_week_df, key="customer_id")
_, gate = fd.evaluate_quality_debt(df, debt_policy="warn_then_fail",
                                   ledger="quality_debt.sqlite")
keys  = fd.suggest_join_keys(left, right, on=["company_name"], exact_within=["country"])
lint  = fd.lint_text_encoding(df, columns=["name", "city"])
brief = fd.stakeholder_summary(report, audience="business", format="markdown")
```

Standard report visualization is included in the base install. Optional
visualization extras only add richer third-party notebook/table integrations:
`pip install 'freshdata[viz]'` (or `[notebook]`, `[all]`).

Domain packs add versioned validation and separately audited repairs:

```python
ledger, report = fd.clean(df, domain="finance", return_report=True)
catalog = fd.clean(df, domain="retail")
stops = fd.clean(stops_df, domain="transport", gtfs_file="stops.txt")
feed, report = fd.clean(
    {"stops.txt": stops_df, "routes.txt": routes_df, "trips.txt": trips_df},
    domain="transport",
    return_report=True,
)
telemetry, report = fd.clean(scada_df, domain="energy", return_report=True)
```

The transport v0.1 pack validates `stops.txt`, `routes.txt`, `trips.txt`, and
`stop_times.txt`. Other GTFS files are preserved and explicitly reported as not
covered, rather than being silently treated as validated.

The energy v0.1 pack validates SCADA / Modbus telemetry — one row per
`(asset_id, register_address, timestamp)` reading — checking the 16-bit register
range, public Modbus function codes, OPC/SCADA point quality (with synonym
coercion), and engineering units. Asset IDs are never imputed, and bad/stale/uncertain
readings are flagged for audit rather than silently dropped. Because the validator is
stateless per frame, `fd.clean(batch, domain="energy")` composes with micro-batch
streaming.

**Format parsers** turn raw messages into DataFrames you can then clean. `fd.parse_domain`
reads FHIR R4 JSON, HL7 v2, GPX, SDMX, and EDIFACT (structural parsing; malformed input is
audited, not fatal), and `fd.clean_domain_file` parses then cleans in one call:

```python
result = fd.parse_domain(fhir_bundle, format="fhir")    # patient/observation/encounter/condition/...
conditions = fd.clean_domain_file("bundle.json", format="fhir", domain="healthcare", frame="condition")
```

The healthcare pack validates **Patient, Observation, Encounter, Condition, and
MedicationRequest** (FHIR R4), with UCUM units, ICD-10/LOINC/SNOMED code systems, and
PHI-safe patient IDs.

**Market tick data** validates with `fd.clean(ticks, domain="finance", finance_mode="tick")`
— ISO-8601 timestamps, positive price/size, ISO-4217 currency, no crossed quotes, and
BCBS-239/SOX-style completeness, with symbols/exchanges treated as never-imputed IDs. See
the [parsers & tick-mode guide](https://freshcode-org.github.io/freshdata/parsers/).

Preview the engine's choices *before* touching your data:

```python
print(fd.profile(df))                    # read-only data-quality report
print(fd.suggest_plan(df).summary())     # the exact plan clean() would run
print(fd.compare_plans(df))              # strategies side by side
```

## 🔁 Before vs after

<table>
<tr><th>Before — raw export</th><th>After — <code>fd.clean(df)</code></th></tr>
<tr><td>

| First Name | AGE | Salary($) | empty |
|---|---|---|---|
| ` Ann ` | `34` | `$1,200.50` | |
| `Bob` | `N/A` | `-` | |
| `Bob` | `N/A` | `-` | |
| `Cara` | `41` | `$2,000` | |

*whitespace, `N/A`/`-` sentinels, currency strings, an all-empty column, a duplicate row, text dtypes*

</td><td>

| first_name | age | salary | age_was_missing |
|---|---|---|---|
| Ann | 34 | 1200.50 | False |
| Bob | 38 | _Missing_ | True |
| Cara | 41 | 2000.00 | False |

*snake_case names, real `Int64`/`float64` dtypes, sentinels → missing → imputed, duplicate dropped, empty column removed, a missingness indicator added*

</td></tr>
</table>

Every one of those changes appears in `report.summary()` with a rationale, risk
level, and confidence score — no silent mutations.

## 🧩 Core API

| name | purpose |
|---|---|
| `fd.clean(df, *, return_report=False, config=None, **options)` | clean, optionally returning a `CleanReport` |
| `fd.profile(df, *, include_plan=False, **options)` | read-only inspection with actionable issues |
| `fd.suggest_plan(df, **options)` | dry-run: primary + alternative models per column |
| `fd.compare_plans(df, *, strategies=...)` | side-by-side models across strategies |
| `fd.compare_clean(df, *, strategies=...)` | side-by-side actual clean outcomes |
| `fd.explain_clean(df, **options)` | what `clean()` did and why, plus inferred roles |
| `fd.Cleaner(config=None, **options)` | reusable configured pipeline (`.clean()`, `.report_`) |
| `fd.CleanConfig` | frozen dataclass holding every option |
| `fd.CleanReport` / `fd.Action` | audit trail with rationale / risk / confidence |

```python
# Tune the engine — explicit choices always override the defaults
cleaned = fd.clean(
    df,
    strategy="balanced",          # "aggressive" | "conservative"
    target_column="churn",        # never modified (no leakage)
    id_columns=("customer_id",),  # never imputed
    preserve_columns=("notes",),  # never dropped
    outlier_method="iqr",         # "zscore" | "auto" | "isolation_forest"
    return_report=True,
)

# Reusable pipeline across many files
cleaner = fd.Cleaner(target_column="churn")
for path in paths:
    out = cleaner.clean(pd.read_csv(path))
    log.info(cleaner.report_.summary())
```

### MissForest-style imputation (optional `[ml]`)

For nonlinear, mixed-type tabular data, opt into the research-inspired
MissForest-style imputer. It trains random forests iteratively across numeric,
categorical, and boolean predictors, while preserving FreshData's role gates:
targets, IDs, and free-text columns are not fabricated.

```python
# pip install "freshdata[ml]"
cleaned, report = fd.clean(
    df,
    impute_method="missforest",
    target_column="churn",
    id_columns=("customer_id",),
    return_report=True,
)
```

You can also use it only where it helps and keep simple fills elsewhere:

```python
cleaned, report = fd.clean(
    df,
    impute_strategy={
        "age": "missforest",
        "income": "median",
        "segment": "missforest",
    },
    return_report=True,
)
```

Use MissForest when missing numeric and categorical fields depend on nonlinear
relationships across other columns. Avoid it for tiny frames, very sparse
columns, high-cardinality identifiers, free text, and latency-sensitive cleaning:
it is slower than median/mode/KNN because it trains random forests and records a
per-column audit trail with model type, iterations, convergence, confidence, and
fallback reasons.

<details>
<summary><b>How the cleaning engine works (two layers)</b></summary>

**Layer 1 — representation repair** (always on):

| order | step | what it does |
|---|---|---|
| 1 | `column_names` | snake_case names, deduplicate collisions (`"a", "a"` → `"a", "a_2"`) |
| 2 | `strip_whitespace` | trim surrounding whitespace in text cells |
| 3 | `normalize_sentinels` | `"N/A"`, `"null"`, `"-"`, `""`, `"#REF!"`, … → missing |
| 4 | `drop_empty_columns` / `drop_empty_rows` | remove all-missing columns and rows |
| 5 | `fix_dtypes` | text → numeric (`"$1,234.56"` works) / datetime / boolean, validated |
| 6 | `drop_duplicates` | resolve duplicate rows (`first`/`last`/`drop`/`aggregate`) |

**Layer 2 — the decision engine** (`strategy="balanced"`, the default) infers
each column's role and applies explicit threshold rules:

| missing ratio | numeric | categorical | datetime |
|---|---|---|---|
| ≤ 5% | mean if ~normal & no outliers, else median | mode if clear majority, else `"Unknown"` | ffill/bfill if time-ordered |
| 5–30% | median (KNN only in aggressive mode) | mode if dominant, else `"Missing"` | ffill/bfill if time-ordered |
| > 30% | **preserved** + warning (balanced) | same | same |

Role gates run first: **targets are never modified**, **IDs are never imputed**,
**free text is never force-filled**. Outliers in ID/target columns,
`preserve_columns`, and domain-sensitive columns (AQI, pollutants, fraud/risk
names) are always preserved — there the extremes usually *are* the signal.

</details>

## ⚡ Performance highlights

Typical throughput on a modern laptop (vectorized pandas/NumPy, one-pass engine
caching — no C extension required):

| Dataset size | Balanced | Aggressive |
|---|---|---|
| 500 rows | < 0.5 s | < 1 s |
| 3,000 rows | < 2.5 s | < 6 s |
| 29k rows (full AQI) | < 5 s | KNN gated |

```bash
python benchmarks/bench_quick.py --fixtures --compare   # quick: tests/fixtures corpus, side by side
python benchmarks/bench.py run                            # full nine-metric harness, writes results/
```

See [docs/benchmarks.md](docs/benchmarks.md) and `benchmarks/README.md` for the
reproducible nine-metric benchmark harness and the enterprise fixture library.

## 🏆 CleanBench: benchmarked, not just claimed

Every trust claim below is backed by a committed result — not a marketing
number. Reproduce it yourself with one command:

```bash
python -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site --reproduce-headline
python -m benchmarks.cleanbench --verify-results benchmarks/cleanbench/results/latest.json
python -m benchmarks.cleanbench.reproducibility audit-readme
```

Committed results: [`benchmarks/cleanbench/results/latest.json`](benchmarks/cleanbench/results/latest.json) ·
[`latest.md`](benchmarks/cleanbench/results/latest.md) · [docs/benchmarks.md](docs/benchmarks.md)

| Release gate | Threshold | This build | Status |
|---|---|---|---|
| Protected-column violation rate | = 0 | 0.0 | ✅ |
| False modification rate | ≤ 0.1% | 0.0% | ✅ |
| Precision @ confidence ≥ 0.95 | ≥ 0.99 | 1.0 | ✅ |
| Confidence ECE | ≤ 0.03 | 0.038 | ❌ (default calibration table; the trained `calib-v1` artifact clears this gate — see [docs/benchmarks.md](docs/benchmarks.md)) |
| Runtime network calls (`fd.clean`, `fd.learn`, `fd.compile_context`) | 0 | 0 | ✅ |
| Model weights in the wheel | 0 | 0 | ✅ |

That ECE row is deliberately left failing here: **this table is generated from
the actual committed result, gate failures included** — CleanBench does not
hide a gate it doesn't pass. See `benchmarks/cleanbench/results/latest.json`
for the full metric set (all five tracks) and `--verify-results` to confirm
this table wasn't hand-edited.

Baseline comparison (T1 representation-repair; pandas hand-written / pyjanitor
where installed / Great Expectations validation-only / disclosed LLM-agent
skipped by default) lives in the same committed result under `"baselines"`.

## 🌊 Streaming / micro-batch cleaning

When a dataset is too big to hold in memory, feed it through `StreamingCleaner` in
**micro-batches**. It keeps *bounded* running statistics across batches (Welford
mean/variance, reservoir-sampled medians, Space-Saving top-k categories), so memory
stays flat whether you push 100k rows or 100M — and every batch still produces the same
explainable `CleanReport`, with a per-batch, rolling, and cumulative trust score.

```python
import freshdata as fd

cleaner = fd.StreamingCleaner(
    target_column="churn",
    id_columns=("customer_id",),
    window_size=100_000,
    warmup_batches=3,
    strategy="balanced",
)

# Any iterable of batches — e.g. pandas chunked CSV reading:
batches = pd.read_csv("events.csv", chunksize=100_000)
for cleaned_batch, report in cleaner.clean_batches(batches):
    write(cleaned_batch)
    log(report.to_dict())          # includes report.streaming: batch_id, trust scores, drift

final_report = cleaner.finalize()  # cumulative summary across the whole stream
```

It accepts **pandas**, and (when installed) **PyArrow** `Table`/`RecordBatch` and
**polars** `DataFrame`/`LazyFrame` batches, coerced safely without ever concatenating the
stream. Optional source connectors live behind extras:

```python
# pip install "freshdata[kafka]"   /   "freshdata[flight]"
cleaner.clean_kafka(topic="events", bootstrap_servers="localhost:9092", batch_size=10_000)
cleaner.clean_arrow_flight("grpc://localhost:8815", batch_size=100_000)
```

From the CLI (reads CSV/Parquet batch-by-batch, writes per-batch + summary reports, and
exits non-zero if the trust gate fails):

```bash
freshdata stream events.csv --batch-size 100000 -o out.parquet --report reports/ \
    --target-column churn --id-columns customer_id --fail-under-trust 80

# Prove stable memory at scale (generates rows lazily — never materialized at once):
python benchmarks/bench_streaming.py --rows 100000000 --batch-size 100000 --cols 20
```

**Be aware** — streaming mode is honest about its trade-offs: it is *micro-batch*, not
true row-by-row real time; cross-batch duplicate detection is a bounded recent-window
approximation (not global) unless explicitly enabled; medians/quantiles are approximate
(reservoir-sampled); Kafka and Arrow Flight are optional integrations; and the
enterprise-scale "100M rows out-of-core with stable memory" claim depends on the
`bench_streaming.py` benchmark passing in *your* environment. See the
[Streaming guide](https://freshcode-org.github.io/freshdata/streaming/) for details.

## 🧠 Semantic cleaning layer

Some defects are not about *representation* (whitespace, dtypes, duplicates) but about
*meaning*: `"twenty"` where a number belongs, `"$1,200.50"` in a money column,
`"yes"`/`"y"` booleans, `"10 kg"` unit strings, or `"M"`/`"male"`/`"Male "` variants. The
**Semantic Cleaning Layer** finds these, proposes scored repairs, and applies only the
safe ones — recording every decision (applied, suggested, or skipped) in the
`CleanReport`.

It is **off by default**. `fd.clean(df)` behaves exactly as before unless you opt in with
`semantic_mode`:

```python
import freshdata as fd

# assist: detect and report only — never mutates your data.
cleaned, report = fd.clean(df, semantic_mode="assist", return_report=True)
for a in report:
    if a.step == "semantic":
        print(a.status, a.column, a.description, a.confidence)
```

```python
# auto: apply high-confidence, low-risk, policy-approved repairs.
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

Modes: `"assist"` (suggest only), `"review"` (apply zero-risk deterministic repairs,
suggest the rest), `"auto"` (apply high-confidence low-risk repairs that pass policy).

### Date phrase repair

`DatePhraseExpert` normalizes ISO/slash/dash dates, month-name dates, and
`"today"`/`"yesterday"`/`"tomorrow"` in date-like columns — but relative phrases only
resolve with an explicit `reference_date` (never the real wall-clock date), and
ambiguous numeric dates (day/month both `<= 12`) are never auto-applied without an
explicit `dayfirst` hint:

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

### Semantic memory replay

Accepted semantic repairs can be learned into a `CleaningMemory` and replayed as
policy-gated candidate proposals on similar future data:

```python
cleaned, report = fd.clean(df, semantic_mode="auto", return_report=True)

memory = fd.learn_cleaning_memory(df, decisions=report, dataset_id="crm_signups")

next_cleaned, next_report = fd.clean(
    next_df, semantic_mode="auto", memory=memory, return_report=True,
)
```

Memory is local/server-free, never overrides policy (protected columns, confidence
floor, and the ambiguity rules above still apply), and every replayed decision is
audited with `memory_influenced=True` and `model_id="semantic:<issue_type>:memory"`.

**Guarantees and honesty about scope:**

- **Off by default** — existing behavior is byte-for-byte unchanged unless enabled.
- **Deterministic and fully offline** — no LLM, no network calls, no API keys, no vector
  database. Semantic memory replay is a local, no-dependency retrieval mechanism, not a
  neural/embedding search; the interfaces still leave a clean extension point for an
  optional future private-LLM candidate generator.
- **An LLM never mutates the DataFrame.** Every change goes through profile → propose →
  score → policy → (maybe) apply.
- **Ambiguous repairs are suggestions, not silent mutations** — they are recorded with
  `status="suggested"` and flagged for human review.
- **ID, target, and `preserve_columns` are protected** — identifier-like columns are
  vetoed unless `semantic_context` explicitly marks them `mutable`.
- **Every proposal is audited** in the report with rationale, risk, confidence, status,
  and `model_id` (e.g. `semantic:spelled_number:v1`, `semantic:date_phrase:memory`).
- On Polars/DuckDB, semantic cleaning runs over a **natively extracted distinct
  table** (no full-frame pandas materialization) and applies repairs natively;
  only a non-default `semantic_backends` choice or a learned `profile=` still
  falls back to pandas, recorded in the report. Spark always falls back to
  pandas today. See [docs/limitations.md](docs/limitations.md#native-engine-semantic-cleaning-phase-6).

## 🔌 Plugins

Extend FreshData with custom **semantic experts**, **semantic backends**, and
**validators** — via entry points (installed packages) or explicit registration:

```python
import freshdata as fd
from my_pkg import MyExpert

fd.testing.expert_contract(MyExpert())   # verify the contract before registering
fd.register_expert(MyExpert())
cleaned, report = fd.clean(df, semantic_mode="auto", return_report=True)
```

Plugins **only propose or validate** — they can never touch the DataFrame
directly. Every proposal flows through the same policy gate and byte-identity
guard as the built-ins, so a plugin can never change a protected column or force
an auto-apply. A declared `max_risk` is a hard ceiling; a plugin that declares
`uses_network = True` is **disabled by default**; a plugin that raises is
isolated (the clean still completes); and every applied plugin repair is
attributed in the report (`metadata["plugin"]`, `model_id="...:plugin:<name>"`).

See the [plugin authoring guide](https://freshcode-org.github.io/freshdata/plugins/)
and runnable [`examples/plugins/`](examples/plugins/).

## 🛡️ Enterprise: drift, privacy & entity resolution

Three opt-in enterprise capabilities sit alongside trust scoring, clustering, and
lineage. All keep the base install dependency-free, never mutate the input frame,
and return JSON-serialisable reports.

### Schema-drift & data-contract monitoring

Record a versioned, **PII-safe** baseline from a trusted dataset, persist it as JSON,
then monitor new batches for schema drift, distribution drift (dependency-free **KS** /
**PSI**), `DataContract` violations, and a trust-score gate.

```python
import freshdata as fd
from freshdata import ColumnContract, DataContract

contract = DataContract(name="customers", trust_score_min=70.0, columns=(
    ColumnContract(name="age", dtype="float64", min_value=0, max_value=120),
    ColumnContract(name="country", allowed_values=("US", "GB", "FR")),
))

baseline = fd.build_baseline(trusted_df, name="customers", contract=contract)
fd.save_baseline(baseline, "customers.baseline.json")            # raw values are NOT stored

report = fd.monitor_contract(new_df, baseline_path="customers.baseline.json")
print(report.summary())          # PASS/FAIL with per-check KS/PSI/contract findings
assert report.passed
```

### Stronger PII detection + reversible / format-preserving anonymization

A Presidio-style but dependency-free detector (regex + context keywords; optional
Presidio NER via `[privacy]`), reversible **tokenization** with a vault, **surrogate**
format-preserving anonymization, HIPAA/GDPR-tagged audit events, and k-anonymity.

```python
import freshdata as fd
from freshdata.enterprise import MaskingRule, PIIDetectionConfig

scan = fd.detect_pii(df, config=PIIDetectionConfig())            # spans redacted by default

rules = (MaskingRule(name="email", columns=("email",), strategy="tokenize",
                     reversible=True, key_env="FD_TOKEN_KEY", entity_types=("EMAIL",)),)
clean, report = fd.anonymize(df, rules=rules, detection_config=PIIDetectionConfig())
print(report.summary())          # events carry hipaa_tag / gdpr_tag / risk_level

fd.check_k_anonymity(df, ["zip", "gender"], k=5)                 # re-identification risk
```

Reports redact raw previews by default (`audit_include_pii=True` to include them). The
`surrogate`/fallback `fpe` mode is *format-preserving but not cryptographic FPE* and is
flagged as such in report metadata.

### Probabilistic entity resolution at scale

A Splink-style, **DuckDB-backed** record-linkage backend (with a pandas fallback) that
blocks candidate pairs via SQL, scores them with weighted comparisons, and clusters
records via connected components — with a hard `max_pairs` safety gate.

```python
import freshdata as fd
from freshdata.enterprise import BlockingRule, ComparisonLevel, EntityResolutionConfig

config = EntityResolutionConfig(
    unique_id_column="id", backend="duckdb",
    blocking_rules=(BlockingRule(sql="l.dob = r.dob"),),       # no full cartesian product
    comparisons=(
        ComparisonLevel(column="name", kind="jaro_winkler", threshold=0.85, weight=3.0),
        ComparisonLevel(column="dob", kind="exact", weight=1.0),
    ),
    match_threshold=0.85,
)
resolved, report = fd.resolve_entities(people_df, config=config)
print(report.summary())          # candidate pairs → matches → entity clusters
```

Built around this core are a full dedupe subsystem: a **human-review queue**
(`build_review_queue` / `export_review_queue` / `load_review_decisions` /
`apply_review_decisions`) with clerical feedback and optional weight
recalibration, **explainable per-field match weights** (`report.to_frame()`,
with PII previews redactable via a privacy config), configurable **golden-record
merge policies** (`GoldenRecordPolicy` / `merge_entities`, with per-field source
lineage), and ready-made **domain templates** for education, healthcare, retail,
and media (`from freshdata.enterprise import healthcare_template`).

> **Splink-class linkage target; current implementation is rule-weighted, not
> EM-trained unless calibration is enabled.** See `benchmarks/bench_entity_resolution.py`
> for candidate-pair reduction, runtime, and precision/recall/F1 (pandas vs DuckDB).

See `examples/schema_drift_monitoring.py`, `examples/privacy_anonymization.py`, and
`examples/entity_resolution_duckdb.py` for runnable end-to-end demos.

## 📊 How FreshData compares

| Capability | **freshdata** | pandas | pyjanitor | Great Expectations | sweetviz | cleanlab |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| One-call automatic cleaning | ✅ | ❌ | ➖ | ❌ | ❌ | ❌ |
| Per-column decisions by inferred role | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Missing-value imputation (smart) | ✅ | ➖ | ➖ | ❌ | ❌ | ❌ |
| Outlier detection & handling | ✅ | ❌ | ❌ | ➖ | ➖ | ✅ |
| Duplicate resolution | ✅ | ➖ | ✅ | ❌ | ❌ | ❌ |
| Dtype / format repair | ✅ | ➖ | ✅ | ❌ | ❌ | ❌ |
| Explainable audit trail | ✅ | ❌ | ❌ | ➖ | ❌ | ➖ |
| Data profiling | ✅ | ➖ | ❌ | ➖ | ✅ | ❌ |
| Data validation / quality gates | ✅¹ | ❌ | ❌ | ✅ | ❌ | ❌ |
| PII masking | ✅¹ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Label-noise (ML) detection | ✅¹ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Polars support | ✅ | ❌ | ❌ | ➖ | ❌ | ❌ |

✅ built-in · ➖ partial / manual · ❌ not a goal · ¹ via the optional enterprise layer

## 🌍 Real-world use cases

- **ML preprocessing** — turn raw CSVs into leakage-aware, typed feature matrices
  before scikit-learn / XGBoost, without imputing IDs or touching the label.
- **Analytics & BI ingestion** — clean CRM, finance, and survey exports
  (currency strings, `N/A` sentinels, duplicate rows) on the way into a warehouse.
- **Data-quality gates in ETL** — run the enterprise CLI in Airflow/Prefect/cron;
  fail the job when the Data Trust Score drops below a threshold.
- **Exploratory data analysis (EDA)** — `fd.profile(df)` surfaces missingness,
  dtype issues, and duplicates before you commit to a modeling approach.
- **Notebook hygiene** — replace ad-hoc `dropna`/`fillna` cells with one
  auditable, reproducible call.

## 🛠️ Example pipeline

```python
import pandas as pd
import freshdata as fd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

raw = pd.read_csv("customers.csv")

# 1. Clean with the target protected from leakage
clean_df, report = fd.clean(raw, target_column="churn", return_report=True)
assert not report.warnings, report.warnings        # gate on data quality

# 2. Split & model on AI-ready data
X = pd.get_dummies(clean_df.drop(columns="churn"))
y = clean_df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

model = RandomForestClassifier(random_state=0).fit(X_tr, y_tr)
print("accuracy:", model.score(X_te, y_te))
```

See [`examples/`](examples/) for 8 runnable scripts and [`notebooks/`](notebooks/)
for narrated walkthroughs.

<details>
<summary><b>Enterprise layer — clustering, PII masking, trust scores, lineage, CLI</b></summary>

```python
from freshdata.enterprise import (
    clean_enterprise, EnterpriseConfig, ClusterConfig, MaskingRule, SemanticValidatorConfig,
)

ec = EnterpriseConfig(
    enable_clustering=True,
    clustering=ClusterConfig(columns=("vendor",)),       # merge "Acme Inc" / "ACME  inc"
    masking=(MaskingRule(name="pii", columns=("email",), strategy="hash", salt="…"),),
    semantic=(SemanticValidatorConfig(name="iso", kind="reference",
              columns=("country",), reference=("US", "CA", "GB")),),
    fail_under_trust=80,                                  # quality gate
)
result = clean_enterprise(df, enterprise=ec)             # df may be pandas OR polars
print(result.quality.to_markdown())                      # before/after trust report
result.lineage.emit("lineage.json")                      # OpenLineage RunEvents
assert result.passed_gate
```

Batch CLI (exits non-zero when the trust gate fails):

```bash
freshdata clean in.csv -o out.parquet --mask email:hash --cluster vendor \
    --report quality.json --lineage lineage.json --fail-under-trust 80
freshdata trust in.csv --fail-under 90
freshdata profile in.csv --json
```

</details>

## 📚 Documentation

Full documentation lives at **<https://freshcode-org.github.io/freshdata/>**:

- [Installation](https://freshcode-org.github.io/freshdata/installation/)
- [Quickstart](https://freshcode-org.github.io/freshdata/quickstart/)
- [Cleaning engine](https://freshcode-org.github.io/freshdata/cleaning-engine/)
- [Data profiling](https://freshcode-org.github.io/freshdata/data-profiling/)
- [API reference](https://freshcode-org.github.io/freshdata/api-reference/)
- [Examples](https://freshcode-org.github.io/freshdata/examples/)
- [Benchmarks](https://freshcode-org.github.io/freshdata/benchmarks/)
- [FAQ](https://freshcode-org.github.io/freshdata/faq/)

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). Quick start:

```bash
git clone https://github.com/FreshCode-Org/freshdata
cd freshdata
pip install -e ".[dev,ml,polars]"
pre-commit install
pytest && ruff check src tests && mypy src/freshdata
```

Security issues: see [SECURITY.md](SECURITY.md) for private disclosure.

## 🗺️ Roadmap

- [x] Per-column decision engine with explainable reports (0.3)
- [x] Enterprise layer: clustering, masking, trust score, lineage, CLI (0.4)
- [x] Documentation site + examples + packaging governance (0.5)
- [ ] Pluggable custom cleaning rules / strategy registry
- [ ] Native Polars cleaning engine (beyond the adapter)
- [ ] HTML/interactive profiling report
- [ ] Config-as-YAML for the core cleaner (not just the CLI)
- [x] 1.0 — stable public API

Have an idea? [Open a discussion or issue.](https://github.com/FreshCode-Org/freshdata/issues)

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Maintainer

Built by **Johnny Wilson Dougherty**
([@JohnnyWilson](https://github.com/JohnnyWilson16)).

Maintained by **Kevin Costner**
([@kevincostner17](https://github.com/kevincostner17)).

If `freshdata` saves you time, please ⭐ the
[repository](https://github.com/FreshCode-Org/freshdata) — it genuinely helps
others discover the project.
