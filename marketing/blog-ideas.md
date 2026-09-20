# Engineering Blog Post Outlines & Technical Backlog

High-depth technical articles for publication on Medium (Towards Data Science), Dev.to, Hacker Noon, and the FreshData engineering blog.

---

## 1. "Why Blind `fillna()` Silently Corrupts Production ML Models"

* **Category**: Problem & Architecture
* **Target Audience**: Data scientists, ML engineers
* **Key Narrative**:
  * The illusion of the clean dataset: why zero-NaN counts give a false sense of security.
  * Real-world disaster stories: imputing customer keys leading to fraudulent account linkages; imputing patient targets causing false test evaluations.
  * Statistical distortion: comparing mean vs median vs mode on log-normal wealth/income distributions.
  * The solution: Role-aware schema profiling and preserving ambiguity when confidence is low.
* **Runnable Code Asset**: `examples/02_missing_values.py` and `examples/05_ml_preprocessing.py`.

---

## 2. "How FreshData Decides What NOT to Change: The Engineering of Safety Invariants"

* **Category**: Engineering Deep Dive
* **Target Audience**: Senior data engineers, software architects
* **Key Narrative**:
  * Why automated cleaners fail: aggressive heuristic overreach.
  * The architecture of FreshData's decision engine:
    * Stage 1: Representation repairs (whitespace, sentinels, snake_case).
    * Stage 2: Profiler evaluation (skewness, uniqueness, cardinality).
    * Stage 3: Role inference (`id`, `target`, `categorical`, `numeric`, `date`).
    * Stage 4: Safety gate verification.
  * Testing for invariants: how we enforce a 0.0% false-repair rate across 100k-row test suites.
* **Runnable Code Asset**: `tests/test_audit_core_defaults.py` and `tests/test_guard_protected.py`.

---

## 3. "Cleaning 25 Million Rows Past RAM: How We Built DuckDB and Polars Out-of-Core Execution"

* **Category**: Benchmarks & Performance
* **Target Audience**: Performance engineers, big data developers
* **Key Narrative**:
  * The memory wall: why a 2 GB CSV consumes 8 GB in pandas RAM.
  * Evaluating streaming and chunking architectures vs. Arrow-native engines.
  * DuckDB and Polars execution backends: memory footprint comparisons (200 MB vs 1,046 MB at 1M rows).
  * Transparent fallback policies: `fallback_policy="warn" | "error"` ensuring developers are never surprised by materialization.
* **Runnable Code Asset**: `benchmarks/bench_outofcore.py` and `examples/integrations/duckdb_workflow.py`.

---

## 4. "Building an Audit Trail for Automated Data Cleaning in Regulated Industries"

* **Category**: Enterprise & Governance
* **Target Audience**: FinTech, HealthTech, and compliance engineers
* **Key Narrative**:
  * Regulatory requirements (BCBS 239, GDPR, HIPAA) around data lineage and automated modifications.
  * Transitioning from black-box transformations to auditable Action records.
  * Exporting `CleanReport` to machine-readable JSON for compliance archives.
* **Runnable Code Asset**: `examples/01_csv_cleaning.py` and `examples/interactive_html_reports.py`.
