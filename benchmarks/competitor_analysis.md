# Competitor analysis

How FreshData relates to adjacent data-quality and data-cleaning tools. This
table is curated from each tool's public documentation and, for the two tools
the harness actually runs (pandas, pyjanitor), from measured results.

**Sourcing rules** (HARD CONSTRAINT 4 — no overclaiming):

- Capability columns (`does_it_repair`, `does_it_explain_repairs`,
  `does_it_preserve_ids`, `audit_trail`, `openlineage`) reflect the tool's
  *documented, out-of-the-box* behaviour, not what a user could hand-build on
  top of it.
- The `benchmark_claim` column only states a FreshData advantage when a
  gold-fixture or harness metric confirms it. Tools the harness does not run
  (HARD CONSTRAINT 2) carry a "static / not benchmarked" note instead of a
  performance number.
- "Repair" means the tool *changes data values to fix defects* as a first-class
  feature. Validation-only tools (assert/expect/test) are marked accordingly.

FreshData's harness-confirmed numbers, at the 10k-row scale, balanced mode
(`benchmarks/results/<run>/summary.json`):

| metric | FreshData result |
|---|---|
| Repair fidelity (gold fixture, all families) | **100%** (≥ 90% required) |
| False-repair rate on id / target / free-text | **0.0%** (all fixtures) |
| Preservation rate on non-null ids | **100%** (all fixtures) |
| Trust-score monotonicity | **valid** on crm, finance, event_log, gold |
| Export completeness | **100%** (all fixtures) |
| Authored-line reduction vs pandas baseline | **88.5%** (3 vs 26 lines) |
| Authored-line reduction vs pyjanitor baseline | **85.0%** (3 vs 20 lines) |

| tool | category | what_it_does | what_fd_does_differently | does_it_repair | does_it_explain_repairs | does_it_preserve_ids | audit_trail | openlineage | benchmark_claim |
|---|---|---|---|---|---|---|---|---|---|
| **pandas** | dataframe library | General-purpose data manipulation; cleaning is hand-authored (`fillna`, `astype`, `drop_duplicates`, sentinel `replace`). | One call (`fd.clean(df, return_report=True)`) chooses and applies the cleaning actions with rationale. | Yes (manual) | No | Only if hand-coded | No | No | **Confirmed**: 88.5% fewer authored lines (3 vs 26) for the same defect set; FreshData also faster + lower peak memory on the crm fixture in `compare`. |
| **pyjanitor** | pandas extension | Chainable cleaning verbs (`clean_names`, `remove_empty`, coercions). | FreshData decides *which* verbs to apply per column and records why; pyjanitor verbs are authored explicitly. | Yes (manual chain) | No | Only if hand-coded | No | No | **Confirmed**: 85.0% fewer authored lines (3 vs 20) for the same defect set. |
| **Great Expectations** | validation framework | Declarative expectation suites that *assert* data quality and produce data docs. | FreshData repairs and explains, not just validates; GE never mutates data. | No (validate only) | N/A | N/A (read-only) | Validation results / data docs | Via integrations | Static / not benchmarked (HARD CONSTRAINT 2). Different category: assertion vs repair. |
| **Soda** | validation / monitoring | SodaCL checks + scans for data-quality monitoring and alerting. | FreshData fixes and audits at clean time; Soda observes and alerts. | No (checks only) | N/A | N/A (read-only) | Scan results | Partial via catalog integrations | Static / not benchmarked. Monitoring, not repair. |
| **dbt** | transformation / testing | SQL transformations with `tests` (not_null, unique, accepted_values, relationships). | FreshData operates on in-memory frames with per-cell rationale; dbt tests pass/fail in-warehouse. | No (tests assert; transforms are user SQL) | N/A | Enforced via tests, not preserved automatically | Run results / manifest | Via dbt + OpenLineage integration | Static / not benchmarked. Test + transform, not automatic repair. |
| **AWS Glue Data Quality** | managed DQ | DQDL rulesets that score datasets and gate pipelines on AWS. | FreshData is library-local, repairs values, and explains each decision; Glue DQ scores and gates. | No (rule scoring) | N/A | N/A (scoring) | Rule outcomes in AWS | Via AWS lineage services | Static / not benchmarked (managed service, HARD CONSTRAINT 2). |
| **Google Dataplex** | governance / DQ | Auto data quality and profiling across a lakehouse. | FreshData is an in-process cleaner with a portable audit trail; Dataplex governs and profiles centrally. | No (profiling / scoring) | N/A | N/A | Catalog / DQ scores | Via Google lineage | Static / not benchmarked (managed service). |
| **OpenRefine** | interactive cleaning | GUI faceting, clustering, transforms for manual data wrangling. | FreshData is programmatic, deterministic and reproducible (seeded); OpenRefine is human-driven and interactive. | Yes (interactive) | Operation history (not per-decision rationale) | Manual | Operation history JSON | No | Static / not benchmarked. Interactive vs automated/reproducible. |
| **Dedupe** | entity resolution | ML-based fuzzy dedup / record linkage requiring labeled training. | FreshData's core clean removes *exact* duplicates safely and explains it; fuzzy ER is a separate enterprise feature (out of this harness's scope). | Yes (dedup) | Match scores | N/A | No | No | Static / not benchmarked. Different scope (probabilistic ER). |
| **ydata-profiling** | profiling (read-only) | Generates an HTML/JSON dataset profile report. | FreshData *acts* on the profile (repairs + report) in one call; ydata only describes. | No | N/A | N/A (read-only) | No (report only) | No | Timing-only baseline in the harness (`ydata_profiling_baseline`); skipped when not installed. Read-only — no repair to compare. |
| **sweetviz** | profiling (read-only) | EDA comparison reports. | FreshData repairs and audits; sweetviz visualises. | No | N/A | N/A (read-only) | No (report only) | No | Timing-only baseline (`sweetviz_baseline`); skipped on mixed-dtype columns it cannot profile. Read-only. |
| **cleanlab** | label quality (ML) | Detects label errors / outliers in ML datasets using model confidence. | FreshData repairs representation/structure defects with rationale and never imputes targets; cleanlab flags noisy *labels*. | No (flags label issues) | Issue scores | N/A | No | No | Static / not benchmarked. Complementary: label-quality vs data-cleaning. |

## Where FreshData does **not** claim an advantage

- **Validation depth**: Great Expectations, Soda and dbt express richer
  declarative assertion suites than FreshData's clean-time checks.
- **Fuzzy entity resolution**: Dedupe and FreshData's own enterprise ER module
  do probabilistic linkage that the core `fd.clean` benchmarked here does not.
- **Governance / cataloguing**: Dataplex and Glue operate at a
  platform/governance layer FreshData does not occupy.
- **Foreign-currency words & accounting negatives**: the generic balanced
  `fd.clean` strips `$` and thousands separators but not `EUR 500` or
  `(1234.56)`; those need the finance domain pack and are intentionally left
  out of the in-scope benchmark (HARD CONSTRAINT 5).

The honest summary: FreshData wins on **automatic, explained, id/target/text-safe
repair with a portable audit trail and minimal authored code** — which is
exactly what the gold-fixture safety metrics and the authored-line metric
confirm. It is not a validation framework, a governance platform, or a fuzzy
entity-resolution engine, and this table does not claim otherwise.
