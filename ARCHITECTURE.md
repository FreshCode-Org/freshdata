# Architecture

This is a map for contributors: what lives where in `src/freshdata`, how a
`clean()` call flows through the code, and where to add new behavior. It is
deliberately high-level — read it once before your first code change, then let
the module docstrings and tests fill in the detail.

For *user*-facing explanations of features, see the
[documentation site](https://freshcode-org.github.io/freshdata/). This file is
about the *code*.

## The one-sentence version

`fd.clean(df)` profiles every column, asks a rule-based **decision engine** which
action each column needs, applies those actions as small vectorized **steps**,
and returns the cleaned frame plus a **report** that explains every change.

## Core flow

```
fd.clean(df)                      # api.py — the public entry point
      │
      ▼
  Cleaner                         # cleaner.py — orchestrates the pipeline
      │
      ├─ profile the frame        # profile.py — per-column stats & inferred role
      │
      ├─ decision engine          # engine/ — pick an action per column
      │     ├─ missing.py         #   imputation strategy
      │     ├─ outliers.py        #   outlier handling
      │     ├─ context.py         #   column-role / context inference
      │     └─ model_select.py    #   when to use model-based imputation
      │
      ├─ apply cleaning steps     # steps/ — pure vectorized transforms
      │     ├─ missing.py  dtypes.py  duplicates.py
      │     ├─ outliers.py strings.py columns.py
      │     └─ prune.py    memory.py
      │
      └─ build the report         # report.py — every Action, with rationale,
                                  #   risk, confidence, affected-cell count
```

Two invariants hold everywhere in this path and are enforced in review (see
[CONTRIBUTING.md](CONTRIBUTING.md)):

1. **Everything is reported.** A transformation that changes data must record an
   `Action` (see `report.py`) with an affected-cell count.
2. **Vectorized only.** No row-wise `apply` or Python loops over rows in the
   cleaning path.

## Top-level modules

| Module | Responsibility |
|---|---|
| `api.py` | The public convenience functions: `clean`, `clean_csv`, `profile`, `plan`, `learn`, `mask_dataframe`, `link`. Start here to trace any public call. |
| `cleaner.py` | The `Cleaner` front-end and the pipeline that wires profiling → engine → steps → report. |
| `profile.py` | Read-only profiling: what is in a frame and what `clean()` *would* do, with no mutation. |
| `plan.py` / `repairplan.py` | Dry-run plans (`plan.py`) and executable, reviewable repair plans (`repairplan.py`). |
| `report.py`, `result.py`, `_reportframe.py` | The report objects returned to users — `Action` records, summaries, `to_dict()`. |
| `provenance.py` | Provenance-aware cleaning for OCR / PDF / document-derived tables. |
| `guard.py` | The hard safety gate that physically protects columns from modification. |
| `findings.py`, `fieldcheck.py`, `textclean.py`, `textlint.py` | Per-cell validation (`fieldcheck`), field-aware text cleaning (`textclean`), and the normalized finding model (`findings`). |
| `insight.py`, `stakeholder.py`, `explain.py` | Presentation layers: action-oriented insight output, business-language stakeholder summaries, and per-decision explanations. |
| `config.py`, `_util.py`, `_sentinels.py` | Configuration and shared internals. |

## Subpackages

Each subpackage has a one-line docstring at the top of its `__init__.py` —
these summaries come from that code.

| Package | What it does |
|---|---|
| `engine/` | The rule-based decision engine behind `strategy=` — picks an action per column. |
| `steps/` | Individual cleaning steps, each a pure vectorized function. |
| `imputation/` | Internal imputation engines (mean/median/mode → KNN/model-based with the `ml` extra). |
| `domains/` | Domain-specific validator packs (finance, healthcare, energy, retail, transport, education, agriculture, media) plus reference data. **The most self-contained place to start** — see [CONTRIBUTING_DOMAINS.md](CONTRIBUTING_DOMAINS.md). |
| `semantic/` | The offline Semantic Cleaning Layer — experts, scoring, routing. Off by default. |
| `context/` | The deterministic natural-language context compiler (plain-English rules → policy). |
| `enterprise/` | Clustering / entity resolution, trust scoring, lineage, and PII masking. |
| `compliance/` | Maps a `CleanReport` to named control frameworks (21 CFR Part 11, GDPR, HIPAA, SOX, ALCOA+). |
| `execution/` | Pluggable out-of-core backends (Polars / DuckDB / Spark) behind a shared logical plan. |
| `streaming/` | Micro-batch and time-series-aware cleaning in bounded memory. |
| `parsers/` | Structural format readers (HL7 v2, GPX, SDMX, EDIFACT, FHIR). |
| `render/` | Interactive HTML rendering of report objects (the `viz` extra). |
| `models/` | Optional local model registry and runtime (`fd.models`). |
| `learning/` | Paired-data learning: `fd.learn` and reusable `.fdprofile` profiles. |
| `integrations/` | Run `clean` + a trust gate inside orchestrators (Dagster / Airflow / dbt). |
| `adapters/` | Optional framework adapters (Polars, etc.). |
| `experimental/` | Features whose APIs may change between minor releases (e.g. the AI Copilot). |
| `plugins.py` + `testing.py` | The entry-point plugin system and `fd.testing` contract helpers for third-party experts/backends/validators. |

## Where do I add…?

- **A new cleaning behavior** → a step in `steps/`, wired through the engine in
  `engine/`, with an `Action` recorded in the report. Add the "does not fire
  when it shouldn't" test.
- **A new domain validator** → a pack in `domains/`. Follow
  [CONTRIBUTING_DOMAINS.md](CONTRIBUTING_DOMAINS.md) — this path is designed to
  be self-contained and is the friendliest first code contribution.
- **A new imputation method** → `imputation/`, selected from `engine/model_select.py`.
- **A new execution backend** → `execution/` + a plugin registration; see
  [`docs/plugins.md`](https://freshcode-org.github.io/freshdata/plugins/).
- **A new output/report surface** → `render/`, `insight.py`, or `stakeholder.py`.
- **A format parser** → `parsers/`.

## What to read next

- [CONTRIBUTING.md](CONTRIBUTING.md) — setup, the checks CI runs, and the PR workflow.
- [CONTRIBUTING_DOMAINS.md](CONTRIBUTING_DOMAINS.md) — the deep dive for domain packs.
- [Contributor roadmap](docs/community/contributor-roadmap.md) — tasks grouped by difficulty.
- The module docstring and the tests next to any file you plan to change.
