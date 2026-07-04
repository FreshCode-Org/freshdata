# FreshData Benchmark Release

A reproducible benchmark harness and an enterprise-shaped fixture library for
FreshData. It measures nine standardized metrics — including the safety
invariants (id / target / free-text are never mutated) that define FreshData's
contract — against known ground truth, and compares FreshData to pandas and
pyjanitor baselines.

> The harness calls FreshData exactly as a user would. It never modifies library
> internals.

## Layout

```
benchmarks/
  bench.py                 # CLI harness (run / compare / report / fixtures / single)
  harness_metrics.py       # the nine metric implementations
  results_schema.py        # stable JSON schema for result files
  baselines/               # pandas / pyjanitor / ydata-profiling / sweetviz
  fixtures/                # crm, finance, event_log, wide_schema, provenance, gold
  results/                 # .gitignored, populated at runtime
  competitor_analysis.md   # static, curated competitor comparison
  README.md
  bench_quick.py           # legacy quick-bench script (kept for convenience)
  bench_missforest.py      # opt-in median/mode vs KNN vs MissForest comparison
```

## Install & run

```bash
pip install -e ".[dev]" jsonschema      # FreshData + dev deps
python benchmarks/bench.py run          # all fixtures, 10k rows, write results/
python benchmarks/bench.py report       # markdown + JSON summary of the latest run
```

Or via the Makefile:

```bash
make benchmark        # full-scale local run + report
make benchmark-ci     # CI-shaped run (10k rows, 3 timing repeats)
make benchmark-test   # the tests/benchmark suite
```

MissForest is intentionally opt-in and slower than the default engine because it
trains random forests. To compare median/mode, aggressive KNN, and MissForest on
small/medium mixed-type synthetic data:

```bash
pip install -e ".[ml]"
python benchmarks/bench_missforest.py
```

FreshData does not choose MissForest by default for large frames; request it
explicitly with `impute_method="missforest"` or per-column `impute_strategy`.

## Subcommands

| command | what it does |
|---|---|
| `run` | Run every fixture at the default (10k) size, write `results/<run_id>/<fixture>/<size>.json` and `summary.json`. `--fixtures crm finance …` to subset; `--aggressive` for the aggressive-mode variant. |
| `compare` | Time FreshData against the baselines on one `--fixture/--size`. Missing competitor libraries are skipped gracefully. |
| `report` | Render `report.md` + `report.json` for a run (`--run-dir`, default latest). |
| `fixtures` | Write fixture CSVs to `--out` (default `benchmarks/generated_fixtures/`). |
| `single` | One `--fixture --size --metric` (`time`/`memory`/`fidelity`/`trust`/…/`all`). `--write` to persist the JSON. |

## The nine metrics

1. **Wall-clock** — p50/p95 seconds for `fd.clean(..., return_report=True)` over 5 repeats (I/O excluded).
2. **Peak memory** — peak/delta MB via `tracemalloc` (dependency-free, reproducible).
3. **Repair fidelity** — cell-level vs the gold oracle; family-level (DEFECT_MANIFEST post-conditions) for the named fixtures.
4. **False-repair rate** — % of protected (id/target/text) cells wrongly changed. **Must be 0.**
5. **Preservation rate** — % of protected cells identical in/out. **Must be 100 on non-null ids.**
6. **Authored-code reduction** — FreshData vs the pandas/pyjanitor baseline line counts.
7. **Diagnosis speed** — `report.summary()` / `to_frame()` / `to_dict()` latency.
8. **Trust-score usefulness** — strict monotonic decrease as injected defect rate rises (0→60%).
9. **Export completeness** — all report fields populated and all export methods non-empty/valid (incl. enterprise `quality.to_markdown()` and `lineage.emit()`).

See `docs/benchmarks.md` for the full definitions and `docs/fixtures.md` for the
fixture schemas.

## Reproducibility

Every fixture is seed-controlled (`generate(n_rows, seed=42, defect_rate=None)`),
so the same `(n_rows, seed, defect_rate)` yields identical data on any machine.
Results carry the FreshData version, Python version and platform so runs can be
diffed across versions. CI runs the 10k-row variants only; the 5M/25M/50M-row
variants are local-only (see each fixture's `SCALE_VARIANTS`).

## Scope notes (no overclaiming)

- Defects are limited to FreshData's declared repair scope. Tokens FreshData
  does not treat as missing (e.g. `999`), foreign-currency words (`EUR 500`) and
  accounting negatives (`(1234.56)`) are intentionally excluded from the
  in-scope generic-clean benchmark.
- Great Expectations, Soda, dbt, AWS Glue DQ, Dataplex, OpenRefine, Dedupe and
  cleanlab are analysed statically in `competitor_analysis.md`, never run here.
