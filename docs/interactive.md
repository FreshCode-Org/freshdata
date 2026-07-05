---
title: Interactive output
description: >-
  freshdata's optional interactive HTML layer — quality cockpit, action timeline,
  audit ledger, decision cards and diff explorer — with zero heavy base deps.
keywords: data cleaning report, interactive data profiling, notebook data quality
---

# Interactive output

Every freshdata report is a **serializable object first** — `.summary()`,
`.to_dict()`, `.to_frame()` keep working with nothing but pandas installed. On
top of that, the same objects render an **interactive HTML view** for notebooks
and stakeholders. Rendering is separated from compute: the renderers live in
`freshdata.render` and are imported lazily, so `import freshdata` never pulls a
visualization library.

```python
import freshdata as fd

result = fd.clean(df)
cleaned = result.data
report = result.report()

result.visualize()       # self-contained report HTML string
report.show()            # collapsible action timeline + filterable audit ledger
fd.profile(df).show()    # inline quality cockpit
fd.suggest_plan(df).show()   # per-column decision cards
fd.explain_clean(df).show()  # before/after diff explorer
fd.compare_plans(df)         # strategy diff grid (rich repr in a notebook)
fd.compare_clean(df)         # outcome dashboard
fd.infer_roles(df)           # role / type confidence cards
```

Each object also exposes `to_html()` for embedding and `_repr_html_()` for
automatic notebook display. Outside a notebook, `.show()` writes a standalone
`.html` file and returns its path.

## Optional extras

The base renderers produce **self-contained HTML** (scoped CSS + a little vanilla
JS for filtering and collapsing) with **no optional dependencies**. Installing an
extra simply upgrades the output:

| Extra | Adds | Upgrades |
|-------|------|----------|
| `freshdata-cleaner[viz]` | `itables`, `plotly`, `great-tables` | richer tables, hoverable charts, stakeholder-clean HTML tables |
| `freshdata-cleaner[notebook]` | the above + `anywidget`, `ipython` | composed notebook widgets |
| `freshdata-cleaner[all]` | everything | — |

```bash
pip install 'freshdata-cleaner[viz]'
```

If you call a path that needs an optional package you don't have, freshdata
raises a clear install hint (e.g. *"install `freshdata-cleaner[viz]`"*) rather than a bare
`ModuleNotFoundError`.

## What each surface shows

- **`report.show()`** — one row per action with before→after impact, rationale,
  risk, confidence, whether it was automatic/suggested/skipped/approved, whether
  it is reversible, and whether memory influenced it. The audit ledger filters by
  column / action / risk and exports to CSV / JSON.
- **`profile.show()`** — overall quality score, missingness / duplicate /
  type-inference summaries, an issue-ranked column list with null bars, and
  cardinality / outlier warnings. Correlations are computed **on demand** to keep
  the first render fast.
- **`suggest_plan(df).show()`** — per-column recommended action, confidence, risk,
  *why*, alternatives, and a "needs review" flag.
- **`explain_clean(df).show()`** — before/after column comparison with risky
  changes separated from safe ones.

## Generating HTML examples

```bash
python scripts/generate_html_examples.py     # writes docs/examples/*.html
```

The committed samples under `docs/examples/` are produced by that script from
synthetic data, so they always match the current renderers.
