---
title: Peel — the output system
description: >-
  Peel is freshdata's progressive-disclosure output system: a glance-level
  answer, an inspect layer of evidence, and a full audit layer with zero
  information loss, shared across the core API, semantic models, the AI
  copilot, and parsing.
keywords: freshdata output, progressive disclosure, clean report, terminal rendering, notebook html, machine readable
---

# Peel — the output system

Peel is how freshdata report objects present themselves. A result is fresh
produce: the **skin** tells you at a glance whether it is good, you can **peel**
to the flesh to inspect it, and the **core** is always intact.

Every report — `CleanReport`, `ParseResult`, the experimental `CopilotReport` —
renders through the same three layers:

| Layer | Name | What it answers |
|---|---|---|
| 1 | **Skin** (glance) | Did it succeed? What changed? Does anything need my attention? What's the one next step? |
| 2 | **Flesh** (inspect) | The evidence — per-column changes, ranked findings, semantic proposals, frame inventory. |
| 3 | **Core** (audit) | Every field, machine-readable, nothing dropped. |

Peel never removes information; it only decides what you see first. Layer 3 is
the existing structured objects and their `to_dict()`/`to_json()` — display can
be turned off entirely and no data is lost.

## Status and severity language

Every result carries a **text status label** (colour and icons only reinforce
it, so output stays readable when piped, in CI, or for screen readers):

`CLEAN` · `CHANGED` · `REVIEW` · `BLOCKED` · `PARTIAL` · `SKIPPED` · `FAILED`

Findings in the attention list are ranked by one shared order:

1. privacy & safety → 2. data-corruption risk → 3. policy/contract → 4. analysis
reliability → 5. cosmetic consistency

so a high trust score can never bury a privacy or policy finding.

## Seeing Peel output

Peel is **opt-in** while it stabilizes; existing output is unchanged by default.

```python
import freshdata as fd

cleaned, report = fd.clean(df, return_report=True)

report.show()                    # legacy behavior unchanged (inline HTML / temp file)
report.show(mode="standard")     # Peel text: glance + attention + next step
report.show(mode="compact")      # two lines, for pipelines
report.show(mode="verbose")      # + per-column, semantic, and action detail
report.show(renderer="terminal") # styled panel when `rich` is installed
```

### Display modes

| Mode | Use |
|---|---|
| `auto` | environment-aware: `standard` in a terminal, `compact` when piped |
| `compact` | one-screen, two lines |
| `standard` | glance + ranked attention + next step |
| `verbose` | full human-readable diagnostics |
| `debug` | + internal audit metadata |
| `json` | stable `to_dict()` on stdout |
| `plain` | no ANSI, ASCII icons |
| `silent` | render nothing |

### Notebook

`fd.set_display("peel")` (or `FRESHDATA_DISPLAY=peel`) switches the notebook
`_repr_html_` to the Peel card; `FRESHDATA_LEGACY_DISPLAY=1` forces the legacy
layout. Objects with only a Peel view (like `ParseResult`) always render as Peel.
Disclosure uses native `<details>`, so it works in static notebook exports with
no JavaScript.

### Command line

`freshdata clean` gains additive display flags; default output is unchanged:

```bash
freshdata clean input.csv                    # unchanged (legacy summary)
freshdata clean input.csv --verbose          # Peel verbose
freshdata clean input.csv -vv                # Peel debug
freshdata clean input.csv --output-format json  # report JSON to stdout
freshdata clean input.csv --no-color         # no ANSI
freshdata clean input.csv --display peel     # Peel standard
```

Display flags never change cleaning behavior. `fd.set_display(...)` sets
process-wide preferences; `NO_COLOR` and `FRESHDATA_NO_PREVIEWS` are honored.

## Machine-readable access is unchanged

`report.to_dict()` / `report.to_json()` schemas are frozen — additive only.
`report.actions`, `report.warnings`, `report.domain_findings`, `report.attention`
(the ranked queue) and every other attribute keep working. Peel is a rendering
layer over these objects, never a replacement for them.

## Privacy

Values shown on evidence cards and previews are escaped before HTML rendering
and never placed in HTML attributes. `fd.set_display(previews=False)` (or
`FRESHDATA_NO_PREVIEWS=1`) switches to schema-only display. `undo_log` is never
serialized, and provider credentials are never captured — only failure states
appear in the audit layer.
