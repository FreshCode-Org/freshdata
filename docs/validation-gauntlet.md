# Validation Gauntlet

The Validation Gauntlet is a gold-labelled disposition benchmark for
FreshData's validation surfaces: `fd.clean`, `fd.validate_fields`,
`fd.clean_text`, the semantic layer, the domain packs, text-encoding linting
and PII detection. It lives in `benchmarks/gauntlet/` and runs on every pull
request (`.github/workflows/gauntlet.yml`).

Where [CleanBench](benchmarks.md) scores whole-frame repair fidelity against a
clean oracle, the gauntlet scores **decisions**. Every injected problem cell
carries the disposition FreshData should choose:

| disposition | meaning |
|---|---|
| `preserve` | valid (often unusual) data — must survive byte-identical, no error-severity issue |
| `repair`   | a safe deterministic repair exists — the gold value is known |
| `flag`     | must be detected, never auto-changed |
| `review`   | ambiguous — must be routed to quarantine / manual review, never guessed |

Automatic removal is never the default correct answer: a `flag` or `review`
cell that the pipeline mutates counts as a **corruption**, the most severe
verdict in the report.

## Fixtures

Five deterministic fixtures (seeded, 300 rows each by default) in
`benchmarks/gauntlet/fixtures.py`: `finance`, `healthcare`, `crm`,
`ecommerce` and `text` (adversarial free text). They cover missing values,
impossible ranges, malformed and impossible dates, invalid identifiers,
duplicates, numbers stored as text, currency/percent formats, unit confusion,
casing and whitespace noise, misspellings, Unicode/encoding noise, emojis,
HTML fragments, mixed-language values, PII, and adversarial traps designed to
trigger false corrections (`X Æ A-12` as a name, `007` as an id, `NA` as a
country, `None` as a brand token, `AB-` as a blood type, `BRK.B` as a ticker).

The flagship case: the string `apple` in a price column must be quarantined
with its original value preserved in `report.coerced_cells` — never silently
imputed — while `Apple` (company), `AAPL` (ticker) survive untouched and the
lowercase ticker `apple` is routed to review with the suggestion `AAPL`.

## Metrics and gates

Per fixture: detection precision / recall / F1, repair accuracy (with the
surface that produced each repair), review-routing rate, preservation rate,
corruption count, escape rate, false-positive rate, audit completeness,
determinism, trust-score monotonicity, wall-clock and peak memory.

CI fails a pull request when any fixture breaks the absolute gates
(`benchmarks/gauntlet/report.py::GATES` — zero corruption, 100% preservation
and audit completeness, F1 ≥ 0.85, repair accuracy ≥ 0.95, escapes ≤ 10%) or
when detection recall/F1 drops below the stored baseline
(`benchmarks/gauntlet/baseline.json`).

## Running locally

```bash
python -m benchmarks.gauntlet run                # JSON + Markdown into benchmarks/gauntlet/results/
python -m benchmarks.gauntlet run --check        # CI mode: exit 1 on gate failure
python -m benchmarks.gauntlet run --rows 2000    # heavier run, manual only
python -m benchmarks.gauntlet run --update-baseline
```

Opt-in behaviour is graded separately from defaults: the safe `clean_text`
config is never scored on lossy operations; HTML stripping and NFKC folding
earn repair credit only from the explicit opt-in pass, and imputation credit
for sentinel cells requires the fill to be audited.

## Known accepted gap

A hostile SQL-injection payload inside a `person_name` field escapes
detection. Any plausibility heuristic tight enough to catch it false-positives
on legally real names (`X Æ A-12`), so the gauntlet documents the escape
rather than forcing a lossy check.
