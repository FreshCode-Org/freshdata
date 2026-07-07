# AI Copilot — explainable data cleaning for messy real-world data

!!! warning "Experimental"
    `freshdata.experimental.ai_copilot` ships under the `experimental`
    namespace: the report shape and rule vocabulary may change between minor
    releases. Everything on this page is implemented and tested today — the
    only part that is a forward-looking interface is the optional LLM
    `provider` hook, which has **no built-in provider yet**.

One call analyzes a dataset and returns everything a human (or a pipeline)
needs to act: a ranked problem list, a PII warning, context-policy
violations, an ordered **explainable cleaning plan**, and **copy-ready
freshdata code** that implements it.

```python
import pandas as pd
from freshdata.experimental.ai_copilot import analyze_dataset

df = pd.read_csv("examples/data/messy_customers.csv")

report = analyze_dataset(
    df,
    goal="Prepare this customer dataset for analytics and ML",
    privacy="mask_pii_before_reasoning",
    context_policy={
        "email": "must_mask",
        "phone": "must_mask",
        "age": "must_be_between_0_and_120",
        "salary": "must_be_positive",
        "city": "normalize_spelling",
    },
)

print(report.summary)
print(report.cleaning_plan)
print(report.recommended_code)
```

Three properties make this different from "ask a chatbot about my data":

- **Deterministic and offline.** The analysis is rule-based, built from
  freshdata's own primitives (profiling, PII detection, the context-policy
  compiler, value clustering, trust scoring). The same input always produces
  the same report; it runs in CI with no API key and no network access.
- **Privacy-first.** Raw cell values never enter `report.model_context` —
  the only payload an LLM provider would ever see. Samples are hashed and
  scrubbed first, or omitted entirely with `privacy="schema_only"`.
- **Actionable.** The output is not advice — it is an ordered plan with a
  rationale per step, plus a generated freshdata pipeline you can run as-is.
  (The test suite literally `exec()`s the generated code and asserts the
  result is clean.)

## What the report contains

| Field | What it is |
|---|---|
| `report.summary` | Printable overview: shape, Data Trust Score, ranked problems, PII warning |
| `report.problems` | Typed `DetectedProblem` list — kind, severity, column, count, detail |
| `report.pii_warning` | Which columns hold personal data and what to do about it |
| `report.policy_violations` | `QualityFinding`s from validating your rules against the data |
| `report.cleaning_plan` | Ordered `PlanStep`s, each with a rationale and the freshdata tool to use |
| `report.recommended_code` | Runnable pipeline generated for *this* dataset |
| `report.trust` | The 0–100 [Data Trust Score](feature-overview.md) with its four dimensions |
| `report.model_context` | The masked schema/stats/sample payload — inspect it to verify nothing leaks |
| `report.audit` | Timestamps, engine, privacy mode, masked columns, model-context SHA-256 |
| `report.narrative` | Provider output, if you plugged an LLM in (else `None`) |

Problem kinds detected today: `pii`, `policy_violation`, `duplicate_rows`,
`missing_values`, `mixed_date_formats`, `category_noise` (near-duplicate
spellings found by the same clustering engine as
`freshdata.enterprise.merge_clusters`).

## The context-policy mini-vocabulary

`context_policy` is a `{column: rule}` mapping (a list of rules per column is
also fine). Supported rules:

| Rule | Effect |
|---|---|
| `must_mask` | Column is masked in the model context and a `MaskingRule` lands in the generated code |
| `must_be_between_<lo>_and_<hi>` | Compiled to a range constraint and validated against the data |
| `must_be_positive` | Compiled to `>= 0` and validated |
| `must_be_unique` | Compiled to a uniqueness constraint and validated |
| `normalize_spelling` | Column is force-clustered for near-duplicate spellings |
| `never_modify` | Column is marked protected in the compiled policy |

Range/uniqueness/protected rules are translated to plain English and compiled
through the deterministic [context-policy compiler](context-policies.md) —
so the rules you pass here produce the *same* reviewable `ContextPolicy`
artifact you would get from `fd.compile_context`. Unknown rules raise a
`ValueError` listing the vocabulary (no silent typos).

## Privacy model

`analyze_dataset` never modifies your DataFrame and never sends it anywhere.
The `privacy` parameter controls what goes into `report.model_context`:

- `"mask_pii_before_reasoning"` (default) — includes `sample_rows` sample
  rows, but every `must_mask` column and every column the PII detector
  flagged is hashed first, and free-text PII spans are scrubbed.
- `"schema_only"` — no cell values at all; only column names, dtypes,
  missing percentages, and aggregate statistics.

Two details worth knowing:

- The PII detector is freshdata's dependency-free detector
  (`fd.detect_pii`), so EMAIL/PHONE/SSN/credit-card detection works out of
  the box. The copilot additionally suppresses a known false positive:
  columns whose only "PHONE" hits are actually dotted/slashed **dates** are
  not treated as PII (the suppression is recorded in
  `report.audit["pii_suppressed_date_like"]`).
- `report.audit["model_context_sha256"]` fingerprints the exact payload a
  provider would have seen, so you can prove after the fact what was (and
  was not) shared.

## Plugging in an LLM (optional, experimental)

The deterministic report needs no model. If you want a natural-language
narrative on top, pass any `Callable[[str], str]`:

```python
def my_provider(prompt: str) -> str:
    # call your LLM of choice here; the prompt contains ONLY the masked
    # model context, never raw data
    ...

report = analyze_dataset(df, provider=my_provider)
print(report.narrative)
```

Passing a provider emits a `FutureWarning` (the prompt contract may evolve).
Provider failures are recorded in `report.audit["provider_error"]` and never
break the deterministic report. First-party provider adapters are a TODO —
the hook is deliberately a plain callable so nothing in freshdata depends on
any LLM SDK.

## The full story, end to end

`examples/freshdata_ai_copilot_demo.py` runs the complete
messy-to-audit-ready pipeline against the bundled
`examples/data/messy_customers.csv` (duplicates, sentinel strings, invalid
ages, negative salaries, mixed date formats, city/plan spelling variants,
raw emails and phones):

```bash
python examples/freshdata_ai_copilot_demo.py
```

It analyzes, masks, cleans under the compiled policy, merges category
variants, and finishes with a before/after scoreboard:

```text
rows                                  50  ->  45
missing cells                         30  ->  11
duplicate rows                         5  ->  0
policy violations                      2  ->  0
PII cells exposed                     89  ->  0
Trust Score               93.9 (grade A)  ->  97.5 (grade A)
```

## Why not just pandas?

You can hand-roll all of this with pandas — people do, for every dataset,
every time. What the copilot (and freshdata underneath it) adds:

- the *decision* layer: which columns are PII, which values violate rules,
  which spellings are the same category — with severity and evidence;
- an audit trail a reviewer can read (`CleanReport` actions with rationale,
  masked-context SHA, privacy events with HIPAA/GDPR tags);
- reproducibility: the same input produces the same report, plan, and code.

## Limitations and responsible use

- The copilot is **heuristic**: PII detection is pattern-based (a person's
  name in a free-text column will not be caught without the optional NER
  extra), date-format detection covers common layouts only, and clustering
  can propose merges you should review before applying.
- The generated code is a starting pipeline, not a substitute for domain
  review — read the plan rationales, then decide.
- There is no LLM in the default path. If you attach one via `provider`,
  you are responsible for that model's terms; freshdata's guarantee is that
  the prompt contains only the masked model context.
- Trust-score deltas depend on your data; the numbers above are from the
  bundled demo dataset, not a general claim.
