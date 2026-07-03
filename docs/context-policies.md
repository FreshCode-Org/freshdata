# Context policies — cleaning rules in English

Write your expectations about a dataset as plain English, and FreshData
**compiles** them — deterministically, offline, with no model and no new
dependency — into a typed, reviewable `ContextPolicy` that governs the clean:

```python
import freshdata as fd

clean_df = fd.clean(
    df,
    context="""
    This is an ecommerce customer dataset.
    CustomerID is unique.
    Emails must be valid.
    Phone numbers are Indian.
    Missing Age should be estimated only if confidence >95%.
    Never modify revenue values.
    """,
)
```

The point is not the prose — it is the artifact. The compiled policy is a JSON
document you can print, diff, commit next to your pipeline, and review in a
pull request like code. `fd.clean(..., context=...)` is exactly equivalent to
compiling the policy yourself and passing it back:

```python
policy = fd.compile_context(ctx_text, df=df)
print(policy.summary())
policy.to_json("policy.json")            # review it, commit it
clean_df = fd.clean(df, policy=policy)   # skip parsing, use it verbatim
```

## What the compiler understands (tier 0)

The parser is a fixed, deterministic lexicon of ~12 intent families. The same
text always compiles to the same policy — there is no LLM, no embedding, no
network call, and no randomness anywhere in this path.

| You write | Compiles to |
|---|---|
| `This is an ecommerce customer dataset.` | dataset domain metadata |
| `CustomerID is unique.` | `unique` — validation + id-column protection |
| `Emails must be valid.` | `valid_format(email)` semantic hint |
| `Phone numbers are Indian.` | `locale_format(phone, region=IN)` hint |
| `Missing Age should be estimated only if confidence >95%.` | per-column imputation confidence gate (0.95) |
| `Never modify revenue values.` / `Do not touch revenue.` | `protected` — hard, never modified |
| `Allowed status values are active, inactive, pending.` | `allowed_values` validation |
| `Age must be between 18 and 100.` | `range` validation |
| `Deduplicate by email and phone.` | duplicate-detection key |
| `Drop rows where age is missing.` | recorded `custom` rule (executed in a later phase) |
| `Rename cust to customer_id.` / `Replace 'M' with 'Male' in gender.` | recorded `custom` rules |

Confidence phrases are normalized: `>95%`, `confidence > 95%`,
`confidence above 0.95`, and `only if confidence >= 95 percent` all compile to
`0.95`.

## Column references resolve against your real schema

You write `CustomerID`; your frame has `cust_id`. The resolver walks a fixed
ladder — exact match, snake_case normalization (the same renaming
`column_names=True` applies), a curated alias lexicon, token containment, and
finally `difflib` similarity (threshold 0.85):

```text
"CustomerID"    → cust_id           (alias)       0.90
"Emails"        → email_addr        (alias)       0.90
"Phone numbers" → mobile            (alias)       0.90
"Age"           → age               (normalized)  1.00
"revenue"       → monthly_revenue   (alias)       0.90
```

**Nothing is ever guessed.** Two candidates that score too close together, or
a reference below the threshold, come back in `policy.unresolved` with the
ranked candidates so *you* disambiguate. Sentences the lexicon cannot parse are
surfaced in `policy.issues` (kind `unparsed_sentence`) — never silently
dropped.

## Strict mode

By default, unresolved references and unparsed sentences become report
warnings and the rest of the policy still applies. With `strict=True` they
raise `fd.PolicyError` *before any data is touched*:

```python
fd.clean(df, context=ctx, strict=True)       # raises PolicyError on any gap
fd.compile_context(ctx, df=df, strict=True)  # same, at compile time
```

## Checking without cleaning: `fd.validate`

```python
findings = fd.validate(df, context=ctx)   # never mutates df
assert not findings.errors
```

Returns a `FindingList` of [`QualityFinding`](api-reference.md)s covering
unresolved references, compile issues, protected columns, and — where
checkable today — `unique`, `allowed_values`, and `range` violations.

## Everywhere the policy threads

```python
fd.clean(df, context=..., policy=..., strict=...)
fd.suggest_plan(df, context=...)          # the plan's config carries the policy
fd.clean_csv("in.csv", context=...)
fd.Cleaner(context=...)                   # reusable, compiled per frame
```

`context=` and `policy=` are mutually exclusive; passing both raises. On the
CLI:

```bash
freshdata clean input.csv -o output.csv --context-file rules.txt
freshdata clean input.csv -o output.csv --context-file rules.txt --strict
freshdata policy compile rules.txt --schema input.csv --output policy.json
```

`policy compile` prints the human-readable summary; `--strict` exits non-zero
on unresolved or unparsed lines (handy as a CI gate for the committed
`rules.txt`).

## How a policy lowers into the engine

`policy.lower(config)` is a pure function producing a new `CleanConfig`:

- `protected` columns → `preserve_columns` + `semantic_context` `mutable=False`
- `unique` columns → `id_columns` (imputation/outlier veto for free) + validation metadata
- `valid_format` / `locale_format` → per-column `semantic_type` (+ `region`) hints
- imputation confidence gates → per-column `impute_min_confidence` metadata
- `allowed_values` / `range` → per-column validation metadata
- dedup keys → `duplicate_subset`

The lowered config is what actually runs — the existing pipeline, unchanged.
Every compile is audited: the report gains a `context` action carrying the full
policy dict, plus a warning per unresolved/unparsed item.

## Guarantees and limitations

- **`context=None` is a no-op.** Not passing a context changes nothing, byte
  for byte — the compiler is never even imported.
- **Deterministic tier-0 only.** This is a fixed pattern lexicon, not natural
  language understanding. Phrasings outside it are surfaced as unparsed, and
  that is by design: a rule the compiler cannot prove it understood is a rule
  it refuses to half-apply.
- **No model, no network, no new dependency.** Pure Python + stdlib.
- **Protection wins ties.** A column that is both protected and asked to be
  repaired compiles with the repair demoted to validation and an explicit
  `protection_conflict` issue (an error under `strict`).
- **Phase-2 enforcement.** `valid_format(email)` and
  `locale_format(phone, region=IN)` now drive deterministic value experts
  (email normalization, Indian phone canonicalization to `+91XXXXXXXXXX`),
  `allowed_values` drives the reference-list expert, imputation confidence
  gates ("only if confidence >95%") are enforced inside the statistical
  engine, and `protected` columns are physically verified **byte-identical**
  before any result is returned (`fd.ProtectedColumnError` on violation).
  See [repair-plans.md](repair-plans.md) for the reviewable plan/apply
  workflow. `drop_if`/`rename`/`map` rules are still compiled and carried in
  the policy but not yet executed.

See the notebook
[`notebooks/06_context_cleaning.ipynb`](https://github.com/FreshCode-Org/freshdata/blob/main/notebooks/06_context_cleaning.ipynb)
for the full ecommerce walkthrough.
