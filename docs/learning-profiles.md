# Learning profiles

`fd.learn()` learns repeatable cleaning behavior from a **paired messy/clean
dataset** — no model training, no LLM, fully offline and deterministic — and
packages it as a reusable, auditable `.fdprofile` file. Replaying a profile on
the next batch of the same corruption family lifts repair F1 without you
hand-writing a single rule.

This is not machine learning in the statistical-model sense: there is no
gradient descent, no embedding fine-tuning, and nothing is sent anywhere.
`fd.learn()` diffs two DataFrames, classifies each observed repair against
freshdata's own deterministic transforms, and keeps only the patterns that
survive a held-out precision check.

## Quickstart

```python
import freshdata as fd

profile = fd.learn(messy_orders, clean_orders, context="status is category", key="order_id")
fd.save_profile(profile, "orders.fdprofile")

# ... later, on a new batch shaped like the training data ...
profile = fd.load_profile("orders.fdprofile")
cleaned, report = fd.clean(new_orders, profile=profile, semantic_mode="auto", return_report=True)
```

`fd.clean` returns a `(cleaned_df, report)` tuple when `return_report=True`.
`semantic_mode="auto"` (or `"assist"`/`"review"`) must be set explicitly —
profile-backed proposals are gathered by the semantic layer, which stays off
by default, same as it does without a profile.

`profile=` is also accepted by `fd.clean_csv`, `fd.Cleaner`, and
`fd.suggest_plan`. Passing `profile=None` (the default everywhere) is
byte-identical to calling these functions today — nothing about the existing
API changes unless you opt in.

## What gets learned

`fd.learn(messy_df, clean_df, *, context=None, policy=None, key=None,
dataset_id=None, privacy="mask", include_sensitive=False, min_support=5,
min_precision=0.98, holdout_fraction=0.2, random_state=0, max_map_size=2000)`
runs a five-stage pipeline:

1. **align** — matches rows by `key` (a column name or list of columns); with
   no key, falls back to positional alignment only when both frames have the
   same length and a compatible index, otherwise degrades to column-level
   learning and says so.
2. **diff** — per shared column, finds every distinct `(raw, clean)` cell pair
   and its support count. Row payloads are never stored — only the cell pairs
   and counts.
3. **classify** — tests each distinct pair against freshdata's own forward
   transforms (the same ones the deterministic and semantic engines already
   use) across 19 families: `email_normalize`, `phone_normalize`,
   `reference_normalize`, `date_dayfirst_inference`, `date_parse`,
   `currency_parse`, `unit_strip`, `spelled_number`, `boolean_synonym`,
   `sentinel_to_missing`, `allowed_value_map`, `category_map`,
   `numeric_rounding`, `dtype_coercion`, `case_fold`, `whitespace`,
   `missing_imputation`, `row_drop_evidence`, and `unexplained`.
4. **extract** — turns classified diffs into rules, config deltas (e.g. a
   phone region, `dayfirst=True`, extra sentinels), and literal value maps
   for the three vocabulary families (`reference_normalize`,
   `allowed_value_map`, `category_map`). Patterns below `min_support`
   (checked per raw value for value maps, per family everywhere else) are
   kept only as suggest-only examples, never as replay rules.
5. **fit_eval** — a deterministic `random_state`-seeded holdout split;
   extracted rules and maps are replayed forward on the held-out rows and
   compared against the held-out clean values. Anything scoring below
   `min_precision` on holdout is demoted to an example with a recorded
   `demotion_reason` — including value-map entries that would otherwise map
   one raw value to two conflicting clean values, which are always resolved
   down to a single best-supported target regardless of the precision
   threshold, since an ambiguous map isn't a function.

The output is a `LearningProfile`: rules (`context.types.ColumnConstraint`),
`ValueMap`s, an embedded `CleaningMemory`, an `ExampleBank` of
unexplained/low-support pairs, and a `ProfileManifest` describing privacy
mode, schema, and provenance.

## Privacy

By default (`privacy="mask"`), columns detected as sensitive — email, phone,
person name, national ID, address, postal code, or free text — never carry
raw literals in the saved profile:

- Rule-level evidence (a phone region, a `dayfirst` flag, a sentinel list)
  carries no literals to begin with and replays normally.
- Literal value-map entries and examples on sensitive columns store an
  HMAC-hashed raw value plus a masked display token (`masked=True`) and are
  excluded from replay — the profile can tell you *that* it saw repairs on
  that column, not *what* they were.
- The salt is stored in the profile so repeated audits of the same profile
  are stable, but it does not let you invert the hash back to raw values.

Pass `privacy="none", include_sensitive=True` to keep raw literals — the
manifest then sets `contains_raw_values=True`, and the CLI (and any audit)
warns loudly rather than staying silent about it.

## The `.fdprofile` format

A `.fdprofile` is a zip archive: `manifest.json` (schema hash, dataset
signature, privacy mode, `profile_version`, and a `member_hashes` map — a
SHA-256 of every other member's bytes), `rules.json`, `value_maps.json`,
`memory.json`, `examples.json`, `audit.json`, and an optional
`examples_vectors.npz` (float16 embeddings, only written when the
`[semantic]` extra produced them).

`load_profile()` verifies every member hash against the manifest and raises a
clear `ProfileError` on a corrupt zip, a hash mismatch, or an unsupported
major `profile_version`. Unknown members (e.g. from a newer profile version)
are ignored with a warning, not an error.

## Replay

`profile=` is folded in ahead of the semantic backends, with a drift gate in
front of it:

- The new frame's schema/signature is compared against the profile's; a
  severe mismatch (low column overlap) blocks replay for the whole frame with
  a report warning, a mild mismatch replays only the columns still present.
- Vetted config deltas (extra sentinels, `dayfirst`, phone region, …) are
  lowered into the run's options **only where you didn't already set them** —
  anything you or your context/policy specify explicitly always wins.
- Protected columns are always excluded, and the executor's byte-identity
  guard still runs underneath — a profile cannot bypass it.
- A `"profile"` backend runs proposals from the profile's rules and value
  maps through the same policy gate and calibration as every other backend;
  accepted actions carry `profile_influenced=True` metadata (and
  `memory_influenced=True` when they came through the profile's embedded
  memory). If you also pass `memory=`, your memory wins over the profile's.

### Example retrieval (optional)

If the `[semantic]` extra is installed (or a local encoder is otherwise
available) and the profile has embedded vectors, unexplained training
examples are retrieved by cosine similarity against new values that didn't
exactly match any rule or value map. This is **suggest-only**: capped
confidence, skipped for protected columns and for high-cardinality
(>200-distinct) columns, and never auto-applied without your review gate
accepting it. Without the extra, this step degrades silently — rules and
value maps still replay.

## Audit, diff, and merge

- `profile.audit()` → a `ProfileAudit` with per-family rule/value-map counts,
  support and precision ranges, masked/raw-literal status, and every
  demotion with its reason. `profile.summary()` is the short human string.
- `profile.diff(other)` → added/removed/changed rules, added/conflicting
  value-map entries, and privacy/schema/embedding-model differences between
  two profiles.
- `profile.merge(other, strategy=...)` combines two profiles. Strategies:
  `"union_min_precision"` (default — keep entries meeting the stricter of the
  two thresholds; conflicting raw→clean mappings are dropped with an audit
  note), `"prefer_self"`, `"prefer_other"`, and `"error_on_conflict"` (raises,
  listing every conflict).

## CLI

```bash
freshdata learn RAW.csv CLEAN.csv --key order_id --context "status is category" \
    --privacy mask --min-support 5 --min-precision 0.98 -o orders.fdprofile

freshdata clean new_orders.csv --profile orders.fdprofile -o cleaned.csv

freshdata profile audit orders.fdprofile
freshdata profile diff orders_v1.fdprofile orders_v2.fdprofile
freshdata profile merge orders_v1.fdprofile orders_v2.fdprofile -o merged.fdprofile
```

`freshdata profile` without one of `audit`/`diff`/`merge` as the first
positional argument keeps its original meaning — data profiling
(`freshdata profile data.csv`) — so existing scripts don't break.

## CleanBench T4

`benchmarks/cleanbench/t4_profiles.py` covers 10 fixture families (category
maps, allowed-value maps, reference typos, sentinels, day-first dates, IN
phone formats, double-`@` emails, an imputation fixture that must *not*
replay literals, and an unexplained-only fixture). Each fixture learns a
profile from a train split and replays it on a held-out batch of the same
corruption family. Gates enforced in `tests/test_cleanbench_t4.py`: mean F1
lift over the replayable fixtures, false-modification rate no worse than the
no-profile baseline, zero protected-column violations, and zero privacy
leaks (no raw sensitive literal ever appears in the serialized profile bytes
under `privacy="mask"`).

## Limitations

- **No model training, no LLM, no cloud.** Every transform a profile can
  learn is one freshdata already ships a deterministic or semantic expert
  for; `fd.learn` only decides *whether* and *how strongly* to apply it.
  Nothing is fine-tuned and nothing leaves the process.
- **Full rows are never stored.** Profiles retain cell-level diffs, counts,
  and (optionally masked) examples — never a training row.
- **Sensitive literals are masked by default,** and raw literal imputation
  values are never learned as a map, mask setting notwithstanding.
- **Low support means suggest-only, always.** A pattern seen fewer than
  `min_support` times (per raw value, for vocabulary maps) never becomes a
  replay rule — it's kept only as an example.
- **A profile cannot bypass policy or the protected-column guard.** Every
  profile-sourced proposal passes through the same policy gate, calibration,
  and byte-identity guard as every other backend.
- **Replay degrades, it doesn't fail loudly by default.** Schema drift
  against the training data blocks replay on the affected columns (or the
  whole frame, if severe) with a report warning rather than raising, unless
  you've asked for strict mode.
- **Example retrieval is capped and optional.** It only fires with a local
  encoder available, only below a 200-distinct-value cardinality ceiling per
  column, and its proposals are never auto-applied outright.
- **A learned profile is a snapshot of one corpus's corruption patterns.**
  It is not a general-purpose cleaner; replaying it on a materially different
  dataset is exactly what the drift gate is designed to catch and limit.
