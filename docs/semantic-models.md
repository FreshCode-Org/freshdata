# Optional semantic models

FreshData's default install is — and will remain — **model-free**. Everything
`fd.clean` does out of the box is deterministic Python: no model weights, no
downloads, no network, no LLM. This page covers the *optional* local model
path added in Phase 3: a small ONNX sentence encoder that upgrades three
things when (and only when) you opt in:

1. **Embedding backend** — extra repair *proposals* for category/reference
   values the deterministic experts abstain on ("aktyve" → "active" when the
   allowed values say so).
2. **Resolver rescue rung** — context phrases like "Phone numbers" can match a
   `mob_no` column that the alias lexicon and fuzzy matching missed.
3. **Semantic-type evidence** — extra signal for `fd.infer_roles`.

Every model output is *evidence, not authority*: proposals still pass the same
policy gate, protected columns keep their byte-identity guarantee, and
ambiguous matches are suggested or skipped, never auto-applied.

## Install

```bash
pip install "freshdata-cleaner[semantic]"   # onnxruntime + tokenizers, CPU only
```

The extra adds the inference runtime only. **Model weights are never bundled
in the wheel and are never downloaded automatically** — not by `fd.clean`, not
by `import freshdata`, not by anything except the explicit pull below.

## Models

```python
import freshdata as fd

fd.models.list_available()        # registry metadata (ids, sizes, licenses)
fd.models.status()                # what is installed and verified, offline
fd.models.pull("fd-col-encoder-v1")   # explicit download (the only network path)
fd.models.path("fd-col-encoder-v1")   # local artifact path
```

| model id | what it is | used by |
|---|---|---|
| `fd-col-encoder-v1` | small sentence encoder, ONNX int8 | embedding backend, resolver rung |
| `fd-intent-v1` | context-sentence intent classifier | registered for forward compatibility; unused in Phase 3 |
| `calib-v1` | isotonic confidence-calibration tables (plain JSON) | proposal scoring; a packaged default ships in the wheel |

Models live in `~/.freshdata/models/<model-id>/`; override the directory with
`FRESHDATA_MODEL_DIR`.

> **Not yet published:** official artifacts are not hosted yet, so
> `fd.models.pull(...)` currently raises `ModelNotPublishedError` with
> instructions. You can point `FRESHDATA_MODEL_URL_BASE` at a mirror that
> hosts the files, or use the air-gapped path below. The registry pins sha256
> checksums as artifacts are published; a pinned checksum that does not match
> is refused at download *and* at load time.

### Air-gapped installs

Copy the model files into the model directory by hand:

```text
$FRESHDATA_MODEL_DIR/fd-col-encoder-v1/model.onnx
$FRESHDATA_MODEL_DIR/fd-col-encoder-v1/tokenizer.json
```

`fd.models.status()` detects them. Manually placed files with no pinned hash
load as `unverified` (visible in `status()`); a pinned mismatch refuses to
load.

## Using the embedding backend

```python
clean_df, report = fd.clean(
    df,
    context="""
    Allowed status values are active, inactive, pending.
    Never modify revenue values.
    """,
    semantic_mode="auto",
    semantic_backends=("deterministic", "memory", "embedding"),
    return_report=True,
)
```

Or from the CLI:

```bash
freshdata models status
freshdata models pull fd-col-encoder-v1
freshdata clean in.csv -o out.csv \
  --context-file rules.txt \
  --semantic-mode auto \
  --semantic-backends deterministic,memory,embedding \
  --report report.json
```

Backends run in trust order. Deterministic experts always go first;
`memory` replays learned repairs when a `CleaningMemory` is supplied; the
embedding backend sees only the *residue* — distinct values nothing earlier
proposed a repair for. It never sees full rows: only distinct values of
eligible columns, and never values from protected, id, target, free-text, or
high-cardinality columns.

### What it proposes, and how carefully

- **With allowed values** (from context or `semantic_context`): a residual
  value whose top cosine to one allowed value clears a threshold *and* leads
  the runner-up by a margin becomes a `reference_value` proposal. At very high
  calibrated confidence the normal gate may auto-apply it in
  `semantic_mode="auto"` — the same rules deterministic proposals live by.
- **Without allowed values** (pure similarity clustering on low-cardinality
  categorical columns): proposals are `category_synonym`, risk `medium` or
  higher, and the packaged calibration caps their confidence *below* the
  default auto threshold — **suggest-only by default**, on purpose.
- **Ambiguous matches** (a close second candidate, e.g. `"nactive"` sitting
  one edit from both `active` and `inactive`, or `Austria`/`Australia`
  clusters) produce **no proposal at all**. Absence is the strongest
  guarantee: nothing downstream can accidentally apply what was never
  proposed.

Every model-assisted action records its provenance in `Action.metadata`:
`backend`, model id and sha, raw score, calibrated confidence, calibration
version, a features hash, and the top/second candidates with the margin.

## Confidence calibration

Expert scores are treated as *raw* scores. A per-(backend, issue-family)
isotonic table maps them to calibrated probabilities before the gate reads
them:

- the packaged default (`calib-default-1`) is **identity for deterministic
  and memory proposals** — calibration can never change a model-free
  install's decisions — and conservative for embedding proposals;
- a pulled `calib-v1` model overrides the packaged default;
- with no table at all, raw scores pass through and actions record
  `calibration_version="uncalibrated"` plus a report warning;
- embedding proposals are never pinned at 1.0, table or no table.

CleanBench gates calibration on every run of the Phase-3 mini-suite:
protected-column violations = 0, false-modification rate ≤ 0.1%, expected
calibration error ≤ 0.05, precision at confidence ≥ 0.95 of at least 0.98,
and zero auto-applied ambiguous merges. (Long-run targets: ECE ≤ 0.03,
P@0.95 ≥ 0.99 — see `docs/benchmarks.md`.)

## Budget

Model work is bounded by `semantic_budget`:

```python
fd.clean(df, semantic_mode="auto",
         semantic_backends=("deterministic", "embedding"),
         semantic_budget={"max_columns": 50, "max_distinct_values": 5000,
                          "max_model_calls": 100, "max_seconds": 10})
```

An exhausted budget stops the embedding backend cleanly (proposals already
made stand; nothing is partially applied) and records a report fallback
event. Deterministic and memory backends are never metered.

## Graceful degradation

Requesting `"embedding"` without the extra or the model **never crashes**:

- the backend self-disables;
- `report.fallback_events` records why;
- one report warning says the backend was skipped and how to enable it;
- deterministic (and memory) behavior is exactly what it would have been.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `Semantic backend 'embedding' skipped: optional dependency missing` | `[semantic]` extra not installed | `pip install "freshdata-cleaner[semantic]"` |
| `... model 'fd-col-encoder-v1' is not installed` | weights never pulled (they never download automatically) | `fd.models.pull("fd-col-encoder-v1")` or place files in `FRESHDATA_MODEL_DIR` |
| `ModelNotPublishedError` from `pull` | no official artifact hosting yet | set `FRESHDATA_MODEL_URL_BASE` to a mirror, or use the air-gapped path |
| `ModelChecksumError` | file does not match the pinned sha256 | re-pull with `--force` / replace the file; FreshData refuses to load mismatches |
| `budget exhausted (...)` fallback event | `semantic_budget` ceiling hit | raise the ceiling or accept the (clean, recorded) early stop |
| `calibration_version="uncalibrated"` in metadata | no calibration table found | reinstall (restores the packaged default) or `fd.models.pull("calib-v1")` |

## Testing without models

CI and local tests run the full embedding path with **no model files**: set
`FRESHDATA_STUB_ENCODER=1` for a deterministic hash-based stub encoder, or
inject an encoder via `freshdata.models.runtime.set_encoder_factory` in
tests. Both are testing seams, not public API.

## What Phase 3 deliberately does not do

No LLM at runtime. No cloud inference. No per-cell model calls (distinct
values only, structurally). No generative repairs. No automatic downloads.
No model weights in the wheel. Protected columns remain byte-identical under
any backend — enforced physically by the executor guard, not by convention.

## How these models are built (Phase 5)

The artifacts themselves (`fd-col-encoder-v1`, `fd-intent-v1`, `calib-v1`)
are produced by the development-time training pipeline in `training/` —
seed-corpus governance, a ~40-corruptor labeling engine, compliance-gated
teacher tasks, human-reviewed eval labels, int8 quantization, and manifested
packaging under `dist/artifacts/`. Every artifact ships with a
[model card](model-cards.md) and is gated by the full
[CleanBench suite](benchmarks.md). See the
[developer training pipeline](developer-training-pipeline.md) guide. None of
this changes the runtime contract above: model-free by default, offline,
no LLM, no automatic downloads, no weights in the wheel.
