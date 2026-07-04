# FreshData training pipeline (Phase 5)

Development-time pipeline that builds, evaluates, and packages FreshData's
small local model artifacts (`fd-col-encoder-v1`, `fd-intent-v1`,
`calib-v1`). This package is **not part of the runtime**: it lives outside
`src/freshdata`, is excluded from the wheel (see `pyproject.toml` and
`.github/workflows/wheel-size.yml`), and the runtime never imports it.

FreshData's runtime stays deterministic, offline, model-free by default, and
free of any LLM or cloud call — nothing here changes that contract. See
[docs/developer-training-pipeline.md](../docs/developer-training-pipeline.md)
for the full guide and [COMPLIANCE.md](COMPLIANCE.md) for the teacher-model
governance rules.

## Layout

```text
training/
  seed/          seed corpus registry, licenses, synthetic PII-shaped generators
  corruptors/    ~40 deterministic, labeled corruptors (representation/semantic/context/scale)
  teacher/       compliance-gated development-time teacher harness (schemas, cache, review)
  datasets/      seed/corrupted dataset builders, splits, license+label validators
  distill/       role-head / intent-head / optional encoder training, ONNX export, quantize, package
  calibration/   feature extraction, isotonic fit, table export, eval
  eval/          human-verified release-gating label store
  model_cards/   the three required model cards
  cache/         teacher prompt/response audit cache (gitignored content, tracked directory)
```

## Quick start

```bash
make training-dev-artifacts       # synthetic-only, offline, seconds
make training-release-artifacts   # full gated pipeline (compliance -> package -> CleanBench)
python -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site
```

Individual stages: `make training-seed`, `training-corrupt`,
`training-teacher-labels`, `training-distill`, `training-eval`,
`training-package-artifacts` (see `training/Makefile`).

## Ground-truth policy

Corruption metadata is the preferred ground truth; teacher output is
training data only and never gates a release on its own — release-gating
eval labels must be 100% human-verified (`training/eval/human_verified.py`).
