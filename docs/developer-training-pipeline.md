# Developer training pipeline (Phase 5)

The `training/` package builds, evaluates, and packages FreshData's small
local model artifacts. It is a **development-time tool**: it lives outside
`src/freshdata`, is never shipped in the wheel, and the runtime never
imports it.

## What stays true at runtime

- `fd.clean` is deterministic, offline, and **model-free by default**
- no LLM is used at runtime, ever — teacher models exist only in `training/`
- no cloud call from `fd.clean`; no automatic model download
- no generative repair model exists anywhere in FreshData
- the protected-column byte-identity guard is absolute
- no model weights ship in the PyPI wheel (enforced by
  `.github/workflows/wheel-size.yml` and `tests/test_semantic_backcompat.py`)

## Pipeline stages

```bash
make training-seed               # license gate + seed corpus
make training-corrupt            # ~40 corruptors -> labeled pairs
make training-teacher-labels     # compliance-gated teacher tasks (optional)
make training-distill            # role/intent heads, export, int8 quantize
make training-eval               # calibration + full CleanBench gates
make training-package-artifacts  # dist/artifacts/ with manifests + cards
```

End-to-end:

```bash
make training-dev-artifacts      # synthetic-only, offline, ~seconds
make training-release-artifacts  # full pipeline; fails on any gate failure
```

## How seed data is licensed

Every source is registered in `training/seed/registry.json` with license,
attribution, explicit `allowed_for_training`, and a resolved PII risk;
`python -m training.datasets.validators --check-licenses` blocks unclear
licenses, share-alike data without legal review, and unresolved PII risk.
See `training/seed/LICENSES.md`.

## How corruptors create ground truth

`training/corruptors/` injects parameterized, deterministic dirt into known
clean frames and emits a machine-verifiable label per mutation (raw value,
clean value, transform family, `should_repair`, `should_auto_apply`, risk,
protected, ambiguous). Ground truth is corruption metadata — never a model
guess. Ambiguous corruptions are labeled *never auto-apply*; protected-column
traps label **any** mutation as failure.

## What teacher models may and may not do

Allowed (development-time only, batched, cached, PII-masked): realism
direction, column-role labeling, context paraphrases, ambiguity
adjudication, rationale templates, benchmark red-teaming.

Never: runtime calls, per-cell calls, generating messy/clean pairs from
scratch, receiving full rows or real PII, or serving as sole release-gating
truth. See `training/COMPLIANCE.md` and
`python -m training.teacher.compliance check`.

## How human review works

≥5% of each teacher batch is sampled for review; >3% disagreement forces a
full batch review; release-gating eval labels are 100% human-verified with
reviewer identity and timestamps (`training/teacher/review.py`,
`training/eval/human_verified.py`).

## How model cards are produced

`training/model_cards/*.md` are authored with the 15 required sections and
copied into each packaged artifact; `training.distill.package_artifacts`
fails when a card, license summary, or eval summary is missing.

## How calibration is trained

`training/calibration/` extracts per-proposal features from CleanBench runs,
fits isotonic curves (pool-adjacent-violators, pure numpy) per (backend,
issue family), and exports the registry-compatible `calib-v1` table. Gates:
ECE ≤ 0.03 and P@0.95 ≥ 0.99 on a held-out split. The runtime only ever
*loads* the JSON — a missing table degrades to `uncalibrated`, and
protected-column checks stay deterministic.

## How CleanBench gates releases

`python -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site
--check-gates` runs representation, semantic-value, context-compliance,
profile-replay, and scale tracks; see [Benchmarks](benchmarks.md) for the
gate list and latest results.

## Reproducing artifact builds

Everything is seeded: corruptors, generators, and training are deterministic
(`sha256`-hashed features, zero-init full-batch training — no RNG). The same
commit + same seeds produce byte-identical weights; manifests record the git
commit and per-file SHA256.

## Running the tests

- model-free (default): `pytest -m "not online and not large"` — this is PR CI
- semantic-with-stub: included in the default run (`FRESHDATA_SEMANTIC_STUB`
  paths and the CleanBench stub encoder need no model files)
- real-model nightly, locally:
  `pip install -e ".[semantic]" onnx && make training-dev-artifacts &&
  pytest tests -k "semantic or models or cleanbench" --no-cov`

## Limitations and known failure modes

Head evals run on synthetic/template corpora — high scores there do not
promise the same on arbitrary real-world data; the encoder contrastive stage
is optional and is recorded as `skipped` without torch; T5 perf gates
compare against a same-machine baseline and are informational until one is
pinned. See [Honest limitations](limitations.md).
