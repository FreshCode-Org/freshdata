# Model card — `calib-v1`

## 1. Purpose
Isotonic confidence-calibration tables mapping raw semantic-proposal scores
to calibrated probabilities of a correct repair, per (backend, issue
family). Plain JSON — not a neural model.

## 2. Non-goals
- Does not create or veto proposals; it only rescales confidence.
- Never makes the deterministic protected-column checks probabilistic.

## 3. Runtime behavior
Loaded via the Phase-3 model registry (or the packaged conservative
default). If the table is missing or corrupt the runtime passes raw scores
through and reports `calibration_version="uncalibrated"`. Embedding-backend
confidences are always capped below 1.0.

## 4. Training data sources
CleanBench T1–T4 fixture runs with corruption-metadata ground truth (never
teacher guesses); the evaluation split is held out from fitting.

## 5. License summary
Apache-2.0; training inputs are in-repo synthetic fixtures.

## 6. PII policy
Fixtures are fully synthetic; no real PII enters calibration training.

## 7. Teacher-model usage
None. Calibration is fitted exclusively on machine-verified outcomes.

## 8. Human-review process
Release gates are computed on machine-verified outcomes over human-reviewed
fixture definitions; the CleanBench release run re-checks all gates.

## 9. Evaluation metrics
`eval_metrics.json`: ECE and precision@0.95 on the held-out split.
Gates: ECE ≤ 0.03, P@0.95 ≥ 0.99, protected-column violation rate = 0,
false modification rate ≤ 0.1% (the last two via CleanBench).

## 10. Known limitations
Curves are fitted on synthetic benchmark distributions; heavily
out-of-distribution data reverts to conservative behavior (embedding cap,
suggest-only below thresholds).

## 11. Failure modes
Missing/corrupt table → uncalibrated passthrough (safe); unseen
(backend, issue) pairs fall back to the per-backend `*` curve or identity.

## 12. Protected-column guarantee
Calibration never touches the deterministic byte-identity guard.

## 13. Calibration version
`version` field inside `calibration.json` (this artifact).

## 14. Artifact hash
See `manifest.json`.

## 15. FreshData version compatibility
`freshdata_min_version` in `manifest.json`; enforced by the model registry.

---
**Explicit statements**: no generative repair model; no runtime LLM; no
automatic model download during cleaning; model output is evidence, not
authority; the protected-column guard remains absolute.
