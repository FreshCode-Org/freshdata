# Model card — `fd-col-encoder-v1` (+ bundled role head)

## 1. Purpose
Local column/value encoder for FreshData's optional embedding backend, with a
bundled semantic-type **role head** used as *additional evidence* for column
role inference. It rescues reference-value repairs the deterministic experts
cannot reach (distance-2 typos of allowed values) and scores column
semantic types.

## 2. Non-goals
- Not a generative model; it never writes values.
- Not a general text embedder; scoped to tabular column names + cell values.
- Not a replacement for FreshData's deterministic experts or detectors.

## 3. Runtime behavior
Loaded only when the user installs the `[semantic]` extra **and** explicitly
pulls the model (`freshdata models pull fd-col-encoder-v1`) or drops files
into `FRESHDATA_MODEL_DIR`. CPU-only, offline, deterministic. **Model output
is evidence, not authority**: every proposal goes through the same policy
gate, calibration, and audit trail as deterministic proposals, and ambiguous
merges are never auto-applied.

## 4. Training data sources
Fully synthetic PII-shaped seed corpus (`training/seed/synthetic/`),
hand-authored fixtures, and corruptor-derived labeled pairs
(`training/seed/registry.json`). No real PII, no customer data, no scraped
private data, no production logs.

## 5. License summary
Model artifact: Apache-2.0. All training data: CC0-1.0 / Apache-2.0 in-repo
sources (see `training/seed/LICENSES.md`).

## 6. PII policy
Training data contains only synthetic, invented identities tagged
`synthetic=true`. Values sent to any development-time teacher model are
masked first (`training/teacher/tasks.py:mask_pii`).

## 7. Teacher-model usage
Teacher models may contribute column-role labels and realism direction at
development time only (cached, compliance-gated, ≥5% human-reviewed). They
never generate messy/clean pairs and never gate a release.

## 8. Human-review process
≥5% of teacher batches human-reviewed; >3% disagreement forces full batch
review; the release-gating eval split (`training/eval/data/`) is 100%
human-verified with reviewer identity and timestamps recorded.

## 9. Evaluation metrics
See `eval_metrics.json` in this artifact (macro-F1, per-class F1, abstention
rate, content-detector contradiction rate, adversarial alias accuracy).
Release gates: macro-F1 ≥ 0.90, contradiction rate ≤ 1%, alias accuracy ≥ 0.85.

## 10. Known limitations
- Eval corpus is synthetic; real-world column vocabularies are broader.
- English + Indian-English column headers only.
- Role head abstains (`unknown`) below its confidence threshold rather than
  guessing; downstream code must handle abstention.

## 11. Failure modes
Opaque codes and short low-support columns fall to `unknown`; heavily mixed
columns may be typed by their majority shape; adversarial headers reduce
confidence rather than flipping labels (gated).

## 12. Protected-column guarantee
Unaffected by this model. The byte-identity protected-column guard is
deterministic, runs regardless of any model, and CleanBench gates its
violation rate at exactly zero.

## 13. Calibration version
Scores are calibrated through `calib-v1` isotonic tables; without them the
runtime reports `calibration_version="uncalibrated"` and stays conservative.

## 14. Artifact hash
See `manifest.json` (`sha256` per file, top-level artifact hash).

## 15. FreshData version compatibility
`freshdata_min_version` in `manifest.json`; enforced by the model registry.

---
**Explicit statements**: no generative repair model; no runtime LLM; no
automatic model download during cleaning; model output is evidence, not
authority; the protected-column guard remains absolute.
