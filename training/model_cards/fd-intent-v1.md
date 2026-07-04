# Model card — `fd-intent-v1`

## 1. Purpose
Small intent classifier for natural-language context sentences (13 intents:
DOMAIN, UNIQUE, VALID_FORMAT, LOCALE_FORMAT, PROTECT, IMPUTE_IF,
ALLOWED_VALUES, RANGE, DEDUP_KEY, DROP_IF, RENAME, MAP, UNKNOWN). It supplies
*optional evidence* to the deterministic context parser, e.g. flagging a
sentence the tier-0 patterns missed.

## 2. Non-goals
- Never silently replaces deterministic parser behavior: a policy rule is
  only created by the deterministic compiler.
- Not a chat model, not a general NLU system, no slot *generation* — slots
  are extracted by deterministic regex/lexicon code.

## 3. Runtime behavior
Optional, offline, CPU-only, explicit opt-in via the model registry. In
strict mode, unresolved or unparsed context is still surfaced to the user
exactly as before. Protection detection favors recall over coverage:
the PROTECT class is trained and gated at recall ≥ 0.99.

## 4. Training data sources
Phase-1 golden context corpus (this repository, Apache-2.0), synthetic
paraphrase templates including the Indian-English/Hinglish set
(`training/seed/synthetic/generators.py`), corruptor context variants, and
optionally teacher paraphrases (cached, reviewed, training-only).

## 5. License summary
Artifact: Apache-2.0. Data: in-repo Apache-2.0 + CC0 synthetic sources.

## 6. PII policy
Context sentences reference column names, never cell values; the corpus is
synthetic/hand-authored. No real PII.

## 7. Teacher-model usage
Paraphrase generation and ambiguity adjudication only, development-time
only, compliance-gated and cached. Teacher labels never gate releases.

## 8. Human-review process
Release-gating eval labels are 100% human-verified; the eval split is
author-disjoint (held-out authors `t2`, `hinglish_b`) so paraphrase
generalization is measured, not memorization.

## 9. Evaluation metrics
See `eval_metrics.json`: exact intent accuracy, slot F1, UNKNOWN precision,
protected-intent recall, conflict-detection accuracy, author-disjoint
paraphrase accuracy. Gates: exact ≥ 0.92, slot F1 ≥ 0.90, UNKNOWN precision
≥ 0.95, PROTECT recall ≥ 0.99.

## 10. Known limitations
- Trained on template-derived paraphrases; entirely novel phrasing may
  abstain to UNKNOWN (by design).
- English and Hinglish only.
- Sentences with multiple intents are classified by the dominant one; the
  deterministic parser remains responsible for multi-intent splitting.

## 11. Failure modes
Low-confidence predictions abstain to UNKNOWN rather than inventing
constraints; a missed PROTECT is the gated worst case (recall ≥ 0.99);
conflicting instructions are surfaced, never resolved silently.

## 12. Protected-column guarantee
The model can only *add* protection evidence, never remove it. The runtime
byte-identity guard is deterministic and absolute.

## 13. Calibration version
Head confidences feed the same `calib-v1` scoring path as other evidence.

## 14. Artifact hash
See `manifest.json`.

## 15. FreshData version compatibility
`freshdata_min_version` in `manifest.json`; enforced by the model registry.

---
**Explicit statements**: no generative repair model; no runtime LLM; no
automatic model download during cleaning; model output is evidence, not
authority; the protected-column guard remains absolute.
