# Training pipeline compliance

This document governs every development-time use of external ("teacher")
models and external data in the FreshData training pipeline. The runtime
library is unaffected: it is deterministic, offline, model-free by default,
and never calls an LLM or any cloud service from `fd.clean`.

## Teacher models — allowed uses

Teacher models may be used **only inside `training/`**, only for:

1. realism direction (tuning corruptor frequency/parameters)
2. column-role labeling
3. NL context paraphrase generation
4. ambiguity adjudication
5. rationale style templates
6. benchmark red-teaming

They must **never** be used to:

- create messy/clean pairs from scratch (unverifiable ground truth)
- label release-gating eval data without human verification
- run at library runtime, per cell, or inside `fd.clean`
- receive full data rows or real PII (values are masked first;
  `training/teacher/tasks.py` enforces both)

## Provider terms checklist (before any teacher run)

Every provider must have a record in `training/teacher/compliance_ledger.json`
containing:

- provider name
- terms URL and a terms snapshot id
- date the terms were checked (records older than 180 days are stale)
- reviewer (name or initials of the human who read the terms)
- allowed/disallowed usage list
- whether outputs may be used for **model training**
- whether outputs may be **redistributed**
- status: `approved` / `rejected` / `pending`

`python -m training.teacher.compliance check` blocks the pipeline when a
snapshot is missing or stale, the intended use is not approved, the reviewer
is missing, or the provider disallows training use. `make
training-release-artifacts` runs this check first.

## Audit trail

Every teacher call is cached under `training/cache/teacher/` with the raw
prompt, raw response, provider, model, timestamp, and terms snapshot id
(key: `provider + model + prompt_template_sha256 + input_sha256 +
schema_version`). Cache entries are the audit record for every
teacher-derived training label.

## Human review

- ≥ 5% of every teacher batch is sampled for human review
  (`training/teacher/review.py`).
- Teacher–human disagreement > 3% forces full review of the batch.
- Release-gating eval labels are 100% human-verified
  (`training/eval/human_verified.py` refuses anything less).
- Reviewer identity, timestamp, and disagreement reasons are stored and
  exported in the review summary JSON.

## Training data

- Every seed source is registered in `training/seed/registry.json` with
  license, attribution, explicit training permission, and resolved PII risk;
  `python -m training.datasets.validators --check-licenses` gates releases.
- No real PII anywhere: identity-shaped data is fully synthetic
  (`training/seed/synthetic/`), tagged `synthetic=true`.
- Corruption metadata — not teacher output — is the preferred ground truth.
