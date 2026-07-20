---
title: Production-readiness checklist
description: >-
  A practical checklist for putting freshdata into a production data or ML
  pipeline — pinning, gating on trust, protecting columns, memory, determinism,
  privacy, observability, and upgrades.
keywords: freshdata production, data cleaning pipeline, trust score gate, reproducible cleaning, production checklist
---

# Production-readiness checklist

freshdata is designed to run unattended in a pipeline, but "runs" and
"production-ready" aren't the same thing. Work through this before you let it
clean data nobody is watching. Each item links to the relevant guarantee on the
[Honest limitations](limitations.md) page.

## Install & pin

- [ ] Pin an exact version (`freshdata-cleaner==2.0.0`) and the extras you use
      (`freshdata-cleaner[polars,privacy]`). Cleaning defaults can tighten between
      minor versions — pinning keeps decisions reproducible.
- [ ] Install only the extras you need. The base install has **no** heavy deps;
      Polars/DuckDB/Spark, NER PII, and orchestration exporters are opt-in.
- [ ] If you use optional learned models, vendor them air-gapped and pin their
      checksums. Cleaning never downloads anything; `fd.models.pull` is the only
      network call and it is explicit.

## Make it deterministic

- [ ] Pass a fixed `seed` anywhere sampling is involved (streaming reservoirs,
      benchmarks) so runs are reproducible.
- [ ] Snapshot the config. Prefer an explicit `CleanConfig` (or a saved
      `context=`/`policy=`) over scattered keyword args, and check it into VCS.
- [ ] Verify byte-for-byte reproducibility on a golden input in CI: same input +
      same version + same config ⇒ same output and same report.

## Gate on quality, don't just clean

- [ ] Set `fail_under_trust=...` so a batch that scores below your bar **fails
      the pipeline** instead of flowing downstream. The 0–100 trust score
      decreases monotonically as data gets dirtier.
- [ ] Add a `contract=` schema/quality gate for the columns your consumers
      depend on (types, ranges, allowed values, nullability).
- [ ] Treat `report.warnings` as signal, not noise — route them to your alerting.
- [ ] Decide your policy for drift: in streaming, `report.streaming` flags schema
      and distribution drift per batch — wire it to an alert.

## Protect what must not change

- [ ] Declare identifiers, labels/targets, and money/PII columns with a
      `context=` policy (`"Never modify revenue. customer_id is unique."`).
      Context-protected columns get a **hard byte-identity guarantee**.
- [ ] Prefer `strict=True` in production so an unparsed or ambiguous policy raises
      instead of silently degrading.
- [ ] In streaming, remember the policy is compiled once against the **first
      batch** — make sure that batch is representative of the stream's schema.
      Inspect `cleaner.policy_` after startup.

## Memory & scale

- [ ] Know whether your path materializes. `report.materialized` is `False` only
      for the native handle outputs (`output_format="duckdb"`/`"polars-lazy"`);
      everything else lands in a pandas frame in RAM.
- [ ] For data larger than memory, use `StreamingCleaner` /
      `fd.clean_timeseries(stream=...)` (bounded, never concatenated) or a native
      backend with a lazy output handle.
- [ ] Watch `report.fallback_events`: a native run that needs the pandas decision
      engine falls back and materializes. Use `strategy="conservative"` to keep
      the native handle if that matters for your memory budget.
- [ ] Size micro-batches and the dedup window deliberately — cross-batch dedup is
      recent-window, not global.

## Privacy & security

- [ ] Confirm no data leaves the process: cleaning makes no network calls.
- [ ] For PII, choose your tier: regex detection needs no extras; NER +
      format-preserving encryption needs `[privacy]`. Review masking rules in the
      report before trusting them.
- [ ] Keep any token vaults / encryption keys out of VCS and injected at runtime.
- [ ] If you accept untrusted files, validate/limit them upstream — parsers are
      robust but you own the ingestion boundary.

## Observability

- [ ] Persist `report.to_dict()` (JSON) per run for audit — it is stable and
      backward-compatible; non-streaming reports never emit a `streaming` key.
- [ ] Log the trust score, `duplicates_removed`, imputation counts, and drift
      events as metrics, not just text.
- [ ] Keep the decision log with the output artifact so any cell change is
      explainable after the fact.

## Calibration & benchmarks — read before you trust confidence

- [ ] Don't rely on the strict ≤ 0.03 ECE tier: out of the box freshdata is
      calibrated to ~0.038 ECE (clears the 0.05 target). The strict tier needs the
      unpublished `calib-v1` artifact — see [limitations](limitations.md#models-and-calibration-honesty).
- [ ] Treat confidence as a conservative floor, not a promise. freshdata
      preserves rather than guesses; the ">95%" imputation clause rarely fires
      outside near-deterministic cases.
- [ ] If you cite a CleanBench number, reproduce it first
      (`--reproduce-headline` / `--verify-results`).

## Release & supply-chain integrity

- [ ] Consume releases from PyPI published via OIDC trusted publishing (no
      long-lived tokens), with SLSA build provenance and PEP 740 attestations
      attached — verify them if your org requires it.
- [ ] Track the [GitHub Releases](https://github.com/FreshCode-Org/freshdata/releases)
      for the signed artifacts and generated notes.

## Upgrades

- [ ] Read the release notes before bumping — quality-debt escalation and
      dirty-join scoring are tuned conservatively and may shift between minor
      versions.
- [ ] Re-run your golden-input reproducibility check after any upgrade and diff
      the report, not just the data.
