# TruthBench results

Artifacts produced by the official release command:

```bash
PYTHONPATH=src python -m benchmarks.truthbench run \
  --profile release \
  --backends pandas,polars,duckdb \
  --require-backends \
  --repeats 2 \
  --check
```

- `latest.json` — the full serialized run (schema-validated, written
  atomically; never claims success for a partial run).
- `latest.md` — the human-readable gate summary for the same run.
- `baseline.json` — gate outcomes of the accepted reference run. The baseline
  is regression *evidence* only: an absolute gate failure is a failure even
  when the baseline also failed, and updating this file requires explicit
  human approval with a documented justification.
- `failures/` — one sanitized, minimized JSON reproduction per real failure
  (stable `tbf-*` id, reduced frame, exact reproduction command, zero raw
  PII).

Artifacts always correspond to the exact commit that produced them; rerun the
command above rather than editing anything here by hand.
