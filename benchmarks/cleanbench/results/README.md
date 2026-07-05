# Committed CleanBench results

`latest.json` / `latest.md` are the current, always-overwritten result of the
public reproduction command:

```bash
python -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site --reproduce-headline
```

Verify a committed result (schema, dataset hashes, environment metadata,
release-gate consistency, README claim links) with:

```bash
python -m benchmarks.cleanbench --verify-results benchmarks/cleanbench/results/latest.json
```

`release/` holds a point-in-time snapshot per tagged release (`vX.Y.Z.json` /
`vX.Y.Z.md`), so a past release's headline numbers stay reproducible even
after `latest.json` moves on. Cut a new one at release time:

```bash
cp benchmarks/cleanbench/results/latest.json benchmarks/cleanbench/results/release/vX.Y.Z.json
cp benchmarks/cleanbench/results/latest.md   benchmarks/cleanbench/results/release/vX.Y.Z.md
```

## Reading a result file

- `release_gates.passed` / `release_gates.failures` — the hard release gates
  (protected-column violations, false modification rate, ECE, P@0.95, runtime/
  memory overhead vs. baseline). **A result can be committed with gates
  failing** — CleanBench does not hide a failure, it discloses it; a build is
  release-ready only when `release_gates.passed` is `true`.
- `environment` — FreshData version, git commit, Python/OS, and optional-engine
  (Polars/DuckDB) versions the run used.
- `dataset_hashes` — sha256 of each track's fixture, so `--verify-results` can
  detect fixture drift between the committed result and a fresh reproduction.
- `baselines` — present only when run with `--reproduce-headline`: pandas /
  pyjanitor / Great Expectations / disclosed LLM-agent comparison, each either
  scored (`status: "ran"`) or honestly skipped (`status: "skipped"`, `reason`).

No result file contains raw PII or secrets: every fixture is synthetic and
seeded in code (see `benchmarks/cleanbench/fixtures/`), and the LLM-agent
baseline is skipped by default everywhere (including CI).
