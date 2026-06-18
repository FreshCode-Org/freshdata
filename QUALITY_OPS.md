# Quality Operations Runbook

This runbook defines the weekly quality cadence and the minimum metrics to track
for release readiness.

## Weekly cadence

- **Monday (scope/risk review)**
  - Review open PR risk levels (engine changes, fixture changes, workflow changes).
  - Confirm required CI lane status and top failure causes from last week.
- **Wednesday (midweek quality check)**
  - Review skip counts and flaky failures.
  - Verify nightly online/large job output and open drift issues.
- **Friday (merge gate)**
  - Merge only PRs with passing required checks.
  - Validate release readiness scorecard before tagging.

## Scorecard metrics

Track these weekly (rolling 4-week trend):

- Required CI flake rate (% reruns needed to pass).
- Median required CI duration (minutes).
- `pytest` skip count in required lane.
- Nightly online/large failures (count + top 3 causes).
- Golden snapshot updates merged (count) with diff summaries attached.
- Release gate pass/fail rate.

## Exit criteria for a release candidate

- Required lane flake rate is near zero for the last 2 weeks.
- No unresolved deterministic regression in outlier/repair tests.
- No unexplained golden drift.
- Release workflow preflight passes on tag candidate.

## Operational commands

Required lane locally:

```bash
ruff check src tests
mypy src/freshdata
pytest -m "not online and not large"
```

Nightly lane locally:

```bash
pytest -m "online or large or tier1"
```
