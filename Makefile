# FreshData developer tasks.

PY ?= python

# training-* targets are matched by the pattern rule below (pattern rules
# cannot be .PHONY; the delegated targets are .PHONY inside training/Makefile).
.PHONY: help benchmark benchmark-ci benchmark-report benchmark-fixtures benchmark-test \
        cleanbench-full truthbench-release truthbench-pr performance-ci \
        performance-baseline performance-profile performance-report

help:
	@echo "Targets:"
	@echo "  benchmark           Full-scale local benchmark run (all fixtures, default sizes)"
	@echo "  benchmark-ci        CI-shaped run: 10k-row variants, 3 timing repeats"
	@echo "  benchmark-report    Render markdown + JSON report for the latest run"
	@echo "  benchmark-fixtures  Write fixture CSVs to benchmarks/generated_fixtures/"
	@echo "  benchmark-test      Run the benchmark test suite"
	@echo "  cleanbench-full     Full CleanBench T1-T5 with release gates + site report"
	@echo "  truthbench-release  Official TruthBench release verification (fail-closed)"
	@echo "  truthbench-pr       TruthBench PR ratchet (fails only on regressions)"
	@echo "  performance-ci      Run the CI-safe performance contract suite"
	@echo "  performance-baseline Run the performance investigation matrix"
	@echo "  performance-profile Profile one 100k-row performance case"
	@echo "  performance-report Analyze and render compact performance evidence"
	@echo "  training-*          Phase-5 training pipeline (see training/Makefile)"

# Full release-gating CleanBench run.
cleanbench-full:
	$(PY) -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site --check-gates

# Official TruthBench release verification. Exact command required by PR CI
# and the release workflow; any partial run, missing backend, or gate failure
# exits nonzero.
truthbench-release:
	PYTHONPATH=src $(PY) -m benchmarks.truthbench run \
	  --profile release \
	  --backends pandas,polars,duckdb \
	  --require-backends \
	  --repeats 2 \
	  --check

# PR ratchet: the same full run, but only a regression against the committed
# baseline.json (or an infrastructure error) fails the job. Known-red gates
# are release blockers enforced by the release workflow's truthbench-release.
truthbench-pr:
	PYTHONPATH=src $(PY) -m benchmarks.truthbench run \
	  --profile release \
	  --backends pandas,polars,duckdb \
	  --require-backends \
	  --repeats 2 \
	  --check-regressions

# Phase-5 training pipeline targets delegate to training/Makefile.
training-%:
	$(MAKE) -C training PY=$(PY) $@

# Full-scale local run. Override sizes per fixture by editing DEFAULT_SIZES in
# benchmarks/bench.py, or call bench.py single --size <n> for the 5M+ variants.
benchmark:
	$(PY) benchmarks/bench.py run
	$(PY) benchmarks/bench.py report

benchmark-ci:
	$(PY) benchmarks/bench.py run --repeat 3
	$(PY) benchmarks/bench.py report

benchmark-report:
	$(PY) benchmarks/bench.py report

benchmark-fixtures:
	$(PY) benchmarks/bench.py fixtures

benchmark-test:
	# --no-cov: the benchmark suite exercises only a slice of freshdata, so it
	# must not be measured against the package-wide --cov-fail-under gate.
	$(PY) -m pytest tests/benchmark -q --no-cov

performance-ci:
	$(PY) -m pytest tests/performance -q --no-cov

performance-baseline:
	$(PY) -m benchmarks.performance run --output benchmarks/results/performance/baseline

performance-profile:
	$(PY) -m benchmarks.performance profile --rows 100000 --widths medium --configs default --report-modes true --output benchmarks/results/performance/baseline

performance-report:
	$(PY) -m benchmarks.performance analyze --input benchmarks/results/performance/baseline --output benchmarks/results/performance/baseline-summary.json
	$(PY) -m benchmarks.performance render --input benchmarks/results/performance/baseline-summary.json --output benchmarks/results/performance/baseline-report.md
