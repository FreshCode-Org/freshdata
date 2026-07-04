# FreshData developer tasks.

PY ?= python

# training-* targets are matched by the pattern rule below (pattern rules
# cannot be .PHONY; the delegated targets are .PHONY inside training/Makefile).
.PHONY: help benchmark benchmark-ci benchmark-report benchmark-fixtures benchmark-test \
        cleanbench-full

help:
	@echo "Targets:"
	@echo "  benchmark           Full-scale local benchmark run (all fixtures, default sizes)"
	@echo "  benchmark-ci        CI-shaped run: 10k-row variants, 3 timing repeats"
	@echo "  benchmark-report    Render markdown + JSON report for the latest run"
	@echo "  benchmark-fixtures  Write fixture CSVs to benchmarks/generated_fixtures/"
	@echo "  benchmark-test      Run the benchmark test suite"
	@echo "  cleanbench-full     Full CleanBench T1-T5 with release gates + site report"
	@echo "  training-*          Phase-5 training pipeline (see training/Makefile)"

# Full release-gating CleanBench run.
cleanbench-full:
	$(PY) -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site --check-gates

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
