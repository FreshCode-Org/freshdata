# FreshData developer tasks.

PY ?= python

.PHONY: help benchmark benchmark-ci benchmark-report benchmark-fixtures benchmark-test

help:
	@echo "Targets:"
	@echo "  benchmark           Full-scale local benchmark run (all fixtures, default sizes)"
	@echo "  benchmark-ci        CI-shaped run: 10k-row variants, 3 timing repeats"
	@echo "  benchmark-report    Render markdown + JSON report for the latest run"
	@echo "  benchmark-fixtures  Write fixture CSVs to benchmarks/generated_fixtures/"
	@echo "  benchmark-test      Run the benchmark test suite"

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
