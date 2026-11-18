# FreshData developer tasks.

PY ?= python

.PHONY: help benchmark-test automation-pr performance-ci \
        performance-baseline performance-profile performance-report coverage-report

help:
	@echo "Targets:"
	@echo "  benchmark-test      Run the benchmark test suite"
	@echo "  automation-pr       Profile + clean all fixtures (PR automation gate)"
	@echo "  performance-ci      Run the CI-safe performance contract suite"
	@echo "  performance-baseline Run the performance investigation matrix"
	@echo "  performance-profile Profile one 100k-row performance case"
	@echo "  performance-report Analyze and render compact performance evidence"
	@echo "  coverage-report     Show per-module coverage for the fast test lane"

# Run the same test scope as the required PR lane and print uncovered lines for
# every measured module so contributors can identify useful testing targets.
coverage-report:
	$(PY) -m pytest -m "not online and not large" --cov-report=term-missing

# Profile + clean every committed fixture; same gate as .github/workflows/automation.yml.
automation-pr:
	$(PY) scripts/automate_freshdata.py --output-dir scripts/.automation_out

benchmark-test:
	$(PY) -m pytest -q

performance-ci:
	$(PY) -m pytest -q

performance-baseline:
	$(PY) -m benchmarks.performance run --output benchmarks/results/performance/baseline

performance-profile:
	$(PY) -m benchmarks.performance profile --rows 100000 --widths medium --configs default --report-modes true --output benchmarks/results/performance/baseline

performance-report:
	$(PY) -m benchmarks.performance analyze --input benchmarks/results/performance/baseline --output benchmarks/results/performance/baseline-summary.json
	$(PY) -m benchmarks.performance render --input benchmarks/results/performance/baseline-summary.json --output benchmarks/results/performance/baseline-report.md
