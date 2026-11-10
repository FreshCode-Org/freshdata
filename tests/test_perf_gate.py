from benchmarks.perf_gate import GateThresholds, evaluate_gate


def test_gate_passes_within_thresholds() -> None:
    baseline = {"plan_ms": 200.0, "compile_ms": 0.45, "total_ms": 800.0, "peak_rss_mb": 200.0}
    candidate = {"plan_ms": 205.0, "compile_ms": 0.50, "total_ms": 840.0, "peak_rss_mb": 205.0}

    failures = evaluate_gate(baseline, candidate, GateThresholds())
    assert failures == []


def test_gate_fails_on_eager_compile_regression() -> None:
    baseline = {"plan_ms": 200.0, "compile_ms": 0.45, "total_ms": 800.0, "peak_rss_mb": 200.0}
    # Simulates an accidental .collect() inside apply_plan()
    candidate = {"plan_ms": 200.0, "compile_ms": 420.0, "total_ms": 810.0, "peak_rss_mb": 200.0}

    failures = evaluate_gate(baseline, candidate, GateThresholds(max_compile_ms=10.0))
    assert len(failures) == 1
    assert "compile_ms" in failures[0]


def test_gate_fails_on_total_ms_regression() -> None:
    baseline = {"plan_ms": 200.0, "compile_ms": 0.45, "total_ms": 800.0, "peak_rss_mb": 200.0}
    # 25% slowdown exceeds the 15% max_total_regression_pct default
    candidate = {"plan_ms": 250.0, "compile_ms": 0.48, "total_ms": 1000.0, "peak_rss_mb": 200.0}

    failures = evaluate_gate(baseline, candidate, GateThresholds(max_total_regression_pct=15.0))
    assert len(failures) == 1
    assert "total_ms regressed by +25.0%" in failures[0]


def test_gate_fails_on_memory_spike() -> None:
    baseline = {"plan_ms": 200.0, "compile_ms": 0.45, "total_ms": 800.0, "peak_rss_mb": 200.0}
    # Simulates broken streaming materializing full table in RAM
    candidate = {"plan_ms": 200.0, "compile_ms": 0.45, "total_ms": 810.0, "peak_rss_mb": 650.0}

    failures = evaluate_gate(baseline, candidate, GateThresholds(max_rss_regression_pct=20.0))
    assert len(failures) == 1
    assert "peak_rss_mb regressed" in failures[0]
