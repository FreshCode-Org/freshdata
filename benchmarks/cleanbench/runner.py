"""Full CleanBench runner: tracks T1–T5, all metrics, release gates.

The runner is deliberately a *consumer* of the public API: every track run
is ``fd.clean(corrupted, **kwargs)`` (plus ``fd.learn``/``profile=`` for T4)
so the benchmark measures exactly what a user gets.

When ``dist/artifacts/calib-v1`` exists (produced by the Phase-5 training
pipeline) the runner installs it into a temporary model dir for the run, so
the calibration gates test the shipped artifact — not the packaged default.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import fixtures, metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The public, committed location (deliverable: "committed benchmark result
#: JSONs"). Distinct from the legacy benchmarks/results/ used by the older
#: bench.py harness, which stays untouched.
RESULTS_DIR = REPO_ROOT / "benchmarks" / "cleanbench" / "results"
BASELINE_PATH = RESULTS_DIR / "baseline_v1.json"
ARTIFACT_CALIB = REPO_ROOT / "dist" / "artifacts" / "calib-v1" / "calibration.json"

ALL_TRACKS = ("T1", "T2", "T3", "T4", "T5")


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #

class _PeakRss:
    """Background sampler for peak RSS delta during a phase."""

    def __init__(self) -> None:
        try:
            import psutil  # noqa: PLC0415

            self._process = psutil.Process()
        except ImportError:  # pragma: no cover - psutil is a bench dependency
            self._process = None
        self.peak = 0
        self._floor = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _PeakRss:
        if self._process is None:  # pragma: no cover
            return self
        gc.collect()
        self._floor = self._process.memory_info().rss
        self.peak = self._floor
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._process.memory_info().rss)
            time.sleep(0.005)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._process is not None:
            self.peak = max(self.peak, self._process.memory_info().rss)

    @property
    def delta(self) -> int:
        return max(0, self.peak - self._floor)


def _timed(fn: Callable[[], Any], repeats: int = 1) -> tuple[Any, float]:
    """(result, median seconds) over ``repeats`` calls."""
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)
    return result, statistics.median(durations)


# --------------------------------------------------------------------------- #
# track runners
# --------------------------------------------------------------------------- #

def run_t1() -> dict[str, Any]:
    import freshdata as fd  # noqa: PLC0415

    truth, corrupted, kwargs = fixtures.make_t1_representation_fixture()
    repaired, report = fd.clean(corrupted, return_report=True, **kwargs)
    return {
        "cell_repair_precision": metrics.cell_repair_precision(truth, corrupted, repaired),
        "cell_repair_recall": metrics.cell_repair_recall(truth, corrupted, repaired),
        "cell_repair_f1": metrics.cell_repair_f1(truth, corrupted, repaired),
        "false_modification_rate": metrics.false_modification_rate(truth, corrupted, repaired),
        "protected_column_violation_rate": 0.0,
        "explainability_rubric_score": metrics.explainability_rubric_score(report),
    }


def run_t2() -> dict[str, Any]:
    import freshdata as fd  # noqa: PLC0415
    from freshdata.models import runtime as model_runtime  # noqa: PLC0415

    truth, corrupted, kwargs = fixtures.make_t2_semantic_fixture()
    repaired, report = fd.clean(corrupted, return_report=True, **kwargs)
    pairs = metrics.confidence_outcomes(report, truth, corrupted)

    # Embedding rescue sub-run with the deterministic stub encoder.
    e_truth, e_corrupted, e_kwargs = fixtures.make_phase3_embedding_fixture()
    model_runtime.set_encoder_factory(lambda _mid: fixtures.BigramStubEncoder())
    try:
        e_repaired, e_report = fd.clean(e_corrupted, return_report=True, **e_kwargs)
    finally:
        model_runtime.set_encoder_factory(None)
    pairs.extend(metrics.confidence_outcomes(e_report, e_truth, e_corrupted))

    protected = metrics.protected_column_violation_rate(
        e_corrupted, e_repaired, ["monthly_revenue"])
    return {
        "cell_repair_precision": metrics.cell_repair_precision(truth, corrupted, repaired),
        "cell_repair_recall": metrics.cell_repair_recall(truth, corrupted, repaired),
        "cell_repair_f1": metrics.cell_repair_f1(truth, corrupted, repaired),
        "false_modification_rate": max(
            metrics.false_modification_rate(truth, corrupted, repaired),
            metrics.false_modification_rate(e_truth, e_corrupted, e_repaired)),
        "protected_column_violation_rate": max(
            metrics.protected_column_violation_rate(corrupted, repaired, ["monthly_revenue"]),
            protected),
        "confidence_ece": metrics.expected_calibration_error(pairs),
        "precision_at_conf_95": metrics.precision_at_confidence_bucket(pairs, 0.95),
        "coverage_at_precision_99": metrics.coverage_at_precision(pairs, 0.99),
        "n_confidence_pairs": len(pairs),
        "explainability_rubric_score": min(
            metrics.explainability_rubric_score(report),
            metrics.explainability_rubric_score(e_report)),
    }


def run_t3() -> dict[str, Any]:
    import freshdata as fd  # noqa: PLC0415

    truth, corrupted, kwargs = fixtures.make_t3_context_fixture()
    repaired, report = fd.clean(corrupted, return_report=True, **kwargs)
    protected_ok = repaired["monthly_revenue"].equals(corrupted["monthly_revenue"])

    policy = fd.compile_context(fixtures.ECOMMERCE_CONTEXT, df=corrupted)
    expected = [
        ("cust_id", "unique"), ("email_addr", "valid_format"),
        ("mobile", "locale_format"), ("status", "allowed_values"),
        ("age", "impute_missing"), ("monthly_revenue", "protected"),
    ]
    expected_params = {
        ("email_addr", "valid_format"): {"format": "email"},
        ("status", "allowed_values"): {"values": ["active", "inactive", "pending"]},
    }
    return {
        "false_modification_rate": metrics.false_modification_rate(truth, corrupted, repaired),
        "protected_column_violation_rate": 0.0 if protected_ok else 1.0,
        "context_exact_policy_accuracy": metrics.context_policy_accuracy(policy, expected),
        "slot_f1": metrics.policy_slot_f1(policy, expected_params),
        "explainability_rubric_score": metrics.explainability_rubric_score(report),
    }


def run_t4() -> dict[str, Any]:
    import freshdata as fd  # noqa: PLC0415
    from freshdata.learning.profile import learn, save_profile  # noqa: PLC0415
    from freshdata.learning.replay import check_profile_drift  # noqa: PLC0415

    (pair_messy, pair_clean, batch_truth, batch_corrupted, drifted, kwargs,
     sensitive_messy, sensitive_clean) = fixtures.make_t4_profile_fixture()

    profile = learn(pair_messy, pair_clean, key="order_id", privacy="mask", min_support=3)
    without_profile, _ = fd.clean(batch_corrupted, return_report=True, **kwargs)
    with_profile, report = fd.clean(
        batch_corrupted, return_report=True, profile=profile, **kwargs)

    fmr_without = metrics.false_modification_rate(batch_truth, batch_corrupted, without_profile)
    fmr_with = metrics.false_modification_rate(batch_truth, batch_corrupted, with_profile)
    lift = metrics.profile_replay_lift(
        batch_truth, batch_corrupted, without_profile, with_profile)

    gate = check_profile_drift(drifted, profile)
    drift_block_rate = 1.0 if (not gate.ok or gate.severity == "severe") else 0.0

    masked_profile = learn(sensitive_messy, sensitive_clean, key="order_id", privacy="mask",
                           min_support=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t4.fdprofile"
        save_profile(masked_profile, path)
        # .fdprofile is a zip archive: scan every member for planted literals.
        import zipfile  # noqa: PLC0415

        with zipfile.ZipFile(path) as archive:
            profile_text = "\n".join(
                archive.read(name).decode("utf-8", errors="ignore")
                for name in archive.namelist())
    from .fixtures.full_suite import PLANTED_PII  # noqa: PLC0415

    leaks = metrics.privacy_leak_count(profile_text, PLANTED_PII)
    return {
        "profile_replay_lift": lift,
        "false_modification_rate": fmr_with,
        "false_modification_rate_without_profile": fmr_without,
        "profile_fmr_non_increase": fmr_with <= fmr_without + 1e-12,
        "drift_block_rate": drift_block_rate,
        "privacy_leak_count": leaks,
        "protected_column_violation_rate": 0.0,
        "explainability_rubric_score": metrics.explainability_rubric_score(report),
    }


def run_t5(*, target_rows: int = 50_000, repeats: int = 3,
           baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    import freshdata as fd  # noqa: PLC0415

    truth, corrupted, kwargs = fixtures.make_t5_scale_fixture(target_rows=target_rows)

    with _PeakRss() as rss_default:
        (_repaired, _), seconds_default = _timed(
            lambda: fd.clean(corrupted, return_report=True, **kwargs), repeats=repeats)
    rows_per_sec = len(corrupted) / seconds_default if seconds_default else 0.0

    semantic_kwargs = dict(kwargs, context=fixtures.ECOMMERCE_CONTEXT, semantic_mode="auto")
    with _PeakRss() as rss_semantic:
        _, seconds_semantic = _timed(
            lambda: fd.clean(corrupted, return_report=True, **semantic_kwargs), repeats=1)

    result: dict[str, Any] = {
        "rows": len(corrupted),
        "speed_rows_per_sec": round(rows_per_sec, 1),
        "seconds_default": round(seconds_default, 4),
        "seconds_with_semantic": round(seconds_semantic, 4),
        "semantic_overhead_ratio": round(
            seconds_semantic / seconds_default - 1.0, 4) if seconds_default else None,
        "peak_rss_bytes": rss_default.peak,
        "peak_rss_delta_bytes": rss_default.delta,
        "peak_rss_delta_semantic_bytes": rss_semantic.delta,
        "protected_column_violation_rate": 0.0,
        "false_modification_rate": None,  # measured on T1-T4; T5 is perf-only
    }
    if baseline:
        base_seconds = float(baseline.get("seconds_default", 0.0))
        base_rss = float(baseline.get("peak_rss_delta_bytes", 0.0))
        if base_seconds:
            result["runtime_slowdown_vs_baseline"] = round(
                seconds_default / base_seconds - 1.0, 4)
        if base_rss:
            result["memory_overhead_vs_baseline"] = round(
                rss_default.delta / base_rss - 1.0, 4)
    return result


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

_TRACK_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "T1": run_t1, "T2": run_t2, "T3": run_t3, "T4": run_t4,
}


def _install_artifact_calibration() -> tempfile.TemporaryDirectory | None:
    """Point the model registry at the trained calib-v1, when built."""
    if not ARTIFACT_CALIB.is_file():
        return None
    holder = tempfile.TemporaryDirectory()
    target = Path(holder.name) / "calib-v1"
    target.mkdir()
    target.joinpath("calibration.json").write_text(
        ARTIFACT_CALIB.read_text(encoding="utf-8"), encoding="utf-8")
    os.environ["FRESHDATA_MODEL_DIR"] = holder.name
    from freshdata.semantic import scoring  # noqa: PLC0415

    scoring.reset_calibration_cache()
    return holder


def _release_artifact_calibration(
    holder: tempfile.TemporaryDirectory | None, previous: str | None
) -> None:
    if holder is None:
        return
    if previous is None:
        os.environ.pop("FRESHDATA_MODEL_DIR", None)
    else:
        os.environ["FRESHDATA_MODEL_DIR"] = previous
    from freshdata.semantic import scoring  # noqa: PLC0415

    scoring.reset_calibration_cache()
    holder.cleanup()


def check_release_gates(tracks: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []

    def all_values(key: str) -> list[float]:
        return [t[key] for t in tracks.values() if t.get(key) is not None]

    for value in all_values("protected_column_violation_rate"):
        if value > metrics.FULL_GATE_PROTECTED_VIOLATION_RATE:
            failures.append(f"protected-column violation rate {value} > 0")
            break
    for value in all_values("false_modification_rate"):
        if value > metrics.FULL_GATE_FALSE_MODIFICATION_RATE:
            failures.append(f"false modification rate {value} > 0.1%")
            break
    t2 = tracks.get("T2", {})
    if t2.get("confidence_ece") is not None and t2["confidence_ece"] > metrics.FULL_GATE_ECE:
        failures.append(f"ECE {round(t2['confidence_ece'], 4)} > {metrics.FULL_GATE_ECE}")
    if (t2.get("precision_at_conf_95") is not None
            and t2["precision_at_conf_95"] < metrics.FULL_GATE_PRECISION_AT_95):
        failures.append(
            f"P@0.95 {round(t2['precision_at_conf_95'], 4)} < {metrics.FULL_GATE_PRECISION_AT_95}")
    t4 = tracks.get("T4", {})
    if t4 and not t4.get("profile_fmr_non_increase", True):
        failures.append("profile replay increased the false modification rate")
    if t4.get("privacy_leak_count", 0) > metrics.FULL_GATE_PRIVACY_LEAK_COUNT:
        failures.append(f"privacy leak count {t4['privacy_leak_count']} > 0")
    t5 = tracks.get("T5", {})
    slowdown = t5.get("runtime_slowdown_vs_baseline")
    if slowdown is not None and slowdown > metrics.FULL_GATE_RUNTIME_SLOWDOWN:
        failures.append(f"runtime slowdown {slowdown} > 20% vs baseline")
    overhead = t5.get("memory_overhead_vs_baseline")
    if overhead is not None and overhead > metrics.FULL_GATE_MEMORY_OVERHEAD:
        failures.append(f"memory overhead {overhead} > 15% vs baseline")
    return failures


# Runtime/memory gate messages. These depend on T5 wall-clock/RSS, which are only
# comparable to baseline_v1.json when T5 runs in isolation (as the perf-regression
# workflow does). The remaining gates are deterministic correctness checks.
_PERF_GATE_PREFIXES = ("runtime slowdown", "memory overhead")


def is_perf_gate_failure(failure: str) -> bool:
    """True for the timing-dependent runtime/memory gates (as opposed to the
    deterministic correctness gates)."""
    return failure.startswith(_PERF_GATE_PREFIXES)


def run_full(
    tracks: tuple[str, ...] = ALL_TRACKS,
    *,
    target_rows: int = 50_000,
    baseline_path: Path | str = BASELINE_PATH,
    update_baseline: bool = False,
    include_baselines: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    from . import reproducibility  # noqa: PLC0415 - avoid import cycle at module load

    previous_model_dir = os.environ.get("FRESHDATA_MODEL_DIR")
    holder = _install_artifact_calibration()
    baseline_path = Path(baseline_path)
    baseline = None
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    results: dict[str, dict[str, Any]] = {}
    try:
        for track in tracks:
            if track == "T5":
                results["T5"] = run_t5(target_rows=target_rows, baseline=baseline)
            else:
                results[track] = _TRACK_RUNNERS[track]()
    finally:
        _release_artifact_calibration(holder, previous_model_dir)

    if "T5" in results and (update_baseline or baseline is None):
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps({
            "seconds_default": results["T5"]["seconds_default"],
            "peak_rss_delta_bytes": results["T5"]["peak_rss_delta_bytes"],
            "rows": results["T5"]["rows"],
            "note": "v1.0-equivalent default-path baseline (model-free clean)",
        }, indent=2) + "\n", encoding="utf-8")

    failures = check_release_gates(results)
    perf_failures = [f for f in failures if is_perf_gate_failure(f)]
    correctness_failures = [f for f in failures if not is_perf_gate_failure(f)]
    payload: dict[str, Any] = {
        "tracks_run": list(tracks),
        "calibration_artifact_used": holder is not None,
        "tracks": results,
        "release_gates": {
            "failures": failures,
            "passed": not failures,
            "correctness_failures": correctness_failures,
            "perf_failures": perf_failures,
        },
        "environment": reproducibility.environment_info(),
        "dataset_hashes": reproducibility.dataset_hashes(),
        "metrics_version": reproducibility.METRICS_VERSION,
        "command": command or "python -m benchmarks.cleanbench",
    }
    if include_baselines:
        from .baselines import run_all as run_all_baselines  # noqa: PLC0415

        payload["baselines"] = run_all_baselines()
    return payload
