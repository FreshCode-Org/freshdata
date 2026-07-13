"""Reproducibility tooling for public CleanBench: environment disclosure,
dataset/task hashing, committed-result verification, and the README
trust-claim audit.

Two entry points:

    python -m benchmarks.cleanbench --verify-results benchmarks/cleanbench/results/latest.json
    python -m benchmarks.cleanbench.reproducibility audit-readme

Both return a non-zero exit code on failure and print every failure reason —
never a bare "failed" with no explanation.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"

#: Metric-definition version: bump when a metric's *meaning* changes (not
#: merely its value), so old committed results stay interpretable.
METRICS_VERSION = "cleanbench-metrics-v1"


# --------------------------------------------------------------------------- #
# environment + hashing
# --------------------------------------------------------------------------- #


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def environment_info() -> dict[str, Any]:
    """Everything a third party needs to judge whether their run should match:
    FreshData version + git commit, Python/OS, and optional-engine versions."""
    import freshdata  # noqa: PLC0415

    return {
        "freshdata_version": freshdata.__version__,
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "polars_version": _optional_version("polars"),
        "duckdb_version": _optional_version("duckdb"),
        "pandas_version": _optional_version("pandas"),
        "metrics_version": METRICS_VERSION,
    }


def frame_hash(*frames: Any) -> str:
    """Stable sha256 over one or more frames' canonical CSV bytes.

    Used to fingerprint a track's ``(truth, corrupted)`` fixture pair so
    :func:`verify_results` can detect fixture drift between the committed
    result and a fresh reproduction — the fixtures are seeded and generated in
    code, so this hash is expected to be byte-identical across machines and
    freshdata versions that have not changed the fixture generators.
    """
    digest = hashlib.sha256()
    for frame in frames:
        if frame is None:
            digest.update(b"\0")
            continue
        digest.update(frame.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def dataset_hashes() -> dict[str, str]:
    """``{track: sha256}`` over every track's fixture, for the committed result."""
    from . import fixtures  # noqa: PLC0415

    truth1, corrupted1, _ = fixtures.make_t1_representation_fixture()
    truth2, corrupted2, _ = fixtures.make_t2_semantic_fixture()
    truth3, corrupted3, _ = fixtures.make_t3_context_fixture()
    (pair_messy, pair_clean, batch_truth, batch_corrupted, *_rest) = (
        fixtures.make_t4_profile_fixture()
    )
    return {
        "T1": frame_hash(truth1, corrupted1),
        "T2": frame_hash(truth2, corrupted2),
        "T3": frame_hash(truth3, corrupted3),
        "T4": frame_hash(pair_messy, pair_clean, batch_truth, batch_corrupted),
        # T5 is generator-parameterized (no fixed frame); its "hash" is the
        # generator identity + row count, recorded in meta.json / task dirs.
        "T5": "generator:make_t5_scale_fixture",
    }


# --------------------------------------------------------------------------- #
# verify-results
# --------------------------------------------------------------------------- #

_REQUIRED_TOP_LEVEL = (
    "tracks_run", "tracks", "release_gates", "environment", "command",
    "dataset_hashes", "metrics_version",
)


def verify_results(path: Path | str) -> list[str]:
    """Verify a committed CleanBench result JSON. Returns failure strings
    (empty list = pass). Checks:

    - the file parses and has every required top-level key (schema validity);
    - dataset hashes match a fresh (re-generated) fixture hash (task/dataset
      hashes are still correct for the installed freshdata version);
    - environment metadata is present;
    - the recorded release-gate verdict matches what the recorded metrics
      actually imply (catches a hand-edited or stale ``passed`` flag);
    - every README trust claim in :data:`CLAIM_REGISTRY` still resolves.
    """
    failures: list[str] = []
    path = Path(path)
    if not path.is_file():
        return [f"{path} does not exist"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    for key in _REQUIRED_TOP_LEVEL:
        if key not in payload:
            failures.append(f"missing required top-level key: {key!r}")

    if "dataset_hashes" in payload:
        fresh = dataset_hashes()
        committed = payload["dataset_hashes"]
        for track, expected in fresh.items():
            got = committed.get(track)
            if got != expected:
                failures.append(
                    f"dataset hash mismatch for {track}: committed={got!r} "
                    f"fresh={expected!r} (fixture drifted or freshdata version differs)"
                )

    if "environment" in payload:
        env = payload["environment"]
        for key in ("freshdata_version", "git_commit", "python_version"):
            if not env.get(key):
                failures.append(f"environment metadata missing/empty: {key!r}")

    gates = payload.get("release_gates", {})
    if isinstance(gates, dict):
        recorded_passed = gates.get("passed")
        recorded_failures = gates.get("failures", [])
        if recorded_passed is not None and recorded_passed != (not recorded_failures):
            failures.append(
                "release_gates.passed is inconsistent with release_gates.failures"
            )

    failures.extend(f"README claim audit: {f}" for f in audit_readme())
    return failures


# --------------------------------------------------------------------------- #
# README trust-claim audit
# --------------------------------------------------------------------------- #


class Claim:
    """One README trust claim: exact text that must appear in README.md, plus
    the artifact(s) that back it."""

    __slots__ = ("readme_text", "backing")

    def __init__(self, readme_text: str, backing: tuple[str, ...]):
        self.readme_text = readme_text
        self.backing = backing


def _test_backing_ok(spec: str) -> str | None:
    """``"tests/foo.py::test_bar"`` -> None if the file exists and defines
    ``test_bar``, else a failure string."""
    file_part, _, func_part = spec.partition("::")
    path = REPO_ROOT / file_part
    if not path.is_file():
        return f"backing test file missing: {file_part}"
    if func_part and f"def {func_part}" not in path.read_text(encoding="utf-8"):
        return f"backing test function {func_part!r} not found in {file_part}"
    return None


def _doc_backing_ok(spec: str) -> str | None:
    """``"docs/limitations.md#anchor"`` -> None if the file exists (and, when
    an anchor is given, a matching heading is present)."""
    file_part, _, anchor = spec.partition("#")
    path = REPO_ROOT / file_part
    if not path.is_file():
        return f"backing doc missing: {file_part}"
    if anchor:
        slug_heading = anchor.replace("-", " ").strip().casefold()
        text = path.read_text(encoding="utf-8").casefold()
        if slug_heading not in text:
            return f"backing doc {file_part} has no heading matching #{anchor}"
    return None


def _benchmark_backing_ok(spec: str) -> str | None:
    """``"benchmark:T2.confidence_ece"`` -> None if the committed
    ``results/latest.json`` has that track/metric key at all (existence, not
    value — the *value* is judged by the release gate, not the claim audit)."""
    _prefix, _, dotted = spec.partition(":")
    track, _, metric = dotted.partition(".")
    latest = RESULTS_DIR / "latest.json"
    if not latest.is_file():
        return f"no committed benchmark result at {latest.relative_to(REPO_ROOT)}"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    if metric not in payload.get("tracks", {}).get(track, {}):
        return f"benchmark result has no {track}.{metric}"
    return None


RESULTS_DIR = REPO_ROOT / "benchmarks" / "cleanbench" / "results"

#: The curated set of README trust claims this audit enforces. Scope note:
#: this is a maintained allow-list, not an NLP claim-extractor — free-text
#: marketing copy is not scanned. Each entry's ``readme_text`` must appear
#: verbatim in README.md (catches wording drift away from what's backed) and
#: every ``backing`` target must exist (catches a claim outliving its proof).
# NOTE: the README was condensed to a short overview (PR #100), so it no longer
# advertises the granular trust claims this audit used to police (no-LLM,
# offline determinism, off-by-default, per-metric ECE/precision, model weights in
# the wheel, ...). The audit mirrors what the README *actually claims*, so those
# entries were removed here — the underlying guarantees remain covered by their
# tests, the release gate (verify-results checks the benchmark metrics exist),
# and docs/limitations.md. Only the safety claims the condensed README still
# makes verbatim are policed below; re-add entries here if a claim returns to the
# README.
CLAIM_REGISTRY: tuple[Claim, ...] = (
    Claim(
        "never imputes an identifier, modifies a target column",
        (
            "tests/test_semantic_cleaning.py::test_id_columns_protected",
            "benchmark:T2.protected_column_violation_rate",
        ),
    ),
    Claim(
        "nothing happens silently",
        ("tests/test_semantic_cleaning.py::test_assist_records_without_mutating",),
    ),
    Claim(
        "raw PII never enters the copilot's model context",
        (
            "tests/test_experimental_ai_copilot.py::test_no_raw_pii_anywhere_in_report",
            "tests/test_experimental_ai_copilot.py::test_provider_hook_receives_only_masked_context",
            "tests/test_experimental_ai_copilot.py::test_undeclared_stringlike_columns_are_masked_in_model_context",
            "tests/test_experimental_ai_copilot.py::test_category_noise_previews_never_reach_model_context",
        ),
    ),
)


def audit_readme() -> list[str]:
    """Verify every :data:`CLAIM_REGISTRY` entry. Empty list = pass."""
    failures: list[str] = []
    if not README_PATH.is_file():
        return ["README.md not found"]
    text = README_PATH.read_text(encoding="utf-8")
    for claim in CLAIM_REGISTRY:
        if claim.readme_text not in text:
            failures.append(
                f"claim text no longer in README.md: {claim.readme_text!r} "
                "(reworded without updating the audit registry?)"
            )
            continue
        for spec in claim.backing:
            if spec.startswith("benchmark:"):
                problem = _benchmark_backing_ok(spec)
            elif spec.startswith("docs/"):
                problem = _doc_backing_ok(spec)
            else:
                problem = _test_backing_ok(spec)
            if problem:
                failures.append(f"claim {claim.readme_text!r}: {problem}")
    return failures


def _main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["audit-readme"]:
        failures = audit_readme()
    elif len(argv) == 2 and argv[0] == "verify-results":
        failures = verify_results(argv[1])
    else:
        print(
            "usage: python -m benchmarks.cleanbench.reproducibility "
            "{audit-readme | verify-results <path>}",
            file=sys.stderr,
        )
        return 2
    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
