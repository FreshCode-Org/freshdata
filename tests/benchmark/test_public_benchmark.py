"""Public benchmark report (#4): rendering, harness orchestration, workflow contract."""

from __future__ import annotations

import json
from pathlib import Path

import public_benchmark as pb
import pytest
import yaml

import freshdata

ENV = {
    "generated_at": "2026-09-15T00:00:00+00:00",
    "freshdata_version": "9.9.9",
    "git_sha": "abcdef1234567890",
    "python": "3.12.0",
    "platform": "Linux-6.8-x86_64",
    "cpu_count": 4,
    "ram_gb": 15.6,
    "packages": {"pandas": "2.3.3", "numpy": "2.5.1"},
}

SCALING = [
    {"benchmark": "csv_ingest", "strategy": "balanced", "rows": 500_000,
     "csv_mb": 48.2, "wall_sec": 3.21, "peak_rss_mb": 410.0},
    {"benchmark": "csv_ingest", "strategy": "aggressive", "rows": 500_000,
     "csv_mb": 48.2, "wall_sec": 3.9, "peak_rss_mb": 455.0},
    {"benchmark": "profile", "strategy": "n/a", "rows": 1_000_000,
     "wall_sec": 1.5, "peak_rss_mb": 120.0},
    {"benchmark": "import_time", "strategy": "n/a", "rows": 0,
     "import_sec_median": 0.412, "import_sec_min": 0.39},
]

FIXTURE_MD = (
    "# FreshData benchmark report — `run1`\n\n"
    "- freshdata: `9.9.9`\n\n"
    "| fixture | n_rows |\n|---|--:|\n| crm | 10,000 |\n"
)


def test_markdown_carries_version_environment_and_both_tables():
    md = pb.render_markdown(ENV, SCALING, FIXTURE_MD)
    assert md.startswith("# FreshData public benchmark — freshdata 9.9.9")
    assert "(commit `abcdef123456`)" in md
    assert "4 CPUs, 15.6 GB RAM" in md
    assert "pandas 2.3.3, numpy 2.5.1" in md
    assert (
        "| CSV ingest + clean | 48.2 MB | 3.21 s, 410 MB peak RSS | 3.90 s, 455 MB peak RSS |"
    ) in md
    assert "| Mixed-schema profile | 1,000,000 rows | 1.50 s, 120 MB peak RSS | n/a |" in md
    assert "| Import time (`import freshdata`) |  | 0.412 s median (0.390 s min) | n/a |" in md
    assert "## Fixture benchmarks" in md
    assert "| crm | 10,000 |" in md
    assert "# FreshData benchmark report" not in md  # nested report title dropped


def test_scaling_table_skips_cases_that_did_not_run():
    table = "\n".join(pb.scaling_table(SCALING))
    assert "Null-fill" not in table
    assert "Peak memory" not in table


def _fake_harness(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(pb, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(pb, "SCALING_JSON", results_dir / "report_bench.json")
    monkeypatch.setattr(pb, "collect_environment", lambda: ENV)
    calls: list[list[str]] = []

    def runner(cmd):
        args = cmd[1:]
        calls.append(args)
        if args[0] == "benchmarks/bench_report.py":
            pb.SCALING_JSON.write_text(json.dumps({"results": SCALING}))
        elif args[:2] == ["benchmarks/bench.py", "run"]:
            (results_dir / "20260915T000000").mkdir()
        elif args[:2] == ["benchmarks/bench.py", "report"]:
            (Path(args[-1]) / "report.md").write_text(FIXTURE_MD)

    return runner, calls


def test_main_runs_both_harnesses_and_writes_report(monkeypatch, tmp_path):
    runner, calls = _fake_harness(monkeypatch, tmp_path)
    out = tmp_path / "public"
    assert pb.main(["--repeat", "1", "--output-dir", str(out)], runner=runner) == 0
    assert calls[0] == ["benchmarks/bench_report.py", "all", "--mb", "50", "--rows", "1000000"]
    assert calls[1] == ["benchmarks/bench.py", "run", "--repeat", "1"]
    assert calls[2][:2] == ["benchmarks/bench.py", "report"]
    md = (out / "public-benchmark.md").read_text()
    assert "| CSV ingest + clean |" in md and "| crm | 10,000 |" in md
    payload = json.loads((out / "public-benchmark.json").read_text())
    assert payload["environment"]["freshdata_version"] == "9.9.9"
    assert payload["options"] == {"scale": "ci", "csv_mb": 50, "rows": 1_000_000,
                                  "repeat": 1, "size": None}


def test_full_scale_uses_harness_default_rows(monkeypatch, tmp_path):
    runner, calls = _fake_harness(monkeypatch, tmp_path)
    pb.main(["--scale", "full", "--skip-fixtures", "--output-dir", str(tmp_path / "o")],
            runner=runner)
    assert calls == [["benchmarks/bench_report.py", "all", "--mb", "100"]]


def test_skip_scaling_runs_only_the_fixture_harness(monkeypatch, tmp_path):
    runner, calls = _fake_harness(monkeypatch, tmp_path)
    out = tmp_path / "public"
    pb.main(["--skip-scaling", "--repeat", "1", "--output-dir", str(out)], runner=runner)
    assert all(call[0] == "benchmarks/bench.py" for call in calls)
    assert "scaling benchmarks" not in (out / "public-benchmark.md").read_text()


def test_skipping_everything_is_an_error():
    with pytest.raises(SystemExit):
        pb.main(["--skip-scaling", "--skip-fixtures"], runner=lambda cmd: None)


def test_environment_reports_the_installed_freshdata_version():
    env = pb.collect_environment()
    assert env["freshdata_version"] == freshdata.__version__
    assert env["cpu_count"]


def test_workflow_is_manual_only_and_uploads_the_report():
    path = Path(pb.REPO_ROOT) / ".github" / "workflows" / "public-benchmark.yml"
    text = path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    # PyYAML 1.1 treats the unquoted GitHub Actions key `on` as a boolean.
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    steps = workflow["jobs"]["public-benchmark"]["steps"]
    assert any("upload-artifact" in step.get("uses", "") for step in steps)
    assert "benchmarks/public_benchmark.py" in text
