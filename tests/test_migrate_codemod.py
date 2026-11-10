from pathlib import Path

import pytest
from libcst.codemod import CodemodContext, transform_module

from freshdata.cli.migrate import FreshDataV3Command, main


def run_codemod(source: str) -> tuple[str, list[str]]:
    context = CodemodContext()
    command = FreshDataV3Command(context)
    result = transform_module(command, source)
    return result.code, context.warnings


def test_rewrites_io_wrappers_and_adds_pandas_import() -> None:
    before = """
import freshdata as fd

# Keep user comment intact
df = fd.clean_csv("sales.csv", impute="median", column_names=True)
excel_df = fd.clean_excel("report.xlsx", outliers="clip")
"""
    expected = """
import freshdata as fd
import pandas as pd

# Keep user comment intact
df = fd.clean(pd.read_csv("sales.csv"), default_impute="median", normalize_names=True)
excel_df = fd.clean(pd.read_excel("report.xlsx"), default_outliers="clip")
"""
    after, warnings = run_codemod(before)
    assert after.strip() == expected.strip()
    assert not warnings


def test_rewrites_compare_plans_and_simple_helpers() -> None:
    before = """
import freshdata as fd

diff = fd.compare_plans(plan_a, plan_b)
df1 = fd.fill_missing(raw_df, method="median")
df2 = fd.remove_outliers(raw_df)
df3 = fd.resolve_duplicates(raw_df, keep="first")
df4 = fd.pipeline_clean(raw_df, impute="mode")
"""
    expected = """
import freshdata as fd

diff = plan_a.diff(plan_b)
df1 = fd.clean(raw_df, default_impute="median")
df2 = fd.clean(raw_df, default_outliers="drop")
df3 = fd.clean(raw_df, drop_duplicates=True)
df4 = fd.clean(raw_df, default_impute="mode")
"""
    after, warnings = run_codemod(before)
    assert after.strip() == expected.strip()
    assert not warnings


def test_rewrites_direct_from_imports_and_config_kwargs() -> None:
    before = """
from freshdata import CleanConfig, clean_csv, fill_missing

cfg = CleanConfig(impute="median", outliers="clip", column_names=False)
df = fill_missing(raw_df, method="mean")
"""
    expected = """
from freshdata import CleanConfig, clean

cfg = CleanConfig(default_impute="median", default_outliers="clip", normalize_names=False)
df = clean(raw_df, default_impute="mean")
"""
    after, warnings = run_codemod(before)
    assert after.strip() == expected.strip()
    assert not warnings


def test_warns_on_unmigratable_nl_and_enterprise_calls() -> None:
    before = """
import freshdata as fd

policy = fd.compile_context("CustomerID is unique", df=df)
score = fd.compute_trust_score(df)
"""
    after, warnings = run_codemod(before)
    assert after.strip() == before.strip()
    assert len(warnings) == 2
    assert "compile_context" in warnings[0]
    assert "compute_trust_score" in warnings[1]


def test_cli_check_and_diff(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "sample.py"
    test_file.write_text('import freshdata as fd\ndf = fd.remove_outliers(data)\n', encoding="utf-8")

    # --check with changes needed returns 1
    code = main([str(test_file), "--check", "--diff"])
    assert code == 1
    captured = capsys.readouterr()
    assert "default_outliers" in captured.out
    assert test_file.read_text(encoding="utf-8") == 'import freshdata as fd\ndf = fd.remove_outliers(data)\n'

    # In-place migration
    code = main([str(test_file)])
    assert code == 0
    assert "default_outliers" in test_file.read_text(encoding="utf-8")

    # Second --check on migrated file returns 0
    code = main([str(test_file), "--check"])
    assert code == 0


def test_cli_directory_and_warnings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    f1 = sub_dir / "a.py"
    f1.write_text('import freshdata as fd\nx = fd.compute_trust_score(df)\n', encoding="utf-8")

    code = main([str(tmp_path)])
    assert code == 0
    captured = capsys.readouterr()
    assert "compute_trust_score" in captured.err
