import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

# Repo-only top-level directories that some suites import. The sdist ships
# ``tests/`` but not these, so suites that depend on a missing directory are
# left out of collection instead of failing with ModuleNotFoundError.
#
# Each entry maps a directory to (test directories that need it, top-level
# modules that come from it). ``expectations`` and ``golden_util`` are
# ``tests/`` helpers that import ``dataset_loader`` / ``fixture_download``
# from ``scripts/``.
_REPO_ONLY_DEPENDENCIES = {
    "benchmarks": (
        ("benchmark", "performance", "truthbench"),
        ("benchmarks", "bench", "harness_metrics", "public_benchmark", "results_schema"),
    ),
    "training": (
        ("training",),
        ("training",),
    ),
    "scripts": (
        (),
        (
            "dataset_loader",
            "fixture_download",
            "fetch_online_fixtures",
            "expectations",
            "golden_util",
        ),
    ),
}


def _missing_repo_only_dependencies():
    dirs = []
    modules = []
    for name, (test_dirs, mods) in _REPO_ONLY_DEPENDENCIES.items():
        if not (ROOT / name).is_dir():
            dirs.extend(TESTS / d for d in test_dirs)
            modules.extend(mods)
    pattern = None
    if modules:
        pattern = re.compile(
            r"^\s*(from|import)\s+(" + "|".join(map(re.escape, modules)) + r")\b",
            re.MULTILINE,
        )
    return tuple(dirs), pattern


_MISSING_TEST_DIRS, _MISSING_IMPORT_RE = _missing_repo_only_dependencies()


def pytest_ignore_collect(collection_path, config):
    """Skip suites whose repo-only dependencies are absent (e.g. from the sdist)."""
    if not _MISSING_TEST_DIRS and _MISSING_IMPORT_RE is None:
        return None
    path = Path(collection_path)
    for test_dir in _MISSING_TEST_DIRS:
        if path == test_dir or test_dir in path.parents:
            return True
    if (
        _MISSING_IMPORT_RE is not None
        and path.suffix == ".py"
        and path.name.startswith("test_")
        and path.is_file()
    ):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        if _MISSING_IMPORT_RE.search(source):
            return True
    return None


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite golden report snapshots in tests/fixtures/golden/ and online/golden/",
    )
    parser.addoption(
        "--require-golden-diff",
        action="store_true",
        default=False,
        help="Fail golden updates unless a machine-readable diff summary is written.",
    )


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")


@pytest.fixture
def require_golden_diff(request):
    return request.config.getoption("--require-golden-diff")


@pytest.fixture
def messy() -> pd.DataFrame:
    """A kitchen-sink frame exercising every default cleaning step."""
    return pd.DataFrame(
        {
            " First Name ": [" alice ", "Bob", "N/A", "Bob", None],
            "AGE": ["25", "30", "-", "30", "40"],
            "Joined Date": ["2021-01-05", "2021-02-11", "", "2021-02-11", "2021-03-09"],
            "Active": ["yes", "no", "no", "no", "YES"],
            "Salary($)": ["$1,200.50", "$2,000.00", "?", "$2,000.00", "$3,500.75"],
            "empty": [None] * 5,
        }
    )


@pytest.fixture
def already_clean() -> pd.DataFrame:
    """A frame on which default cleaning should be a no-op."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.5, 2.5, 3.5],
            "c": ["x", "y", "z"],
        }
    )
