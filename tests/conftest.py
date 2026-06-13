import pandas as pd
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite golden report snapshots in tests/fixtures/golden/ and online/golden/",
    )


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")


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
