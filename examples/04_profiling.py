"""Read-only data profiling and EDA with freshdata.

`fd.profile` inspects data quality without changing anything, using the same
inference engine as `fd.clean`. Run:

    python examples/04_profiling.py
"""

import pandas as pd

import freshdata as fd


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "First Name": [" Ann ", "Bob", "Bob", "Cara", None],
            "AGE": ["34", "N/A", "N/A", "41", "29"],
            "Joined Date": ["2021-01-01", "2021-02-15", "2021-02-15", "-", "2021-03-30"],
            "Active": ["yes", "no", "no", "yes", "no"],
            "Salary($)": ["$1,200.50", "-", "-", "$2,000", "$1,750"],
            "empty": [None, None, None, None, None],
        }
    )


def main() -> None:
    df = make_data()

    # Human-readable profile
    print(fd.profile(df))

    # With a dry-run plan attached
    profile = fd.profile(df, include_plan=True)
    print("\nPlanned primary model per column:")
    print(profile.plan.summary())

    # Inferred roles
    print("\nInferred roles:")
    print(fd.infer_roles(df))


if __name__ == "__main__":
    main()
