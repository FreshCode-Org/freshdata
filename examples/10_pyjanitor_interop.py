"""Use FreshData and PyJanitor in the same pandas workflow.

PyJanitor is optional; install it alongside FreshData to run this example:

    pip install freshdata-cleaner "pyjanitor<0.32"

FreshData 2.0 supports pandas 1.5–2.x; PyJanitor 0.31 is the compatible line
verified for this example.

Use PyJanitor for explicit DataFrame reshaping and method-style transforms.
Use FreshData for evidence-based quality detection, conservative repair, and
an audit report. Either tool can go first, depending on which step needs the
other tool's output.
"""

import pandas as pd
from janitor import clean_names, transform_column  # type: ignore[import-untyped]

import freshdata as fd


def build_frame() -> pd.DataFrame:
    """Return one small frame used by both ordering examples."""
    return pd.DataFrame(
        {
            " Customer ID ": ["C-01", "C-02", "C-03", "C-03"],
            "Order Amount": ["12.50", "n/a", "18.00", "18.00"],
            "Region Name": [" North ", "south", "NORTH", "NORTH"],
        }
    )


def pyjanitor_then_freshdata(raw: pd.DataFrame) -> tuple[pd.DataFrame, fd.CleanReport]:
    """Shape labels explicitly, then detect and repair quality issues."""
    shaped = clean_names(raw, remove_special=True, strip_underscores="both")
    return fd.clean(
        shaped,
        id_columns=("customer_id",),
        return_report=True,
    )


def freshdata_then_pyjanitor(raw: pd.DataFrame) -> tuple[pd.DataFrame, fd.CleanReport]:
    """Repair with an audit trail, then add an explicit presentation column."""
    cleaned, report = fd.clean(
        raw,
        id_columns=(" Customer ID ",),
        return_report=True,
    )
    enriched = transform_column(
        cleaned,
        column_name="region_name",
        function=lambda value: value.strip().title(),
        dest_column_name="region_label",
    )
    return enriched, report


def main() -> None:
    raw = build_frame()

    cleaned, clean_report = pyjanitor_then_freshdata(raw)
    print("=== PyJanitor then FreshData ===")
    print(cleaned)
    print(clean_report.summary())

    enriched, enriched_report = freshdata_then_pyjanitor(raw)
    print("\n=== FreshData then PyJanitor ===")
    print(enriched)
    print(enriched_report.summary())


if __name__ == "__main__":
    main()
