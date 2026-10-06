"""Interactive Streamlit data-cleaning recipe.

Install dependencies with::

    pip install streamlit freshdata-cleaner

Run the app with::

    streamlit run examples/integrations/streamlit_app.py
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st

import freshdata as fd


def demo_data() -> pd.DataFrame:
    """Return a small messy dataset for trying the app without an upload."""
    return pd.DataFrame(
        {
            "Customer Name": ["  Ava Chen ", "Noah Patel", "Mia Lopez", "Noah Patel"],
            "Orders": ["3", "N/A", "5", "N/A"],
            "Region": ["West", "East", "West", "East"],
        }
    )


def load_uploaded_csv(contents: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes into a DataFrame."""
    return pd.read_csv(BytesIO(contents))


def count_modified_columns(report: Any) -> int:
    """Count columns modified by applied cleaning actions, imputations, or drops."""
    modified = set(report.columns_imputed) | set(report.columns_dropped)
    modified.update(
        action.column
        for action in report.actions
        if action.column is not None
        and action.count > 0
        and getattr(action, "status", "automatic") in {"automatic", "approved"}
    )
    return len(modified)


def main() -> None:
    st.set_page_config(page_title="FreshData Cleaning Lab", layout="wide")
    st.title("FreshData Cleaning Lab")
    st.caption(
        "Upload a CSV to compare its original data with FreshData's cleaned output."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is None:
        raw_df = demo_data()
        st.info("Showing sample data. Upload a CSV to clean your own dataset.")
    else:
        try:
            raw_df = load_uploaded_csv(uploaded_file.getvalue())
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            st.error(f"Could not read this CSV: {exc}")
            st.stop()

    st.sidebar.header("Cleaning Settings")
    strategy = st.sidebar.selectbox(
        "Strategy", ["balanced", "conservative", "aggressive"], index=0
    )
    drop_duplicates = st.sidebar.checkbox("Drop duplicate rows", value=True)

    try:
        cleaned_df, report = fd.clean(
            raw_df,
            strategy=strategy,
            drop_duplicates=drop_duplicates,
            return_report=True,
        )
    except Exception as exc:
        st.error(f"FreshData could not clean this dataset: {exc}")
        st.stop()

    st.subheader("Cleaning summary")
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Rows removed", max(report.rows_before - report.rows_after, 0)
    )
    metric_columns[1].metric("Columns modified", count_modified_columns(report))
    imputed_values = sum(
        action.count
        for action in report.actions
        if (
            action.step == "impute"
            or (
                action.step == "missing"
                and action.column in report.columns_imputed
            )
        )
        and getattr(action, "status", "automatic") in {"automatic", "approved"}
        and action.count > 0
    )
    metric_columns[2].metric("Missing values imputed", imputed_values)
    metric_columns[3].metric("Duplicates dropped", report.duplicates_removed)

    raw_column, cleaned_column = st.columns(2)
    with raw_column:
        st.subheader("Raw data")
        st.dataframe(raw_df, use_container_width=True)
    with cleaned_column:
        st.subheader("Cleaned data")
        st.dataframe(cleaned_df, use_container_width=True)

    st.download_button(
        "Download cleaned CSV",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    with st.expander("Quality report"):
        st.code(report.summary(), language="text")
        if report.actions:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Step": action.step,
                            "Column": action.column or "",
                            "Change": action.description,
                            "Count": action.count,
                        }
                        for action in report.actions
                    ]
                ),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
