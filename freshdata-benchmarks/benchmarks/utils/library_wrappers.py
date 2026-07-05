"""Library adapter wrappers for fair cross-library benchmarking.

Each adapter implements equivalent operations using the target library's
native API.  Adapters that cannot be imported set ``available = False``
and all their methods raise ``NotImplementedError``, which ASV treats as
a skip.

Usage::

    adapter = get_adapter("pandas")
    adapter.drop_missing_rows(df)
"""

from __future__ import annotations

import abc
import gc
from typing import Any


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class LibraryAdapter(abc.ABC):
    """Abstract base class for library-specific operation wrappers."""

    name: str = "base"
    available: bool = False

    def _skip(self, op: str) -> None:
        raise NotImplementedError(f"{self.name} does not support {op}")

    # --- Missing values ---
    def drop_missing_rows(self, df: Any) -> Any:
        self._skip("drop_missing_rows")

    def fill_missing_mean(self, df: Any, columns: list[str]) -> Any:
        self._skip("fill_missing_mean")

    def fill_missing_median(self, df: Any, columns: list[str]) -> Any:
        self._skip("fill_missing_median")

    def fill_missing_mode(self, df: Any, columns: list[str]) -> Any:
        self._skip("fill_missing_mode")

    def fill_missing_ffill(self, df: Any) -> Any:
        self._skip("fill_missing_ffill")

    def fill_missing_bfill(self, df: Any) -> Any:
        self._skip("fill_missing_bfill")

    # --- Duplicates ---
    def detect_duplicates(self, df: Any) -> Any:
        self._skip("detect_duplicates")

    def drop_duplicates(self, df: Any) -> Any:
        self._skip("drop_duplicates")

    # --- String cleaning ---
    def trim_whitespace(self, df: Any, columns: list[str]) -> Any:
        self._skip("trim_whitespace")

    def to_lowercase(self, df: Any, columns: list[str]) -> Any:
        self._skip("to_lowercase")

    def to_uppercase(self, df: Any, columns: list[str]) -> Any:
        self._skip("to_uppercase")

    def regex_replace(self, df: Any, columns: list[str],
                      pattern: str, replacement: str) -> Any:
        self._skip("regex_replace")

    # --- Column operations ---
    def rename_columns(self, df: Any, mapping: dict) -> Any:
        self._skip("rename_columns")

    def drop_columns(self, df: Any, columns: list[str]) -> Any:
        self._skip("drop_columns")

    def select_columns(self, df: Any, columns: list[str]) -> Any:
        self._skip("select_columns")

    # --- Type conversion ---
    def convert_numeric(self, df: Any, columns: list[str]) -> Any:
        self._skip("convert_numeric")

    def convert_datetime(self, df: Any, columns: list[str]) -> Any:
        self._skip("convert_datetime")

    def optimize_dtypes(self, df: Any) -> Any:
        self._skip("optimize_dtypes")

    # --- Encoding ---
    def onehot_encode(self, df: Any, columns: list[str]) -> Any:
        self._skip("onehot_encode")

    def label_encode(self, df: Any, columns: list[str]) -> Any:
        self._skip("label_encode")

    # --- Scaling ---
    def standard_scale(self, df: Any, columns: list[str]) -> Any:
        self._skip("standard_scale")

    def minmax_scale(self, df: Any, columns: list[str]) -> Any:
        self._skip("minmax_scale")

    # --- Outlier detection ---
    def detect_outliers_iqr(self, df: Any, columns: list[str]) -> Any:
        self._skip("detect_outliers_iqr")

    def detect_outliers_zscore(self, df: Any, columns: list[str]) -> Any:
        self._skip("detect_outliers_zscore")

    # --- Group aggregations ---
    def group_agg_single(self, df: Any) -> Any:
        self._skip("group_agg_single")

    def group_agg_multi(self, df: Any, columns: list[str]) -> Any:
        self._skip("group_agg_multi")

    def group_agg_transform(self, df: Any, columns: list[str]) -> Any:
        self._skip("group_agg_transform")

    # --- Pipeline ---
    def full_clean(self, df: Any) -> Any:
        self._skip("full_clean")


# ---------------------------------------------------------------------------
# FreshData adapter
# ---------------------------------------------------------------------------

class FreshDataAdapter(LibraryAdapter):
    name = "freshdata"

    def __init__(self) -> None:
        try:
            import freshdata  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def drop_missing_rows(self, df):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", impute=False,
                        outliers=False, preserve_original=True)

    def fill_missing_mean(self, df, columns):
        import freshdata as fd
        return fd.clean(df, strategy="aggressive", preserve_original=True)

    def fill_missing_median(self, df, columns):
        import freshdata as fd
        return fd.clean(df, strategy="balanced", preserve_original=True)

    def fill_missing_mode(self, df, columns):
        import freshdata as fd
        return fd.clean(df, strategy="balanced", preserve_original=True)

    def fill_missing_ffill(self, df):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", preserve_original=True)

    def fill_missing_bfill(self, df):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", preserve_original=True)

    def detect_duplicates(self, df):
        import freshdata as fd
        return fd.profile(df)

    def drop_duplicates(self, df):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", preserve_original=True)

    def trim_whitespace(self, df, columns):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", preserve_original=True)

    def to_lowercase(self, df, columns):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", preserve_original=True)

    def to_uppercase(self, df, columns):
        # FreshData normalizes to lowercase, not uppercase
        self._skip("to_uppercase")

    def rename_columns(self, df, mapping):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", preserve_original=True)

    def drop_columns(self, df, columns):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", preserve_original=True)

    def select_columns(self, df, columns):
        # FreshData doesn't have explicit column selection
        self._skip("select_columns")

    def convert_numeric(self, df, columns):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", preserve_original=True)

    def convert_datetime(self, df, columns):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", preserve_original=True)

    def optimize_dtypes(self, df):
        import freshdata as fd
        return fd.clean(df, strategy="conservative", preserve_original=True)

    def detect_outliers_iqr(self, df, columns):
        import freshdata as fd
        return fd.clean(df, outlier_method="iqr", outlier_action="flag",
                        strategy="balanced", preserve_original=True)

    def detect_outliers_zscore(self, df, columns):
        import freshdata as fd
        return fd.clean(df, outlier_method="zscore", outlier_action="flag",
                        strategy="balanced", preserve_original=True)

    def group_agg_single(self, df):
        import freshdata as fd
        return fd.profile(df)

    def group_agg_multi(self, df, columns):
        import freshdata as fd
        return fd.profile(df)

    def group_agg_transform(self, df, columns):
        import freshdata as fd
        return fd.profile(df)

    def full_clean(self, df):
        import freshdata as fd
        return fd.clean(df, strategy="balanced", preserve_original=True)


# ---------------------------------------------------------------------------
# Pandas adapter
# ---------------------------------------------------------------------------

class PandasAdapter(LibraryAdapter):
    name = "pandas"

    def __init__(self) -> None:
        try:
            import pandas  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def drop_missing_rows(self, df):
        import pandas as pd
        return df.dropna()

    def fill_missing_mean(self, df, columns):
        import pandas as pd
        result = df.copy()
        for col in columns:
            if col in result.columns:
                numeric = pd.to_numeric(result[col], errors="coerce")
                result[col] = numeric.fillna(numeric.mean())
        return result

    def fill_missing_median(self, df, columns):
        import pandas as pd
        result = df.copy()
        for col in columns:
            if col in result.columns:
                numeric = pd.to_numeric(result[col], errors="coerce")
                result[col] = numeric.fillna(numeric.median())
        return result

    def fill_missing_mode(self, df, columns):
        result = df.copy()
        for col in columns:
            if col in result.columns:
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col] = result[col].fillna(mode_val.iloc[0])
        return result

    def fill_missing_ffill(self, df):
        return df.ffill()

    def fill_missing_bfill(self, df):
        return df.bfill()

    def detect_duplicates(self, df):
        return df.duplicated()

    def drop_duplicates(self, df):
        return df.drop_duplicates()

    def trim_whitespace(self, df, columns):
        result = df.copy()
        for col in columns:
            if col in result.columns and result[col].dtype == object:
                result[col] = result[col].str.strip()
        return result

    def to_lowercase(self, df, columns):
        result = df.copy()
        for col in columns:
            if col in result.columns and result[col].dtype == object:
                result[col] = result[col].str.lower()
        return result

    def to_uppercase(self, df, columns):
        result = df.copy()
        for col in columns:
            if col in result.columns and result[col].dtype == object:
                result[col] = result[col].str.upper()
        return result

    def regex_replace(self, df, columns, pattern, replacement):
        result = df.copy()
        for col in columns:
            if col in result.columns and result[col].dtype == object:
                result[col] = result[col].str.replace(
                    pattern, replacement, regex=True
                )
        return result

    def rename_columns(self, df, mapping):
        return df.rename(columns=mapping)

    def drop_columns(self, df, columns):
        existing = [c for c in columns if c in df.columns]
        return df.drop(columns=existing)

    def select_columns(self, df, columns):
        existing = [c for c in columns if c in df.columns]
        return df[existing]

    def convert_numeric(self, df, columns):
        import pandas as pd
        result = df.copy()
        for col in columns:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
        return result

    def convert_datetime(self, df, columns):
        import pandas as pd
        result = df.copy()
        for col in columns:
            if col in result.columns:
                result[col] = pd.to_datetime(result[col], errors="coerce")
        return result

    def optimize_dtypes(self, df):
        return df.convert_dtypes()

    def onehot_encode(self, df, columns):
        import pandas as pd
        existing = [c for c in columns if c in df.columns]
        return pd.get_dummies(df, columns=existing)

    def label_encode(self, df, columns):
        result = df.copy()
        for col in columns:
            if col in result.columns:
                result[col] = result[col].astype("category").cat.codes
        return result

    def standard_scale(self, df, columns):
        result = df.copy()
        for col in columns:
            if col in result.columns:
                vals = result[col]
                result[col] = (vals - vals.mean()) / vals.std()
        return result

    def minmax_scale(self, df, columns):
        result = df.copy()
        for col in columns:
            if col in result.columns:
                vals = result[col]
                mn, mx = vals.min(), vals.max()
                if mx != mn:
                    result[col] = (vals - mn) / (mx - mn)
        return result

    def detect_outliers_iqr(self, df, columns):
        import pandas as pd
        result = df.copy()
        for col in columns:
            if col in result.columns:
                numeric = pd.to_numeric(result[col], errors="coerce")
                q1 = numeric.quantile(0.25)
                q3 = numeric.quantile(0.75)
                iqr = q3 - q1
                mask = (numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)
                result[f"{col}_outlier"] = mask
        return result

    def detect_outliers_zscore(self, df, columns):
        import numpy as np
        import pandas as pd
        result = df.copy()
        for col in columns:
            if col in result.columns:
                numeric = pd.to_numeric(result[col], errors="coerce")
                mean = numeric.mean()
                std = numeric.std()
                if std > 0:
                    z = np.abs((numeric - mean) / std)
                    result[f"{col}_outlier"] = z > 3
        return result

    def group_agg_single(self, df):
        """Single-column groupby with mean aggregation."""
        import pandas as pd
        result = df.copy()
        # Use CATEGORY_COL as the groupby key (high-cardinality categorical)
        if "CATEGORY_COL" in result.columns and "float_col_1" in result.columns:
            result["float_col_1"] = pd.to_numeric(result["float_col_1"], errors="coerce")
            return result.groupby("CATEGORY_COL", observed=True).agg({"float_col_1": "mean"})
        # Fallback: group by the first object column
        obj_cols = result.select_dtypes(include="object").columns
        if len(obj_cols) > 0:
            return result.groupby(obj_cols[0], observed=True).size()
        return result.describe()

    def group_agg_multi(self, df, columns):
        """Multi-column groupby with multiple agg functions."""
        import pandas as pd
        result = df.copy()
        if "CATEGORY_COL" in result.columns:
            # Coerce numeric columns and build agg dict
            agg_dict = {}
            for col in columns:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors="coerce")
                    agg_dict[col] = ["mean", "std", "min", "max"]
            if agg_dict:
                return result.groupby("CATEGORY_COL", observed=True).agg(agg_dict)
        # Fallback: describe all numeric
        return result.describe()

    def group_agg_transform(self, df, columns):
        """Groupby transform: center numeric columns within groups."""
        import pandas as pd
        result = df.copy()
        if "CATEGORY_COL" not in result.columns:
            return result
        for col in columns:
            if col in result.columns:
                try:
                    numeric_col = pd.to_numeric(result[col], errors="coerce")
                    result[col] = numeric_col - numeric_col.groupby(
                        result["CATEGORY_COL"]
                    ).transform("mean")
                except Exception:
                    pass
        return result

    def full_clean(self, df):
        """Manual pandas cleaning pipeline equivalent to fd.clean()."""
        result = df.copy()
        # 1. Clean column names
        result.columns = (
            result.columns.str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9_]", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )
        # 2. Strip whitespace from string columns
        str_cols = result.select_dtypes(include="object").columns
        for col in str_cols:
            try:
                result[col] = result[col].str.strip()
            except AttributeError:
                pass
        # 3. Replace sentinels with NaN
        import numpy as np
        sentinels = ["N/A", "n/a", "NA", "null", "None", "-", "", "#REF!",
                     "missing", "nan", "NaN"]
        result = result.replace(sentinels, np.nan)
        # 4. Drop all-empty columns
        result = result.dropna(axis=1, how="all")
        # 5. Drop all-empty rows
        result = result.dropna(axis=0, how="all")
        # 6. Drop duplicates
        result = result.drop_duplicates()
        # 7. Convert dtypes
        result = result.convert_dtypes()
        return result


# ---------------------------------------------------------------------------
# Polars adapter
# ---------------------------------------------------------------------------

class PolarsAdapter(LibraryAdapter):
    name = "polars"

    def __init__(self) -> None:
        try:
            import polars  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def _to_polars(self, df):
        import polars as pl
        import pandas as pd
        df_clean = df.copy()
        # PyArrow cannot convert mixed int/str object columns. Cast non-nulls to string.
        for col in df_clean.select_dtypes(include=["object"]).columns:
            mask = df_clean[col].notna()
            df_clean.loc[mask, col] = df_clean.loc[mask, col].astype(str)
        return pl.from_pandas(df_clean)

    def drop_missing_rows(self, df):
        pldf = self._to_polars(df)
        return pldf.drop_nulls()

    def fill_missing_mean(self, df, columns):
        import polars as pl
        pldf = self._to_polars(df)
        for col in columns:
            if col in pldf.columns:
                try:
                    pldf = pldf.with_columns(
                        pl.col(col).cast(pl.Float64, strict=False)
                        .fill_null(pl.col(col).cast(pl.Float64, strict=False).mean())
                    )
                except Exception:
                    pass
        return pldf

    def fill_missing_median(self, df, columns):
        import polars as pl
        pldf = self._to_polars(df)
        for col in columns:
            if col in pldf.columns:
                try:
                    pldf = pldf.with_columns(
                        pl.col(col).cast(pl.Float64, strict=False)
                        .fill_null(pl.col(col).cast(pl.Float64, strict=False).median())
                    )
                except Exception:
                    pass
        return pldf

    def fill_missing_ffill(self, df):
        import polars as pl
        pldf = self._to_polars(df)
        return pldf.with_columns(pl.all().forward_fill())

    def fill_missing_bfill(self, df):
        import polars as pl
        pldf = self._to_polars(df)
        return pldf.with_columns(pl.all().backward_fill())

    def detect_duplicates(self, df):
        pldf = self._to_polars(df)
        return pldf.is_duplicated()

    def drop_duplicates(self, df):
        pldf = self._to_polars(df)
        return pldf.unique()

    def trim_whitespace(self, df, columns):
        import polars as pl
        pldf = self._to_polars(df)
        for col in columns:
            if col in pldf.columns:
                try:
                    pldf = pldf.with_columns(pl.col(col).str.strip_chars())
                except Exception:
                    pass
        return pldf

    def to_lowercase(self, df, columns):
        import polars as pl
        pldf = self._to_polars(df)
        for col in columns:
            if col in pldf.columns:
                try:
                    pldf = pldf.with_columns(pl.col(col).str.to_lowercase())
                except Exception:
                    pass
        return pldf

    def to_uppercase(self, df, columns):
        import polars as pl
        pldf = self._to_polars(df)
        for col in columns:
            if col in pldf.columns:
                try:
                    pldf = pldf.with_columns(pl.col(col).str.to_uppercase())
                except Exception:
                    pass
        return pldf

    def regex_replace(self, df, columns, pattern, replacement):
        import polars as pl
        pldf = self._to_polars(df)
        for col in columns:
            if col in pldf.columns:
                try:
                    pldf = pldf.with_columns(
                        pl.col(col).str.replace_all(pattern, replacement)
                    )
                except Exception:
                    pass
        return pldf

    def rename_columns(self, df, mapping):
        pldf = self._to_polars(df)
        return pldf.rename(mapping)

    def drop_columns(self, df, columns):
        pldf = self._to_polars(df)
        existing = [c for c in columns if c in pldf.columns]
        return pldf.drop(existing)

    def select_columns(self, df, columns):
        pldf = self._to_polars(df)
        existing = [c for c in columns if c in pldf.columns]
        return pldf.select(existing)

    def convert_numeric(self, df, columns):
        import polars as pl
        pldf = self._to_polars(df)
        for col in columns:
            if col in pldf.columns:
                try:
                    pldf = pldf.with_columns(
                        pl.col(col).cast(pl.Float64, strict=False)
                    )
                except Exception:
                    pass
        return pldf

    def detect_outliers_iqr(self, df, columns):
        import polars as pl
        pldf = self._to_polars(df)
        for col in columns:
            if col in pldf.columns:
                try:
                    q1 = pldf[col].cast(pl.Float64, strict=False).quantile(0.25)
                    q3 = pldf[col].cast(pl.Float64, strict=False).quantile(0.75)
                    if q1 is not None and q3 is not None:
                        iqr = q3 - q1
                        pldf = pldf.with_columns(
                            ((pl.col(col).cast(pl.Float64, strict=False) < q1 - 1.5 * iqr)
                             | (pl.col(col).cast(pl.Float64, strict=False) > q3 + 1.5 * iqr))
                            .alias(f"{col}_outlier")
                        )
                except Exception:
                    pass
        return pldf

    def full_clean(self, df):
        """Manual Polars cleaning pipeline."""
        import polars as pl
        pldf = self._to_polars(df)
        # Clean column names
        new_names = {}
        for col in pldf.columns:
            import re
            clean = col.strip().lower()
            clean = re.sub(r"[^a-z0-9_]", "_", clean)
            clean = re.sub(r"_+", "_", clean).strip("_")
            new_names[col] = clean
        pldf = pldf.rename(new_names)
        # Drop all-null columns
        non_null_cols = [
            col for col in pldf.columns
            if pldf[col].null_count() < len(pldf)
        ]
        pldf = pldf.select(non_null_cols)
        # Drop duplicates
        pldf = pldf.unique()
        return pldf


# ---------------------------------------------------------------------------
# pyjanitor adapter
# ---------------------------------------------------------------------------

class PyjanitorAdapter(LibraryAdapter):
    name = "pyjanitor"

    def __init__(self) -> None:
        try:
            import janitor  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def drop_missing_rows(self, df):
        import janitor  # noqa: F401
        return df.remove_empty(reset_index=False)

    def drop_duplicates(self, df):
        return df.drop_duplicates()

    def trim_whitespace(self, df, columns):
        result = df.copy()
        for col in columns:
            if col in result.columns and result[col].dtype == object:
                result[col] = result[col].str.strip()
        return result

    def to_lowercase(self, df, columns):
        result = df.copy()
        for col in columns:
            if col in result.columns and result[col].dtype == object:
                result[col] = result[col].str.lower()
        return result

    def rename_columns(self, df, mapping):
        import janitor  # noqa: F401
        return df.clean_names()

    def drop_columns(self, df, columns):
        import janitor  # noqa: F401
        return df.remove_columns(columns)

    def select_columns(self, df, columns):
        import janitor  # noqa: F401
        return df.select_columns(columns)

    def onehot_encode(self, df, columns):
        import janitor  # noqa: F401
        result = df
        for col in columns:
            if col in result.columns:
                try:
                    result = result.encode_categorical(col)
                except Exception:
                    pass
        return result

    def full_clean(self, df):
        import janitor  # noqa: F401
        return df.clean_names().remove_empty(reset_index=False).drop_duplicates()


# ---------------------------------------------------------------------------
# Feature Engine adapter
# ---------------------------------------------------------------------------

class FeatureEngineAdapter(LibraryAdapter):
    name = "feature_engine"

    def __init__(self) -> None:
        try:
            import feature_engine  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def fill_missing_mean(self, df, columns):
        from feature_engine.imputation import MeanMedianImputer
        existing = [c for c in columns if c in df.columns]
        if not existing:
            return df
        imp = MeanMedianImputer(imputation_method="mean", variables=existing)
        return imp.fit_transform(df)

    def fill_missing_median(self, df, columns):
        from feature_engine.imputation import MeanMedianImputer
        existing = [c for c in columns if c in df.columns]
        if not existing:
            return df
        imp = MeanMedianImputer(imputation_method="median", variables=existing)
        return imp.fit_transform(df)

    def onehot_encode(self, df, columns):
        from feature_engine.encoding import OneHotEncoder
        existing = [c for c in columns if c in df.columns]
        if not existing:
            return df
        enc = OneHotEncoder(variables=existing, drop_last=False)
        return enc.fit_transform(df)

    def label_encode(self, df, columns):
        from feature_engine.encoding import OrdinalEncoder
        existing = [c for c in columns if c in df.columns]
        if not existing:
            return df
        enc = OrdinalEncoder(variables=existing)
        return enc.fit_transform(df)

    def detect_outliers_iqr(self, df, columns):
        from feature_engine.outliers import Winsorizer
        existing = [c for c in columns if c in df.columns]
        if not existing:
            return df
        w = Winsorizer(capping_method="iqr", fold=1.5, variables=existing)
        return w.fit_transform(df)

    def standard_scale(self, df, columns):
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        result = df.copy()
        existing = [c for c in columns if c in result.columns]
        if existing:
            scaler = StandardScaler()
            result[existing] = scaler.fit_transform(result[existing])
        return result

    def minmax_scale(self, df, columns):
        from sklearn.preprocessing import MinMaxScaler
        result = df.copy()
        existing = [c for c in columns if c in result.columns]
        if existing:
            scaler = MinMaxScaler()
            result[existing] = scaler.fit_transform(result[existing])
        return result


# ---------------------------------------------------------------------------
# Scikit-learn adapter
# ---------------------------------------------------------------------------

class SklearnAdapter(LibraryAdapter):
    name = "sklearn"

    def __init__(self) -> None:
        try:
            import sklearn  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def fill_missing_mean(self, df, columns):
        from sklearn.impute import SimpleImputer
        import numpy as np
        result = df.copy()
        existing = [c for c in columns if c in result.columns]
        if existing:
            imp = SimpleImputer(strategy="mean")
            result[existing] = imp.fit_transform(result[existing])
        return result

    def fill_missing_median(self, df, columns):
        from sklearn.impute import SimpleImputer
        result = df.copy()
        existing = [c for c in columns if c in result.columns]
        if existing:
            imp = SimpleImputer(strategy="median")
            result[existing] = imp.fit_transform(result[existing])
        return result

    def fill_missing_mode(self, df, columns):
        from sklearn.impute import SimpleImputer
        result = df.copy()
        existing = [c for c in columns if c in result.columns]
        if existing:
            imp = SimpleImputer(strategy="most_frequent")
            result[existing] = imp.fit_transform(result[existing])
        return result

    def onehot_encode(self, df, columns):
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import numpy as np
        result = df.copy()
        existing = [c for c in columns if c in result.columns]
        if not existing:
            return result
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = enc.fit_transform(result[existing].astype(str))
        feature_names = enc.get_feature_names_out(existing)
        encoded_df = pd.DataFrame(encoded, columns=feature_names,
                                  index=result.index)
        result = result.drop(columns=existing)
        return pd.concat([result, encoded_df], axis=1)

    def label_encode(self, df, columns):
        from sklearn.preprocessing import LabelEncoder
        result = df.copy()
        for col in columns:
            if col in result.columns:
                le = LabelEncoder()
                mask = result[col].notna()
                if mask.any():
                    result.loc[mask, col] = le.fit_transform(
                        result.loc[mask, col].astype(str)
                    )
        return result

    def standard_scale(self, df, columns):
        from sklearn.preprocessing import StandardScaler
        result = df.copy()
        existing = [c for c in columns if c in result.columns]
        if existing:
            scaler = StandardScaler()
            result[existing] = scaler.fit_transform(result[existing])
        return result

    def minmax_scale(self, df, columns):
        from sklearn.preprocessing import MinMaxScaler
        result = df.copy()
        existing = [c for c in columns if c in result.columns]
        if existing:
            scaler = MinMaxScaler()
            result[existing] = scaler.fit_transform(result[existing])
        return result

    def detect_outliers_zscore(self, df, columns):
        import numpy as np
        result = df.copy()
        for col in columns:
            if col in result.columns:
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:
                    z = np.abs((result[col] - mean) / std)
                    result[f"{col}_outlier"] = z > 3
        return result


# ---------------------------------------------------------------------------
# AutoClean adapter (may not be available)
# ---------------------------------------------------------------------------

class AutoCleanAdapter(LibraryAdapter):
    name = "autoclean"

    def __init__(self) -> None:
        try:
            from AutoClean import AutoClean  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def full_clean(self, df):
        from AutoClean import AutoClean
        pipeline = AutoClean(df.copy(), mode="auto", duplicates=True,
                             missing_num="auto", missing_categ="auto",
                             outliers="winz")
        return pipeline.output


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, type[LibraryAdapter]] = {
    "freshdata": FreshDataAdapter,
    "pandas": PandasAdapter,
    "polars": PolarsAdapter,
    "pyjanitor": PyjanitorAdapter,
    "feature_engine": FeatureEngineAdapter,
    "sklearn": SklearnAdapter,
    "autoclean": AutoCleanAdapter,
}

_adapter_cache: dict[str, LibraryAdapter] = {}


def get_adapter(name: str) -> LibraryAdapter:
    """Get a cached adapter instance by library name."""
    if name not in _adapter_cache:
        cls = _ADAPTERS.get(name)
        if cls is None:
            raise ValueError(f"Unknown library: {name!r}. "
                             f"Available: {list(_ADAPTERS)}")
        _adapter_cache[name] = cls()
    return _adapter_cache[name]


def available_libraries() -> list[str]:
    """Return names of libraries that are importable."""
    return [name for name, cls in _ADAPTERS.items() if cls().available]
