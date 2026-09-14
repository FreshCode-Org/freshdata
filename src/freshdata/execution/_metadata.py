"""Cheap, backend-native column statistics.

:class:`ColumnMetadata` is everything the planner and the selector need to make
decisions without materialising a dataset. Each scanner uses the cheapest path
its backend offers: pandas describe on a sample, polars lazy aggregates, one
DuckDB aggregate query, or the Parquet footer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._lazy import require_duckdb, require_polars, require_pyarrow

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: Above this row count, the pandas scanner samples instead of scanning fully.
_PANDAS_SAMPLE_THRESHOLD = 100_000
_SAMPLE_FRAC = 0.10


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def is_duckdb_float(native_dtype: str) -> bool:
    """True for DuckDB floating-point types, the only ones that can hold ``NaN``."""
    return native_dtype.upper() in ("FLOAT", "DOUBLE", "REAL", "FLOAT4", "FLOAT8")


def _canonical_dtype(kind: str) -> str:
    """Map an arbitrary dtype string to freshdata's canonical buckets."""
    k = kind.lower()
    if "int" in k:
        return "int64"
    if "float" in k or "double" in k or "decimal" in k:
        return "float64"
    if "bool" in k:
        return "bool"
    if "date" in k or "time" in k:
        return "datetime"
    if "str" in k or "utf8" in k or "object" in k or "char" in k:
        return "string"
    return "object"


@dataclass
class ColumnMetadata:
    """Per-column statistics computed without a full materialisation."""

    name: str
    dtype_str: str
    row_count: int
    null_ratio: float
    non_null_count: int
    n_unique: int = -1  # -1 = not computed / unknown
    is_numeric: bool = False
    is_string: bool = False
    sample_values: list[Any] = field(default_factory=list)
    #: The backend's own type name (e.g. DuckDB ``DOUBLE``); empty when unknown.
    native_dtype: str = ""

    @property
    def is_empty(self) -> bool:
        """True when the column holds no non-null values."""
        return self.non_null_count == 0


class MetadataScanner:
    """Compute :class:`ColumnMetadata` per backend, cheaply."""

    @staticmethod
    def from_pandas(df: pd.DataFrame) -> list[ColumnMetadata]:
        from pandas.api.types import is_numeric_dtype, is_string_dtype

        n = len(df)
        sample = df
        if n > _PANDAS_SAMPLE_THRESHOLD:
            sample = df.sample(frac=_SAMPLE_FRAC, random_state=0)

        out: list[ColumnMetadata] = []
        for col in df.columns:
            s = df[col]
            non_null = int(s.notna().sum())
            null_ratio = 0.0 if n == 0 else 1.0 - non_null / n
            samp = sample[col].dropna()
            try:
                n_unique = int(samp.nunique())
            except TypeError:  # unhashable values
                n_unique = -1
            out.append(
                ColumnMetadata(
                    name=str(col),
                    dtype_str=_canonical_dtype(str(s.dtype)),
                    row_count=n,
                    null_ratio=null_ratio,
                    non_null_count=non_null,
                    n_unique=n_unique,
                    is_numeric=bool(is_numeric_dtype(s)),
                    is_string=bool(is_string_dtype(s) or s.dtype == object),
                    sample_values=list(samp.head(5).tolist()),
                )
            )
        return out

    @staticmethod
    def from_polars_lazy(lf: Any) -> list[ColumnMetadata]:
        """Scan a polars LazyFrame using only aggregate collects (constant memory)."""
        pl = require_polars()

        schema = lf.collect_schema()
        names = list(schema.names())
        if not names:
            return []

        # One aggregate pass for height + per-column null counts + n_unique.
        aggs = [pl.len().alias("__height__")]
        for name in names:
            aggs.append(pl.col(name).null_count().alias(f"__nulls__{name}"))
            aggs.append(pl.col(name).n_unique().alias(f"__nuniq__{name}"))
        stats = lf.select(aggs).collect()
        row = stats.row(0, named=True)
        n = int(row["__height__"])

        out: list[ColumnMetadata] = []
        for name in names:
            dtype = schema[name]
            nulls = int(row[f"__nulls__{name}"])
            non_null = n - nulls
            null_ratio = 0.0 if n == 0 else nulls / n
            out.append(
                ColumnMetadata(
                    name=name,
                    dtype_str=_canonical_dtype(str(dtype)),
                    row_count=n,
                    null_ratio=null_ratio,
                    non_null_count=non_null,
                    n_unique=int(row[f"__nuniq__{name}"]),
                    is_numeric=dtype.is_numeric(),
                    is_string=(dtype == pl.Utf8),
                )
            )
        return out

    @staticmethod
    def from_duckdb(conn: Any, table_name: str) -> list[ColumnMetadata]:
        """Scan a registered DuckDB table/view with one aggregate query (no Python scan).

        Null counts are exact, and float ``NaN`` counts as missing as it does in
        pandas. ``SUMMARIZE`` is not used: its ``stddev_samp`` raises on
        non-finite floats and it only reports a rounded null percentage.
        """
        require_duckdb()
        described = conn.execute(f"DESCRIBE {table_name}").fetchall()
        columns = [(str(r[0]), str(r[1])) for r in described]
        aggs = ["COUNT(*)"]
        for name, native in columns:
            col = _quote_identifier(name)
            present = f"CASE WHEN NOT isnan({col}) THEN 1 END" if is_duckdb_float(native) else col
            aggs.append(f"COUNT({present})")
            aggs.append(f"approx_count_distinct({col})")
        row = conn.execute(f"SELECT {', '.join(aggs)} FROM {table_name}").fetchone()
        n = int(row[0])

        out: list[ColumnMetadata] = []
        for i, (name, native) in enumerate(columns):
            non_null = int(row[1 + 2 * i])
            approx_unique = row[2 + 2 * i]
            canonical = _canonical_dtype(native)
            out.append(
                ColumnMetadata(
                    name=name,
                    dtype_str=canonical,
                    row_count=n,
                    null_ratio=0.0 if n == 0 else 1.0 - non_null / n,
                    non_null_count=non_null,
                    n_unique=int(approx_unique) if approx_unique is not None else -1,
                    is_numeric=canonical in ("int64", "float64"),
                    is_string=canonical == "string",
                    native_dtype=native,
                )
            )
        return out

    @staticmethod
    def from_parquet_path(path: str) -> list[ColumnMetadata]:
        """Read the exact row count from the Parquet footer; null stats via DuckDB.

        The footer gives the row count for free (no data scan); DuckDB streams the
        file for null/value statistics without loading it into Python.
        """
        require_pyarrow("Reading Parquet metadata")
        # ``pyarrow.parquet`` is a submodule: importing ``pyarrow`` alone does
        # not expose it as an attribute.
        import pyarrow.parquet as pq

        n_rows = pq.read_metadata(path).num_rows
        duckdb = require_duckdb()
        escaped = path.replace("'", "''")
        conn = duckdb.connect()
        try:
            conn.execute(
                f"CREATE VIEW _fd_meta AS SELECT * FROM read_parquet('{escaped}')"
            )
            meta = MetadataScanner.from_duckdb(conn, "_fd_meta")
        finally:
            conn.close()
        for m in meta:  # trust the footer's exact count over DuckDB's
            m.row_count = n_rows
        return meta

    @staticmethod
    def from_source(source: Any, engine: str) -> list[ColumnMetadata]:
        """Dispatch to the right scanner for *source* given the resolved *engine*."""
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            return MetadataScanner.from_pandas(source)
        if isinstance(source, str):
            return MetadataScanner.from_parquet_path(source)
        try:
            import pyarrow as pa

            if isinstance(source, (pa.Table, pa.RecordBatch)):
                table = source if isinstance(source, pa.Table) else pa.Table.from_batches([source])
                pl = require_polars()
                return MetadataScanner.from_polars_lazy(pl.from_arrow(table).lazy())
        except ImportError:
            pass
        try:
            pl = require_polars()
            if isinstance(source, pl.LazyFrame):
                return MetadataScanner.from_polars_lazy(source)
            if isinstance(source, pl.DataFrame):
                return MetadataScanner.from_polars_lazy(source.lazy())
        except ImportError:
            pass
        raise TypeError(f"cannot scan metadata for source of type {type(source).__name__}")
