"""Cheap, backend-native column statistics.

:class:`ColumnMetadata` is everything the planner and the selector need to make
decisions without materialising a dataset. Each scanner uses the cheapest path
its backend offers: pandas describe on a sample, polars lazy aggregates, DuckDB
``SUMMARIZE``, or the Parquet footer.
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
        """Scan a registered DuckDB table/view via ``SUMMARIZE`` (no Python scan)."""
        require_duckdb()
        summary = conn.execute(f"SUMMARIZE {table_name}").fetchall()
        cols = [d[0] for d in conn.execute(f"SUMMARIZE {table_name}").description]
        idx = {name: i for i, name in enumerate(cols)}

        (n,) = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        n = int(n)

        out: list[ColumnMetadata] = []
        for r in summary:
            name = r[idx["column_name"]]
            dtype = str(r[idx["column_type"]])
            null_pct = r[idx.get("null_percentage", -1)] if "null_percentage" in idx else None
            null_ratio = float(null_pct) / 100.0 if null_pct is not None else 0.0
            approx_unique = r[idx["approx_unique"]] if "approx_unique" in idx else -1
            canonical = _canonical_dtype(dtype)
            non_null = int(round(n * (1.0 - null_ratio)))
            out.append(
                ColumnMetadata(
                    name=str(name),
                    dtype_str=canonical,
                    row_count=n,
                    null_ratio=null_ratio,
                    non_null_count=non_null,
                    n_unique=int(approx_unique) if approx_unique is not None else -1,
                    is_numeric=canonical in ("int64", "float64"),
                    is_string=canonical == "string",
                )
            )
        return out

    @staticmethod
    def from_parquet_path(path: str) -> list[ColumnMetadata]:
        """Read the exact row count from the Parquet footer; null stats via DuckDB.

        The footer gives the row count for free (no data scan); DuckDB streams the
        file for null/value statistics without loading it into Python.
        """
        n_rows = require_pyarrow().parquet.read_metadata(path).num_rows
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
