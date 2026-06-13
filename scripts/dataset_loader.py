"""Shared online dataset load/normalize helpers for fetch + tests."""

from __future__ import annotations

import io
import zipfile

import pandas as pd


def payload_bytes(raw: bytes, entry: dict) -> bytes:
    fmt = entry.get("format", "csv")
    if fmt == "zip":
        member = entry.get("zip_member")
        if not member:
            raise ValueError("zip format requires zip_member")
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            return zf.read(member)
    return raw


def flatten_json_frame(df: pd.DataFrame, *, max_columns: int = 50) -> pd.DataFrame:
    """Flatten one level of dict-valued columns for tabular cache."""
    out = df.copy()
    for col in list(out.columns):
        sample = out[col].dropna().head(20)
        if sample.empty:
            continue
        if sample.map(lambda x: isinstance(x, dict)).any():
            expanded = pd.json_normalize(out[col].tolist())
            expanded.columns = [f"{col}_{sub}" for sub in expanded.columns]
            out = out.drop(columns=[col]).join(expanded)
        if out.shape[1] > max_columns:
            out = out.iloc[:, :max_columns]
            break
    return out


def load_dataframe(payload: bytes, entry: dict) -> pd.DataFrame:
    fmt = entry.get("format", "csv")
    normalize = entry.get("normalize") or {}

    if fmt in ("csv", "tsv", "zip"):
        read_csv = dict(entry.get("read_csv") or {})
        if fmt == "tsv" and "sep" not in read_csv:
            read_csv["sep"] = "\t"
        if read_csv.get("header") is None and "header" in read_csv:
            read_csv["header"] = None
        df = pd.read_csv(io.BytesIO(payload), **read_csv)
    elif fmt in ("json", "jsonl"):
        read_json = dict(entry.get("read_json") or {})
        if fmt == "jsonl":
            read_json.setdefault("lines", True)
        df = pd.read_json(io.BytesIO(payload), **read_json)
    else:
        raise ValueError(f"unsupported format: {fmt}")

    if normalize.get("flatten"):
        df = flatten_json_frame(df, max_columns=int(normalize.get("max_columns", 50)))
    return df


def registry_entry_to_manifest(entry: dict) -> dict:
    """Copy fetch-relevant fields from registry into manifest entry."""
    keys = (
        "url",
        "description",
        "format",
        "max_rows",
        "read_csv",
        "read_json",
        "normalize",
        "zip_member",
        "tags",
        "domain",
        "tier",
    )
    return {k: entry[k] for k in keys if k in entry}
