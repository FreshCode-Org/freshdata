"""The finance domain pack: ledger and transactional dataset hygiene.

Validates accounting/finance frames (transactions with debit/credit/currency)
against the rules in ``rules.yaml``. Standard checks (presence, regex, ISO 4217
reference) come from :class:`~freshdata.domains.base.ConfigDrivenValidator`;
the accounting-specific checks (date format, 2-decimal amounts, per-transaction
balance, single-sided entries, future dates) live here as custom functions.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from ..base import ColumnMapping, ConfigDrivenValidator, Rule, RuleResult

_PACK_DIR = Path(__file__).resolve().parent
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Numeric date with both leading components <= 12 is genuinely ambiguous
# (could be DD/MM or MM/DD), so we refuse to coerce it.
_AMBIGUOUS_DATE_RE = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\s*$")


@lru_cache(maxsize=1)
def _iso4217() -> dict[str, Any]:
    with open(_PACK_DIR / "reference" / "iso4217.json", encoding="utf-8") as handle:
        return json.load(handle)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_iso(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(value, format="%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    return ts if not pd.isna(ts) else None


def _loose_datetime(value: Any) -> Any:
    """Best-effort date parse, suppressing pandas' format-inference UserWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(value, errors="coerce")


class FinanceValidator(ConfigDrivenValidator):
    """Validator for finance/accounting ledger frames."""

    domain_name = "finance"
    version = "0.1.0"
    schema_version = "2024-01"

    canonical_fields = (
        "transaction_id", "date", "account_code", "debit", "credit",
        "currency", "description", "entity_id",
    )
    required_fields = ("transaction_id", "date", "debit", "credit", "currency")
    id_fields = ("transaction_id", "entity_id")
    aliases = {
        "transaction_id": (r"txn_?id", r"transaction_?id", r"trans_?id", r"tx_?id"),
        "date": (r"date", r"txn_?date", r"transaction_?date", r"posting_?date", r"value_?date"),
        "account_code": (r"account_?code", r"acct_?code", r"gl_?account", r"account_?no"),
        "debit": (r"debit", r"dr", r"debit_?amount"),
        "credit": (r"credit", r"cr", r"credit_?amount"),
        "currency": (r"currency", r"ccy", r"curr", r"currency_?code"),
        "description": (r"description", r"memo", r"narration", r"details"),
        "entity_id": (r"entity_?id", r"party_?id", r"counterparty_?id", r"vendor_?id"),
    }
    rules_path = str(_PACK_DIR / "rules.yaml")

    def register_extensions(self) -> None:
        self.register_check("iso8601_date", self._check_iso8601)
        self.register_check("nonneg_2dp", self._check_nonneg_2dp)
        self.register_check("balanced_by_transaction", self._check_balanced)
        self.register_check("not_both_sided", self._check_both_sided)
        self.register_check("not_future_date", self._check_future)
        self.register_repair("coerce_iso8601_date", self._repair_iso8601)

    def load_reference_values(self, name: str):
        if name == "iso4217":
            return _iso4217()["codes"]
        return super().load_reference_values(name)

    def reference_sources(self) -> list[dict[str, Any]]:
        return [{"name": "iso4217", **_iso4217()["_meta"]}]

    # -- custom checks ------------------------------------------------------

    def _check_iso8601(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        col = mapping.actual("date")
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            return []  # already real dates — valid by construction
        present = series.notna()
        text = series.astype("string")
        well_formed = text.str.fullmatch(_ISO_DATE_RE.pattern).fillna(False)
        bad = present & ~well_formed
        # A YYYY-MM-DD shape can still be an impossible date (2024-13-40).
        for idx in df.index[present & well_formed]:
            if _parse_iso(series.at[idx]) is None:
                bad.at[idx] = True
        return df.index[bad].tolist()

    def _check_nonneg_2dp(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        rows: set[Any] = set()
        for field_name in rule.fields:
            col = mapping.actual(field_name)
            series = df[col]
            present = series.notna()
            numeric = _to_numeric(series)
            bad = present & numeric.isna()  # non-numeric where a value exists
            finite = numeric.map(lambda value: pd.isna(value) or math.isfinite(float(value)))
            bad = bad | (present & ~finite)
            bad = bad | (present & (numeric < 0))
            # More than 2 decimal places (tolerant of float noise).
            scaled = (numeric * 100).round()
            over = present & numeric.notna() & ((numeric * 100 - scaled).abs() > 1e-6)
            bad = bad | over
            rows.update(df.index[bad].tolist())
        return sorted(rows, key=_sort_key)

    def _check_balanced(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        txn = mapping.actual("transaction_id")
        debit = _to_numeric(df[mapping.actual("debit")]).fillna(0.0)
        credit = _to_numeric(df[mapping.actual("credit")]).fillna(0.0)
        tolerance = float(rule.params.get("tolerance", 0.0))
        work = pd.DataFrame({"_txn": df[txn], "_d": debit, "_c": credit}, index=df.index)
        sums = work.groupby("_txn")[["_d", "_c"]].transform("sum")
        unbalanced = (sums["_d"] - sums["_c"]).abs() > tolerance
        # Only flag rows that belong to an identified transaction.
        unbalanced = unbalanced & df[txn].notna()
        return df.index[unbalanced].tolist()

    def _check_both_sided(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        debit = _to_numeric(df[mapping.actual("debit")])
        credit = _to_numeric(df[mapping.actual("credit")])
        both = debit.notna() & credit.notna() & (debit != 0) & (credit != 0)
        return df.index[both].tolist()

    def _check_future(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        col = mapping.actual("date")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        today = pd.Timestamp.now(tz="UTC").normalize()
        future = parsed.notna() & (parsed.dt.normalize() > today)
        return df.index[future].tolist()

    # -- custom repair ------------------------------------------------------

    def _repair_iso8601(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule, result: RuleResult
    ) -> dict[Any, Any]:
        col = mapping.actual("date")
        fixes: dict[Any, Any] = {}
        for row in result.violation_rows:
            if row not in df.index:
                continue
            value = df.at[row, col]
            if pd.isna(value):
                continue
            text = str(value).strip()
            if _AMBIGUOUS_DATE_RE.match(text) and _is_ambiguous(text):
                continue  # refuse to guess DD/MM vs MM/DD
            ts = _loose_datetime(text)
            if not pd.isna(ts):
                fixes[row] = ts.strftime("%Y-%m-%d")
        return fixes


def _is_ambiguous(text: str) -> bool:
    match = _AMBIGUOUS_DATE_RE.match(text)
    if not match:
        return False
    first, second = int(match.group(1)), int(match.group(2))
    return first <= 12 and second <= 12 and first != second


def _sort_key(value: Any) -> tuple[int, Any]:
    """Sort row labels of mixed type deterministically (numbers, then strings)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, value)
    return (1, str(value))
