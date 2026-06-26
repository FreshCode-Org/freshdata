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
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .._common import check_iso_datetime, check_not_future
from ..base import ColumnMapping, ConfigDrivenValidator, Rule, RuleResult
from ..reference import load_reference

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


# Canonical model for finance_mode="tick" (market tick / trade data).
_TICK_FIELDS = ("timestamp", "symbol", "exchange", "price", "size", "bid", "ask", "currency")
_TICK_REQUIRED = ("timestamp", "symbol", "price", "size")
_TICK_IDS = ("symbol", "exchange")
_TICK_ALIASES = {
    "timestamp": (r"timestamp", r"ts", r"time", r"datetime", r"trade_?time", r"exec_?time"),
    "symbol": (r"symbol", r"ticker", r"instrument", r"sym", r"ric", r"isin"),
    "exchange": (r"exchange", r"venue", r"mic", r"market"),
    "price": (r"price", r"px", r"last", r"trade_?price"),
    "size": (r"size", r"qty", r"quantity", r"volume", r"shares"),
    "bid": (r"bid", r"bid_?px", r"bid_?price"),
    "ask": (r"ask", r"offer", r"ask_?px", r"ask_?price"),
    "currency": (r"currency", r"ccy", r"curr", r"currency_?code"),
}


class FinanceValidator(ConfigDrivenValidator):
    """Validator for finance/accounting ledger frames, or market tick data.

    Pass ``finance_mode="tick"`` (e.g. ``fd.clean(df, domain="finance",
    finance_mode="tick")``) to validate market tick/trade data — timestamp, symbol,
    price, size, bid/ask, currency — with BCBS-239 / SOX-style control checks instead
    of the default double-entry ledger model.
    """

    domain_name = "finance"
    version = "0.1.0"
    schema_version = "2024-01"

    canonical_fields: tuple[str, ...] = (
        "transaction_id", "date", "account_code", "debit", "credit",
        "currency", "description", "entity_id",
    )
    required_fields: tuple[str, ...] = ("transaction_id", "date", "debit", "credit", "currency")
    id_fields: tuple[str, ...] = ("transaction_id", "entity_id")
    aliases: Mapping[str, Sequence[str]] = {
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

    def __init__(self, *, column_map: Any = None, finance_mode: str = "ledger") -> None:
        if finance_mode not in ("ledger", "tick"):
            raise ValueError(f"finance_mode must be 'ledger' or 'tick', got {finance_mode!r}")
        self.finance_mode = finance_mode
        if finance_mode == "tick":
            self.schema_version = "finance-tick/2025.06"
            self.canonical_fields = _TICK_FIELDS
            self.required_fields = _TICK_REQUIRED
            self.id_fields = _TICK_IDS
            self.aliases = _TICK_ALIASES
            self.rules_path = str(_PACK_DIR / "rules_tick.yaml")
        super().__init__(column_map=column_map)

    def register_extensions(self) -> None:
        self.register_check("iso8601_date", self._check_iso8601)
        self.register_check("nonneg_2dp", self._check_nonneg_2dp)
        self.register_check("balanced_by_transaction", self._check_balanced)
        self.register_check("not_both_sided", self._check_both_sided)
        self.register_check("not_future_date", self._check_future)
        self.register_repair("coerce_iso8601_date", self._repair_iso8601)
        # finance_mode="tick" checks
        self.register_check("iso8601_datetime", check_iso_datetime)
        self.register_check("not_future_ts", check_not_future)
        self.register_check("tick_positive", self._check_tick_positive)
        self.register_check("currency_iso4217", self._check_currency_iso4217)
        self.register_check("bid_le_ask", self._check_bid_le_ask)
        self.register_check("unique_tick", self._check_unique_tick)
        self.register_check("control_completeness", self._check_control_completeness)

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

    # -- finance_mode="tick" checks -----------------------------------------

    def _check_tick_positive(self, df: pd.DataFrame, mapping: ColumnMapping,
                             rule: Rule) -> list[Any]:
        """Flag present price/size values that are non-numeric or not strictly positive."""
        rows: set[Any] = set()
        for field_name in rule.fields:
            col = mapping.actual(field_name)
            if col is None:
                continue
            series = df[col]
            num = _to_numeric(series)
            bad = series.notna() & (num.isna() | (num <= 0))
            rows.update(df.index[bad].tolist())
        return sorted(rows, key=_sort_key)

    def _check_currency_iso4217(self, df: pd.DataFrame, mapping: ColumnMapping,
                               rule: Rule) -> list[Any]:
        """Flag currency codes outside ISO-4217, via the bundled reference layer."""
        col = mapping.actual("currency")
        if col is None:
            return []
        ref = load_reference("iso4217", normalizer="upper")
        return df.index[ref.invalid_mask(df[col])].tolist()

    def _check_bid_le_ask(self, df: pd.DataFrame, mapping: ColumnMapping,
                          rule: Rule) -> list[Any]:
        """Flag crossed quotes (bid > ask) where both sides are present."""
        bcol, acol = mapping.actual("bid"), mapping.actual("ask")
        if bcol is None or acol is None:
            return []
        bid, ask = _to_numeric(df[bcol]), _to_numeric(df[acol])
        bad = bid.notna() & ask.notna() & (bid > ask)
        return df.index[bad].tolist()

    def _check_unique_tick(self, df: pd.DataFrame, mapping: ColumnMapping,
                           rule: Rule) -> list[Any]:
        """Flag duplicate ticks on the rule's key fields (completeness / SOX audit)."""
        cols = [mapping.actual(f) for f in rule.fields]
        if any(c is None for c in cols):
            return []
        keyed = df[cols]
        dup = keyed.duplicated(keep="first") & keyed.notna().all(axis=1)
        return df.index[dup].tolist()

    def _check_control_completeness(self, df: pd.DataFrame, mapping: ColumnMapping,
                                    rule: Rule) -> list[Any]:
        """BCBS-239 / SOX: flag ticks missing the currency/exchange needed to roll up
        control totals by venue and currency. Audit-only — nothing is dropped."""
        rows: set[Any] = set()
        for field_name in rule.fields:
            col = mapping.actual(field_name)
            if col is None:
                continue
            rows.update(df.index[df[col].isna()].tolist())
        return sorted(rows, key=_sort_key)


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
