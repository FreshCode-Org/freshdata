"""Gold-labelled gauntlet fixtures.

Every fixture is deterministic for a given seed: a base frame of valid
records, plus injected problem cells whose *expected disposition* is known.

Dispositions (``GoldCell.expect``):

- ``preserve`` — valid (often unusual) data; must survive every cleaning
  surface byte-identical and must not draw an error-severity issue.
- ``repair`` — a safe deterministic repair exists; ``repaired`` is the gold
  value. Silently leaving it is an escape; changing it to anything else is a
  corruption.
- ``flag`` — must be *detected* (issue / warning / finding) but never
  auto-changed by the default pipeline.
- ``review`` — ambiguous; must be routed to a quarantine / manual-review
  action, never auto-deleted or auto-accepted.

Whole-row duplicate injections are tracked separately in ``dup_row_count``
because row removal is a row-level (not cell-level) disposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from freshdata import FieldSpec

DEFAULT_ROWS = 300
DEFAULT_SEED = 42


@dataclass(frozen=True)
class GoldCell:
    row: int
    column: str
    kind: str          #: defect / trap family, e.g. "text_in_numeric"
    dirty: Any         #: the value placed in the frame
    expect: str        #: preserve | repair | flag | review
    repaired: Any = None
    #: the cell may legitimately end up imputed by the audited missing-value
    #: engine after its repair-to-missing (sentinels, empty strings): both
    #: "left missing" and "audited fill" count as the gold repair.
    accept_impute: bool = False
    pii: str | None = None  #: expected PII entity_type, when the cell is PII
    note: str = ""
    replaced: Any = None  #: the valid value the injection overwrote


@dataclass
class GauntletFixture:
    name: str
    df: pd.DataFrame
    cells: list[GoldCell]
    schema: dict[str, FieldSpec | str]
    field_types: dict[str, str] = field(default_factory=dict)
    domain: str | None = None
    dup_row_count: int = 0
    n_rows: int = 0

    def labelled(self, *expects: str) -> list[GoldCell]:
        return [c for c in self.cells if not expects or c.expect in expects]

    def pristine(self) -> pd.DataFrame:
        """The base frame before injection: labelled cells restored, dups gone."""
        base = self.df.iloc[: self.n_rows].copy()
        for c in self.cells:
            base.iloc[c.row, base.columns.get_loc(c.column)] = c.replaced
        return base


class _Injector:
    """Places labelled values on distinct (row, column) slots, deterministically."""

    def __init__(self, df: pd.DataFrame, seed: int) -> None:
        self.df = df
        self.cells: list[GoldCell] = []
        self._order = {
            col: list(np.random.default_rng(seed + i).permutation(len(df)))
            for i, col in enumerate(df.columns)
        }

    def place(self, column: str, dirty: Any, kind: str, expect: str, *,
              repaired: Any = None, accept_impute: bool = False,
              pii: str | None = None, note: str = "") -> int:
        row = self._order[column].pop()
        loc = self.df.columns.get_loc(column)
        replaced = self.df.iloc[row, loc]
        self.df.iloc[row, loc] = dirty
        self.cells.append(GoldCell(row=row, column=column, kind=kind, dirty=dirty,
                                   expect=expect, repaired=repaired,
                                   accept_impute=accept_impute, pii=pii, note=note,
                                   replaced=replaced))
        return row


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# finance
# ---------------------------------------------------------------------------

_TICKERS = ("AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "NVDA", "JPM", "V")
_COMPANIES = ("Apple", "Microsoft", "Tesla", "Alphabet", "Amazon",
              "Nvidia", "JPMorgan", "Visa")
_CURRENCIES = frozenset({"USD", "EUR", "GBP", "JPY"})


def _finance(n: int, seed: int) -> GauntletFixture:
    r = _rng(seed)
    idx = r.integers(0, len(_TICKERS), n)
    df = pd.DataFrame({
        "txn_id": [f"TXN{100000 + i}" for i in range(n)],
        "company": [_COMPANIES[i] for i in idx],
        "ticker": [_TICKERS[i] for i in idx],
        "price": np.round(r.uniform(10, 900, n), 2).astype(object),
        "revenue": np.round(r.uniform(1e4, 5e6, n), 2).astype(object),
        "currency": r.choice(sorted(_CURRENCIES), n),
        "trade_date": pd.to_datetime("2025-01-01")
        + pd.to_timedelta(r.integers(0, 365, n), unit="D"),
        "pct_change": np.round(r.uniform(-9, 9, n), 3).astype(object),
    })
    df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    inj = _Injector(df, seed)

    # -- the flagship case: 'apple' across financial columns -------------------
    inj.place("price", "apple", "text_in_numeric", "flag",
              note="company name in a price column; never auto-delete")
    inj.place("revenue", "apple", "text_in_numeric", "flag")
    inj.place("company", "Apple", "valid_company_name", "preserve")
    inj.place("ticker", "AAPL", "valid_ticker", "preserve")
    inj.place("ticker", "apple", "lowercase_ticker", "review",
              note="structurally invalid ticker; plausible fix exists -> review")

    # numeric problems
    inj.place("price", -12.5, "impossible_range", "flag", note="negative price")
    inj.place("price", 9_999_999.0, "extreme_outlier", "flag")
    inj.place("price", "402.10", "numeric_as_text", "repair", repaired=402.1)
    inj.place("revenue", "1,200,500.00", "thousands_separator", "repair",
              repaired=1200500.0)
    inj.place("revenue", "$3,400.50", "currency_symbol", "repair", repaired=3400.5)
    inj.place("pct_change", "12.5%", "percent_as_text", "review",
              note="'12.5%' could mean 12.5 or 0.125 in a change column — "
                   "quarantine for review, never guess")
    inj.place("price", " 212.0 ", "whitespace_numeric", "repair", repaired=212.0)
    inj.place("revenue", "N/A", "sentinel", "repair", repaired=None,
              accept_impute=True)
    inj.place("price", "null", "sentinel", "repair", repaired=None,
              accept_impute=True)

    # dates: coercing garbage to missing is an acceptable audited repair;
    # inventing a date is not.
    inj.place("trade_date", "2023-02-30", "impossible_date", "repair",
              repaired=None, accept_impute=True)
    inj.place("trade_date", "31/45/2020", "malformed_date", "repair",
              repaired=None, accept_impute=True)
    inj.place("trade_date", "not a date", "text_in_date", "repair",
              repaired=None, accept_impute=True)

    # identifiers / vocabulary
    inj.place("txn_id", "TXN 12@34!", "malformed_id", "flag")
    inj.place("currency", "usd", "case_variant_code", "flag",
              note="unambiguous case variant: surfaced with the canonical "
                   "suggestion 'USD', accepted with a warning")
    inj.place("currency", "US Dollar", "verbose_code", "review")
    inj.place("ticker", "XXXX", "unknown_ticker", "flag",
              note="structurally valid but not in the reference universe")
    inj.place("ticker", "BRK.B", "rare_valid_ticker", "flag",
              note="valid class-B share; outside this fund's trading universe, "
                   "so the reference lookup flags it — but it must NOT be changed")

    # adversarial traps
    inj.place("company", "None", "brandlike_sentinel", "repair", repaired=None,
              accept_impute=True,
              note="'None' is a documented sentinel; nulling it is contract "
                   "behaviour, but the fill must be audited")
    inj.place("company", "Ünïcode Holdings ÅB", "unicode_company", "preserve")
    inj.place("pct_change", 0.0, "zero_is_valid", "preserve")

    schema: dict[str, FieldSpec | str] = {
        "txn_id": FieldSpec(semantic_type="identifier", required=True),
        "company": FieldSpec(semantic_type="company_name"),
        "ticker": FieldSpec(semantic_type="stock_ticker", reference=set(_TICKERS),
                            suggest={"apple": "AAPL"}),
        "price": FieldSpec(semantic_type="numeric", min_value=0.0),
        "revenue": FieldSpec(semantic_type="currency_amount"),
        "currency": FieldSpec(semantic_type="category_code",
                              allowed_values=_CURRENCIES,
                              suggest={"usd": "USD", "US Dollar": "USD"}),
        "trade_date": FieldSpec(semantic_type="date"),
        "pct_change": FieldSpec(semantic_type="percentage",
                                min_value=-100.0, max_value=100.0),
    }
    fx = GauntletFixture(
        name="finance", df=df, cells=inj.cells, schema=schema,
        field_types={"company": "company_name", "ticker": "stock_ticker"},
        domain="finance", n_rows=n,
    )
    return _append_duplicates(fx, rows=3, seed=seed)


# ---------------------------------------------------------------------------
# healthcare
# ---------------------------------------------------------------------------

_BLOOD = frozenset({"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"})
_ICD10 = ("E11.9", "I10", "J45.909", "M54.5", "F41.1", "K21.9")


def _healthcare(n: int, seed: int) -> GauntletFixture:
    r = _rng(seed + 1)
    df = pd.DataFrame({
        "mrn": [f"MRN{500000 + i}" for i in range(n)],
        "icd10": r.choice(_ICD10, n),
        "dob": pd.to_datetime("1950-01-01")
        + pd.to_timedelta(r.integers(0, 20000, n), unit="D"),
        "age": r.integers(18, 90, n).astype(object),
        "temp_c": np.round(r.uniform(36.0, 39.5, n), 1).astype(object),
        "heart_rate": r.integers(48, 130, n).astype(object),
        "blood_type": r.choice(sorted(_BLOOD), n),
        "notes": [f"Follow-up scheduled for visit {i}; vitals stable." for i in range(n)],
    })
    df["dob"] = df["dob"].dt.strftime("%Y-%m-%d")
    inj = _Injector(df, seed + 1)

    inj.place("age", 400, "impossible_range", "flag")
    inj.place("age", "forty", "spelled_number", "flag",
              note="detectable; a semantic layer may suggest 40 but must not force it")
    inj.place("temp_c", 98.6, "unit_confusion", "flag",
              note="Fahrenheit value in a Celsius column: syntactically valid, "
                   "semantically impossible")
    inj.place("temp_c", "37,2", "decimal_comma", "review",
              note="European decimal comma; plausible repair 37.2 needs review")
    inj.place("heart_rate", 0, "impossible_range", "flag")
    inj.place("dob", "2031-05-01", "future_dob", "flag")
    inj.place("dob", "1875-01-01", "implausible_dob", "flag")
    inj.place("icd10", "ZZZ.99.9X", "invalid_code", "flag")
    inj.place("icd10", "e11.9", "case_variant_code", "review")
    inj.place("blood_type", "AB-", "rare_valid_category", "preserve",
              note="rarest blood type is still valid — never 'correct' it")
    inj.place("blood_type", "abplus", "invalid_category", "flag")
    inj.place("mrn", "", "missing_required", "flag")
    inj.place("notes", "Patient SSN 123-45-6789 noted.", "pii_ssn", "flag",
              pii="SSN")
    inj.place("notes", "Call 555-867-5309 to reschedule.", "pii_phone", "flag",
              pii="PHONE")
    inj.place("notes", "  Routine   visit.\u200b  ", "whitespace_noise", "repair",
              repaired="Routine visit.")
    inj.place("age", " 45 ", "whitespace_numeric", "repair", repaired=45)

    schema: dict[str, FieldSpec | str] = {
        "mrn": FieldSpec(semantic_type="identifier", required=True, nullable=False),
        "icd10": FieldSpec(semantic_type="category_code",
                           pattern=r"[A-Z]\d{2}(?:\.\d{1,4})?"),
        "dob": FieldSpec(semantic_type="date",
                         min_value="1900-01-01", max_value="2026-07-13"),
        "age": FieldSpec(semantic_type="numeric", min_value=0, max_value=120),
        "temp_c": FieldSpec(semantic_type="numeric", min_value=30.0, max_value=45.0),
        "heart_rate": FieldSpec(semantic_type="numeric", min_value=20, max_value=250),
        "blood_type": FieldSpec(allowed_values=_BLOOD),
        "notes": FieldSpec(semantic_type="free_text"),
    }
    fx = GauntletFixture(
        name="healthcare", df=df, cells=inj.cells, schema=schema,
        field_types={"notes": "free_text"}, domain="healthcare", n_rows=n,
    )
    return _append_duplicates(fx, rows=3, seed=seed + 1)


# ---------------------------------------------------------------------------
# crm
# ---------------------------------------------------------------------------

_COUNTRIES = frozenset({
    "United States", "Germany", "France", "Japan", "Brazil", "Namibia", "India",
})


def _crm(n: int, seed: int) -> GauntletFixture:
    r = _rng(seed + 2)
    first = ("Ana", "Bob", "Chloé", "Dmitri", "Emeka", "Fatima", "Göran", "Hana")
    last = ("Ivanov", "Jones", "Kowalski", "López", "Müller", "Ncube", "O'Brien")
    df = pd.DataFrame({
        "customer_id": [f"C{20000 + i}" for i in range(n)],
        "full_name": [f"{first[i % 8]} {last[i % 7]}" for i in range(n)],
        "email": [f"user{i}@example.com" for i in range(n)],
        "phone": [f"+1-555-{1000 + i:04d}" for i in range(n)],
        "country": r.choice(sorted(_COUNTRIES), n),
        "signup_date": (pd.to_datetime("2024-01-01")
                        + pd.to_timedelta(r.integers(0, 500, n), unit="D")
                        ).strftime("%Y-%m-%d"),
        "status": r.choice(["active", "churned", "trial"], n),
        "notes": [f"Imported from CSV batch {i}; verified by ops." for i in range(n)],
    })
    inj = _Injector(df, seed + 2)

    inj.place("email", "not-an-email", "invalid_email", "flag")
    inj.place("email", "jane@@corp..com", "invalid_email", "flag")
    inj.place("email", "o'brien+crm@sub.domain.co.uk", "unusual_valid_email",
              "preserve", note="RFC-valid oddball address")
    inj.place("phone", "12", "invalid_phone", "flag")
    inj.place("phone", "  +1-555-0199  ", "whitespace_phone", "repair",
              repaired="+1-555-0199")
    inj.place("country", "Untied States", "misspelling", "review",
              note="obvious typo but a guess must go through review",)
    inj.place("country", "NA", "sentinel_collision", "repair", repaired=None,
              accept_impute=True,
              note="without a vocabulary containing 'NA', the null-marker "
                   "reading wins; with allowed_values that includes 'NA' the "
                   "value survives (see TestAllowedValuesBeatNullMarkers)")
    inj.place("country", "Namibia", "rare_valid_country", "preserve")
    inj.place("full_name", "Ýrsa Þorsteinsdóttir", "unicode_name", "preserve")
    inj.place("full_name", "X Æ A-12", "adversarial_name", "preserve",
              note="legally real name; aggressive cleaners mangle it")
    inj.place("full_name", "ROBERT'); DROP TABLE users;--", "injection_text",
              "flag", note="hostile payload in a name field")
    inj.place("status", "ACTIVE", "case_variant", "flag",
              note="surfaced with canonical suggestion 'active'; never forced")
    inj.place("status", "cancelled", "unknown_category", "flag")
    inj.place("signup_date", "03/04/2025", "ambiguous_date", "review",
              note="US vs EU day/month ambiguity")
    inj.place("customer_id", "C20001 ", "trailing_space_id", "repair",
              repaired="C20001")
    inj.place("notes", "Great customer 👍🎉", "emoji_text", "preserve")
    inj.place("notes", "<div>copied from web</div>", "html_fragment", "repair",
              repaired="copied from web")
    inj.place("notes", "très bien — merci béaucoup", "mixed_language", "preserve")
    inj.place("email", "USER42@EXAMPLE.COM", "case_variant_email", "preserve",
              note="uppercase emails are deliverable; do not force-lower silently")

    schema: dict[str, FieldSpec | str] = {
        "customer_id": FieldSpec(semantic_type="identifier", required=True),
        "full_name": FieldSpec(semantic_type="person_name"),
        "email": FieldSpec(semantic_type="email"),
        "phone": FieldSpec(semantic_type="phone"),
        "country": FieldSpec(allowed_values=_COUNTRIES,
                             suggest={"Untied States": "United States"}),
        "signup_date": FieldSpec(semantic_type="date"),
        "status": FieldSpec(allowed_values=frozenset({"active", "churned", "trial"})),
        "notes": FieldSpec(semantic_type="free_text"),
    }
    fx = GauntletFixture(
        name="crm", df=df, cells=inj.cells, schema=schema,
        field_types={"notes": "free_text", "full_name": "person_name",
                     "email": "email", "phone": "phone"},
        n_rows=n,
    )
    return _append_duplicates(fx, rows=4, seed=seed + 2)


# ---------------------------------------------------------------------------
# ecommerce
# ---------------------------------------------------------------------------


def _ecommerce(n: int, seed: int) -> GauntletFixture:
    r = _rng(seed + 3)
    products = ("USB-C Cable", "Desk Lamp", "Notebook", "Water Bottle",
                "Backpack", "Monitor Stand")
    df = pd.DataFrame({
        "order_id": [f"ORD-{700000 + i}" for i in range(n)],
        "sku": [f"SKU-{r.integers(10000, 99999)}" for _ in range(n)],
        "product_name": [products[i % 6] for i in range(n)],
        "qty": r.integers(1, 12, n).astype(object),
        "unit_price": np.round(r.uniform(3, 250, n), 2).astype(object),
        "discount_pct": np.round(r.uniform(0, 40, n), 1).astype(object),
        "order_date": (pd.to_datetime("2025-06-01")
                       + pd.to_timedelta(r.integers(0, 200, n), unit="D")
                       ).strftime("%Y-%m-%d"),
        "review": [f"Arrived on time; order {i} matched the listing." for i in range(n)],
    })
    inj = _Injector(df, seed + 3)

    inj.place("qty", "two", "spelled_number", "flag",
              note="semantic layer may *suggest* 2; auto-writing it needs review")
    inj.place("qty", -3, "impossible_range", "flag")
    inj.place("qty", "3 pcs", "unit_suffix", "review", note="repairable to 3 via review")
    inj.place("unit_price", "€49.99", "currency_symbol", "repair", repaired=49.99)
    inj.place("unit_price", "12,99", "decimal_comma", "review")
    inj.place("unit_price", 0.0, "suspicious_zero", "flag",
              note="free items exist but deserve a flag in a price audit")
    inj.place("discount_pct", 150.0, "impossible_range", "flag")
    inj.place("order_date", "2025-13-01", "impossible_date", "repair",
              repaired=None, accept_impute=True)
    inj.place("sku", "sku-1234", "case_variant_id", "review")
    inj.place("product_name", "Café Press — 12″ (limited)", "unicode_product",
              "preserve")
    inj.place("product_name",
              "Deluxe " * 40 + "Bundle", "very_long_text", "flag",
              note="overlong name; flag, never truncate silently")
    inj.place("review", "GREAT!!!!!!! 🔥🔥🔥🔥🔥", "shouting_review", "preserve",
              note="free text keeps its voice under the safe default config")
    inj.place("review", "b&#8217;uy n&#8217;ow", "html_entities", "repair",
              repaired="b’uy n’ow")
    inj.place("review", "Visit http://spam.example.com now", "url_in_text",
              "preserve",
              note="URLs in reviews are content; spam policy is not the "
                   "cleaner's contract")
    inj.place("review", "", "empty_string", "preserve",
              note="empty free text: canonically missing either way; text-role "
                   "columns are never filled with fabricated content")

    schema: dict[str, FieldSpec | str] = {
        "order_id": FieldSpec(semantic_type="identifier", required=True),
        "sku": FieldSpec(semantic_type="identifier",
                         pattern=r"SKU-\d{5}"),
        "product_name": FieldSpec(semantic_type="entity_name", max_length=120),
        "qty": FieldSpec(semantic_type="numeric", min_value=0),
        "unit_price": FieldSpec(semantic_type="currency_amount", min_value=0.01),
        "discount_pct": FieldSpec(semantic_type="percentage",
                                  min_value=0.0, max_value=100.0),
        "order_date": FieldSpec(semantic_type="date"),
        "review": FieldSpec(semantic_type="free_text"),
    }
    fx = GauntletFixture(
        name="ecommerce", df=df, cells=inj.cells, schema=schema,
        field_types={"review": "free_text", "product_name": "entity_name"},
        n_rows=n,
    )
    return _append_duplicates(fx, rows=3, seed=seed + 3)


# ---------------------------------------------------------------------------
# text (adversarial free text)
# ---------------------------------------------------------------------------


def _text(n: int, seed: int) -> GauntletFixture:
    r = _rng(seed + 4)
    df = pd.DataFrame({
        "doc_id": [f"{i:03d}" for i in range(n)],
        "category": r.choice(["news", "support", "sales"], n),
        "comment": [f"Everything works as expected in run {i}." for i in range(n)],
    })
    inj = _Injector(df, seed + 4)

    inj.place("comment", "zero\u200bwidth‌joined", "zero_width", "repair",
              repaired="zerowidthjoined")
    inj.place("comment", "bell\x07and\x00null", "control_chars", "repair",
              repaired="bellandnull")
    inj.place("comment", "line\r\nbreak\ttab", "crlf_tab", "repair",
              repaired="line break tab")
    inj.place("comment", "  spaced   out  ", "whitespace", "repair",
              repaired="spaced out")
    inj.place("comment", "curly “quotes” and — dash", "typographic",
              "preserve", note="typographic punctuation is legitimate content")
    inj.place("comment", "ＦＵＬＬＷＩＤＴＨ ｔｅｘｔ", "fullwidth", "repair",
              repaired="FULLWIDTH text",
              note="safe default (NFC) keeps fullwidth forms; the opt-in NFKC "
                   "pass folds them")
    inj.place("comment", "cafÃ© au lait", "mojibake", "flag",
              note="classic UTF-8-as-Latin-1; detection wanted, silent guess not")
    inj.place("comment", "sooooo cooool!!!!!!!!", "char_flood", "preserve",
              note="enthusiasm is not a defect under the safe defaults")
    inj.place("comment", "नमस्ते + hello + שלום", "mixed_scripts", "preserve")
    inj.place("comment", "🙂🙂🙂", "emoji_only", "preserve")
    inj.place("comment", "<script>alert(1)</script>ok", "script_tag", "repair",
              repaired="ok",
              note="hostile HTML; kept under the safe default, stripped (with "
                   "script content dropped) only under the opt-in HTML pass")
    inj.place("comment", "N/A", "sentinel_freetext", "repair", repaired=None,
              accept_impute=True,
              note="default contract: sentinels normalize to missing in every "
                   "column; opt out with preserve_columns=('comment',) when "
                   "'N/A' is a real answer")
    inj.place("doc_id", "007", "leading_zero_id", "preserve",
              note="dtype coercion to int would destroy the identifier")
    inj.place("doc_id", "1e5", "scientific_lookalike", "preserve")
    inj.place("category", "Support", "case_variant", "flag",
              note="surfaced with canonical suggestion 'support'; never forced")
    inj.place("category", "spam", "unknown_category", "flag")

    schema: dict[str, FieldSpec | str] = {
        "doc_id": FieldSpec(semantic_type="identifier", required=True),
        "category": FieldSpec(allowed_values=frozenset({"news", "support", "sales"})),
        "comment": FieldSpec(semantic_type="free_text"),
    }
    fx = GauntletFixture(
        name="text", df=df, cells=inj.cells, schema=schema,
        field_types={"comment": "free_text"}, n_rows=n,
    )
    return _append_duplicates(fx, rows=2, seed=seed + 4)


# ---------------------------------------------------------------------------


def _append_duplicates(fx: GauntletFixture, *, rows: int, seed: int) -> GauntletFixture:
    """Duplicate ``rows`` untouched records at the end of the frame."""
    labelled_rows = {c.row for c in fx.cells}
    clean_rows = [i for i in range(len(fx.df)) if i not in labelled_rows]
    picks = list(np.random.default_rng(seed + 99).choice(clean_rows, rows, replace=False))
    dup = fx.df.iloc[picks].copy()
    fx.df = pd.concat([fx.df, dup], ignore_index=True)
    fx.dup_row_count = rows
    return fx


FIXTURES: dict[str, Callable[[int, int], GauntletFixture]] = {
    "finance": _finance,
    "healthcare": _healthcare,
    "crm": _crm,
    "ecommerce": _ecommerce,
    "text": _text,
}


def build_fixture(name: str, n_rows: int = DEFAULT_ROWS,
                  seed: int = DEFAULT_SEED) -> GauntletFixture:
    try:
        builder = FIXTURES[name]
    except KeyError:
        raise KeyError(f"unknown gauntlet fixture {name!r}; "
                       f"available: {sorted(FIXTURES)}") from None
    return builder(n_rows, seed)
