"""Deterministic synthetic dataset generator for benchmarks.

Generates realistic tabular data with controllable data quality issues.
All generation uses fixed random seeds for reproducibility across runs.

Usage::

    from benchmarks.datasets import get_dataset
    df = get_dataset(100_000)  # 100K rows, cached
"""

from __future__ import annotations

import functools
import string

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
SENTINEL_VALUES = ["N/A", "-", "", "null", "NA", "#REF!", "missing", "n/a", "None"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "company.org", "test.io"]
FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj", "Oscar", "Peggy",
    "Quinn", "Rupert", "Sybil", "Trent", "Ursula", "Victor", "Wendy",
]
CATEGORIES_50 = [f"category_{i:03d}" for i in range(50)]
URL_PREFIXES = ["http://", "https://", "ftp://", ""]
CURRENCY_SYMBOLS = ["$", "€", "£", "¥", ""]

# Dataset size tiers
SIZES = {
    "tiny": 10_000,
    "small": 100_000,
    "medium": 1_000_000,
    "large": 5_000_000,
    "xlarge": 10_000_000,
}


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


def generate_dataset(n_rows: int, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic messy dataset with *n_rows* rows.

    The dataset contains 20+ columns with realistic data quality issues:
    missing values, duplicates, wrong dtypes, whitespace, case
    inconsistencies, Unicode, outliers, invalid patterns, and more.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A deliberately messy DataFrame suitable for cleaning benchmarks.
    """
    rng = np.random.default_rng(seed)

    # --- Unique ID column ---
    ids = np.arange(1, n_rows + 1)

    # --- Numeric columns with NaN and outliers ---
    float_col_1 = _numeric_with_issues(rng, n_rows, loc=100.0, scale=25.0,
                                        nan_rate=0.12, outlier_rate=0.02)
    float_col_2 = _numeric_with_issues(rng, n_rows, loc=50.0, scale=15.0,
                                        nan_rate=0.18, outlier_rate=0.015)
    float_col_3 = _skewed_numeric(rng, n_rows, nan_rate=0.10)

    int_col_1 = _integer_with_contamination(rng, n_rows, low=0, high=1000,
                                             nan_rate=0.08, contam_rate=0.03)
    int_col_2 = _integer_with_contamination(rng, n_rows, low=1, high=100,
                                             nan_rate=0.05, contam_rate=0.02)

    # --- String columns ---
    string_col = _messy_strings(rng, n_rows, FIRST_NAMES,
                                 whitespace_rate=0.15, case_rate=0.20,
                                 sentinel_rate=0.05)

    # --- High-cardinality categorical ---
    category_col = _messy_categorical(rng, n_rows, CATEGORIES_50,
                                       nan_rate=0.10)

    # --- Date column with mixed formats and invalid values ---
    date_col = _messy_dates(rng, n_rows, invalid_rate=0.05, nan_rate=0.08)

    # --- Boolean column with mixed representations ---
    bool_col = _messy_booleans(rng, n_rows, nan_rate=0.06)

    # --- Email column with invalid entries ---
    email_col = _messy_emails(rng, n_rows, invalid_rate=0.10, nan_rate=0.05)

    # --- Phone column with mixed formats ---
    phone_col = _messy_phones(rng, n_rows, invalid_rate=0.08, nan_rate=0.07)

    # --- URL column ---
    url_col = _messy_urls(rng, n_rows, invalid_rate=0.10, nan_rate=0.06)

    # --- Currency column ---
    currency_col = _messy_currency(rng, n_rows, nan_rate=0.08)

    # --- Unicode column ---
    unicode_col = _messy_unicode(rng, n_rows, nan_rate=0.05)

    # --- Constant column (all same value) ---
    constant_col = np.full(n_rows, "CONSTANT_VALUE", dtype=object)

    # --- Null-heavy column (90%+ null) ---
    null_heavy = _null_heavy_column(rng, n_rows, null_rate=0.92)

    # --- Mixed-type column (ints, floats, strings) ---
    mixed_type = _mixed_type_column(rng, n_rows)

    # --- Negative values column ---
    negative_col = rng.normal(-50.0, 30.0, n_rows)
    neg_mask = rng.random(n_rows) < 0.10
    negative_col = negative_col.astype(object)
    negative_col[neg_mask] = None

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "id": ids,
        "float_col_1": float_col_1,
        "float_col_2": float_col_2,
        "float_col_3": float_col_3,
        "int_col_1": int_col_1,
        "int_col_2": int_col_2,
        " String Col ": string_col,       # intentional whitespace in name
        "CATEGORY_COL": category_col,      # intentional uppercase name
        "date_col": date_col,
        "bool_col": bool_col,
        "email_col": email_col,
        "phone_col": phone_col,
        "url_col": url_col,
        "currency_col": currency_col,
        "unicode_col": unicode_col,
        "constant_col": constant_col,
        "null_heavy_col": null_heavy,
        "mixed_type_col": mixed_type,
        "negative_col": negative_col,
    })

    # --- Inject duplicate rows (~5% of dataset) ---
    n_dupes = max(1, int(n_rows * 0.05))
    dupe_idx = rng.choice(n_rows, size=n_dupes, replace=True)
    dupes = df.iloc[dupe_idx].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    # Shuffle to distribute duplicates
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Column generators (private)
# ---------------------------------------------------------------------------


def _numeric_with_issues(
    rng: np.random.Generator,
    n: int,
    loc: float,
    scale: float,
    nan_rate: float,
    outlier_rate: float,
) -> np.ndarray:
    """Normal-distributed floats with NaN and extreme outliers."""
    values = rng.normal(loc, scale, n).astype(object)
    # Inject outliers (8–15× the value)
    outlier_mask = rng.random(n) < outlier_rate
    outlier_count = outlier_mask.sum()
    if outlier_count > 0:
        values[outlier_mask] = (
            rng.normal(loc, scale, outlier_count) * rng.uniform(8, 15, outlier_count)
        )
    # Inject NaN
    nan_mask = rng.random(n) < nan_rate
    values[nan_mask] = None
    return values


def _skewed_numeric(
    rng: np.random.Generator, n: int, nan_rate: float
) -> np.ndarray:
    """Log-normal (right-skewed) distribution with NaN."""
    values = rng.lognormal(mean=3.0, sigma=1.0, size=n).astype(object)
    values[rng.random(n) < nan_rate] = None
    return values


def _integer_with_contamination(
    rng: np.random.Generator,
    n: int,
    low: int,
    high: int,
    nan_rate: float,
    contam_rate: float,
) -> np.ndarray:
    """Integers stored as strings, with some non-numeric contamination."""
    values = rng.integers(low, high, n).astype(str).astype(object)
    # Contaminate with non-numeric strings
    contam_mask = rng.random(n) < contam_rate
    contam_count = contam_mask.sum()
    if contam_count > 0:
        values[contam_mask] = rng.choice(["abc", "N/A", "??", "--", ""], contam_count)
    # Inject NaN
    values[rng.random(n) < nan_rate] = None
    return values


def _messy_strings(
    rng: np.random.Generator,
    n: int,
    base_values: list[str],
    whitespace_rate: float,
    case_rate: float,
    sentinel_rate: float,
) -> np.ndarray:
    """Strings with whitespace, case inconsistencies, and sentinels."""
    values = rng.choice(base_values, n).astype(object)
    # Add whitespace
    ws_mask = rng.random(n) < whitespace_rate
    for i in np.where(ws_mask)[0]:
        pad_left = " " * rng.integers(1, 4)
        pad_right = " " * rng.integers(1, 4)
        values[i] = f"{pad_left}{values[i]}{pad_right}"
    # Random case
    case_mask = rng.random(n) < case_rate
    for i in np.where(case_mask)[0]:
        choice = rng.integers(0, 3)
        if choice == 0:
            values[i] = str(values[i]).upper()
        elif choice == 1:
            values[i] = str(values[i]).lower()
        else:
            values[i] = str(values[i]).title()
    # Replace some with sentinels
    sent_mask = rng.random(n) < sentinel_rate
    values[sent_mask] = rng.choice(SENTINEL_VALUES, sent_mask.sum())
    return values


def _messy_categorical(
    rng: np.random.Generator,
    n: int,
    categories: list[str],
    nan_rate: float,
) -> np.ndarray:
    """High-cardinality categorical with NaN."""
    values = rng.choice(categories, n).astype(object)
    values[rng.random(n) < nan_rate] = None
    return values


def _messy_dates(
    rng: np.random.Generator,
    n: int,
    invalid_rate: float,
    nan_rate: float,
) -> np.ndarray:
    """Date strings in mixed formats with invalid entries."""
    base_dates = pd.date_range("2020-01-01", periods=n, freq="h")
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d %H:%M:%S"]
    values = np.empty(n, dtype=object)
    for i in range(n):
        fmt = formats[i % len(formats)]
        values[i] = base_dates[i % len(base_dates)].strftime(fmt)
    # Invalid dates
    inv_mask = rng.random(n) < invalid_rate
    inv_values = ["2020-13-45", "not-a-date", "31/02/2020", "00/00/0000",
                  "2020-02-30", "abc", "9999-99-99"]
    values[inv_mask] = rng.choice(inv_values, inv_mask.sum())
    # NaN
    values[rng.random(n) < nan_rate] = None
    return values


def _messy_booleans(
    rng: np.random.Generator, n: int, nan_rate: float
) -> np.ndarray:
    """Booleans with mixed representations."""
    representations = ["yes", "no", "Yes", "No", "YES", "NO", "true", "false",
                       "True", "False", "TRUE", "FALSE", "1", "0", "Y", "N",
                       "y", "n", "T", "F"]
    values = rng.choice(representations, n).astype(object)
    values[rng.random(n) < nan_rate] = None
    return values


def _messy_emails(
    rng: np.random.Generator, n: int, invalid_rate: float, nan_rate: float
) -> np.ndarray:
    """Email addresses with invalid entries."""
    values = np.empty(n, dtype=object)
    for i in range(n):
        user_len = rng.integers(5, 12)
        user = "".join(rng.choice(list(string.ascii_lowercase), user_len))
        domain = EMAIL_DOMAINS[i % len(EMAIL_DOMAINS)]
        values[i] = f"{user}@{domain}"
    # Invalidate some
    inv_mask = rng.random(n) < invalid_rate
    invalid_emails = ["notanemail", "missing@", "@nodomain.com", "spaces in@email.com",
                      "double@@at.com", "no.tld@", ""]
    values[inv_mask] = rng.choice(invalid_emails, inv_mask.sum())
    values[rng.random(n) < nan_rate] = None
    return values


def _messy_phones(
    rng: np.random.Generator, n: int, invalid_rate: float, nan_rate: float
) -> np.ndarray:
    """Phone numbers in mixed formats."""
    values = np.empty(n, dtype=object)
    formats_tpl = [
        "{area}-{pre}-{suf}", "({area}) {pre}-{suf}",
        "+1{area}{pre}{suf}", "{area}.{pre}.{suf}",
        "{area}{pre}{suf}",
    ]
    for i in range(n):
        area = f"{rng.integers(200, 999)}"
        pre = f"{rng.integers(200, 999)}"
        suf = f"{rng.integers(1000, 9999)}"
        fmt = formats_tpl[i % len(formats_tpl)]
        values[i] = fmt.format(area=area, pre=pre, suf=suf)
    inv_mask = rng.random(n) < invalid_rate
    invalid_phones = ["123", "abc-def-ghij", "00000000000000", "+", "555"]
    values[inv_mask] = rng.choice(invalid_phones, inv_mask.sum())
    values[rng.random(n) < nan_rate] = None
    return values


def _messy_urls(
    rng: np.random.Generator, n: int, invalid_rate: float, nan_rate: float
) -> np.ndarray:
    """URLs with invalid entries."""
    tlds = ["com", "org", "net", "io", "dev"]
    values = np.empty(n, dtype=object)
    for i in range(n):
        prefix = URL_PREFIXES[i % len(URL_PREFIXES)]
        domain_len = rng.integers(4, 10)
        domain = "".join(rng.choice(list(string.ascii_lowercase), domain_len))
        tld = tlds[i % len(tlds)]
        values[i] = f"{prefix}{domain}.{tld}/path"
    inv_mask = rng.random(n) < invalid_rate
    invalid_urls = ["not a url", "://missing-scheme", "http://", "ftp:///",
                    "just-text", "http://a b c.com"]
    values[inv_mask] = rng.choice(invalid_urls, inv_mask.sum())
    values[rng.random(n) < nan_rate] = None
    return values


def _messy_currency(
    rng: np.random.Generator, n: int, nan_rate: float
) -> np.ndarray:
    """Currency strings in mixed formats."""
    values = np.empty(n, dtype=object)
    for i in range(n):
        amount = rng.uniform(1.0, 99999.99)
        symbol = CURRENCY_SYMBOLS[i % len(CURRENCY_SYMBOLS)]
        # Randomly format with/without commas
        if rng.random() < 0.5:
            values[i] = f"{symbol}{amount:,.2f}"
        else:
            values[i] = f"{symbol}{amount:.2f}"
    values[rng.random(n) < nan_rate] = None
    return values


def _messy_unicode(
    rng: np.random.Generator, n: int, nan_rate: float
) -> np.ndarray:
    """Strings with mixed Unicode normalization and accented chars."""
    base_words = [
        "café", "naïve", "résumé", "über", "señor",
        "Ångström", "Zürich", "São Paulo", "Malmö", "日本語",
        "中文", "한국어", "العربية", "हिन्दी", "normal_ascii",
    ]
    values = rng.choice(base_words, n).astype(object)
    values[rng.random(n) < nan_rate] = None
    return values


def _null_heavy_column(
    rng: np.random.Generator, n: int, null_rate: float
) -> np.ndarray:
    """Column that is almost entirely null."""
    values = rng.normal(0, 1, n).astype(object)
    values[rng.random(n) < null_rate] = None
    return values


def _mixed_type_column(rng: np.random.Generator, n: int) -> np.ndarray:
    """Column with mixed types: ints, floats, strings."""
    values = np.empty(n, dtype=object)
    for i in range(n):
        choice = i % 4
        if choice == 0:
            values[i] = int(rng.integers(0, 1000))
        elif choice == 1:
            values[i] = float(rng.normal(50, 10))
        elif choice == 2:
            values[i] = f"text_{rng.integers(0, 100)}"
        else:
            values[i] = None
    return values
