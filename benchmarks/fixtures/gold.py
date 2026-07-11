"""Fixture 6 — Gold-labelled repair-fidelity fixture.

Ground truth for the repair-fidelity (Metric 3), false-repair-rate (Metric 4)
and preservation (Metric 5) metrics. Every cell's correct post-clean value is
known, derived *analytically* from first principles (never by running the
library under test), so the metrics measure FreshData against an independent
oracle.

``generate(n_rows=10_000, seed=42, defect_rate=None)`` returns a
:class:`GoldBundle` with:

* ``dirty_df`` — input with injected defects (includes appended duplicate rows).
* ``clean_df`` — expected post-clean output (deduped, repaired), one row per
  surviving record, in the order FreshData's ``drop_duplicates(keep="first")``
  yields (the original records in order).
* ``preservation_mask`` — True where a protected (id/target/text) cell must be
  byte-identical between input and output.
* ``repair_mask`` — True where a cell must be changed to a known correct value.
* ``false_repair_traps`` — True where changing the cell is a *failure*
  (protected cells, including nulls; target NaNs; whitespace-only text).

The masks all share ``clean_df``'s shape (the deduplicated frame). The harness
compares ``fd.clean(dirty_df)`` (which drops the duplicates) against
``clean_df`` cell-by-cell under these masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._common import (
    ROLE_CATEGORICAL,
    ROLE_DATETIME,
    ROLE_ID,
    ROLE_NUMERIC,
    ROLE_TARGET,
    ROLE_TEXT,
    Defect,
    GoldLabel,
    defect_mask,
    format_iso_date,
    gold_to_records,
    manifest_to_records,
    pick,
    resolve_rate,
    rng_for,
    uuid_series,
)

# Only tokens FreshData's normalize_sentinels recognises (see
# freshdata._sentinels.DEFAULT_SENTINELS). "999" is excluded on purpose — the
# library keeps it as a number, so it is not an in-scope sentinel defect.
_NUM_SENTINELS = ("N/A", "null", "--", "n.a.")
_CAT_POOL = ("north", "south", "east", "west")
_WORDS = ("lorem", "ipsum", "dolor", "sit", "amet", "consect", "adipisc", "elit")

ID_COLUMNS = ("gid",)
TARGET_COLUMN = "gtarget"
TEXT_COLUMNS = ("gtext",)
PROTECTED = ("gid", "gtarget", "gtext")
DEFAULT_ROWS = 10_000


@dataclass
class GoldBundle:
    dirty_df: pd.DataFrame
    clean_df: pd.DataFrame
    preservation_mask: pd.DataFrame
    repair_mask: pd.DataFrame
    false_repair_traps: pd.DataFrame
    fills: dict[str, Any]
    n_duplicates: int


def _free_text(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.array(
        [" ".join(rng.choice(_WORDS, size=5)) + f" {rng.integers(0, 10**7)}" for _ in range(n)],
        dtype=object,
    )


def generate(n_rows: int = DEFAULT_ROWS, seed: int = 42, defect_rate: float | None = None) -> GoldBundle:
    rng = rng_for(seed)
    n = int(n_rows)

    def rate(base: float) -> float:
        return resolve_rate(base, defect_rate)

    # -- clean base columns (the oracle, before defects) -------------------
    gid = np.array(uuid_series(rng, n, prefix="g-"), dtype=object)
    gtarget = np.round(rng.normal(10.0, 3.0, size=n), 5)
    gtext = _free_text(rng, n)
    # skewed numerics so the engine's robust default is the median
    num_sent_vals = np.round(rng.lognormal(4.0, 0.9, size=n), 2)
    num_miss_vals = np.round(rng.lognormal(3.5, 0.8, size=n), 2)
    # Skewed so a single modal value is >= 50% of non-missing: this puts the
    # categorical sentinel repair on FreshData's deterministic mode-fill path
    # (engine.missing fills the mode only when mode_ratio >= 0.5, else "Unknown").
    gcat = rng.choice(_CAT_POOL, size=n, p=[0.55, 0.20, 0.15, 0.10]).astype(object)
    base = np.datetime64("2022-01-01")
    gdate_days = rng.integers(0, 900, size=n)
    gdate_dt = base + gdate_days.astype("timedelta64[D]")

    clean = pd.DataFrame(
        {
            "gid": gid.copy(),
            "gtarget": gtarget.copy(),
            "gtext": gtext.copy(),
            "gnum_sentinel": num_sent_vals.astype(float).copy(),
            "gnum_missing": num_miss_vals.astype(float).copy(),
            "gcat_sentinel": gcat.copy(),
            "gdate": pd.to_datetime(pd.Series(gdate_dt)),
        }
    )

    # dirty starts as a copy with string-typed numeric/date columns
    dirty = pd.DataFrame(
        {
            "gid": gid.copy(),
            "gtarget": gtarget.copy(),
            "gtext": gtext.copy(),
            "gnum_sentinel": np.array([f"{v:.2f}" for v in num_sent_vals], dtype=object),
            "gnum_missing": num_miss_vals.astype(float).copy(),
            "gcat_sentinel": gcat.copy().astype(object),
            "gdate": np.array([str(d) for d in gdate_dt], dtype=object),
        }
    )

    repair = pd.DataFrame(False, index=clean.index, columns=clean.columns)
    preserve = pd.DataFrame(False, index=clean.index, columns=clean.columns)
    traps = pd.DataFrame(False, index=clean.index, columns=clean.columns)

    # protected columns: preserve + trap everywhere
    for c in PROTECTED:
        preserve[c] = True
        traps[c] = True

    # -- defect: null IDs (preservation trap — must stay null) -------------
    m = defect_mask(rng, n, rate(0.01))
    dirty.loc[m, "gid"] = None
    clean.loc[m, "gid"] = None  # oracle: still null

    # -- defect: target NaNs (must never be imputed) ----------------------
    m = defect_mask(rng, n, rate(0.05))
    dirty.loc[m, "gtarget"] = np.nan
    clean.loc[m, "gtarget"] = np.nan  # oracle: still NaN

    # -- defect: missing free text (must never be force-filled) ------------
    # NB: whitespace-only text is *not* used here — FreshData legitimately
    # strips surrounding whitespace (in-scope representation repair), so a
    # "   " cell correctly becomes empty/NA and is not a false-repair trap.
    # The real free-text contract is "never force-filled": a missing text cell
    # must stay missing.
    m = defect_mask(rng, n, rate(0.01))
    dirty.loc[m, "gtext"] = np.nan
    clean.loc[m, "gtext"] = np.nan  # oracle: still missing, not fabricated

    # -- repair: numeric sentinels -> NaN -> median fill -------------------
    m = defect_mask(rng, n, rate(0.08))
    sent = pick(rng, _NUM_SENTINELS, int(m.sum()))
    dvals = dirty["gnum_sentinel"].to_numpy(copy=True)
    dvals[m] = sent
    dirty["gnum_sentinel"] = dvals
    # oracle fill: median over the *valid* numeric values (post-normalisation)
    valid = num_sent_vals[~m]
    fill_sent = float(np.median(valid))
    clean.loc[m, "gnum_sentinel"] = fill_sent
    repair.loc[m, "gnum_sentinel"] = True

    # -- repair: numeric missing -> median fill ---------------------------
    m = defect_mask(rng, n, rate(0.04))
    dirty.loc[m, "gnum_missing"] = np.nan
    valid2 = num_miss_vals[~m]
    fill_miss = float(np.median(valid2))
    clean.loc[m, "gnum_missing"] = fill_miss
    repair.loc[m, "gnum_missing"] = True

    # -- repair: categorical sentinels -> normalized to mode --------------
    # 4% keeps the column in the low-missingness band (<= 5%) where the
    # mode-fill rule applies.
    m = defect_mask(rng, n, rate(0.04))
    dvals = dirty["gcat_sentinel"].to_numpy(copy=True)
    dvals[m] = pick(rng, _NUM_SENTINELS, int(m.sum()))
    dirty["gcat_sentinel"] = dvals
    valid_cat = pd.Series(gcat[~m])
    fill_cat = valid_cat.mode().iloc[0]
    clean.loc[m, "gcat_sentinel"] = fill_cat
    repair.loc[m, "gcat_sentinel"] = True

    # -- repair: datetime dtype coercion (string -> datetime64) -----------
    # The whole gdate column arrives as uniform ISO date *strings*; the repair
    # is the dtype coercion to datetime64. A single, unambiguous format is used
    # on purpose: FreshData coerces a date column with one inferred format and
    # would NaT a minority format, so mixed-format coercion is out of scope.
    # When defect_rate == 0 there is no defect and the column is already typed.
    coerce_dates = defect_rate is None or float(defect_rate) > 0.0
    if coerce_dates:
        dirty["gdate"] = np.array(
            [format_iso_date(d, "%Y-%m-%d") for d in gdate_dt], dtype=object
        )
        repair["gdate"] = True  # clean already holds the parsed Timestamp
    else:
        dirty["gdate"] = clean["gdate"]

    fills = {
        "gnum_sentinel_median": fill_sent,
        "gnum_missing_median": fill_miss,
        "gcat_sentinel_mode": fill_cat,
    }

    # -- inject exact duplicate rows (dropped on clean) -------------------
    k = int(round(rate(0.02) * n))
    if k:
        dup_idx = rng.permutation(n)[:k]
        dirty = pd.concat([dirty, dirty.iloc[dup_idx].copy()], ignore_index=True)

    return GoldBundle(
        dirty_df=dirty.reset_index(drop=True),
        clean_df=clean.reset_index(drop=True),
        preservation_mask=preserve.reset_index(drop=True),
        repair_mask=repair.reset_index(drop=True),
        false_repair_traps=traps.reset_index(drop=True),
        fills=fills,
        n_duplicates=k,
    )


GOLD_LABELS: dict[str, dict[str, Any]] = gold_to_records(
    {
        "gid": GoldLabel(ROLE_ID, "object", "preserve", True),
        "gtarget": GoldLabel(ROLE_TARGET, "float64", "preserve", True),
        "gtext": GoldLabel(ROLE_TEXT, "object", "preserve", True),
        "gnum_sentinel": GoldLabel(ROLE_NUMERIC, "float64", "median_fill", False),
        "gnum_missing": GoldLabel(ROLE_NUMERIC, "float64", "median_fill", False),
        "gcat_sentinel": GoldLabel(ROLE_CATEGORICAL, "object", "sentinel_normalize", False),
        "gdate": GoldLabel(ROLE_DATETIME, "datetime64[ns]", "coerce_datetime", False),
    }
)

DEFECT_MANIFEST: list[dict[str, Any]] = manifest_to_records(
    [
        Defect("gold-null-id", "gid", "null_id", 0.01, "preserve", preservation=True),
        Defect("gold-target-nan", "gtarget", "target_missing", 0.05, "preserve", preservation=True),
        Defect("gold-missing-text", "gtext", "missing_text", 0.01, "preserve", preservation=True),
        Defect("gold-num-sentinel", "gnum_sentinel", "numeric_sentinel", 0.08, "median_fill"),
        Defect("gold-num-missing", "gnum_missing", "missing_numeric", 0.04, "median_fill"),
        Defect("gold-cat-sentinel", "gcat_sentinel", "categorical_sentinel", 0.04, "sentinel_normalize"),
        Defect("gold-date-coerce", "gdate", "string_typed_date", 1.0, "dtype_coerce"),
        Defect("gold-dupes", "*", "exact_duplicate_row", 0.02, "drop_duplicate"),
    ]
)

SCALE_VARIANTS = (10_000,)
N_COLS = 7
