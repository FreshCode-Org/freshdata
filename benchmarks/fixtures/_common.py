"""Shared helpers for the enterprise fixture generators.

Every fixture in this package is *deterministic*: the same ``(n_rows, seed,
defect_rate)`` triple always yields byte-for-byte identical output. That is what
makes the benchmark harness reproducible across machines (see
``benchmarks/README.md``).

Two conventions hold across all fixtures:

* ``generate(n_rows, seed=42, defect_rate=None)`` returns a ``pandas.DataFrame``.
  When ``defect_rate is None`` each defect family is injected at its
  *documented* base rate (the rate recorded in that fixture's
  ``DEFECT_MANIFEST``). When ``defect_rate`` is a float in ``[0, 1]`` every
  defect family is injected at that single uniform rate instead — this is the
  knob the trust-monotonicity metric (Metric 8) sweeps over.
* ``GOLD_LABELS`` records, per column, the expected post-clean ``role``,
  ``dtype``, ``fill_action`` and whether the column must be ``preserved``.
* ``DEFECT_MANIFEST`` is a list of :class:`Defect` records the harness uses to
  attribute repair fidelity to a known ground-truth defect family.

None of these helpers import :mod:`freshdata`; fixtures describe *data*, the
harness is the only thing that calls the library under test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# -- roles -----------------------------------------------------------------
# These mirror the roles FreshData's decision engine infers
# (freshdata.engine.context): id / target / text columns are never mutated,
# everything else is fair game for repair.
ROLE_ID = "id"
ROLE_TARGET = "target"
ROLE_TEXT = "text"
ROLE_NUMERIC = "numeric"
ROLE_CATEGORICAL = "categorical"
ROLE_DATETIME = "datetime"
ROLE_BOOL = "bool"

PROTECTED_ROLES = frozenset({ROLE_ID, ROLE_TARGET, ROLE_TEXT})

# Sentinel tokens FreshData's normalize_sentinels step recognises. Keep this in
# sync with what the library actually treats as missing; injecting a token the
# library does *not* recognise would be testing out-of-scope behaviour, which
# the benchmark must never do (see HARD CONSTRAINT 5).
SENTINELS = ("N/A", "null", "--", "n.a.", "999", "NULL", "NaN", "")


@dataclass
class Defect:
    """One documented defect family injected into a fixture column.

    ``in_scope_repair`` names the FreshData repair the harness expects to fix
    this defect ("sentinel_normalize", "dtype_coerce", "median_fill",
    "drop_duplicate", "reference_flag", ...) or the literal ``"preserve"`` when
    the correct behaviour is to leave the value untouched and merely flag it.
    """

    id: str
    column: str
    defect_type: str
    rate: float
    in_scope_repair: str
    preservation: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GoldLabel:
    """Per-column post-clean expectation."""

    role: str
    expected_dtype: str
    fill_action: str
    preserve: bool
    reference: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["reference"] is None:
            d.pop("reference")
        return d


def rng_for(seed: int) -> np.random.Generator:
    """A NumPy generator seeded deterministically."""
    return np.random.default_rng(seed)


def defect_mask(
    rng: np.random.Generator, n: int, rate: float
) -> np.ndarray:
    """A boolean mask selecting ``round(rate * n)`` rows deterministically.

    Uses a fixed permutation rather than per-row Bernoulli draws so the *count*
    of injected defects is exact (``DEFECT_MANIFEST`` validation asserts counts
    within tolerance) and stable regardless of ``n``.
    """
    rate = max(0.0, min(1.0, float(rate)))
    k = int(round(rate * n))
    mask = np.zeros(n, dtype=bool)
    if k:
        idx = rng.permutation(n)[:k]
        mask[idx] = True
    return mask


def resolve_rate(base_rate: float, defect_rate: float | None) -> float:
    """Pick the injection rate for a defect family.

    ``defect_rate is None`` -> use the documented base rate. Otherwise every
    family is injected at the uniform ``defect_rate`` (the Metric 8 sweep).
    """
    return base_rate if defect_rate is None else float(defect_rate)


def uuid_series(rng: np.random.Generator, n: int, prefix: str = "") -> list[str]:
    """Deterministic UUID-like id strings (not RFC4122, but stable & unique)."""
    hi = rng.integers(0, 2**32, size=n, dtype=np.uint64)
    lo = rng.integers(0, 2**32, size=n, dtype=np.uint64)
    out = []
    for h, l in zip(hi.tolist(), lo.tolist()):
        out.append(f"{prefix}{h:08x}-{l:08x}")
    return out


def manifest_to_records(defects: list[Defect]) -> list[dict[str, Any]]:
    return [d.as_dict() for d in defects]


def gold_to_records(labels: dict[str, GoldLabel]) -> dict[str, dict[str, Any]]:
    return {col: lbl.as_dict() for col, lbl in labels.items()}


# -- reference category pools ---------------------------------------------
COUNTRY_REF = ("US", "CA", "GB", "DE", "FR", "AU")
ACCOUNT_STATUS_REF = ("active", "inactive", "suspended")
CURRENCY_REF = ("USD", "EUR", "GBP", "JPY", "CHF")
DEBIT_CREDIT_REF = ("D", "C", "DR", "CR", "debit", "credit")
OPERATION_REF = ("INSERT", "UPDATE", "DELETE", "UPSERT")

# Tokens deliberately outside the approved reference sets, used to inject
# reference-set violations the harness expects FreshData to flag (not silently
# rewrite).
BAD_COUNTRY = ("XX", "ZZ", "Narnia", "us-east-1")
BAD_CURRENCY = ("BTC", "XYZ", "DOGE")
BAD_OPERATION = ("MERGE", "SYNC", None)


def pick(rng: np.random.Generator, pool, n: int) -> np.ndarray:
    """Vectorised uniform choice from a small pool."""
    idx = rng.integers(0, len(pool), size=n)
    return np.array([pool[i] for i in idx], dtype=object)
