"""CI-safe mini fixtures for CleanBench tiers T2 and T3.

Fixtures are generated in code (no data files), deterministic, and small
enough to run in every CI job. Each builder returns
``(truth, corrupted, kwargs)`` where *kwargs* are the exact arguments to pass
to ``freshdata.clean`` — so a benchmark run is literally
``fd.clean(corrupted, **kwargs)``.
"""

from __future__ import annotations

import pandas as pd

from ..corruptors import (
    allowed_value_casing,
    allowed_value_punctuation,
    email_at_spacing,
    email_double_at,
    phone_format_corruption,
    whitespace_corruption,
)

#: The Phase-2 reference context (mirrors the docs' ecommerce example).
ECOMMERCE_CONTEXT = """\
This is an ecommerce customer dataset.
CustomerID is unique.
Emails must be valid.
Phone numbers are Indian.
Allowed status values are active, inactive, pending.
Missing Age should be estimated only if confidence >95%.
Never modify revenue values.
"""

_FIRST = ("asha", "ravi", "neha", "vikram", "priya", "arjun", "meera", "kiran",
          "divya", "rohan", "sneha", "amit")
_DOMAINS = ("gmail.com", "example.com", "shop.in", "mail.org")
_STATUS = ("active", "inactive", "pending")


def _base_frame(n: int = 24) -> pd.DataFrame:
    rows = []
    for i in range(n):
        name = _FIRST[i % len(_FIRST)]
        rows.append(
            {
                "cust_id": f"C{i + 1:03d}",
                "email_addr": f"{name}{i}@{_DOMAINS[i % len(_DOMAINS)]}",
                "mobile": f"+91{9000000000 + i * 137}",
                "age": 20.0 + (i % 40),
                "monthly_revenue": str(1000 + 250 * (i % 8)),
                "status": _STATUS[i % len(_STATUS)],
            }
        )
    return pd.DataFrame(rows)


def make_t2_semantic_fixture(seed: int = 0):
    """T2: email / phone / reference-list value repairs.

    Only the three semantic columns are corrupted, so every other cell must
    survive untouched (that is what the false-modification gate measures).
    """
    truth = _base_frame()
    corrupted = email_double_at(truth, ["email_addr"], seed=seed, share=0.4)
    corrupted = email_at_spacing(corrupted, ["email_addr"], seed=seed + 1, share=0.3)
    corrupted = whitespace_corruption(corrupted, ["email_addr"], seed=seed + 2, share=0.3)
    corrupted = phone_format_corruption(corrupted, ["mobile"], seed=seed + 3, share=0.7)
    corrupted = allowed_value_casing(corrupted, ["status"], seed=seed + 4, share=0.5)
    corrupted = allowed_value_punctuation(corrupted, ["status"], seed=seed + 5, share=0.4)
    kwargs = {
        "context": ECOMMERCE_CONTEXT,
        "semantic_mode": "auto",
        "verbose": False,
        "drop_duplicates": False,
        "reset_index": False,
    }
    return truth, corrupted, kwargs


def make_t3_context_fixture(seed: int = 10):
    """T3: context compliance — protection, thresholds, allowed values, uniqueness.

    The revenue column is corrupted *on purpose*: a compliant cleaner must
    leave the corruption in place (protected means protected), and the
    duplicate ``cust_id`` must be surfaced by validation, not silently fixed.
    """
    truth = _base_frame()
    corrupted = truth.copy(deep=True)
    # Protected column: whitespace dirt that must NOT be repaired.
    corrupted.loc[0, "monthly_revenue"] = " 1000 "
    corrupted.loc[5, "monthly_revenue"] = "2,250"
    # Age gaps that must stay missing (threshold 0.95 beats every fill).
    corrupted.loc[[2, 7, 11], "age"] = None
    # Allowed-value dirt.
    corrupted = allowed_value_casing(corrupted, ["status"], seed=seed, share=0.5)
    # Uniqueness violation to be validated, never dropped.
    corrupted.loc[3, "cust_id"] = corrupted.loc[2, "cust_id"]
    kwargs = {
        "context": ECOMMERCE_CONTEXT,
        "semantic_mode": "auto",
        "verbose": False,
        "drop_duplicates": False,
        "reset_index": False,
    }
    return truth, corrupted, kwargs
