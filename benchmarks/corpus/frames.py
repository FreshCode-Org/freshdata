"""Build a realistic frame for one :class:`TrapCase`.

A trap token on its own teaches the library nothing: role inference and the
semantic layer both need evidence, and ``semantic_types.MIN_DISTINCT_SUPPORT``
is 5, so a one-row column is deliberately ignored.  Every case therefore runs
inside a column of plausible values for its role, with the trap appended as the
last row.

The filler is also what makes a *preservation* result meaningful: if the whole
column were traps, "nothing changed" could just mean "the library gave up".
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .traps import TrapCase

__all__ = ["FILLER", "frame_for", "trap_position", "supported_roles"]

#: Eight plausible values per role. Eight is above MIN_DISTINCT_SUPPORT (5) so
#: the semantic layer engages, and small enough to keep the corpus fast.
FILLER: dict[str, list[Any]] = {
    "customer_id": ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008"],
    "invoice_no": ["9001", "9002", "9003", "9004", "9005", "9006", "9007", "9008"],
    "account_id": ["5001", "5002", "5003", "5004", "5005", "5006", "5007", "5008"],
    "unit_number": ["101A", "102B", "103C", "104D", "105E", "106F", "107G", "108H"],
    "postal_code": ["02134", "10001", "94105", "60601", "30301", "98101", "19103", "33101"],
    "product_code": ["A100", "A101", "B200", "B201", "C300", "C301", "D400", "D401"],
    "part_number": ["P-100", "P-101", "P-200", "P-201", "P-300", "P-301", "P-400", "P-401"],
    "quantity": [1, 2, 3, 4, 5, 6, 7, 8],
    "login_count": [1, 2, 3, 4, 5, 6, 7, 8],
    "price": [10.5, 20.25, 30.0, 40.75, 50.5, 60.25, 70.0, 80.5],
    "amount": [10.5, 20.25, 30.0, 40.75, 50.5, 60.25, 70.0, 80.5],
    "rate": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "weight_kg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    "shirt_size": ["Small", "Large", "Small", "Large", "Small", "Large", "Small", "Large"],
    "marital_status": [
        "Single",
        "Married",
        "Single",
        "Married",
        "Single",
        "Married",
        "Single",
        "Single",
    ],
    "initial": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "status": [
        "active",
        "inactive",
        "pending",
        "active",
        "inactive",
        "pending",
        "active",
        "active",
    ],
    "is_active": [True, False, True, False, True, False, True, False],
    "country": ["US", "GB", "FR", "DE", "JP", "CA", "AU", "BR"],
    "last_name": ["Smith", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"],
    "first_name": ["Ann", "Bob", "Cara", "Dan", "Eve", "Finn", "Gail", "Hugo"],
    "middle_name": ["Lee", "Kay", "Ray", "Jo", "Max", "Sam", "Tom", "Wes"],
    "company_name": [
        "Acme",
        "Globex",
        "Initech",
        "Umbrella",
        "Stark",
        "Wayne",
        "Hooli",
        "Vandelay",
    ],
    "product_name": ["Widget", "Gadget", "Doohickey", "Thing", "Gizmo", "Device", "Tool", "Part"],
    "ticker": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NFLX", "TSLA", "NVDA"],
    "email": [f"{c}@example.com" for c in "abcdefgh"],
    "notes": ["ok", "fine", "good", "great", "nice", "sure", "fine too", "all good"],
    "comment": ["yes", "no", "maybe", "later", "soon", "done", "not yet", "ready"],
    "description": ["red", "blue", "green", "black", "white", "grey", "pink", "teal"],
    "order_date": [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ],
}

#: A second column so the frame is never single-column. Without it a row whose
#: only cell becomes missing is removed by ``drop_empty_rows``, which would
#: confound "the value was nulled" with "the row was dropped".
_ANCHOR = "row_key"


def supported_roles() -> tuple[str, ...]:
    """Roles the corpus can build a frame for."""
    return tuple(sorted(FILLER))


def trap_position() -> int:
    """Positional index of the trap row in every built frame."""
    return len(next(iter(FILLER.values())))


def frame_for(case: TrapCase) -> pd.DataFrame | None:
    """A frame whose last row holds ``case.token`` in ``case.role``.

    Returns ``None`` when the corpus has no filler for the role, so a caller
    can report the case as unmeasured rather than inventing a column.
    """
    filler = FILLER.get(case.role)
    if filler is None:
        return None
    column = [*filler, case.token]
    return pd.DataFrame(
        {
            _ANCHOR: [f"r{i}" for i in range(len(column))],
            case.role: column,
        }
    )
