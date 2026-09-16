"""Re-export the existing gold corpora through :class:`TrapCase`.

Gauntlet and TruthBench each own a corpus that is authoritative for its own
harness.  Nothing here replaces them: these adapters read their cases and
present them in one shape, so a test can ask "every leading-zero trap this
repository knows about" without caring which harness defined it.

Both source harnesses use the same four-value vocabulary, so the mapping onto
:class:`~.dispositions.Disposition` is exact and lossless.  The two additional
values (``QUARANTINE``, ``REJECT``) only ever come from the hand-written corpus
in :mod:`.traps` -- which is precisely why they were added.

Building a fixture is not free (each generates a frame), so both readers are
cached and both degrade to an empty tuple when their optional dependencies are
missing, rather than breaking collection of every test that imports the corpus.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .dispositions import Disposition
from .traps import UNSET, TrapCase

__all__ = ["from_gauntlet", "from_truthbench", "all_cases"]

#: Both harnesses spell the four shared dispositions identically.
_SHARED = {
    "preserve": Disposition.PRESERVE,
    "repair": Disposition.REPAIR,
    "flag": Disposition.FLAG,
    "review": Disposition.REVIEW,
}


def _disposition(value: Any) -> Disposition:
    key = value.value if hasattr(value, "value") else str(value)
    try:
        return _SHARED[key]
    except KeyError:
        raise KeyError(
            f"unmapped source disposition {value!r}; if a harness gained a new "
            "disposition, add it to benchmarks/corpus/adapters._SHARED rather "
            "than letting it score as something it is not"
        ) from None


@lru_cache(maxsize=1)
def from_gauntlet() -> tuple[TrapCase, ...]:
    """Gauntlet's labelled cells as TrapCases.

    Gauntlet labels a cell by ``(row, column)`` in a generated frame, so
    ``role`` becomes the column name; the frame itself is not carried over.
    """
    try:
        from benchmarks.gauntlet.fixtures import FIXTURES, build_fixture
    except Exception:  # pragma: no cover - optional dependency
        return ()

    cases: list[TrapCase] = []
    for name in sorted(FIXTURES):
        try:
            fixture = build_fixture(name)
        except Exception:  # pragma: no cover - fixture needs an optional extra
            continue
        for cell in fixture.cells:
            expected = _disposition(cell.expect)
            cases.append(
                TrapCase(
                    token=cell.dirty,
                    family=cell.kind,
                    role=cell.column,
                    semantic_type=fixture.field_types.get(cell.column),
                    expected=expected,
                    rationale=f"Gauntlet {name} gold cell ({cell.kind}).",
                    repaired=cell.repaired if expected is Disposition.REPAIR else UNSET,
                    source="gauntlet",
                )
            )
    return tuple(cases)


@lru_cache(maxsize=1)
def from_truthbench() -> tuple[TrapCase, ...]:
    """TruthBench's gold cells as TrapCases.

    TruthBench already carries a ``family`` per cell, which maps straight onto
    ``TrapCase.family``.  Its ``expected_output`` is a ``TypedValue``; only the
    plain value is carried across, because the dtype fidelity that type encodes
    is TruthBench's own contract to enforce, not this corpus's.

    The dirty token is read out of the adversarial frame by ``(row_id, column)``
    so the case is usable standalone.
    """
    try:
        from benchmarks.truthbench.fixtures import DOMAINS, build_fixture
    except Exception:  # pragma: no cover - optional dependency
        return ()

    cases: list[TrapCase] = []
    for domain in DOMAINS:
        try:
            fixture = build_fixture(domain)
        except Exception:  # pragma: no cover
            continue
        frame = fixture.adversarial
        for cell in fixture.cells:
            expected = _disposition(cell.disposition)
            repaired = UNSET
            if expected is Disposition.REPAIR:
                # expected_output of None is a real gold value here: "repair
                # this cell to missing". UNSET would fail TrapCase validation.
                repaired = getattr(cell.expected_output, "value", cell.expected_output)
            token = None
            try:
                if cell.column in frame.columns:
                    token = frame.at[cell.row_id, cell.column]
            except Exception:  # pragma: no cover - non-scalar or odd label
                token = None
            cases.append(
                TrapCase(
                    token=token,
                    family=cell.family or "unfamilied",
                    role=cell.column,
                    semantic_type=None,
                    expected=expected,
                    rationale=f"TruthBench {cell.domain} gold cell.",
                    repaired=repaired,
                    source="truthbench",
                )
            )
    return tuple(cases)


def all_cases(include_harnesses: bool = True) -> tuple[TrapCase, ...]:
    """The hand-written corpus, optionally plus every harness corpus."""
    from .traps import TRAPS

    if not include_harnesses:
        return TRAPS
    return TRAPS + from_gauntlet() + from_truthbench()
