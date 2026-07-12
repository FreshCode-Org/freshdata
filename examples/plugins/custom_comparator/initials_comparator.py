"""Example FreshData plugin: a custom entity-resolution *comparator*.

A comparator is a named callable ``(a: str, b: str) -> float`` in ``[0, 1]``,
used as a :class:`~freshdata.enterprise.ComparisonLevel` kind. This one scores
initials agreement — useful when one source abbreviates first names
("J. Smith" vs "John Smith").

Try it::

    import freshdata as fd
    from initials_comparator import InitialsComparator

    fd.testing.comparator_contract(InitialsComparator())
    fd.register_comparator(InitialsComparator())

    from freshdata.enterprise import ComparisonLevel
    level = ComparisonLevel("name", kind="initials", weight=1.0)

Package it::

    [project.entry-points."freshdata.comparators"]
    initials = "initials_comparator:InitialsComparator"
"""

from __future__ import annotations


class InitialsComparator:
    """1.0 when every token's initial matches in order, partial otherwise."""

    name = "initials"
    max_risk = "low"
    uses_network = False
    requires: tuple[str, ...] = ()

    def __call__(self, a: str, b: str) -> float:
        ia = [t[0] for t in a.lower().split() if t]
        ib = [t[0] for t in b.lower().split() if t]
        if not ia and not ib:
            return 1.0
        if not ia or not ib:
            return 0.0
        matches = sum(1 for x, y in zip(ia, ib) if x == y)
        return matches / max(len(ia), len(ib))
