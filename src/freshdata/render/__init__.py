"""Interactive rendering for freshdata report objects.

This package is **rendering only** — it consumes already-computed, serializable
report objects and turns them into HTML. It is imported lazily (never by
``import freshdata``) so the base install stays at pandas + numpy.

The renderers produce **self-contained HTML** (scoped inline CSS + a little
vanilla JS for filtering/collapsing) with *zero* optional dependencies. The
``freshdata-cleaner[viz]`` / ``freshdata-cleaner[notebook]`` extras (itables, plotly,
great-tables, anywidget) merely *upgrade* the output when installed.

See `ARCHITECTURE.md <https://github.com/FreshCode-Org/freshdata/blob/main/ARCHITECTURE.md>`_
for how this package fits into the overall cleaning flow.
"""

from __future__ import annotations

from ._optional import has, require
from .mixins import HtmlReprMixin

__all__ = ["HtmlReprMixin", "has", "require"]
