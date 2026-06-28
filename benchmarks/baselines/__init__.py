"""Competitor baselines for the FreshData benchmark harness.

Each baseline module exposes:

* ``run(df: pd.DataFrame) -> pd.DataFrame`` — produce a cleaned (or, for the
  read-only profilers, simply profiled) frame.
* ``AUTHORED_LINES: int`` — non-blank, non-comment source lines in ``run``,
  used by the authored-code-reduction metric (Metric 6). Counted from source
  with :func:`count_authored_lines` so it can never drift from the code.

Competitor libraries are **not** installed by default. Baselines import them
lazily and raise a clear :class:`ImportError` when missing; the harness's
baseline runner catches that and skips the baseline rather than failing the
whole run.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable


def count_authored_lines(func: Callable) -> int:
    """Count non-blank, non-comment, non-docstring lines in ``func``'s body.

    This is the line metric Metric 6 reports. It excludes the ``def`` line, the
    docstring, blank lines and comment-only lines so the count reflects authored
    logic, not formatting.
    """
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    fn = tree.body[0]
    body = fn.body
    # drop a leading docstring
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return 0
    lines = set()
    for node in body:
        for n in ast.walk(node):
            if hasattr(n, "lineno"):
                lines.add(n.lineno)
    # subtract comment-only physical lines that ast never sees anyway; comments
    # are already excluded because ast has no nodes for them.
    return len(lines)


REGISTRY = (
    "pandas_baseline",
    "pyjanitor_baseline",
    "ydata_profiling_baseline",
    "sweetviz_baseline",
)

#: Baselines that actually repair (so their output can be compared), vs the
#: read-only profilers that are timing-only.
REPAIR_BASELINES = ("pandas_baseline", "pyjanitor_baseline")
READONLY_BASELINES = ("ydata_profiling_baseline", "sweetviz_baseline")

__all__ = ["count_authored_lines", "REGISTRY", "REPAIR_BASELINES", "READONLY_BASELINES"]
