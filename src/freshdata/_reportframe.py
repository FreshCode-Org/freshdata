"""``ReportFrame`` — a transparent ``pandas.DataFrame`` subclass that adds an
interactive notebook view to the tabular results of ``compare_plans``,
``compare_clean`` and ``infer_roles`` *without* changing their data.

It **is** a DataFrame: all existing usage (indexing, ``.columns``, ``.loc``,
``len``, ``to_csv``/``to_html``, ``assert_frame_equal``) works unchanged. The
only additions are a richer ``_repr_html_`` and a ``.show()`` method. Derived
frames (slices, ``set_index``, …) deliberately fall back to a plain DataFrame via
``_constructor`` so the rich behaviour never leaks into unrelated objects.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class ReportFrame(pd.DataFrame):
    """DataFrame carrying a ``_render_kind`` for the interactive renderer."""

    # Preserve our marker across the few operations pandas copies metadata for.
    _metadata = ["_render_kind"]

    #: Plain DataFrame for derived objects — keeps the rich repr from spreading.
    @property
    def _constructor(self) -> type[pd.DataFrame]:
        return pd.DataFrame

    @classmethod
    def wrap(cls, frame: pd.DataFrame, render_kind: str) -> ReportFrame:
        out = cls(frame)
        out._render_kind = render_kind
        return out

    def _repr_html_(self) -> str | None:
        kind = getattr(self, "_render_kind", "")
        if not kind:
            return None  # pandas default repr
        try:
            from .render import renderers

            return renderers.render(self, kind)
        except Exception:  # pragma: no cover - display must never raise
            return None

    def show(self) -> Any:
        """Display the interactive view (notebook) or write an HTML file."""
        from .render.mixins import HtmlReprMixin

        proxy = HtmlReprMixin()
        proxy._render_kind = getattr(self, "_render_kind", "table")
        # Reuse the mixin's display/file logic against this frame's HTML.
        proxy.to_html = lambda: self._repr_html_() or self.to_html()  # type: ignore[method-assign]
        return proxy.show()
