"""``HtmlReprMixin`` — adds ``to_html()`` / ``_repr_html_()`` / ``show()`` to report
objects without putting any rendering logic in the core compute classes.

A class opts in by inheriting the mixin and setting ``_render_kind`` to a key the
:mod:`freshdata.render.renderers` dispatcher knows. Compute stays fully
serializable: the renderers are imported lazily, only when HTML is requested.
"""

from __future__ import annotations

from typing import Any


class HtmlReprMixin:
    """Mixin providing notebook/HTML output. Requires ``_render_kind`` on the class."""

    #: Dispatch key into :func:`freshdata.render.renderers.render`.
    _render_kind: str = ""

    def to_html(self) -> str:
        """Return a self-contained HTML fragment for this report object."""
        from . import renderers

        return renderers.render(self, self._render_kind)

    def _repr_html_(self) -> str | None:
        """Rich display hook for Jupyter; falls back to text on any failure."""
        try:
            return self.to_html()
        except Exception:  # pragma: no cover - display must never raise
            return None

    def show(self, mode: str | None = None, *, renderer: str | None = None) -> Any:
        """Display this report.

        With no arguments the behavior is unchanged: in Jupyter/IPython the
        HTML renders inline; outside a notebook a standalone ``.html`` file is
        written to a temp location and its path returned.

        ``mode`` (``"compact"``/``"standard"``/``"verbose"``/``"debug"``/
        ``"json"``/``"plain"``/``"silent"``) or ``renderer="terminal"`` selects
        the Peel text output instead; ``renderer="notebook"`` keeps the HTML
        path. Display never raises: on any failure the Peel path falls back to
        the object's ``summary()``/``repr``.
        """
        if mode is not None or renderer == "terminal":
            text = self._peel_text(mode)
            if text:
                print(text)
            return None
        html = self.to_html()
        try:
            from ._optional import require

            ipython = require("IPython")
            display_mod = require("IPython.display")

            if ipython.get_ipython() is not None:
                display_mod.display(display_mod.HTML(html))
                return None
        except Exception:  # pragma: no cover - not in IPython
            pass

        import tempfile

        kind = (self._render_kind or "freshdata").lstrip("_")
        with tempfile.NamedTemporaryFile(
            "w", suffix=f"_{kind}.html", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(f"<!doctype html><meta charset='utf-8'>{html}")
            path = fh.name
        print(f"freshdata: wrote {kind} report to {path}")
        return path

    def _peel_text(self, mode: str | None) -> str:
        """Peel plain-text rendering with the never-raise fallback chain."""
        try:
            from . import normalize, plain
            from .options import get_display

            options = get_display(mode=mode) if mode is not None else get_display()
            return plain.render_plain(normalize.normalize(self), options)
        except Exception:
            summary = getattr(self, "summary", None)
            if callable(summary):
                try:
                    return str(summary())
                except Exception:  # pragma: no cover - summary must not raise
                    pass
            return repr(self)


class SimpleHtmlReport(HtmlReprMixin):
    """Base for report objects that build their own HTML from the primitives.

    Subclasses implement :meth:`_html_title` and :meth:`_html_sections`; they get
    ``to_html()`` / ``_repr_html_()`` / ``show()`` for free. This keeps each new
    report's layout next to its data instead of in a central dispatcher.
    """

    _render_kind = "_simple"

    def _html_title(self) -> str:  # pragma: no cover - overridden
        return type(self).__name__

    def _html_subtitle(self) -> str | None:
        return None

    def _html_sections(self) -> list[str]:  # pragma: no cover - overridden
        return []

    def to_html(self) -> str:
        from . import html as H

        return H.document(
            self._html_title(), *self._html_sections(), subtitle=self._html_subtitle()
        )
