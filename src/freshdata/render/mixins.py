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

    def show(self) -> Any:
        """Display in a notebook, or write an HTML file and return its path.

        In Jupyter/IPython this renders inline. Outside a notebook it writes a
        standalone ``.html`` file to a temp location and returns the path, so the
        same call works from scripts and the REPL.
        """
        html = self.to_html()
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display

            if get_ipython() is not None:
                display(HTML(html))
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

