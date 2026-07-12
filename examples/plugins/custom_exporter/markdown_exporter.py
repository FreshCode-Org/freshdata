"""Example FreshData plugin: a custom report *exporter*.

An exporter renders a report (``CleanReport`` / ``DriftReport`` — anything
with ``to_dict()``) into another format. It is invoked explicitly via
``fd.export(report, format=<name>)`` and never runs inside the pipeline.

Try it::

    import freshdata as fd
    from markdown_exporter import MarkdownExporter

    fd.testing.exporter_contract(MarkdownExporter())
    fd.register_exporter(MarkdownExporter())

    cleaned, report = fd.clean(df, return_report=True)
    print(fd.export(report, format="markdown"))
    fd.export(report, format="markdown", path="clean_report.md")

Package it::

    [project.entry-points."freshdata.exporters"]
    markdown = "markdown_exporter:MarkdownExporter"
"""

from __future__ import annotations


class MarkdownExporter:
    """Render the report's headline numbers and action log as Markdown."""

    name = "markdown"
    uses_network = False
    requires: tuple[str, ...] = ()

    def export(self, report: object) -> str:
        d = report.to_dict()  # type: ignore[attr-defined]
        lines = [
            "# freshdata clean report",
            "",
            f"- rows: {d.get('rows_before')} -> {d.get('rows_after')}",
            f"- columns: {d.get('cols_before')} -> {d.get('cols_after')}",
            f"- missing cells: {d.get('missing_before')} -> {d.get('missing_after')}",
        ]
        if d.get("backend"):
            lines.append(f"- backend: {d['backend']}")
        actions = d.get("actions", [])
        if actions:
            lines += ["", "## Actions", ""]
            lines += [
                f"- `{a.get('step')}` {a.get('description', '')}".rstrip()
                for a in actions
            ]
        return "\n".join(lines) + "\n"
