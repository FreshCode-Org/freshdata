"""Interactive HTML reports via ``freshdata.render`` (new in 1.1).

``CleanReport``, ``Profile``, ``CleanPlan`` and ``ExplainReport`` all gain
``to_html()`` / ``_repr_html_()`` / ``.show()``: a collapsible action timeline
and filterable audit ledger, an inline quality cockpit, decision cards for a
proposed plan, and a before/after diff explorer. Everything is self-contained
HTML (scoped CSS + a little vanilla JS) — no optional deps required. Installing
``freshdata-cleaner[viz]`` / ``[notebook]`` only upgrades the output (itables,
plotly, great-tables, anywidget). Run:

    python examples/interactive_html_reports.py
"""

import pandas as pd

import freshdata as fd


def main() -> None:
    df = pd.DataFrame(
        {
            "Customer Name": [" Alice ", "Bob", "N/A", "Alice", "  Carol"],
            "Signup Date": ["2024-01-05", "2024-02-30", "2024/03/01", "2024-01-05", None],
            "Revenue": ["1200", "980", "-", "1200", "15000"],
        }
    )

    # 1. CleanReport — action timeline + audit ledger.
    cleaned, report = fd.clean(df, return_report=True)
    report_path = report.show()  # opens inline in Jupyter, else writes an .html file
    print(f"CleanReport -> {report_path}")

    # 2. Profile — inline quality cockpit for the raw data.
    prof = fd.profile(df)
    profile_path = prof.show()
    print(f"Profile -> {profile_path}")

    # 3. CleanPlan — decision cards / strategy diff grid for a *proposed*, not-yet-
    #    applied plan, so a reviewer can approve/reject before anything runs.
    plan = fd.suggest_plan(df)
    plan_path = plan.show()
    print(f"CleanPlan -> {plan_path}")

    # 4. ExplainReport — before/after diff explorer for the whole run (per-column
    #    dtype/changed-cell stats via .to_frame(), narratives, warnings).
    explain = fd.explain_clean(df)
    explain_path = explain.show()
    print(explain.summary())
    print(f"ExplainReport -> {explain_path}")

    # 5. compare_plans / compare_clean / infer_roles — return a ``ReportFrame``
    #    (a DataFrame subclass): behaves like a normal frame but also renders
    #    as HTML via .to_html()/.show().
    strategy_diff = fd.compare_plans(df, strategies=("conservative", "balanced", "aggressive"))
    print(strategy_diff)
    strategy_diff.show()

    roles = fd.infer_roles(df)
    print(roles)


if __name__ == "__main__":
    main()
