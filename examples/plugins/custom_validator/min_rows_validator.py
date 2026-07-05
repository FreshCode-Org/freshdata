"""Example FreshData plugin: a custom *validator*.

A validator is **read-only**: it inspects the frame (and the compiled context
policy) and returns :class:`freshdata.QualityFinding` records. It never repairs
anything — validation is not repair. This one flags frames with too few rows to
reason about, and any fully-constant column.

Try it::

    import freshdata as fd
    from min_rows_validator import MinRowsValidator

    fd.testing.validator_contract(MinRowsValidator())
    fd.register_validator(MinRowsValidator())

    import pandas as pd
    df = pd.DataFrame({"x": [1, 1, 1]})
    findings = fd.validate(df, context="x is unique.")   # validators run inside fd.validate

Package it::

    [project.entry-points."freshdata.validators"]
    min_rows = "min_rows_validator:MinRowsValidator"
"""

from __future__ import annotations

import pandas as pd

from freshdata import QualityFinding

_MIN_ROWS = 30


class MinRowsValidator:
    name = "min_rows"
    max_risk = "low"
    uses_network = False
    requires = ()

    def validate(self, df: pd.DataFrame, policy: object, ctx: object) -> list[QualityFinding]:
        findings: list[QualityFinding] = []
        if len(df) < _MIN_ROWS:
            findings.append(QualityFinding.create(
                severity="warning",
                step="plugin",
                rule_name="plugin.min_rows",
                message=f"only {len(df)} rows (< {_MIN_ROWS}); statistics may be unreliable",
                expected_condition=f">= {_MIN_ROWS} rows",
                extra={"n_rows": int(len(df))},
            ))
        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 1 and non_null.nunique() == 1:
                findings.append(QualityFinding.create(
                    severity="info",
                    step="plugin",
                    rule_name="plugin.constant_column",
                    column=str(col),
                    message=f"column {col!r} is constant",
                    expected_condition="more than one distinct value",
                ))
        return findings
