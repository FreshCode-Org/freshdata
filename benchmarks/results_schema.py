"""Stable JSON schema for benchmark result files.

Every ``results/<run_id>/<fixture>/<size>.json`` validates against
:data:`RESULTS_SCHEMA`. Keeping the schema versioned and explicit is what lets
runs be diffed across FreshData versions and across machines.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

#: The nine standardized metrics, in reporting order. The ``single`` subcommand
#: may emit a subset (with the rest null); ``run`` emits all of them.
METRIC_FIELDS = (
    "wall_clock_p50_sec",
    "wall_clock_p95_sec",
    "peak_memory_mb",
    "repair_fidelity_pct",
    "false_repair_rate_pct",
    "preservation_rate_pct",
    "authored_lines_fd",
    "authored_lines_pandas",
    "reduction_vs_pandas_pct",
    "diagnosis_summary_sec",
    "diagnosis_to_frame_sec",
    "diagnosis_to_dict_sec",
    "trust_score",
    "trust_monotonic_valid",
    "export_completeness_pct",
)

RESULTS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "freshdata_version",
        "python_version",
        "platform",
        "fixture",
        "n_rows",
        "n_cols",
        "mode",
        "metrics",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "run_id": {"type": "string"},
        "freshdata_version": {"type": "string"},
        "python_version": {"type": "string"},
        "platform": {"type": "string"},
        "fixture": {"type": "string"},
        "n_rows": {"type": "integer", "minimum": 0},
        "n_cols": {"type": "integer", "minimum": 0},
        "mode": {"type": "string", "enum": ["balanced", "aggressive"]},
        "seed": {"type": "integer"},
        "metrics": {
            "type": "object",
            "required": list(METRIC_FIELDS),
            "properties": {
                "wall_clock_p50_sec": {"type": ["number", "null"], "minimum": 0},
                "wall_clock_p95_sec": {"type": ["number", "null"], "minimum": 0},
                "peak_memory_mb": {"type": ["number", "null"], "minimum": 0},
                "repair_fidelity_pct": {"type": ["number", "null"], "minimum": 0, "maximum": 100},
                "false_repair_rate_pct": {
                    "type": ["number", "null"], "minimum": 0, "maximum": 100
                },
                "preservation_rate_pct": {
                    "type": ["number", "null"], "minimum": 0, "maximum": 100
                },
                "authored_lines_fd": {"type": ["integer", "null"], "minimum": 0},
                "authored_lines_pandas": {"type": ["integer", "null"], "minimum": 0},
                "reduction_vs_pandas_pct": {"type": ["number", "null"]},
                "diagnosis_summary_sec": {"type": ["number", "null"], "minimum": 0},
                "diagnosis_to_frame_sec": {"type": ["number", "null"], "minimum": 0},
                "diagnosis_to_dict_sec": {"type": ["number", "null"], "minimum": 0},
                "trust_score": {"type": ["number", "null"], "minimum": 0, "maximum": 100},
                "trust_monotonic_valid": {"type": ["boolean", "null"]},
                "export_completeness_pct": {
                    "type": ["number", "null"], "minimum": 0, "maximum": 100
                },
            },
        },
        "details": {"type": "object"},
    },
}
