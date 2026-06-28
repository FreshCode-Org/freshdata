"""CDC / event-time quality gate with ``fd.cdc_profile``.

Change-data-capture streams fail in ways that are *not* missing values: records
arrive stale, late, out of order, with duplicate keys, bad operation codes, or as
a replayed batch. ``cdc_profile`` classifies those freshness/ordering defects
separately from nulls and reports trust penalties you can gate on. Run:

    python examples/cdc_event_time_gate.py
"""

import pandas as pd

import freshdata as fd


def main() -> None:
    # An order-change feed captured from a CDC connector, in arrival order.
    changes = pd.DataFrame(
        {
            "order_id": [101, 101, 102, 103, 103, 101, 104],
            "event_ts": [
                "2024-03-01 09:00:00",
                "2024-03-01 09:05:00",
                "2024-03-01 09:06:00",
                "2024-03-01 09:05:30",  # 102/103 interleave is fine (different keys)
                None,                    # missing event time
                "2024-03-01 08:30:00",  # 101 arrives 35m late, past the watermark
                "2024-03-01 09:05:30",
            ],
            "op": ["c", "u", "c", "c", "u", "u", "Q"],  # 'Q' is not a valid op code
        }
    )

    report = fd.cdc_profile(
        changes,
        event_time="event_ts",
        key="order_id",
        lateness="10m",
        operation_col="op",
        now=pd.Timestamp("2024-03-01 09:10:00"),
        stale_after="30m",
    )

    print(report.summary())
    print("\ntrust penalties (0=clean, 1=worst):", report.trust_penalties)
    print(f"freshness: {report.freshness_seconds:.0f}s behind reference")
    print(f"\ngate passed: {report.passed}\n")
    print("defect table:")
    print(report.to_frame()[["kind", "level", "n_rows", "rationale"]].to_string(index=False))


if __name__ == "__main__":
    main()
