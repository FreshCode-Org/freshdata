"""Cleaning memory + replay via ``fd.learn_cleaning_memory`` (new in 1.1).

Record a reviewed ``CleanReport`` as reusable, server-free memory (JSON or
SQLite), then replay the accepted decisions on new batches of the *same*
dataset. Replay only happens when the new data still looks like what the
memory learned (column set + coarse dtypes); if it drifted too far, replay is
**blocked and explained** on the report rather than applied blindly. Run:

    python examples/cleaning_memory_replay.py
"""

import pandas as pd

import freshdata as fd


def main() -> None:
    # Day 1: a human reviews and accepts freshdata's cleaning decisions.
    day1 = pd.DataFrame(
        {
            "Customer Name": [" Alice ", "Bob", "N/A", "Alice"],
            "Signup Date": ["2024-01-05", "2024-02-30", "2024/03/01", "2024-01-05"],
            "Revenue": ["1200", "980", "-", "1200"],
        }
    )
    cleaned1, report1 = fd.clean(day1, return_report=True)

    memory = fd.learn_cleaning_memory(day1, decisions=report1, dataset_id="crm_daily")
    memory.to_json("crm_daily_memory.json")
    print(memory.summary())

    # Day 2: same shape/dtypes -> memory matches, decisions replay automatically.
    day2 = pd.DataFrame(
        {
            "Customer Name": [" Dave ", "Erin", "N/A"],
            "Signup Date": ["2024-04-01", "2024-04-02", None],
            "Revenue": ["2100", "-", "1750"],
        }
    )
    loaded = fd.load_cleaning_memory("crm_daily_memory.json")
    cleaned2, report2 = fd.clean(day2, memory=loaded, return_report=True)

    replayed = [a for a in report2.actions if a.memory_influenced]
    print(f"day 2: {len(replayed)} decision(s) replayed from memory")
    for w in report2.warnings:
        print("warning:", w)

    # Day 3: the feed drifted (a learned column is gone) -> replay is blocked,
    # not silently skipped — the report says exactly why.
    day3 = pd.DataFrame({"Revenue": ["500", "600"]})
    cleaned3, report3 = fd.clean(day3, memory=loaded, return_report=True)
    print("day 3 warnings:", report3.warnings)
    print("day 3 recommendations:", report3.recommendations)

    # Inspect what changed between two learned memories (e.g. after a re-review).
    day2_review_report = fd.clean(day2, return_report=True)[1]
    memory2 = fd.learn_cleaning_memory(day2, decisions=day2_review_report, dataset_id="crm_daily")
    print(memory.diff(memory2))


if __name__ == "__main__":
    main()
