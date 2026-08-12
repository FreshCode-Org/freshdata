"""Individual cleaning steps. Each is a pure function:

``step(df, config, report) -> df``

Steps never mutate the caller's original DataFrame: the pipeline hands them a
frame it owns, and steps only ever rebind whole columns or produce new frames
(``.loc`` row selection), never write into shared blocks in place.

See `ARCHITECTURE.md <https://github.com/FreshCode-Org/freshdata/blob/main/ARCHITECTURE.md>`_
for how this package fits into the overall cleaning flow.
"""
