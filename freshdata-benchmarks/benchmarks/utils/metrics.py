"""Custom metric calculators for benchmark reports."""

from __future__ import annotations


def throughput(n_rows: int, elapsed_seconds: float) -> float:
    """Compute throughput in rows per second."""
    if elapsed_seconds <= 0:
        return float("inf")
    return n_rows / elapsed_seconds


def speedup(baseline_time: float, comparison_time: float) -> float:
    """Compute speedup ratio (baseline / comparison).

    A value > 1.0 means *comparison* is faster than *baseline*.
    """
    if comparison_time <= 0:
        return float("inf")
    return baseline_time / comparison_time


def memory_ratio(baseline_bytes: float, comparison_bytes: float) -> float:
    """Compute relative memory usage (comparison / baseline).

    A value < 1.0 means *comparison* uses less memory.
    """
    if baseline_bytes <= 0:
        return float("inf")
    return comparison_bytes / baseline_bytes


def scaling_efficiency(
    time_small: float,
    time_large: float,
    n_small: int,
    n_large: int,
) -> float:
    """Estimate scaling efficiency.

    Perfect linear scaling = 1.0. Sublinear > 1.0. Superlinear < 1.0.
    """
    if time_small <= 0 or time_large <= 0:
        return float("nan")
    data_ratio = n_large / n_small
    time_ratio = time_large / time_small
    return data_ratio / time_ratio


def estimate_bigO(times: list[float], sizes: list[int]) -> str:
    """Estimate Big-O complexity from empirical timing data.

    Returns a human-readable string like "O(n)", "O(n log n)", "O(n²)".
    """
    import math

    if len(times) < 2 or len(sizes) < 2:
        return "insufficient data"

    # Compute growth rates between consecutive points
    ratios = []
    for i in range(1, len(times)):
        if times[i - 1] > 0 and sizes[i - 1] > 0:
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = times[i] / times[i - 1]
            if size_ratio > 0 and time_ratio > 0:
                exponent = math.log(time_ratio) / math.log(size_ratio)
                ratios.append(exponent)

    if not ratios:
        return "unknown"

    avg_exp = sum(ratios) / len(ratios)

    if avg_exp < 0.3:
        return "O(1)"
    elif avg_exp < 0.7:
        return "O(√n)"
    elif avg_exp < 1.3:
        return "O(n)"
    elif avg_exp < 1.7:
        return "O(n log n)"
    elif avg_exp < 2.3:
        return "O(n²)"
    else:
        return f"O(n^{avg_exp:.1f})"
