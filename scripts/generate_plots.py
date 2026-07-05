"""Generate visualization plots from ASV benchmark results."""

import json
import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"


def load_extracted_results():
    json_path = REPORTS_DIR / "results_extracted.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found.", file=sys.stderr)
        print("Run 'python scripts/generate_reports.py' first.", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        return json.load(f)


def _is_missing(val):
    return val is None or str(val) == "nan"


def _value_at(raw_results, row_sizes, libraries, size_idx, lib_idx):
    """Index into ASV's flat result list.

    ASV flattens the cartesian product of params with the FIRST param
    (n_rows) varying slowest and the LAST param (library) varying fastest:
    idx = size_idx * len(libraries) + lib_idx.
    """
    idx = size_idx * len(libraries) + lib_idx
    return raw_results[idx] if idx < len(raw_results) else None


def plot_scaling_curves(results):
    """Plot execution time vs dataset size for the full pipeline."""
    pipeline_data = results.get("BenchmarkFreshDataVsPandas", {}).get("time_full_pipeline", {})
    if not pipeline_data:
        print("No BenchmarkFreshDataVsPandas data found. Skipping scaling curves.")
        return

    row_sizes_str = pipeline_data.get("params", [])[0]
    libraries = pipeline_data.get("params", [])[1]
    raw_results = pipeline_data.get("result", [])

    if not row_sizes_str or not libraries or not raw_results:
        return

    row_sizes = [int(s.replace("_", "")) for s in row_sizes_str]

    plt.figure(figsize=(10, 6))

    for lib_idx, lib in enumerate(libraries):
        times = []
        for size_idx in range(len(row_sizes)):
            val = _value_at(raw_results, row_sizes, libraries, size_idx, lib_idx)
            times.append(val if not _is_missing(val) else np.nan)

        plt.plot(row_sizes, times, marker='o', linewidth=2, label=lib)

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Execution Time vs. Dataset Size (Full Preprocessing Pipeline)')
    plt.xlabel('Number of Rows (Log Scale)')
    plt.ylabel('Execution Time in Seconds (Log Scale)')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()

    out_path = PLOTS_DIR / "scaling_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_execution_time_bar(results):
    """Plot absolute execution time per operation, per library, at the largest dataset size."""
    ops_to_plot = {
        "Load CSV": ("BenchmarkFreshDataVsPandas", "time_load_csv"),
        "Missing Values": ("BenchmarkFreshDataVsPandas", "time_handle_missing"),
        "Outliers": ("BenchmarkFreshDataVsPandas", "time_detect_outliers"),
        "Duplicates": ("BenchmarkFreshDataVsPandas", "time_resolve_duplicates"),
        "Group Agg": ("BenchmarkFreshDataVsPandas", "time_group_aggregations"),
        "Full Pipeline": ("BenchmarkFreshDataVsPandas", "time_full_pipeline"),
    }

    times_by_op = defaultdict(dict)

    for op_name, (group, method) in ops_to_plot.items():
        data = results.get(group, {}).get(method, {})
        if not data:
            continue

        libraries = [lib.strip("'") for lib in data.get("params", [])[1]]
        row_sizes = data.get("params", [])[0]
        raw_results = data.get("result", [])
        if not row_sizes or not libraries:
            continue
        largest_idx = len(row_sizes) - 1

        for lib_idx, lib in enumerate(libraries):
            val = _value_at(raw_results, row_sizes, libraries, largest_idx, lib_idx)
            if not _is_missing(val):
                times_by_op[op_name][lib] = val

    if not times_by_op:
        print("No execution time data available.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ops = list(times_by_op.keys())
    all_libs = sorted({lib for s in times_by_op.values() for lib in s})

    x = np.arange(len(ops))
    width = 0.8 / len(all_libs)

    for i, lib in enumerate(all_libs):
        values = [times_by_op[op].get(lib, 0) for op in ops]
        ax.bar(x + i * width - width * (len(all_libs) / 2 - 0.5), values, width, label=lib)

    ax.set_ylabel('Execution Time (seconds, largest dataset size)')
    ax.set_title('Execution Time Comparison by Operation')
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    out_path = PLOTS_DIR / "execution_time_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()


def plot_speedup_bar(results):
    """Plot speedup vs Pandas for different operations on the largest dataset."""
    ops_to_plot = {
        "Load CSV": ("BenchmarkFreshDataVsPandas", "time_load_csv"),
        "Missing Values": ("BenchmarkFreshDataVsPandas", "time_handle_missing"),
        "Outliers": ("BenchmarkFreshDataVsPandas", "time_detect_outliers"),
        "Duplicates": ("BenchmarkFreshDataVsPandas", "time_resolve_duplicates"),
        "Group Agg": ("BenchmarkFreshDataVsPandas", "time_group_aggregations"),
        "Full Pipeline": ("BenchmarkFreshDataVsPandas", "time_full_pipeline"),
    }

    speedups = defaultdict(dict)

    for op_name, (group, method) in ops_to_plot.items():
        data = results.get(group, {}).get(method, {})
        if not data:
            continue

        libraries = [lib.strip("'") for lib in data.get("params", [])[1]]
        raw_results = data.get("result", [])
        row_sizes = data.get("params", [])[0]

        if "pandas" not in libraries or not row_sizes:
            continue

        largest_idx = len(row_sizes) - 1
        pandas_time = _value_at(
            raw_results, row_sizes, libraries, largest_idx, libraries.index("pandas")
        )

        if _is_missing(pandas_time):
            continue

        for lib_idx, lib in enumerate(libraries):
            if lib == "pandas":
                continue
            val_time = _value_at(raw_results, row_sizes, libraries, largest_idx, lib_idx)

            if not _is_missing(val_time) and val_time > 0:
                speedups[op_name][lib] = pandas_time / val_time

    if not speedups:
        print("No speedup data available.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ops = list(speedups.keys())
    # Find all unique libraries across operations
    all_libs = set()
    for s in speedups.values():
        all_libs.update(s.keys())
    all_libs = sorted(list(all_libs))

    x = np.arange(len(ops))
    width = 0.8 / len(all_libs)

    for i, lib in enumerate(all_libs):
        values = [speedups[op].get(lib, 0) for op in ops]
        ax.bar(x + i * width - width * (len(all_libs) / 2 - 0.5), values, width, label=lib)

    # Add baseline
    ax.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='Pandas Baseline (1.0x)')

    ax.set_ylabel('Speedup Ratio (Higher is Better)')
    ax.set_title('Speedup vs Pandas (Largest Dataset Size)')
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    out_path = PLOTS_DIR / "speedup_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()


def plot_memory_bar(results):
    """Plot memory reduction vs Pandas for different operations on the largest dataset."""
    ops_to_plot = {
        "Load CSV": ("BenchmarkFreshDataVsPandas", "peakmem_load_csv"),
        "Missing Values": ("BenchmarkFreshDataVsPandas", "peakmem_handle_missing"),
        "Outliers": ("BenchmarkFreshDataVsPandas", "peakmem_detect_outliers"),
        "Duplicates": ("BenchmarkFreshDataVsPandas", "peakmem_resolve_duplicates"),
        "Group Agg": ("BenchmarkFreshDataVsPandas", "peakmem_group_aggregations"),
        "Full Pipeline": ("BenchmarkFreshDataVsPandas", "peakmem_full_pipeline"),
    }

    memory_ratios = defaultdict(dict)

    for op_name, (group, method) in ops_to_plot.items():
        data = results.get(group, {}).get(method, {})
        if not data:
            continue

        libraries = [lib.strip("'") for lib in data.get("params", [])[1]]
        raw_results = data.get("result", [])
        row_sizes = data.get("params", [])[0]

        if "pandas" not in libraries or not row_sizes:
            continue

        largest_idx = len(row_sizes) - 1
        pandas_mem = _value_at(
            raw_results, row_sizes, libraries, largest_idx, libraries.index("pandas")
        )

        if _is_missing(pandas_mem) or pandas_mem == 0:
            continue

        for lib_idx, lib in enumerate(libraries):
            if lib == "pandas":
                continue
            val_mem = _value_at(raw_results, row_sizes, libraries, largest_idx, lib_idx)

            if not _is_missing(val_mem) and val_mem > 0:
                memory_ratios[op_name][lib] = pandas_mem / val_mem

    if not memory_ratios:
        print("No memory data available.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ops = list(memory_ratios.keys())
    all_libs = set()
    for s in memory_ratios.values():
        all_libs.update(s.keys())
    all_libs = sorted(list(all_libs))

    x = np.arange(len(ops))
    width = 0.8 / len(all_libs)

    for i, lib in enumerate(all_libs):
        values = [memory_ratios[op].get(lib, 0) for op in ops]
        ax.bar(x + i * width - width * (len(all_libs) / 2 - 0.5), values, width, label=lib)

    ax.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='Pandas Baseline (1.0x Memory)')

    ax.set_ylabel('Memory Efficiency Ratio (Higher is Better, 2.0 = half memory)')
    ax.set_title('Memory Efficiency vs Pandas (Largest Dataset Size)')
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    out_path = PLOTS_DIR / "memory_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()


def generate_plots():
    """Main entry point."""
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    results = load_extracted_results()

    plot_scaling_curves(results)
    plot_execution_time_bar(results)
    plot_speedup_bar(results)
    plot_memory_bar(results)


if __name__ == "__main__":
    generate_plots()
