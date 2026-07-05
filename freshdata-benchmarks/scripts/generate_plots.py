"""Generate visualization plots from ASV benchmark results."""

import json
import os
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


def plot_scaling_curves(results):
    """Plot execution time vs dataset size for the full pipeline."""
    pipeline_data = results.get("BenchmarkPipeline", {}).get("time_full_clean", {})
    if not pipeline_data:
        print("No BenchmarkPipeline data found. Skipping scaling curves.")
        return
        
    row_sizes_str = pipeline_data.get("params", [])[0]
    libraries = pipeline_data.get("params", [])[1]
    raw_results = pipeline_data.get("result", [])
    
    if not row_sizes_str or not libraries or not raw_results:
        return
        
    row_sizes = [int(s.replace("_", "")) for s in row_sizes_str]
    
    plt.figure(figsize=(10, 6))
    
    idx = 0
    for lib in libraries:
        times = []
        for _ in row_sizes:
            val = raw_results[idx]
            times.append(val if val is not None and str(val) != "nan" else np.nan)
            idx += 1
            
        plt.plot(row_sizes, times, marker='o', linewidth=2, label=lib)

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Execution Time vs. Dataset Size (Full Clean Pipeline)')
    plt.xlabel('Number of Rows (Log Scale)')
    plt.ylabel('Execution Time in Seconds (Log Scale)')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    
    out_path = PLOTS_DIR / "scaling_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_speedup_bar(results):
    """Plot speedup vs Pandas for different operations on the largest dataset."""
    # Find operations to plot
    ops_to_plot = {
        "Drop Missing": ("BenchmarkMissing", "time_drop_missing"),
        "Detect Duplicates": ("BenchmarkDuplicates", "time_detect_duplicates"),
        "Regex Replace": ("BenchmarkStrings", "time_regex_replace"),
        "Standard Scale": ("BenchmarkScaling", "time_standard_scale"),
        "Full Pipeline": ("BenchmarkPipeline", "time_full_clean"),
    }
    
    speedups = defaultdict(dict)
    
    for op_name, (group, method) in ops_to_plot.items():
        data = results.get(group, {}).get(method, {})
        if not data:
            continue
            
        libraries = data.get("params", [])[1]
        raw_results = data.get("result", [])
        row_sizes = data.get("params", [])[0]
        
        if "pandas" not in libraries:
            continue
            
        # Get data for largest dataset (last index)
        pandas_idx = libraries.index("pandas") * len(row_sizes) + (len(row_sizes) - 1)
        pandas_time = raw_results[pandas_idx]
        
        if pandas_time is None or str(pandas_time) == "nan":
            continue
            
        for lib_idx, lib in enumerate(libraries):
            if lib == "pandas":
                continue
            val_idx = lib_idx * len(row_sizes) + (len(row_sizes) - 1)
            val_time = raw_results[val_idx]
            
            if val_time is not None and str(val_time) != "nan" and val_time > 0:
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
        ax.bar(x + i*width - width*(len(all_libs)/2 - 0.5), values, width, label=lib)

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


def generate_plots():
    """Main entry point."""
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    results = load_extracted_results()
    
    plot_scaling_curves(results)
    plot_speedup_bar(results)


if __name__ == "__main__":
    generate_plots()
