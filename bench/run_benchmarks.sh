#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# bench/run_benchmarks.sh
#
# One-shot script that installs dependencies in an isolated
# virtualenv, runs the focused freshdata-vs-pandas ASV benchmarks,
# generates the HTML report and a JSON/CSV summary, and collects
# all artifacts under bench/artifacts/.
#
# Usage:
#   bash bench/run_benchmarks.sh              # full run
#   bash bench/run_benchmarks.sh --quick      # quick (dry-run) mode
#
# Requirements:
#   - Python 3.9+ available as `python3`
#   - git (required by ASV)
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv-bench"
ARTIFACTS_DIR="${PROJECT_ROOT}/bench/artifacts"
QUICK_MODE=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK_MODE=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════"
echo "  FreshData vs Pandas — ASV Benchmark Runner"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Project root : ${PROJECT_ROOT}"
echo "  Quick mode   : ${QUICK_MODE}"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ── 1. Create / activate isolated virtualenv ─────────────────────
echo "▸ Step 1/7: Setting up isolated virtualenv …"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created ${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

# ── 2. Install dependencies ─────────────────────────────────────
echo "▸ Step 2/7: Installing dependencies …"
pip install -r "${PROJECT_ROOT}/requirements.txt" --quiet
pip install asv virtualenv --quiet
echo "  Done."

# ── 3. Capture environment details ──────────────────────────────
echo "▸ Step 3/7: Capturing environment …"
mkdir -p "${ARTIFACTS_DIR}"
python "${PROJECT_ROOT}/scripts/capture_env.py" "${ARTIFACTS_DIR}/env_info.json"

# ── 4. Configure ASV machine ────────────────────────────────────
echo "▸ Step 4/7: Configuring ASV machine …"
cd "${PROJECT_ROOT}"
asv machine --yes 2>/dev/null || true

# ── 5. Pin reproducibility settings ─────────────────────────────
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=42

# ── 6. Run benchmarks ───────────────────────────────────────────
echo "▸ Step 5/7: Running benchmarks …"
if [ "${QUICK_MODE}" = true ]; then
    asv run --quick \
        --bench "BenchmarkFreshDataVsPandas|BenchmarkGroupAgg" \
        -e 2>&1 | tee "${ARTIFACTS_DIR}/benchmark_output.txt"
else
    # --record-samples keeps every raw per-repeat timing (not just
    # ASV's default quartile summary), so generate_reports.py can report
    # true min/max/std instead of falling back to the coarser IQR.
    asv run \
        --skip-existing-successful \
        --record-samples \
        --bench "BenchmarkFreshDataVsPandas|BenchmarkGroupAgg" \
        -e 2>&1 | tee "${ARTIFACTS_DIR}/benchmark_output.txt"
fi

# ── 7. Generate reports & HTML ───────────────────────────────────
echo "▸ Step 6/7: Generating reports …"
python "${PROJECT_ROOT}/scripts/generate_reports.py" || true
python "${PROJECT_ROOT}/scripts/generate_plots.py" || true
asv publish || true

# Copy HTML report to artifacts
if [ -d "${PROJECT_ROOT}/.asv/html" ]; then
    cp -r "${PROJECT_ROOT}/.asv/html" "${ARTIFACTS_DIR}/html_report"
    echo "  HTML report → ${ARTIFACTS_DIR}/html_report/"
fi

# Copy JSON results to artifacts
if [ -d "${PROJECT_ROOT}/.asv/results" ]; then
    cp -r "${PROJECT_ROOT}/.asv/results" "${ARTIFACTS_DIR}/asv_results"
    echo "  ASV results → ${ARTIFACTS_DIR}/asv_results/"
fi

# Copy generated reports
if [ -d "${PROJECT_ROOT}/reports" ]; then
    cp -r "${PROJECT_ROOT}/reports" "${ARTIFACTS_DIR}/reports"
    echo "  Reports     → ${ARTIFACTS_DIR}/reports/"
fi

echo ""
echo "▸ Step 7/7: Summary"
echo "══════════════════════════════════════════════════════════════"
echo "  Artifacts directory : ${ARTIFACTS_DIR}"
echo "  Environment info    : ${ARTIFACTS_DIR}/env_info.json"
echo "  Benchmark log       : ${ARTIFACTS_DIR}/benchmark_output.txt"
echo ""
echo "  To view the HTML report locally:"
echo "    cd ${PROJECT_ROOT} && asv preview"
echo ""
echo "  Exact command to reproduce:"
echo "    PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\"
echo "    OPENBLAS_NUM_THREADS=1 bash bench/run_benchmarks.sh"
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✓ Benchmark run complete."
echo "══════════════════════════════════════════════════════════════"
