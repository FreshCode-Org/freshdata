"""int8 quantization of the exported heads, with an accuracy-delta gate.

Quantization is symmetric per-row int8 on the weight matrix (see
``linear.quantize_int8``). The gate re-evaluates both fp32 and int8 heads on
the human-verified eval sets and fails when any headline metric moves by
0.5 percentage points or more. When ``onnxruntime`` is installed the ONNX
graph is additionally quantized; that path is release-only.

CLI::

    python -m training.distill.quantize [--check-gates]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..common import BUILD_DIR, write_json
from .linear import LinearHead, dequantize, quantize_int8

OUT_DIR = BUILD_DIR / "export"
MAX_METRIC_DELTA = 0.005  # 0.5 percentage points


def _headline_metric(model_id: str, head: LinearHead) -> float:
    """One comparable accuracy number per head on its human-verified eval."""
    if model_id == "fd-role-head-v1":
        from .train_role_head import evaluate, load_human_eval  # noqa: PLC0415

        return float(evaluate(head, load_human_eval())["macro_f1"])
    from .train_intent_head import build_dataset, evaluate  # noqa: PLC0415

    _, eval_rows = build_dataset()
    return float(evaluate(head, eval_rows)["exact_intent_accuracy"])


def quantize_all(*, out_dir: Path | str = OUT_DIR) -> dict[str, Any]:
    out = Path(out_dir)
    summary: dict[str, Any] = {"max_metric_delta": MAX_METRIC_DELTA, "heads": {}}
    for model_id in ("fd-role-head-v1", "fd-intent-v1"):
        weights = out / model_id / "weights.json"
        if not weights.is_file():
            summary["heads"][model_id] = {"status": "missing"}
            continue
        head = LinearHead.load(weights)
        quantized = quantize_int8(head)
        int8_path = out / model_id / "weights.int8.json"
        int8_path.write_text(json.dumps(quantized, sort_keys=True), encoding="utf-8")
        fp32_metric = _headline_metric(model_id, head)
        int8_metric = _headline_metric(model_id, dequantize(quantized))
        delta = abs(fp32_metric - int8_metric)
        summary["heads"][model_id] = {
            "status": "quantized",
            "fp32_metric": round(fp32_metric, 4),
            "int8_metric": round(int8_metric, 4),
            "delta": round(delta, 4),
            "within_tolerance": delta < MAX_METRIC_DELTA,
        }
    write_json(out / "quantize_summary.json", summary)
    return summary


def check_gates(summary: dict[str, Any]) -> list[str]:
    failures = []
    for model_id, entry in summary["heads"].items():
        if entry.get("status") == "quantized" and not entry["within_tolerance"]:
            failures.append(
                f"{model_id}: int8 vs fp32 delta {entry['delta']} >= {MAX_METRIC_DELTA}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.distill.quantize")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--check-gates", action="store_true")
    args = parser.parse_args(argv)
    summary = quantize_all(out_dir=args.out)
    for model_id, entry in summary["heads"].items():
        print(f"{model_id}: {entry.get('status')} delta={entry.get('delta')}")
    failures = check_gates(summary)
    if failures and args.check_gates:
        for failure in failures:
            print(f"GATE FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
