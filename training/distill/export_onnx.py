"""Export trained heads to ONNX (release) or a documented dev stub.

The heads are linear (``softmax(x @ W.T + b)``) so the graph is three nodes.
Building it needs the ``onnx`` package, which is deliberately **not** a core
dependency: dev environments export the portable weights JSON plus a
``tokenizer.json`` (featurizer config) and skip the ``.onnx`` file; the
release pipeline (``--release``) hard-fails without ``onnx`` so a published
artifact can never silently miss its graph.

CLI::

    python -m training.distill.export_onnx [--release]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from ..common import BUILD_DIR, write_json
from .linear import LinearHead

OUT_DIR = BUILD_DIR / "export"

HEADS = {
    "fd-role-head-v1": BUILD_DIR / "role_head" / "role_head.weights.json",
    "fd-intent-v1": BUILD_DIR / "intent_head" / "intent_head.weights.json",
}


def onnx_available() -> bool:
    return importlib.util.find_spec("onnx") is not None


def export_head_onnx(head: LinearHead, path: Path) -> Path:  # pragma: no cover - needs onnx
    """Build the 3-node linear-head graph with the real onnx package."""
    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    dim = head.weights.shape[1]
    n_classes = head.weights.shape[0]
    weight_init = numpy_helper.from_array(head.weights.T.copy(), name="W")
    bias_init = numpy_helper.from_array(head.bias.copy(), name="b")
    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["features", "W"], ["logits_raw"]),
            helper.make_node("Add", ["logits_raw", "b"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=-1),
        ],
        name="freshdata_linear_head",
        inputs=[helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, dim])],
        outputs=[helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, [None, n_classes])],
        initializer=[weight_init, bias_init],
    )
    model = helper.make_model(graph, producer_name="freshdata-training")
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def tokenizer_config(head: LinearHead) -> dict[str, Any]:
    """The featurizer config, playing the role of ``tokenizer.json``."""
    return {
        "type": "freshdata-hashed-ngram-featurizer",
        "featurizer": head.featurizer.to_dict(),
        "classes": list(head.classes),
        "abstain_class": head.abstain_class,
        "abstain_threshold": head.abstain_threshold,
    }


def export_all(*, out_dir: Path | str = OUT_DIR, release: bool = False) -> dict[str, Any]:
    out = Path(out_dir)
    have_onnx = onnx_available()
    if release and not have_onnx:
        raise SystemExit(
            "release export requires the 'onnx' package: pip install onnx "
            "(dev exports run without it, but a release artifact must ship a real graph)"
        )
    summary: dict[str, Any] = {"onnx_available": have_onnx, "release": release, "heads": {}}
    for model_id, weights_path in HEADS.items():
        if not weights_path.is_file():
            summary["heads"][model_id] = {"status": "missing", "weights": str(weights_path)}
            continue
        head = LinearHead.load(weights_path)
        model_dir = out / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(weights_path, model_dir / "weights.json")
        (model_dir / "tokenizer.json").write_text(
            json.dumps(tokenizer_config(head), indent=2, sort_keys=True), encoding="utf-8")
        entry: dict[str, Any] = {"status": "exported", "files": ["weights.json", "tokenizer.json"]}
        if have_onnx:  # pragma: no cover - needs onnx installed
            export_head_onnx(head, model_dir / "model.onnx")
            entry["files"].append("model.onnx")
        else:
            entry["note"] = "dev export: model.onnx omitted (onnx package not installed)"
        summary["heads"][model_id] = entry
    write_json(out / "export_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.distill.export_onnx")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args(argv)
    summary = export_all(out_dir=args.out, release=args.release)
    for model_id, entry in summary["heads"].items():
        print(f"{model_id}: {entry['status']} {entry.get('files', '')}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
