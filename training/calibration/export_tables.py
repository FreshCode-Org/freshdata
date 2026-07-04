"""Export fitted curves as the runtime-loadable ``calib-v1`` artifact.

The primary file is ``calibration.json`` — exactly the filename the Phase-3
model registry expects for ``calib-v1`` — validated by round-tripping it
through the runtime's own ``_IsotonicTable.from_json``. A ``calib.json``
alias is written alongside for the packaged artifact layout.

CLI::

    python -m training.calibration.export_tables [--version calib-v1]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..common import BUILD_DIR, read_json, utc_now_iso, write_json

OUT_DIR = BUILD_DIR / "calibration"


def export(
    curves_path: Path | str = OUT_DIR / "curves.json",
    *,
    version: str = "calib-v1",
    out_dir: Path | str = OUT_DIR,
) -> Path:
    curves = read_json(curves_path)
    table = {
        "version": version,
        "comment": (
            "Trained offline by training/calibration (Phase 5). Deterministic "
            "and memory backends are intentionally absent (identity mapping) "
            "unless explicitly fitted; embedding curves are capped below 1.0."
        ),
        "tables": curves["tables"],
    }
    # Round-trip through the runtime loader: an artifact the runtime cannot
    # parse must never be exported.
    from freshdata.semantic.scoring import _IsotonicTable  # noqa: PLC0415

    _IsotonicTable.from_json(json.dumps(table))

    out = Path(out_dir)
    primary = write_json(out / "calibration.json", table)
    shutil.copy(primary, out / "calib.json")
    write_json(out / "export_meta.json", {
        "version": version,
        "exported_at": utc_now_iso(),
        "n_backends": len(curves["tables"]),
        "n_records": curves.get("n_records"),
    })
    return primary


def metrics_path(out_dir: Path | str = OUT_DIR) -> Path:
    return Path(out_dir) / "calib-v1.metrics.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.calibration.export_tables")
    parser.add_argument("--curves", default=str(OUT_DIR / "curves.json"))
    parser.add_argument("--version", default="calib-v1")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    path = export(args.curves, version=args.version, out_dir=args.out)
    print(f"calibration table exported: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
