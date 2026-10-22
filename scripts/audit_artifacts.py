# scripts/audit_artifacts.py
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)
    return res


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = repo_root / "dist"
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    print("[1/4] Building wheel and sdist via `python -m build --outdir dist/`...")
    run_cmd([sys.executable, "-m", "build", "--outdir", str(dist_dir)], cwd=repo_root)

    wheels = sorted(glob.glob(str(dist_dir / "*.whl")))
    sdists = sorted(glob.glob(str(dist_dir / "*.tar.gz")))
    assert wheels, "No wheel built in dist/"
    assert sdists, "No sdist built in dist/"

    whl_path = Path(wheels[0])
    sdist_path = Path(sdists[0])

    print(f"\n[2/4] Inspecting Wheel Archive ({whl_path.name}):")
    with zipfile.ZipFile(whl_path, "r") as zf:
        whl_members = zf.namelist()
        for m in whl_members:
            print(f"  WHL: {m}")

    print(f"\n[3/4] Inspecting Sdist Archive ({sdist_path.name}):")
    with tarfile.open(sdist_path, "r:gz") as tf:
        sdist_members = tf.getnames()
        for m in sdist_members:
            print(f"  SDIST: {m}")

    has_rust_crate = any("crates/freshcore" in m for m in sdist_members)
    has_native_ext = any(m.endswith((".so", ".pyd", ".dylib")) for m in whl_members)
    print(f"\nArtifact Backend Telemetry: crates/freshcore in sdist={has_rust_crate}, native_ext in wheel={has_native_ext}")

    print("\n[4/4] Clean-Room Isolated Venv Smoke Test in /tmp...")
    with tempfile.TemporaryDirectory(prefix="freshdata_cleanroom_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        venv_dir = tmp_path / "venv"
        venv.create(venv_dir, with_pip=True)
        py_bin = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

        run_cmd([str(py_bin), "-m", "pip", "install", "--quiet", f"{whl_path}[polars,duckdb]"])
        # Install runtime engine dependencies from local environment site-packages if needed
        smoke_code = """
import sys, os, pandas as pd
import freshdata as fd
assert os.getcwd() != sys.path[0] or "freshdata_cleanroom_" in os.getcwd()
df = pd.DataFrame({
    "user_id": pd.Series([101, 102, 102, 104], dtype="Int64"),
    "name": pd.Series([" Alice ", "Bob", "Bob", "N/A"], dtype="string"),
    "score": pd.Series([10.0, 20.0, 20.0, pd.NA], dtype="Float64"),
})
for eng in ("pandas", "polars", "duckdb"):
    res, rep = fd.clean(df, engine=eng, drop_duplicates=True, return_report=True)
    assert len(res) == 3, f"Engine {eng} failed row count"
    assert str(res["user_id"].dtype) == "Int64", f"Engine {eng} lost Int64 dtype"
print("CLEANROOM_SMOKE_PASS: pandas, polars, duckdb verified")
"""
        res = run_cmd([str(py_bin), "-c", smoke_code], cwd=tmp_path)
        print(res.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
