"""Production-grade verification of Copilot-generated code.

``ast.parse`` alone proves nothing about behavior.  Verification here is a
pipeline: parse, strict AST allowlist, compile, then execution with
``python -I`` in a throwaway working directory with a restricted environment,
an import poison for network/process modules, a hard timeout, and a defined
input contract (the fixture written as the CSV the generated script reads).
Afterwards the protected cells must be untouched and no PII canary may appear
in the source, stdout, stderr, produced files, or the exception text.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .privacy import SinkScanner

#: Modules the generated pipeline may import.  Everything else is rejected
#: statically and poisoned at runtime.
ALLOWED_IMPORTS: frozenset[str] = frozenset({"pandas", "numpy", "freshdata"})

#: Builtins whose call is forbidden inside generated code.
BANNED_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
    }
)

#: Modules poisoned inside the sandbox so even a missed static path cannot
#: reach the network, spawn processes, or load native code.
POISONED_MODULES: tuple[str, ...] = (
    "socket",
    "ssl",
    "http",
    "urllib",
    "urllib3",
    "requests",
    "ftplib",
    "smtplib",
    "telnetlib",
    "subprocess",
    "ctypes",
    "multiprocessing",
    "webbrowser",
)

_INPUT_BASENAME = "your_data.csv"


@dataclass(frozen=True)
class GeneratedCodeResult:
    """Outcome of one full verification run."""

    passed: bool
    failures: tuple[str, ...]
    stdout: str = ""
    stderr: str = ""
    produced_files: tuple[str, ...] = ()
    stages: tuple[str, ...] = field(default=())


def _allowlist_failures(tree: ast.AST) -> list[str]:
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    failures.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                failures.append(f"forbidden import: from {node.module or '.'}")
        elif isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name) and callee.id in BANNED_CALLS:
                failures.append(f"forbidden call: {callee.id}()")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            failures.append(f"forbidden dunder attribute access: .{node.attr}")
        elif isinstance(node, ast.Name) and node.id == "__builtins__":
            failures.append("forbidden name: __builtins__")
    return failures


def _harness(code_path: str, input_name: str) -> str:
    poison = ", ".join(repr(name) for name in POISONED_MODULES)
    return f"""
import sys

# Pre-import the allowed libraries so their own internal use of stdlib
# modules (pandas imports subprocess for locale probing) completes first...
import pandas  # noqa: F401
import freshdata  # noqa: F401

# ...then deny-by-poison: any attempt by the generated code to import
# network/process/native modules fails immediately.
for _name in ({poison},):
    sys.modules[_name] = None

code = open({code_path!r}, encoding="utf-8").read()
namespace = {{"__name__": "__main__", "__file__": {code_path!r}}}
exec(compile(code, "generated_pipeline.py", "exec"), namespace)
"""


def verify_generated_code(  # noqa: PLR0915 - one linear verification pipeline
    code: str,
    fixture: Any,
    *,
    timeout: float = 60.0,
    python: str | None = None,
) -> GeneratedCodeResult:
    """Run the full verification pipeline for one generated script."""

    failures: list[str] = []
    stages: list[str] = []
    scanner = None
    canaries = dict(getattr(fixture, "pii_canaries", {}) or {})
    if canaries:
        scanner = SinkScanner.from_canaries(canaries, key=b"truthbench-generated-code-v1")

    def leaks(label: str, payload: Any) -> None:
        if scanner is None:
            return
        for leak in scanner.scan(payload):
            failures.append(f"{label} leaked canary {leak.canary_id} at {leak.path}")

    # 1. parse
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return GeneratedCodeResult(
            False, (f"generated code does not parse: {exc.msg}",), stages=("parse",)
        )
    stages.append("parse")

    # 2. strict AST allowlist
    allow = _allowlist_failures(tree)
    failures.extend(allow)
    stages.append("allowlist")

    # 3. compile
    try:
        compile(code, "generated_pipeline.py", "exec")
    except (SyntaxError, ValueError) as exc:
        failures.append(f"generated code does not compile: {exc}")
        return GeneratedCodeResult(False, tuple(failures), stages=tuple(stages))
    stages.append("compile")

    leaks("generated source", code)
    if failures:
        return GeneratedCodeResult(False, tuple(failures), stages=tuple(stages))

    frame = getattr(fixture, "frame", fixture)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("generated-code verification requires a pandas fixture frame")

    # 4-9. isolated execution with the defined input contract
    with tempfile.TemporaryDirectory(prefix="truthbench-gen-") as tmp:
        workdir = Path(tmp)
        input_path = workdir / _INPUT_BASENAME
        frame.to_csv(input_path, index=False)
        code_path = workdir / "generated_pipeline.py"
        code_path.write_text(code, encoding="utf-8")
        harness_path = workdir / "harness.py"
        harness_path.write_text(
            _harness(str(code_path), _INPUT_BASENAME), encoding="utf-8"
        )
        env = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "FRESHDATA_NO_NETWORK": "1",
            "HOME": str(workdir),
            "TMPDIR": str(workdir),
        }
        interpreter = python or sys.executable
        try:
            proc = subprocess.run(  # noqa: PLW1510 - exit code inspected below
                [interpreter, "-I", str(harness_path)],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return GeneratedCodeResult(
                False,
                (*failures, f"generated code exceeded the {timeout:.0f}s timeout"),
                stages=(*stages, "execute"),
            )
        stages.append("execute")
        stdout, stderr = proc.stdout, proc.stderr
        if proc.returncode != 0:
            safe_stderr: Any = stderr
            if scanner is not None:
                safe_stderr = scanner.redact(stderr)
            failures.append(
                f"generated code exited {proc.returncode}: {str(safe_stderr)[-800:]}"
            )

        produced = tuple(
            sorted(
                str(path.relative_to(workdir))
                for path in workdir.rglob("*")
                if path.is_file()
                and path not in {input_path, code_path, harness_path}
            )
        )

        # 10-11. input contract and protected-cell identity
        after = pd.read_csv(input_path, dtype=str, keep_default_na=False)
        before = pd.read_csv(
            _rewrite(frame, workdir), dtype=str, keep_default_na=False
        )
        if not after.equals(before):
            failures.append("generated code modified its input file in place")
        protected = tuple(getattr(fixture, "protected_columns", ()) or ())
        for column in protected:
            if column in after.columns and not after[column].equals(before[column]):
                failures.append(f"generated code modified protected column {column!r}")
        # 12. PII canaries in any external sink
        leaks("stdout", stdout)
        leaks("stderr", stderr)
        for name in produced:
            try:
                payload = (workdir / name).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            leaks(f"produced file {name}", payload)

        safe_out: Any = stdout
        safe_err: Any = stderr
        if scanner is not None:
            safe_out = scanner.redact(stdout)
            safe_err = scanner.redact(stderr)
        return GeneratedCodeResult(
            not failures,
            tuple(failures),
            stdout=str(safe_out),
            stderr=str(safe_err),
            produced_files=produced,
            stages=tuple(stages),
        )


def _rewrite(frame: pd.DataFrame, workdir: Path) -> Path:
    """Write a reference copy of the input CSV for byte-comparison."""

    reference = workdir / ".reference_input.csv"
    frame.to_csv(reference, index=False)
    return reference


__all__ = [
    "ALLOWED_IMPORTS",
    "BANNED_CALLS",
    "POISONED_MODULES",
    "GeneratedCodeResult",
    "verify_generated_code",
]
