"""Generated-code sandbox: positive contract and one negative test per
prohibited action."""

from __future__ import annotations

import json
import signal
import subprocess
import sys

import pandas as pd
import pytest
from benchmarks.truthbench import generated_code as gc
from benchmarks.truthbench.fixtures.base import FixtureBuilder
from benchmarks.truthbench.generated_code import (
    GeneratedCodeResult,
    verify_generated_code,
)
from benchmarks.truthbench.models import Disposition


def _fixture():
    frame = pd.DataFrame(
        {
            "account": ["TB-1", "TB-2"],
            "amount": [1.5, 2.5],
            "memo": ["tb.gen+leak@example.invalid", "fine"],
        },
        index=["r1", "r2"],
    )
    builder = FixtureBuilder("v1", "gen", frame, protected_columns=("account",))
    builder.inject(
        "r1",
        "memo",
        "tb.gen+leak@example.invalid",
        Disposition.FLAG,
        family="pii-memo",
        sensitive=True,
    )
    return builder.build()


GOOD = """
import pandas as pd

df = pd.read_csv("your_data.csv")
print("rows:", len(df))
"""

#: What a child that started the harness prints first.
_STARTED = gc._STARTED_MARKER + "\n"


def test_wellformed_code_passes_all_stages():
    result = verify_generated_code(GOOD, _fixture())
    assert isinstance(result, GeneratedCodeResult)
    assert result.stages == ("parse", "allowlist", "compile", "execute")
    assert result.passed, result.failures


def test_syntax_error_fails_at_parse():
    result = verify_generated_code("def broken(:\n  pass", _fixture())
    assert not result.passed
    assert any("does not parse" in f for f in result.failures)


@pytest.mark.parametrize(
    ("snippet", "needle"),
    [
        ("import socket\n", "forbidden import: socket"),
        ("import subprocess\n", "forbidden import: subprocess"),
        ("from os import system\n", "forbidden import"),
        ("import urllib.request\n", "forbidden import"),
        ("eval('1+1')\n", "forbidden call: eval"),
        ("exec('x = 1')\n", "forbidden call: exec"),
        ("open('/etc/passwd')\n", "forbidden call: open"),
        ("__import__('socket')\n", "forbidden call: __import__"),
        ("getattr(object, 'x', None)\n", "forbidden call: getattr"),
        ("x = ().__class__\n", "forbidden dunder attribute"),
        ("b = __builtins__\n", "forbidden name: __builtins__"),
    ],
)
def test_prohibited_actions_are_rejected_statically(snippet, needle):
    result = verify_generated_code(snippet, _fixture())
    assert not result.passed
    assert any(needle in failure for failure in result.failures), result.failures


def test_runtime_poison_blocks_banned_module_even_if_statically_allowed(monkeypatch):
    # Defence in depth: even if the static allowlist were relaxed (or missed a
    # path), importing a poisoned module still fails inside the sandbox.
    monkeypatch.setattr(
        gc, "ALLOWED_IMPORTS", frozenset({*gc.ALLOWED_IMPORTS, "socket"})
    )
    result = verify_generated_code("import socket\n", _fixture())
    assert result.infrastructure_failure is None, result.failures
    assert not result.passed
    assert any("generated code exited" in f for f in result.failures)


def test_timeout_is_enforced():
    result = verify_generated_code(
        "while True:\n    pass\n", _fixture(), timeout=3.0
    )
    assert result.infrastructure_failure is None, result.failures
    assert not result.passed
    assert any("timeout" in f for f in result.failures)


def test_pii_canary_in_source_is_reported():
    leaking = 'print("contact: tb.gen+leak@example.invalid")\n'
    result = verify_generated_code(leaking, _fixture())
    assert not result.passed
    assert any("generated source leaked canary" in f for f in result.failures)


def test_pii_canary_in_stdout_is_reported():
    code = (
        "import pandas as pd\n"
        'df = pd.read_csv("your_data.csv")\n'
        'print(df["memo"].tolist())\n'
    )
    result = verify_generated_code(code, _fixture())
    assert result.infrastructure_failure is None, result.failures
    assert not result.passed
    assert any("stdout leaked canary" in f for f in result.failures)
    assert "example.invalid" not in result.stdout  # evidence itself is redacted


def test_input_file_overwrite_is_reported():
    code = (
        "import pandas as pd\n"
        'df = pd.read_csv("your_data.csv")\n'
        'df.head(1).to_csv("your_data.csv", index=False)\n'
    )
    result = verify_generated_code(code, _fixture())
    assert result.infrastructure_failure is None, result.failures
    assert not result.passed
    assert any("modified its input file" in f for f in result.failures)


def _fake_interpreter(tmp_path, body):
    """An executable standing in for ``python`` that runs *body* instead."""
    script = tmp_path / "fake_python"
    script.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shebang interpreter")
def test_sandbox_pins_native_threads_and_enables_faulthandler(tmp_path):
    python = _fake_interpreter(
        tmp_path,
        "import json, os, sys\n"
        f"print({gc._STARTED_MARKER!r})\n"
        "print(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}))\n",
    )
    result = verify_generated_code(GOOD, _fixture(), python=python)
    assert result.passed, result.failures
    seen = json.loads(result.stdout)
    assert seen["argv"][:3] == ["-I", "-X", "faulthandler"]
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "RAYON_NUM_THREADS",
    ):
        assert seen["env"][var] == "1", var


def test_native_crash_reports_signal_exit_and_faulthandler_dump(monkeypatch):
    # The flake seen in CI was a bare "generated code exited -11: " with no
    # evidence; a crash must now carry the exit signal and the native dump.
    # The crash is simulated: killing a real child with SIGSEGV makes macOS
    # file a crash report (and may pop a dialog) on every local test run.
    # That -X faulthandler reaches the child is covered by the test above.
    dump = (
        "Fatal Python error: Segmentation fault\n\n"
        "Current thread 0x0000000000000001 (most recent call first):\n"
        '  File "harness.py", line 1 in <module>\n'
    )

    def crashed_child(args, **kwargs):
        return subprocess.CompletedProcess(args, -signal.SIGSEGV, "", dump)

    monkeypatch.setattr(gc.subprocess, "run", crashed_child)
    result = verify_generated_code(GOOD, _fixture())
    assert not result.passed
    [failure] = [f for f in result.failures if "exited" in f]
    assert f"exited {-signal.SIGSEGV}" in failure
    assert "Segmentation fault" in failure


def test_native_crash_failure_keeps_stack_not_module_list(monkeypatch):
    # faulthandler writes the stack first and a long "Extension modules" line
    # last. Keeping only the tail of stderr reported just the module list and
    # lost the frame that located the crash (seen in CI on a copilot case).
    modules = ", ".join(f"pandas._libs.module_{i}" for i in range(53))
    dump = (
        "Fatal Python error: Segmentation fault\n\n"
        "Current thread 0x0000000000000001 (most recent call first):\n"
        '  File "pandas/core/tools/numeric.py", line 235 in to_numeric\n'
        '  File "generated_pipeline.py", line 20 in <module>\n\n'
        f"Extension modules: {modules} (total: 53)\n"
    )
    assert len(dump) > 800

    def crashed_child(args, **kwargs):
        return subprocess.CompletedProcess(args, -signal.SIGSEGV, "", dump)

    monkeypatch.setattr(gc.subprocess, "run", crashed_child)
    result = verify_generated_code(GOOD, _fixture())
    [failure] = [f for f in result.failures if "exited" in f]
    assert "line 235 in to_numeric" in failure
    assert "Extension modules" not in failure


def test_ordinary_failure_keeps_traceback_tail(monkeypatch):
    stderr = "noise\n" * 300 + "ValueError: bad column\n"

    def failed_child(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, _STARTED, stderr)

    monkeypatch.setattr(gc.subprocess, "run", failed_child)
    result = verify_generated_code(GOOD, _fixture())
    [failure] = [f for f in result.failures if "exited" in f]
    assert failure.endswith("ValueError: bad column\n")


_SEGV_DUMP = (
    "Fatal Python error: Segmentation fault\n\n"
    "Current thread 0x0000000000000001 (most recent call first):\n"
    '  File "pandas/core/tools/numeric.py", line 235 in to_numeric\n'
)


def _scripted_children(monkeypatch, outcomes):
    """Replace the sandbox child with *outcomes* (return code, stderr) in order."""
    calls = []

    def child(args, **kwargs):
        code, stderr = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(code)
        return subprocess.CompletedProcess(args, code, _STARTED, stderr)

    monkeypatch.setattr(gc.subprocess, "run", child)
    return calls


def test_native_crash_is_retried_once_and_recorded(monkeypatch):
    calls = _scripted_children(monkeypatch, [(-signal.SIGSEGV, _SEGV_DUMP), (0, "")])
    result = verify_generated_code(GOOD, _fixture())
    assert calls == [-signal.SIGSEGV, 0]
    assert result.passed, result.failures
    [crash] = result.native_crashes
    assert f"exited {-signal.SIGSEGV}" in crash
    assert "line 235 in to_numeric" in crash


def test_repeated_native_crash_still_fails(monkeypatch):
    calls = _scripted_children(monkeypatch, [(-signal.SIGSEGV, _SEGV_DUMP)])
    result = verify_generated_code(GOOD, _fixture())
    assert len(calls) == 1 + gc._NATIVE_CRASH_RETRIES
    assert not result.passed
    assert any(f"exited {-signal.SIGSEGV}" in f for f in result.failures)
    assert len(result.native_crashes) == len(calls)


def test_ordinary_failure_is_not_retried(monkeypatch):
    calls = _scripted_children(monkeypatch, [(1, "ValueError: bad column\n")])
    result = verify_generated_code(GOOD, _fixture())
    assert calls == [1]
    assert not result.passed
    assert result.native_crashes == ()


_POSIX_ENV_KEYS = {
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "FRESHDATA_NO_NETWORK",
    "HOME",
    "TMPDIR",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "LANG",
}
_HOST_ENV = {"SYSTEMROOT": r"C:\Windows", "PATH": r"C:\bin", "USERPROFILE": r"C:\u"}


@pytest.mark.parametrize("key", ["SYSTEMROOT", "SystemRoot"])
def test_sandbox_env_keeps_system_root_on_windows(tmp_path, key):
    # CPython <= 3.10 on Windows cannot seed hash randomization without it.
    host = {key: r"C:\Windows", "PATH": r"C:\bin", "USERPROFILE": r"C:\u"}
    env = gc._sandbox_env(tmp_path, os_name="nt", environ=host)
    assert env["SystemRoot"] == r"C:\Windows"
    assert set(env) == _POSIX_ENV_KEYS | {"SystemRoot"}


def test_sandbox_env_on_posix_is_unchanged(tmp_path):
    env = gc._sandbox_env(tmp_path, os_name="posix", environ=_HOST_ENV)
    assert env == {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "FRESHDATA_NO_NETWORK": "1",
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "LANG": "C.UTF-8",
    }


def test_windows_child_receives_system_root(monkeypatch):
    class _WindowsOs:
        name = "nt"
        environ = _HOST_ENV

    seen = {}

    def child(args, **kwargs):
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(args, 0, _STARTED + "rows: 2\n", "")

    monkeypatch.setattr(gc, "os", _WindowsOs)
    monkeypatch.setattr(gc.subprocess, "run", child)
    result = verify_generated_code(GOOD, _fixture())
    assert result.passed, result.failures
    assert seen["SystemRoot"] == r"C:\Windows"
    assert "PATH" not in seen
    assert result.stdout == "rows: 2\n"


_STARTUP_FATAL = (
    "Fatal Python error: _Py_HashRandomization_Init: failed to get random "
    "numbers to initialize Python\nPython runtime state: preinitialized\n\n"
)


@pytest.mark.parametrize(
    "code",
    [
        GOOD,
        'import pandas as pd\nprint(pd.read_csv("your_data.csv")["memo"].tolist())\n',
        'import pandas as pd\npd.DataFrame().to_csv("your_data.csv")\n',
    ],
)
def test_child_that_cannot_start_is_an_infrastructure_failure(monkeypatch, code):
    # Seen on Windows + CPython 3.9: the child died at startup, and each case
    # reported an ordinary "generated code exited 1", so the canary and
    # overwrite checks passed vacuously.
    calls = []

    def child_never_started(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", _STARTUP_FATAL)

    monkeypatch.setattr(gc.subprocess, "run", child_never_started)
    result = verify_generated_code(code, _fixture())
    assert not result.passed
    assert result.infrastructure_failure
    assert "_Py_HashRandomization_Init" in result.infrastructure_failure
    assert result.failures == (result.infrastructure_failure,)
    assert "execute" not in result.stages
    assert not any("generated code exited" in f for f in result.failures)
    assert len(calls) == 1  # a startup failure is not retried as a native crash


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shebang interpreter")
def test_child_exiting_cleanly_without_running_the_harness_fails_closed(tmp_path):
    python = _fake_interpreter(tmp_path, "import sys\nsys.exit(0)\n")
    result = verify_generated_code(GOOD, _fixture(), python=python)
    assert not result.passed
    assert "exited with code 0 before starting the harness" in (
        result.infrastructure_failure or ""
    )


def test_normal_run_reports_generated_stdout_without_the_start_marker():
    result = verify_generated_code(GOOD, _fixture())
    assert result.passed, result.failures
    assert result.infrastructure_failure is None
    assert result.stdout == "rows: 2\n"
