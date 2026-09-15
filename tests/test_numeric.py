"""safe_to_numeric: pandas parity, crash safety and call-site coverage.

pandas < 3 can segfault in ``to_numeric`` on a cell that starts with scientific
notation whose exponent overflows a C int (pandas-dev/pandas#62617). Tests
that hand such a token to a *raw* ``pd.to_numeric`` run in a child
interpreter; in-process tests only use exponents far below the C-int limit, or
a tripwire that stops before pandas parses anything.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import freshdata
from freshdata._numeric import _has_unsafe_scientific_exponent, safe_to_numeric

PANDAS_MAJOR = int(pd.__version__.split(".")[0])

# Each of the first four segfaulted pd.to_numeric on Linux x86_64 / pandas 2.3.3.
_CRASH_TOKENS = [
    "81e3104049863b72",
    "4e492493924924",
    "1e3104049863",
    "1e2147483648",
    "  -7.5E+99999999999xyz",
    ".5e-3104049863 tail",
]
# Mirrors precise_xstrtod as reported in pandas-dev/pandas#62617: up to 17
# exponent digits accumulated in a C int.
_C_INT_EXPONENT = re.compile(r"\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?(\d{1,17})")


def _overflows_c_int_exponent(value: object) -> bool:
    if isinstance(value, bytes):
        value = value.decode("latin-1")
    match = _C_INT_EXPONENT.match(value) if isinstance(value, str) else None
    return match is not None and int(match.group(1)) > 2**31 - 1


def _run_child(code: str, stdin: str | None = None) -> str:
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(code)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    return proc.stdout


# -- parity with pandas --------------------------------------------------------


def _outcome(fn, values, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # errors="ignore" on pandas >= 2.2
        try:
            return "ok", fn(values, **kwargs)
        except (TypeError, ValueError) as exc:
            return "err", (type(exc), str(exc))


def _assert_same_outcome(values, **kwargs):
    kind, expected = _outcome(pd.to_numeric, values, kwargs)
    got_kind, got = _outcome(safe_to_numeric, values, kwargs)
    assert (got_kind, type(got)) == (kind, type(expected)), (kwargs, expected, got)
    if kind == "err":
        assert got == expected, (kwargs, expected, got)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(got, expected)
    elif isinstance(expected, pd.Index):
        pd.testing.assert_index_equal(got, expected, exact=True)
    elif isinstance(expected, np.ndarray):  # Series comparison handles pd.NA cells
        assert got.dtype == expected.dtype
        pd.testing.assert_series_equal(pd.Series(got), pd.Series(expected))
    elif isinstance(expected, pd.api.extensions.ExtensionArray):
        pd.testing.assert_extension_array_equal(got, expected)
    else:  # scalar
        both_missing = pd.api.types.is_scalar(got) and pd.isna(got) and pd.isna(expected)
        assert both_missing or got == expected, (kwargs, expected, got)


_MODES = [
    {"errors": "coerce"},
    {"errors": "raise"},
    {"errors": "ignore"},
    {},
    {"errors": "coerce", "downcast": "integer"},
    {"errors": "coerce", "downcast": "float"},
    {"errors": "raise", "downcast": "unsigned"},
]
if PANDAS_MAJOR >= 2:
    _MODES.append({"errors": "coerce", "dtype_backend": "numpy_nullable"})


def _edge_inputs():
    # The guarded exponents here have ten digits (so the guard masks them) but
    # stay far below INT_MAX, the limit of pandas' exponent accumulator
    # (pandas-dev/pandas#62617), and are positive, so pandas rejects them too.
    return {
        "object_mixed": pd.Series(
            ["1", "2.5", "1e1000000000abc", None, np.nan, " 7 ", 4, 5.5],
            dtype=object, name="v", index=list("abcdefgh"),
        ),
        "object_clean_ints": pd.Series(["1", "2", "3"], name="n", index=[9, 8, 7]),
        "object_bad_before_unsafe": pd.Series(["abc", "1e1000000000"]),
        "object_list_cell_before_unsafe": pd.Series(["1", ["x"], "1e1000000000"], dtype=object),
        "string_dtype": pd.Series(["1", "1e1000000000", None, "2e3"], dtype="string"),
        "string_dtype_clean": pd.Series(["1", None, "2e3"], dtype="string", name="s"),
        "bytes_cells": pd.Series([b"12", "3", b"1e1000000000"], dtype=object),
        "bytes_clean": pd.Series([b"12", "3"], dtype=object),
        "categorical": pd.Series(["1", "1e1000000000x", "2"], dtype="category"),
        "float": pd.Series([1.5, np.nan, 3.0], name="f"),
        "nullable_int": pd.Series([1, None, 300], dtype="Int64"),
        "all_missing": pd.Series([None, np.nan], dtype=object),
        "empty": pd.Series([], dtype=object),
        "list": ["1", "1e1000000000x", None],
        "tuple": ("4", "5e+1000000000"),
        "ndarray_object": np.array(["1", "9e1000000000", None], dtype=object),
        "ndarray_unicode": np.array(["1", "9e1000000000"]),
        "ndarray_float": np.array([1.0, 2.0]),
        "index": pd.Index(["1", "2e1000000000", "3"], name="i"),
        "string_array": pd.array(["1", "7e1000000000", None], dtype="string"),
        "scalar_unsafe": "1e1000000000",
        "scalar_unsafe_bytes": b"1e1000000000",
        "scalar_clean": "12",
        "scalar_none": None,
        "scalar_float": 2.5,
    }


@pytest.mark.parametrize("mode", _MODES, ids=lambda m: ",".join(f"{k}={v}" for k, v in m.items()))
@pytest.mark.parametrize("name", list(_edge_inputs()))
def test_matches_pandas_on_edge_inputs(name, mode):
    _assert_same_outcome(_edge_inputs()[name], **mode)


@pytest.mark.parametrize(
    ("token", "unsafe"),
    [
        ("1e999999999", False),  # nine exponent digits: left to pandas
        ("-1e-999999999", False),
        ("1e0000000001", False),  # leading zeros keep pandas' accumulator at 0
        ("4.9e-324", False),
        ("1e400", False),
        ("1e1000000000", True),  # ten digits: may overflow after the mantissa adjustment
        ("1e-1000000000", True),
        ("1e2147483648", True),
        ("1e" + "9" * 5000, True),  # beyond int()'s string limit on 3.11+: ValueError path
        (b"1e1000000000", True),
    ],
)
def test_unsafe_exponent_bound_is_ten_significant_digits(token, unsafe):
    assert _has_unsafe_scientific_exponent(token) is unsafe


def test_ten_digit_negative_exponent_is_unparseable():
    # The one intended difference from pandas: it would underflow this cell to
    # 0.0, but a ten-digit exponent is not proven safe, so it stays text.
    parsed = safe_to_numeric(pd.Series(["1e-1000000000", "2"]), errors="coerce")
    assert pd.isna(parsed.iloc[0]) and parsed.iloc[1] == 2


def test_rejects_what_pandas_rejects():
    for kwargs in ({"errors": "bogus"}, {"downcast": "bogus"}):
        _assert_same_outcome(pd.Series(["1", "1e1000000000"]), **kwargs)
    _assert_same_outcome(np.array([["1", "1e1000000000"]], dtype=object))  # 2-D


def test_matches_pandas_on_random_hex_tokens():
    """4,000 seeded hash-like tokens plus boundary cases, compared in every
    mode. The raw pandas baseline runs in a child interpreter: the tokens that
    overflow pandas' exponent parser are filtered out here, and if that filter
    ever misses one, the child crashes and this test fails instead of the run."""
    rng = np.random.default_rng(20260915)
    hexdigits = np.array(list("0123456789abcdef"))
    tokens = ["".join(rng.choice(hexdigits, 16)) for _ in range(4000)]
    tokens += ["1e880f3f8de2590b", "1e1000000000abc", "2.5e-309x", "7e123456789z", "1e308x"]
    tokens += ["1e308", " 1e5 ", "12e3", "e999", "1e", "1e+", "3.5", "-0", "abc", None]
    # Valid subnormal, underflow and out-of-range values parse exactly as pandas.
    tokens += ["4.9e-324", "1e-320", "1e-310", "5e-400", "1e309", "1e400", "-1e400"]
    tokens += ["2.2e-308", "7e123456789", "1e999999999", "-1e-999999999", "1e0000000001"]
    # Guarded (ten exponent digits, below INT_MAX) and rejected by pandas as well.
    tokens += ["5e2000000000x", "3.5e-1999999999 tail", "4e1000000000", "7e+1000000000"]
    safe = [t for t in tokens if not _overflows_c_int_exponent(t)]
    assert sum(1 for t in safe if _has_unsafe_scientific_exponent(t)) >= 5
    assert len(safe) < len(tokens)  # crash tokens were generated and held back
    out = _run_child(
        """
        import json
        import sys
        import warnings

        import numpy as np
        import pandas as pd
        from freshdata._numeric import safe_to_numeric

        warnings.simplefilter("ignore", FutureWarning)
        tokens = json.load(sys.stdin)
        numeric = [t for t in tokens if t is not None and t.strip().lstrip("-").isdigit()]

        def outcome(fn, values, **kw):
            try:
                return "ok", fn(values, **kw)
            except (TypeError, ValueError) as exc:
                return "err", (type(exc), str(exc))

        def same(values, **kw):
            kind, expected = outcome(pd.to_numeric, values, **kw)
            got_kind, got = outcome(safe_to_numeric, values, **kw)
            assert got_kind == kind, (kw, expected, got)
            if kind == "err":
                assert got == expected, (kw, expected, got)
            elif isinstance(expected, pd.Series):
                pd.testing.assert_series_equal(got, expected)
            elif isinstance(expected, pd.Index):
                pd.testing.assert_index_equal(got, expected, exact=True)
            elif isinstance(expected, np.ndarray):
                assert got.dtype == expected.dtype
                pd.testing.assert_series_equal(pd.Series(got), pd.Series(expected))
            else:
                assert type(got) is type(expected)
                assert (pd.isna(got) and pd.isna(expected)) or got == expected

        index = pd.RangeIndex(10, 10 + len(tokens))
        for dtype in (object, "string"):
            values = pd.Series(tokens, dtype=dtype, name="token", index=index)
            for errors in ("coerce", "raise", "ignore"):
                same(values, errors=errors)
            same(values, errors="coerce", downcast="float")
            same(pd.Series(numeric, dtype=dtype), downcast="integer")
        same(pd.Index(tokens, name="token"), errors="coerce")
        same(list(tokens), errors="coerce")
        same(np.array(tokens, dtype=object), errors="ignore")
        for token in tokens[:400] + tokens[-10:]:
            for errors in ("coerce", "raise", "ignore"):
                same(token, errors=errors)
        print("ok")
        """,
        stdin=json.dumps(safe),
    )
    assert out.strip().endswith("ok")


# -- crash tokens never reach the parser ---------------------------------------


@pytest.fixture
def tripwire(monkeypatch):
    """Fail (instead of crashing) if an overflowing exponent reaches pandas."""
    real = pd.to_numeric

    def guarded(arg, *args, **kwargs):
        cells = [arg] if np.ndim(arg) == 0 else list(np.asarray(arg, dtype=object).ravel())
        leaked = [cell for cell in cells if _overflows_c_int_exponent(cell)]
        assert not leaked, f"overflowing exponent reached pandas: {leaked!r}"
        return real(arg, *args, **kwargs)

    monkeypatch.setattr(pd, "to_numeric", guarded)


@pytest.mark.parametrize(
    "container",
    [
        lambda v: pd.Series(v, dtype=object, name="t"),
        lambda v: pd.Series(v, dtype="string"),
        lambda v: pd.Series(v, dtype="category"),
        list,
        tuple,
        lambda v: np.array(v, dtype=object),
        lambda v: pd.Index(v, dtype=object),
        lambda v: pd.array(v, dtype="string"),
    ],
    ids=["object", "string", "category", "list", "tuple", "ndarray", "index", "string_array"],
)
def test_crash_tokens_are_kept_from_pandas(tripwire, container):
    values = container(["1", *_CRASH_TOKENS, "3"])
    parsed = np.asarray(safe_to_numeric(values, errors="coerce"), dtype=float)
    assert parsed[0] == 1 and parsed[-1] == 3
    assert np.isnan(parsed[1:-1]).all()

    message = r'Unable to parse string "81e3104049863b72" at position 1'
    with pytest.raises(ValueError, match=message):
        safe_to_numeric(values)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ignored = safe_to_numeric(values, errors="ignore")
    assert list(np.asarray(ignored, dtype=object)) == ["1", *_CRASH_TOKENS, "3"]


@pytest.mark.parametrize("token", [*_CRASH_TOKENS, b"81e3104049863b72"])
def test_crash_token_scalars_are_kept_from_pandas(tripwire, token):
    assert np.isnan(safe_to_numeric(token, errors="coerce"))
    with pytest.raises(ValueError, match="Unable to parse string"):
        safe_to_numeric(token)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert safe_to_numeric(token, errors="ignore") is token


def test_migrated_public_apis_survive_crash_tokens():
    """Runs in a child interpreter: on pandas < 3 an unguarded call site can
    kill the process with SIGSEGV. The tripwire makes a bypass fail on every
    platform, including those where the overflow happens not to crash."""
    out = _run_child(
        f"""
        import os
        import re
        import sys

        import numpy as np
        import pandas as pd

        _real = pd.to_numeric
        _exp = re.compile({_C_INT_EXPONENT.pattern!r})

        def _tripwire(arg, *args, **kwargs):
            cells = [arg] if np.ndim(arg) == 0 else np.asarray(arg, dtype=object).ravel()
            for cell in cells:
                text = cell.decode("latin-1") if isinstance(cell, bytes) else cell
                match = _exp.match(text) if isinstance(text, str) else None
                if match and int(match.group(1)) > 2**31 - 1:
                    sys.stderr.write(f"unguarded to_numeric reached {{cell!r}}\\n")
                    sys.stderr.flush()
                    os._exit(97)
            return _real(arg, *args, **kwargs)

        pd.to_numeric = _tripwire

        import freshdata as fd
        from freshdata.domains import run_domain

        tokens = {_CRASH_TOKENS!r}
        column = (tokens + ["12.5", "7", "3.25"]) * 3

        report = fd.validate_fields(
            pd.DataFrame({{"amount": column}}), {{"amount": "currency_amount"}}
        )
        assert report is not None

        frame = pd.DataFrame({{
            "transaction_id": range(len(column)),
            "debit": column,
            "credit": column,
            "amount": column,
        }})
        _, outcome = run_domain(frame, "finance")
        assert outcome is not None

        out = fd.clean(pd.DataFrame({{"token": tokens, "n": range(len(tokens))}}))
        assert out["token"].notna().all()
        print("ok")
        """
    )
    assert out.strip().endswith("ok")


# -- no call site bypasses the guard -------------------------------------------

_PACKAGE = Path(freshdata.__file__).resolve().parent

# Files whose direct calls are left for a follow-up PR.
_DEFERRED = frozenset({"enterprise/contracts.py"})

# Direct calls whose argument is provably numeric: pandas never runs its
# string parser on them. Counts must match exactly, so a new call in the same
# file has to be reviewed (and normally routed through safe_to_numeric).
_NUMERIC_ONLY = {
    # guarded by is_numeric_dtype(s) and not is_bool_dtype(s)
    "streaming/_state.py": 1,
    "streaming/_drift.py": 1,
    "imputation/missforest.py": 1,
    # is_integer_dtype / float64 branches of _downcast_numeric
    "steps/memory.py": 2,
}


def _to_numeric_references(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    count = 0
    for node in ast.walk(tree):
        name = getattr(node, "attr", None) or getattr(node, "id", None)
        if isinstance(node, (ast.Attribute, ast.Name)) and name == "to_numeric":
            count += 1
        elif isinstance(node, ast.ImportFrom):
            count += sum(alias.name == "to_numeric" for alias in node.names)
    return count


def test_every_to_numeric_call_goes_through_safe_to_numeric():
    found = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        rel = path.relative_to(_PACKAGE).as_posix()
        if rel == "_numeric.py" or rel in _DEFERRED:
            continue
        count = _to_numeric_references(path)
        if count:
            found[rel] = count
    unguarded = {rel: n for rel, n in found.items() if _NUMERIC_ONLY.get(rel) != n}
    assert not unguarded, (
        "pd.to_numeric can segfault on pandas < 3 (pandas-dev/pandas#62617); "
        f"use freshdata._numeric.safe_to_numeric instead: {unguarded}"
    )
    stale = {rel for rel in _NUMERIC_ONLY if rel not in found}
    assert not stale, f"update _NUMERIC_ONLY: {stale}"
    stale_deferred = {rel for rel in _DEFERRED if not _to_numeric_references(_PACKAGE / rel)}
    assert not stale_deferred, f"remove from _DEFERRED: {stale_deferred}"
