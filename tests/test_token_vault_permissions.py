"""Token vault files are created owner-only.

JsonTokenVault and SqliteTokenVault hold the plaintext token-to-value mapping, so
they must not be readable by other local users under a permissive umask.
"""

from __future__ import annotations

import json
import os
import stat
import warnings

import pandas as pd
import pytest

from freshdata.enterprise import (
    JsonTokenVault,
    MaskingRule,
    PrivacyPolicy,
    PrivacyRule,
    SqliteTokenVault,
    anonymize,
    apply_privacy_policy,
    make_vault,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")

KEY = "unit-test-key"
SSN = "123-45-6789"


@pytest.fixture
def umask():
    """Run with umask 022 (the common default); the test may change it. Restored after."""
    previous = os.umask(0o022)
    try:
        yield os.umask
    finally:
        os.umask(previous)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _make(backend: str, path):
    return JsonTokenVault(path) if backend == "json" else SqliteTokenVault(path)


def _close(vault) -> None:
    close = getattr(vault, "close", None)
    if close is not None:
        close()


def test_poc_anonymize_json_vault_and_sqlite_vault_are_0600(tmp_path, umask):
    jp, sp = tmp_path / "vault.json", tmp_path / "vault.db"
    rule = MaskingRule(
        name="t", columns=("ssn",), strategy="tokenize", reversible=True, key="k",
        token_vault_path=str(jp),
    )
    anonymize(pd.DataFrame({"ssn": [SSN]}), rules=(rule,))
    sv = SqliteTokenVault(sp)
    sv.put("tok_1", SSN)
    sv.close()
    assert SSN in jp.read_text(encoding="utf-8")
    assert {p.name: oct(_mode(p)) for p in (jp, sp)} == {
        "vault.json": "0o600", "vault.db": "0o600",
    }


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_make_vault_creates_owner_only_file(tmp_path, umask, backend):
    path = tmp_path / f"vault.{backend}"
    vault = make_vault(backend, path=path)
    vault.put("tok_1", SSN)
    _close(vault)
    assert _mode(path) == 0o600


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_policy_vault_path_creates_owner_only_file(tmp_path, umask, backend):
    path = tmp_path / f"policy-vault.{backend}"
    rule = PrivacyRule(id="ssn", action="tokenize", reversible=True, columns=("ssn",))
    policy = PrivacyPolicy(
        name="p", rules=(rule,), key=KEY, vault_backend=backend, vault_path=str(path)
    )
    out, report = apply_privacy_policy(pd.DataFrame({"ssn": [SSN]}), policy)
    assert out["ssn"].iloc[0].startswith("tok_")
    assert report.vault_info["backend"] == backend
    assert _mode(path) == 0o600


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_umask_000_still_gives_0600(tmp_path, umask, backend):
    umask(0o000)
    path = tmp_path / f"vault.{backend}"
    vault = _make(backend, path)
    vault.put("tok_1", SSN)
    _close(vault)
    assert _mode(path) == 0o600


def test_json_rewrites_keep_mode_and_content(tmp_path, umask):
    path = tmp_path / "vault.json"
    vault = JsonTokenVault(path)
    vault.put("tok_long", "x" * 200)
    vault.put("tok_b", "b")
    other = JsonTokenVault(path)
    other.put("tok_c", "c")
    vault.save()  # rewrites in place: truncate, then write at offset 0
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "tok_long": "x" * 200, "tok_b": "b", "tok_c": "c",
    }
    assert _mode(path) == 0o600


def test_sqlite_wal_and_shm_files_inherit_0600(tmp_path, umask):
    path = tmp_path / "vault.db"
    vault = SqliteTokenVault(path)
    try:
        vault._conn.execute("PRAGMA journal_mode=WAL")
        vault.put("tok_1", SSN)
        wal = tmp_path / "vault.db-wal"
        assert wal.exists()
        assert _mode(wal) == 0o600
        shm = tmp_path / "vault.db-shm"
        if shm.exists():
            assert _mode(shm) == 0o600
    finally:
        vault.close()
    assert _mode(path) == 0o600


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_new_parent_directory_is_0700(tmp_path, umask, backend):
    parent = tmp_path / "private"
    path = parent / f"vault.{backend}"
    vault = _make(backend, path)
    vault.put("tok_1", SSN)
    _close(vault)
    assert _mode(parent) == 0o700
    assert _mode(path) == 0o600


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_fresh_vault_emits_no_warning(tmp_path, umask, backend):
    path = tmp_path / f"vault.{backend}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vault = _make(backend, path)
        vault.put("tok_1", SSN)
        _close(vault)
        _close(_make(backend, path))  # reopening an owner-only file is silent too
    assert [str(w.message) for w in caught] == []


def test_existing_loose_json_vault_warns_once_and_is_left_unmodified(tmp_path, umask):
    path = tmp_path / "vault.json"
    path.write_text(json.dumps({"tok_old": "old"}), encoding="utf-8")
    os.chmod(path, 0o644)
    with pytest.warns(UserWarning, match=r"group/other-accessible \(0644\); .*chmod 600"):
        vault = JsonTokenVault(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vault.put("tok_new", "new")
        assert vault.get("tok_old") == "old"
    assert caught == []  # one warning per instance
    assert _mode(path) == 0o644
    assert json.loads(path.read_text(encoding="utf-8")) == {"tok_old": "old", "tok_new": "new"}


def test_existing_loose_json_vault_warns_on_first_write(tmp_path, umask):
    path = tmp_path / "vault.json"
    vault = JsonTokenVault(path)  # file does not exist yet: nothing to check
    path.write_text("", encoding="utf-8")
    os.chmod(path, 0o640)
    with pytest.warns(UserWarning, match=r"group/other-accessible \(0640\)"):
        vault.put("tok_1", SSN)
    assert _mode(path) == 0o640


def test_existing_loose_sqlite_vault_warns_and_is_left_unmodified(tmp_path, umask):
    path = tmp_path / "vault.db"
    path.touch()
    os.chmod(path, 0o644)
    with pytest.warns(UserWarning, match=r"group/other-accessible \(0644\)"):
        vault = SqliteTokenVault(path)
    try:
        vault.put("tok_1", SSN)
        assert vault.get("tok_1") == SSN
    finally:
        vault.close()
    assert _mode(path) == 0o644


def test_sqlite_memory_vault_creates_no_file(tmp_path, umask, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vault = SqliteTokenVault(":memory:")
    vault.put("tok_1", SSN)
    assert len(vault) == 1
    vault.close()
    assert list(tmp_path.iterdir()) == []
