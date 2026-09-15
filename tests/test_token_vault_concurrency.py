"""Token vaults shared across instances and threads (#279).

JsonTokenVault merges and serialises writes from several instances on one path;
SqliteTokenVault's connection is usable from any thread.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from freshdata.enterprise import (
    JsonTokenVault,
    PrivacyPolicy,
    PrivacyRule,
    SqliteTokenVault,
    apply_privacy_policy,
    detokenize_series,
)

KEY = "unit-test-key"
N_THREADS = 8
PUTS_PER_THREAD = 10
JOIN_TIMEOUT = 30.0


def _run_threads(target, n: int = N_THREADS) -> None:
    """Start ``n`` threads together (via a barrier) and re-raise any worker error."""
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=JOIN_TIMEOUT)
            target(i)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(JOIN_TIMEOUT)
    assert not any(t.is_alive() for t in threads), "worker thread did not finish"
    if errors:
        raise errors[0]


# --------------------------------------------------------------------------
# JsonTokenVault
# --------------------------------------------------------------------------


def test_json_two_instances_keep_both_mappings(tmp_path):
    # Issue #279 repro: the second instance used to overwrite the first's entry.
    path = tmp_path / "vault.json"
    a, b = JsonTokenVault(path), JsonTokenVault(path)
    a.put("tok_a", "111-11-1111")
    b.put("tok_b", "222-22-2222")
    assert JsonTokenVault(path).get("tok_a") == "111-11-1111"
    assert JsonTokenVault(path).get("tok_b") == "222-22-2222"


def test_json_interleaved_puts_from_two_instances(tmp_path):
    path = tmp_path / "vault.json"
    a, b = JsonTokenVault(path), JsonTokenVault(path)
    expected = {}
    for i in range(6):
        vault = a if i % 2 == 0 else b
        vault.put(f"tok_{i}", f"value-{i}")
        expected[f"tok_{i}"] = f"value-{i}"
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    for token, value in expected.items():
        assert a.get(token) == value
        assert b.get(token) == value


def test_json_get_sees_entry_written_by_other_instance(tmp_path):
    path = tmp_path / "vault.json"
    b = JsonTokenVault(path)  # created before the file exists
    a = JsonTokenVault(path)
    assert b.get("tok_x") is None
    a.put("tok_x", "x-value")
    assert b.get("tok_x") == "x-value"
    a.put("tok_y", "y-value")
    assert b.get("tok_y") == "y-value"


def test_json_threads_with_own_instances_keep_all_entries(tmp_path):
    path = tmp_path / "vault.json"

    def work(i: int) -> None:
        vault = JsonTokenVault(path)
        for j in range(PUTS_PER_THREAD):
            vault.put(f"tok_{i}_{j}", f"value-{i}-{j}")

    _run_threads(work)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert len(on_disk) == N_THREADS * PUTS_PER_THREAD
    fresh = JsonTokenVault(path)
    for i in range(N_THREADS):
        for j in range(PUTS_PER_THREAD):
            assert fresh.get(f"tok_{i}_{j}") == f"value-{i}-{j}"


def test_json_threads_sharing_one_instance(tmp_path):
    path = tmp_path / "vault.json"
    vault = JsonTokenVault(path)

    def work(i: int) -> None:
        for j in range(PUTS_PER_THREAD):
            vault.put(f"tok_{i}_{j}", f"value-{i}-{j}")

    _run_threads(work)
    assert len(json.loads(path.read_text(encoding="utf-8"))) == N_THREADS * PUTS_PER_THREAD


def test_json_empty_file_loads_as_empty_vault(tmp_path):
    path = tmp_path / "vault.json"
    path.write_text("", encoding="utf-8")
    vault = JsonTokenVault(path)
    assert vault.get("tok_missing") is None
    vault.put("tok_a", "a")
    assert json.loads(path.read_text(encoding="utf-8")) == {"tok_a": "a"}


def test_json_nothing_written_until_put(tmp_path):
    path = tmp_path / "sub" / "vault.json"
    vault = JsonTokenVault(path)
    assert vault.get("tok_a") is None
    assert not path.exists()
    vault.put("tok_a", "a")
    assert path.read_text(encoding="utf-8") == json.dumps({"tok_a": "a"}, indent=2)


def test_json_repeated_put_of_same_value_skips_rewrite(tmp_path, monkeypatch):
    path = tmp_path / "vault.json"
    vault = JsonTokenVault(path)
    vault.put("tok_a", "a")
    calls = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real_fsync(fd)))
    vault.put("tok_a", "a")
    JsonTokenVault(path).put("tok_a", "a")
    assert calls == []
    vault.put("tok_b", "b")
    assert len(calls) == 1


def test_json_save_keeps_entries_from_other_instances(tmp_path):
    path = tmp_path / "vault.json"
    a, b = JsonTokenVault(path), JsonTokenVault(path)
    a.put("tok_a", "a")
    b.save()
    assert json.loads(path.read_text(encoding="utf-8")) == {"tok_a": "a"}


def test_json_tokenize_round_trip_through_two_vault_instances(tmp_path):
    path = tmp_path / "vault.json"
    rule = PrivacyRule(id="ssn", action="tokenize", reversible=True, columns=("ssn",))
    policy = PrivacyPolicy(name="p", rules=(rule,), key=KEY)
    out_a, _ = apply_privacy_policy(
        pd.DataFrame({"ssn": ["123-45-6789"]}), policy, vault=JsonTokenVault(path)
    )
    out_b, _ = apply_privacy_policy(
        pd.DataFrame({"ssn": ["987-65-4321"]}), policy, vault=JsonTokenVault(path)
    )
    fresh = JsonTokenVault(path)
    assert list(detokenize_series(out_a["ssn"], fresh, KEY)) == ["123-45-6789"]
    assert list(detokenize_series(out_b["ssn"], fresh, KEY)) == ["987-65-4321"]


def test_json_uses_exclusive_flock_for_writes_and_shared_for_reads(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    ops = []

    def recording_flock(fd, op):
        ops.append(op)
        return fcntl.flock(fd, op)

    stub = types.SimpleNamespace(
        LOCK_EX=fcntl.LOCK_EX, LOCK_SH=fcntl.LOCK_SH, LOCK_UN=fcntl.LOCK_UN, flock=recording_flock
    )
    monkeypatch.setitem(sys.modules, "fcntl", stub)
    path = tmp_path / "vault.json"
    JsonTokenVault(path).put("tok_a", "a")
    assert ops == [fcntl.LOCK_EX, fcntl.LOCK_UN]
    ops.clear()
    JsonTokenVault(path)
    assert ops == [fcntl.LOCK_SH, fcntl.LOCK_UN]


def test_json_windows_lock_covers_byte_zero(tmp_path, monkeypatch):
    calls = []

    def locking(fd, mode, nbytes):
        calls.append((mode, nbytes, os.lseek(fd, 0, os.SEEK_CUR)))

    stub = types.SimpleNamespace(LK_LOCK=1, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setitem(sys.modules, "msvcrt", stub)
    path = tmp_path / "vault.json"
    a, b = JsonTokenVault(path), JsonTokenVault(path)
    a.put("tok_a", "a")
    b.put("tok_b", "b")
    assert calls == [(1, 1, 0), (0, 1, 0)] * 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"tok_a": "a", "tok_b": "b"}


def test_json_without_lock_modules_falls_back_to_thread_lock(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setitem(sys.modules, "msvcrt", None)
    path = tmp_path / "vault.json"
    vault = JsonTokenVault(path)

    def work(i: int) -> None:
        for j in range(PUTS_PER_THREAD):
            vault.put(f"tok_{i}_{j}", f"value-{i}-{j}")

    _run_threads(work)
    other = JsonTokenVault(path)
    other.put("tok_other", "other")
    assert len(json.loads(path.read_text(encoding="utf-8"))) == N_THREADS * PUTS_PER_THREAD + 1


# --------------------------------------------------------------------------
# SqliteTokenVault
# --------------------------------------------------------------------------


def test_sqlite_vault_works_from_worker_thread(tmp_path):
    # Issue #279 repro: a vault created in the main thread used in a thread pool.
    vault = SqliteTokenVault(tmp_path / "v.db")
    policy = PrivacyPolicy(
        rules=(PrivacyRule(id="t", action="tokenize", columns=("ssn",), key="k"),)
    )
    frame = pd.DataFrame({"ssn": ["123-45-6789"]})
    with ThreadPoolExecutor(1) as ex:
        out, report = ex.submit(apply_privacy_policy, frame, policy, vault=vault).result()
    assert out["ssn"].iloc[0].startswith("tok_")
    assert report.vault_info["entries"] == 1
    assert len(vault) == 1
    vault.close()


def test_sqlite_get_put_len_from_other_thread(tmp_path):
    vault = SqliteTokenVault(tmp_path / "v.db")
    vault.put("tok_main", "main")

    def use() -> tuple[str | None, int]:
        vault.put("tok_worker", "worker")
        return vault.get("tok_main"), len(vault)

    with ThreadPoolExecutor(1) as ex:
        assert ex.submit(use).result() == ("main", 2)
    assert vault.get("tok_worker") == "worker"
    vault.close()


def test_sqlite_eight_threads_share_one_vault(tmp_path):
    vault = SqliteTokenVault(tmp_path / "v.db")

    def work(i: int) -> None:
        for j in range(PUTS_PER_THREAD):
            vault.put(f"tok_{i}_{j}", f"value-{i}-{j}")
            assert vault.get(f"tok_{i}_{j}") == f"value-{i}-{j}"
            assert len(vault) >= 1

    _run_threads(work)
    assert len(vault) == N_THREADS * PUTS_PER_THREAD
    vault.close()
    reopened = SqliteTokenVault(tmp_path / "v.db")
    assert reopened.get("tok_7_9") == "value-7-9"
    reopened.close()


def test_sqlite_close_twice_is_harmless(tmp_path):
    vault = SqliteTokenVault(tmp_path / "v.db")
    vault.close()
    vault.close()
