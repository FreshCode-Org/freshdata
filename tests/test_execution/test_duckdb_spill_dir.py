"""DuckDB spills into a private, per-run directory that is removed afterwards.

Regression tests for the shared, fixed ``/tmp/freshdata_spill`` default, where
spill files holding dataset rows were readable by other local users and
concurrent runs collided. Every test redirects HOME, XDG_CACHE_HOME and
FRESHDATA_SPILL_DIR into ``tmp_path``; nothing here touches the real /tmp.
"""

from __future__ import annotations

import gc
import os
import stat
import sys
import threading

import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution import EngineConfig, _spill
from freshdata.execution._spill import (
    SPILL_DIR_ENV,
    UnsafeSpillDirectoryError,
    _user_spill_base,
    create_run_spill_dir,
)

duckdb = pytest.importorskip("duckdb")

posix_only = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "geteuid"),
    reason="POSIX ownership and permission bits",
)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A private HOME/cache layout inside tmp_path, with no spill override."""
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.delenv(SPILL_DIR_ENV, raising=False)
    return tmp_path


def _expected_cache_base(root) -> str:
    if sys.platform == "win32":
        return str(root / "localappdata" / "freshdata" / "spill")
    if sys.platform == "darwin":
        return str(root / "home" / "Library" / "Caches" / "freshdata" / "spill")
    return str(root / "xdg-cache" / "freshdata" / "spill")


class _Captured(list):
    """Connections seen so far; ``hook`` (if set) runs right after each connect."""

    hook = None


@pytest.fixture
def captured(monkeypatch):
    """Record the ``temp_directory`` DuckDB is connected with, while it is live."""
    seen = _Captured()
    real_connect = duckdb.connect

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        config = kwargs.get("config") or {}
        if "temp_directory" in config:
            path = config["temp_directory"]
            seen.append({"path": path, "isdir": os.path.isdir(path), "mode": _mode(path)})
            if seen.hook is not None:
                seen.hook()
        return conn

    monkeypatch.setattr(duckdb, "connect", connect)
    return seen


def test_temp_directory_defaults_to_none():
    assert EngineConfig().temp_directory is None
    assert EngineConfig(engine="duckdb").temp_directory is None


def test_default_run_uses_private_dir_under_user_cache_and_removes_it(
    home, captured, small_df, native_config
):
    out = fd.clean(small_df.copy(), config=native_config, engine="duckdb")
    assert isinstance(out, pd.DataFrame)

    [run] = captured
    base = _expected_cache_base(home)
    assert os.path.dirname(run["path"]) == base
    assert os.path.basename(run["path"]).startswith("freshdata_spill_")
    assert run["isdir"]
    if os.name == "posix":
        assert run["mode"] == 0o700
        assert _mode(base) == 0o700
    assert not os.path.exists(run["path"])
    assert os.listdir(base) == []
    assert _user_spill_base() == base


def test_spill_dir_env_override(home, monkeypatch, captured, small_df, native_config):
    override = home / "env-spill"
    monkeypatch.setenv(SPILL_DIR_ENV, str(override))
    fd.clean(small_df.copy(), config=native_config, engine="duckdb")
    [run] = captured
    assert os.path.dirname(run["path"]) == str(override)
    assert not os.path.exists(run["path"])
    assert not os.path.exists(_expected_cache_base(home))


def test_explicit_temp_directory_gets_a_private_run_subdirectory(
    home, captured, small_df, native_config
):
    spill = home / "explicit" / "spill"
    ec = EngineConfig(engine="duckdb", memory_limit_gb=0.5, temp_directory=str(spill))
    fd.clean(small_df.copy(), config=native_config, engine_config=ec)
    [run] = captured
    assert os.path.dirname(run["path"]) == str(spill)
    if os.name == "posix":
        assert run["mode"] == 0o700
        assert _mode(spill) == 0o700
    assert not os.path.exists(run["path"])
    assert spill.is_dir() and os.listdir(spill) == []


def test_parallel_runs_get_distinct_directories(home, captured, small_df, native_config):
    barrier = threading.Barrier(2)
    captured.hook = lambda: barrier.wait(timeout=60)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            fd.clean(small_df.copy(), config=native_config, engine="duckdb")
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors
    paths = [r["path"] for r in captured]
    assert len(paths) == 2 and len(set(paths)) == 2
    assert all(r["isdir"] for r in captured)
    assert not any(os.path.exists(p) for p in paths)


def test_native_relation_keeps_directory_until_released(
    home, captured, small_df, native_config
):
    ec = EngineConfig(engine="duckdb", output_format="duckdb")
    relation = fd.clean(small_df.copy(), config=native_config, engine_config=ec)
    [run] = captured
    assert os.path.isdir(run["path"])
    assert len(relation.fetchdf()) > 0
    assert os.path.isdir(run["path"])

    del relation
    gc.collect()
    assert not os.path.exists(run["path"])


def test_run_directory_removed_when_connect_fails(home, monkeypatch, small_df, native_config):
    spill = home / "spill"

    def broken_connect(*args, **kwargs):
        raise duckdb.IOException("boom")

    monkeypatch.setattr(duckdb, "connect", broken_connect)
    ec = EngineConfig(engine="duckdb", temp_directory=str(spill))
    with pytest.raises(duckdb.IOException):
        fd.clean(small_df.copy(), config=native_config, engine_config=ec)
    assert os.listdir(spill) == []


@posix_only
def test_world_writable_base_without_sticky_bit_raises(home, small_df, native_config):
    spill = home / "shared"
    spill.mkdir()
    os.chmod(spill, 0o777)
    with pytest.raises(PermissionError, match="0o777"):
        create_run_spill_dir(EngineConfig(temp_directory=str(spill)))
    ec = EngineConfig(engine="duckdb", temp_directory=str(spill))
    with pytest.raises(UnsafeSpillDirectoryError):
        fd.clean(small_df.copy(), config=native_config, engine_config=ec)
    assert os.listdir(spill) == []


@posix_only
def test_group_writable_base_raises(home):
    spill = home / "group"
    spill.mkdir()
    os.chmod(spill, 0o770)
    with pytest.raises(UnsafeSpillDirectoryError):
        create_run_spill_dir(EngineConfig(temp_directory=str(spill)))


@posix_only
def test_sticky_world_writable_base_is_accepted(home):
    spill = home / "sticky"
    spill.mkdir()
    os.chmod(spill, 0o1777)
    run = create_run_spill_dir(EngineConfig(temp_directory=str(spill)))
    try:
        assert os.path.dirname(run) == str(spill)
        assert _mode(run) == 0o700
    finally:
        _spill.remove_run_spill_dir(run)
    assert not os.path.exists(run)


@posix_only
def test_foreign_owner_raises(home, monkeypatch):
    spill = home / "foreign"
    spill.mkdir(mode=0o700)
    real_euid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: real_euid + 4242)
    with pytest.raises(UnsafeSpillDirectoryError, match=f"uid {real_euid}"):
        create_run_spill_dir(EngineConfig(temp_directory=str(spill)))
    assert os.listdir(spill) == []


@posix_only
def test_symlink_to_unsafe_directory_raises(home):
    target = home / "target"
    target.mkdir()
    os.chmod(target, 0o777)
    link = home / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(UnsafeSpillDirectoryError):
        create_run_spill_dir(EngineConfig(temp_directory=str(link)))


@posix_only
def test_unsafe_env_and_default_cache_dirs_raise(home, monkeypatch):
    shared = home / "env-shared"
    shared.mkdir()
    os.chmod(shared, 0o777)
    monkeypatch.setenv(SPILL_DIR_ENV, str(shared))
    with pytest.raises(UnsafeSpillDirectoryError):
        create_run_spill_dir(EngineConfig())

    monkeypatch.delenv(SPILL_DIR_ENV)
    cache = _expected_cache_base(home)
    os.makedirs(cache)
    os.chmod(cache, 0o777)
    with pytest.raises(UnsafeSpillDirectoryError):
        create_run_spill_dir(EngineConfig())


@posix_only
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root ignores directory permissions")
def test_unwritable_cache_falls_back_to_private_dir_in_temp(home, monkeypatch):
    readonly = home / "readonly"
    readonly.mkdir()
    monkeypatch.setenv("HOME", str(readonly / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(readonly / "cache"))
    systmp = home / "systmp"
    systmp.mkdir(mode=0o700)
    monkeypatch.setattr(_spill.tempfile, "gettempdir", lambda: str(systmp))
    os.chmod(readonly, 0o500)
    try:
        run = create_run_spill_dir(EngineConfig())
    finally:
        os.chmod(readonly, 0o700)
    try:
        assert os.path.dirname(run) == str(systmp)
        assert _mode(run) == 0o700
    finally:
        _spill.remove_run_spill_dir(run)
