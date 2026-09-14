"""Retry and failure classification for the online fixture download tooling."""

from __future__ import annotations

import http.client
import io
import sys
import urllib.error
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fetch_online_fixtures as fof  # noqa: E402
import fixture_download as net  # noqa: E402

URL = "https://example.invalid/data.csv"
BODY = b"a,b\n1,2\n"


def _flaky_urlopen(monkeypatch, errors):
    """Patch urlopen to raise *errors* in order, then return BODY."""
    calls: list[str] = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return io.BytesIO(BODY)

    monkeypatch.setattr(net.urllib.request, "urlopen", urlopen)
    return calls


def test_download_retries_dropped_connections(monkeypatch):
    calls = _flaky_urlopen(
        monkeypatch,
        [http.client.RemoteDisconnected("closed"), ConnectionResetError(104, "reset")],
    )
    sleeps: list[float] = []
    assert net.download(URL, sleep=sleeps.append) == BODY
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # exponential backoff


def test_download_gives_up_after_all_attempts(monkeypatch):
    calls = _flaky_urlopen(monkeypatch, [http.client.IncompleteRead(b"")] * 5)
    with pytest.raises(net.TransientFetchError, match="after 3 attempts"):
        net.download(URL, attempts=3, sleep=lambda _: None)
    assert len(calls) == 3


@pytest.mark.parametrize("code", [429, 503])
def test_download_retries_throttling_and_server_errors(monkeypatch, code):
    error = urllib.error.HTTPError(URL, code, "busy", hdrs=None, fp=None)
    calls = _flaky_urlopen(monkeypatch, [error])
    assert net.download(URL, sleep=lambda _: None) == BODY
    assert len(calls) == 2


def test_download_does_not_retry_permanent_http_errors(monkeypatch):
    error = urllib.error.HTTPError(URL, 404, "Not Found", hdrs=None, fp=None)
    calls = _flaky_urlopen(monkeypatch, [error])
    with pytest.raises(urllib.error.HTTPError):
        net.download(URL, sleep=lambda _: None)
    assert len(calls) == 1


def _run_main(monkeypatch, tmp_path, outcomes, *argv):
    """Run main() over fake datasets whose fetch_one result/exception is *outcomes*."""
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")
    monkeypatch.setattr(fof, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(fof, "_load_registry", lambda: {name: {"url": URL} for name in outcomes})
    monkeypatch.setattr(fof, "_load_manifest", dict)
    monkeypatch.setattr(fof, "_save_manifest", lambda manifest: None)
    attempted: list[str] = []

    def fake_fetch_one(name, entry, manifest, *, refresh, update_manifest):
        attempted.append(name)
        outcome = outcomes[name]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(fof, "fetch_one", fake_fetch_one)
    return fof.main(list(argv)), attempted


def test_main_keeps_going_after_a_network_failure(monkeypatch, tmp_path):
    outcomes = {
        "a_flaky": net.TransientFetchError("reset"),
        "b_ok": tmp_path / "b_ok.csv",
    }
    code, attempted = _run_main(monkeypatch, tmp_path, outcomes)
    assert attempted == ["a_flaky", "b_ok"]
    assert code == 1  # default budget is zero


def test_main_tolerates_network_failures_within_budget(monkeypatch, tmp_path):
    outcomes = {
        "a_flaky": net.TransientFetchError("reset"),
        "b_flaky": net.TransientFetchError("reset"),
        "c_ok": tmp_path / "c_ok.csv",
    }
    assert _run_main(monkeypatch, tmp_path, outcomes, "--max-failures", "2")[0] == 0
    assert _run_main(monkeypatch, tmp_path, outcomes, "--max-failures", "1")[0] == 1


@pytest.mark.parametrize(
    "outcome",
    [ValueError("unparseable"), urllib.error.URLError("404"), None],
    ids=["parse-error", "permanent-http", "empty-after-parse"],
)
def test_main_never_tolerates_dataset_failures(monkeypatch, tmp_path, outcome):
    outcomes = {"bad": outcome, "ok": tmp_path / "ok.csv"}
    assert _run_main(monkeypatch, tmp_path, outcomes, "--max-failures", "5")[0] == 1


def test_main_annotates_failures_in_github_actions(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    outcomes = {"flaky": net.TransientFetchError("reset")}
    _run_main(monkeypatch, tmp_path, outcomes, "--max-failures", "1")
    assert "::warning::online fixture 'flaky'" in capsys.readouterr().out
