"""FreshData's runtime must never touch the network (offline-by-default,
no runtime LLM, no cloud call). This patches every socket-construction entry
point so any accidental network attempt raises loudly instead of silently
succeeding or silently timing out.

The one documented exception, ``fd.models.pull(...)``, is explicit/opt-in and
is not exercised here — these tests cover the default, always-on runtime
paths: ``fd.clean``, ``fd.learn``, ``fd.compile_context``, and semantic
cleaning (deterministic + memory + profile replay).
"""

from __future__ import annotations

import socket

import pandas as pd
import pytest

import freshdata as fd


@pytest.fixture(autouse=True)
def _fail_on_network(monkeypatch):
    def _blocked(*_args, **_kwargs):
        raise AssertionError("unexpected network call from the FreshData runtime")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def _messy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "  Name  ": ["alice", " bob ", "carol", None],
            "age": ["twenty", "30", "N/A", "40"],
            "email": ["a@@b.com", "c@d.com", " e@f.com ", None],
            "customer_id": ["C1", "C2", "C3", "C4"],
        }
    )


def test_clean_makes_no_network_calls():
    fd.clean(_messy_frame(), semantic_mode="auto", id_columns=("customer_id",))


def test_clean_with_context_makes_no_network_calls():
    context = "Emails must be valid.\ncustomer_id is unique.\n"
    fd.clean(_messy_frame(), context=context, semantic_mode="auto")


def test_compile_context_makes_no_network_calls():
    fd.compile_context("Emails must be valid.\n", df=_messy_frame())


def test_learn_and_replay_make_no_network_calls():
    messy = _messy_frame()
    clean = messy.copy()
    clean["age"] = [20, 30, None, 40]
    profile = fd.learn(messy, clean, key="customer_id", min_support=1)
    fd.clean(messy, semantic_mode="auto", profile=profile)
