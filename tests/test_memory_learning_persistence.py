"""Persistence regressions for cleaning memory and learning profiles.

Covers #306 (explicit SQLite memory selection, no file creation on a missing
path), #308 (merged profiles keep their source schema), #309 (NaN decision
columns and NaN-free memory JSON) and #311 (strict ``.fdprofile`` manifests).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.learning import ProfileFormatError
from freshdata.learning.profile import LearningProfile, load_profile, save_profile
from freshdata.learning.replay import _profile_schema, check_profile_drift
from freshdata.learning.types import ExampleBank, ExamplePair
from freshdata.memory import CleaningMemory, _normalize_decisions


def _strict_json(text: str) -> object:
    def _reject(token: str) -> object:
        raise AssertionError(f"non-standard JSON token {token}")

    return json.loads(text, parse_constant=_reject)


# ---------------------------------------------------------------------------
# #306 load_cleaning_memory on SQLite stores
# ---------------------------------------------------------------------------


@pytest.fixture
def two_memory_store(tmp_path: Path) -> Path:
    df = pd.DataFrame({"a": [1, 2]})
    store = tmp_path / "memory.sqlite"
    fd.learn_cleaning_memory(df, [], "sales", roles={}).to_json(str(store))
    fd.learn_cleaning_memory(df, [], "crm", roles={}).to_json(str(store))
    return store


def test_multi_memory_store_requires_dataset_id(two_memory_store: Path) -> None:
    with pytest.raises(ValueError, match="dataset_id") as info:
        fd.load_cleaning_memory(str(two_memory_store))
    assert "'crm'" in str(info.value)
    assert "'sales'" in str(info.value)


def test_multi_memory_store_selects_dataset_id(two_memory_store: Path) -> None:
    assert fd.load_cleaning_memory(str(two_memory_store), dataset_id="crm").dataset_id == "crm"
    assert fd.load_cleaning_memory(two_memory_store, dataset_id="sales").dataset_id == "sales"


def test_unknown_dataset_id_raises_key_error(two_memory_store: Path) -> None:
    with pytest.raises(KeyError, match="billing"):
        fd.load_cleaning_memory(str(two_memory_store), dataset_id="billing")


def test_single_memory_store_loads_without_dataset_id(tmp_path: Path) -> None:
    store = tmp_path / "one.db"
    fd.learn_cleaning_memory(pd.DataFrame({"a": [1]}), [], "only", roles={}).to_json(str(store))
    assert fd.load_cleaning_memory(str(store)).dataset_id == "only"


def test_store_path_with_uri_special_characters(tmp_path: Path) -> None:
    store = tmp_path / "my store #1 %20.sqlite"
    fd.learn_cleaning_memory(pd.DataFrame({"a": [1]}), [], "odd", roles={}).to_json(str(store))
    assert fd.load_cleaning_memory(str(store)).dataset_id == "odd"


@pytest.mark.parametrize("name", ["typo.sqlite", "typo.db", "typo.json"])
def test_missing_path_raises_without_creating_file(tmp_path: Path, name: str) -> None:
    target = tmp_path / name
    with pytest.raises(FileNotFoundError):
        fd.load_cleaning_memory(str(target))
    assert not target.exists()


def test_json_memory_dataset_id_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "mem.json"
    fd.learn_cleaning_memory(pd.DataFrame({"a": [1]}), [], "crm", roles={}).to_json(str(path))
    assert fd.load_cleaning_memory(str(path), dataset_id="crm").dataset_id == "crm"
    with pytest.raises(KeyError, match="sales"):
        fd.load_cleaning_memory(str(path), dataset_id="sales")


# ---------------------------------------------------------------------------
# #308 merged LearningProfile drift
# ---------------------------------------------------------------------------


def _email_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    emails = ["a@x.com", "B@Y.COM ", "c@z.org", "d@w.net"] * 5
    messy = pd.DataFrame(
        {
            "id": range(20),
            "email": emails,
            "status": ["pend-ing"] * 10 + ["done"] * 10,
            "n": range(20),
        }
    )
    clean = pd.DataFrame(
        {
            "id": range(20),
            "email": [e.strip().lower() for e in emails],
            "status": ["pending"] * 10 + ["done"] * 10,
            "n": range(20),
        }
    )
    return messy, clean


def test_merged_profile_replays_without_drift_on_training_frame() -> None:
    messy, clean = _email_pair()
    profile = fd.learn(messy, clean, key="id", min_support=2)
    merged = profile.merge(profile)

    _, parent_report = fd.clean(messy, profile=profile, return_report=True, verbose=False)
    _, merged_report = fd.clean(messy, profile=merged, return_report=True, verbose=False)
    assert parent_report.warnings == []
    assert merged_report.warnings == []

    gate = check_profile_drift(messy, merged)
    assert gate.severity == "none"
    assert set(gate.compatible_columns) == {"id", "email", "status", "n"}
    assert merged.audit().alignment["source_schema"] == profile.audit().alignment["source_schema"]


def test_merged_source_schema_is_union_with_base_precedence() -> None:
    messy, clean = _email_pair()
    left = fd.learn(messy, clean, key="id", min_support=2)
    wider_messy = messy.assign(extra=[1.5] * 20, n=messy["n"].astype("float64"))
    wider_clean = clean.assign(extra=[1.5] * 20, n=clean["n"].astype("float64"))
    right = fd.learn(wider_messy, wider_clean, key="id", min_support=2)

    union = left.merge(right).audit().alignment["source_schema"]
    assert union["extra"] == "float64"
    assert union["n"] == "int64"  # self wins a dtype disagreement
    other_first = left.merge(right, strategy="prefer_other").audit().alignment["source_schema"]
    assert other_first["n"] == "float64"


def test_memory_signature_fallback_maps_coarse_types() -> None:
    messy, clean = _email_pair()
    profile = fd.learn(messy, clean, key="id", min_support=2)
    memory = CleaningMemory(
        dataset_id="sig",
        signature={
            "columns": {"id": "integer", "email": "text", "status": "text", "n": "integer"}
        },
    )
    bare = dataclasses.replace(profile, audit_info=None, memory=memory)
    assert _profile_schema(bare) == {
        "id": "int64",
        "email": "object",
        "status": "object",
        "n": "int64",
    }
    gate = check_profile_drift(messy, bare)
    assert gate.severity == "none", gate.reasons
    assert "status" in gate.compatible_columns


# ---------------------------------------------------------------------------
# #309 NaN decision columns
# ---------------------------------------------------------------------------


def test_csv_round_tripped_decisions_replay_table_level_steps() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    _, report = fd.clean(df, return_report=True, verbose=False)
    reviewed = pd.read_csv(io.StringIO(report.to_frame().to_csv(index=False)))

    from_report = fd.learn_cleaning_memory(df, report, "d", roles={})
    from_csv = fd.learn_cleaning_memory(df, reviewed, "d", roles={})

    def replayed(memory: CleaningMemory) -> list[str]:
        _, rep = fd.clean(df, memory=memory, return_report=True, verbose=False)
        return [a.step for a in rep.actions if a.memory_influenced and a.step != "memory"]

    assert "drop_duplicates" in replayed(from_report)
    assert replayed(from_csv) == replayed(from_report)
    assert all(d["column"] is None or isinstance(d["column"], str) for d in from_csv.accepted)


def test_missing_decision_columns_normalise_to_none() -> None:
    rows = pd.DataFrame(
        {
            "column": [float("nan"), "", "  ", None, "amount"],
            "step": ["drop_duplicates", "a", "b", "c", "d"],
        }
    )
    accepted, rejected = _normalize_decisions(rows)
    assert rejected == []
    assert [d["column"] for d in accepted] == [None, None, None, None, "amount"]
    listed, _ = _normalize_decisions([{"column": float("nan"), "step": "x"}])
    assert listed[0]["column"] is None


def test_memory_json_never_emits_nan_tokens(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 1, 2]})
    reviewed = pd.DataFrame(
        {"column": [float("nan")], "step": ["drop_duplicates"], "description": [float("nan")]}
    )
    memory = fd.learn_cleaning_memory(
        df, reviewed, "d", roles={}, thresholds={"outlier_factor": float("inf")}
    )
    payload = _strict_json(memory.to_json())
    assert isinstance(payload, dict)
    assert payload["thresholds"]["outlier_factor"] is None
    assert payload["accepted"][0]["column"] is None
    assert payload["accepted"][0]["description"] is None

    path = tmp_path / "mem.json"
    memory.to_json(str(path))
    _strict_json(path.read_text(encoding="utf-8"))
    assert fd.load_cleaning_memory(str(path)).accepted[0]["column"] is None


# ---------------------------------------------------------------------------
# #311 strict .fdprofile manifests
# ---------------------------------------------------------------------------


@pytest.fixture
def good_profile(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    messy = pd.DataFrame({"id": range(20), "s": ["pend-ing"] * 10 + ["done"] * 10})
    clean = pd.DataFrame({"id": range(20), "s": ["pending"] * 10 + ["done"] * 10})
    path = tmp_path / "good.fdprofile"
    save_profile(fd.learn(messy, clean, key="id", min_support=2), path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    return path, members


def _write(target: Path, members: dict[str, bytes], replace: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(target, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, replace.get(name, payload))
    return target


@pytest.mark.parametrize(
    "manifest",
    [
        b'{"profile_ver',
        b"{}",
        b"[]",
        b"\xff\xfe",
        b'{"profile_version": "1.0", "compartments": 3}',
    ],
    ids=["truncated", "empty-object", "array", "not-utf8", "bad-field-type"],
)
def test_bad_manifest_raises_profile_format_error(
    good_profile: tuple[Path, dict[str, bytes]], tmp_path: Path, manifest: bytes
) -> None:
    _, members = good_profile
    bad = _write(tmp_path / "bad.fdprofile", members, {"manifest.json": manifest})
    with pytest.raises(ProfileFormatError) as info:
        load_profile(bad)
    assert info.value.__cause__ is not None


def test_missing_member_hashes_fail_closed(
    good_profile: tuple[Path, dict[str, bytes]], tmp_path: Path
) -> None:
    _, members = good_profile
    manifest = dict(json.loads(members["manifest.json"]), member_hashes={})
    bad = _write(
        tmp_path / "bad.fdprofile",
        members,
        {"manifest.json": json.dumps(manifest).encode(), "value_maps.json": b'{"value_maps": {}}'},
    )
    with pytest.raises(ProfileFormatError, match="no member hash"):
        load_profile(bad)


def test_one_missing_member_hash_fails_closed(
    good_profile: tuple[Path, dict[str, bytes]], tmp_path: Path
) -> None:
    _, members = good_profile
    manifest = json.loads(members["manifest.json"])
    del manifest["member_hashes"]["rules.json"]
    bad = _write(
        tmp_path / "bad.fdprofile", members, {"manifest.json": json.dumps(manifest).encode()}
    )
    with pytest.raises(ProfileFormatError, match="rules.json"):
        load_profile(bad)


def test_malformed_hashed_member_raises_profile_format_error(
    good_profile: tuple[Path, dict[str, bytes]], tmp_path: Path
) -> None:
    _, members = good_profile
    rules = b"[]"
    manifest = json.loads(members["manifest.json"])
    manifest["member_hashes"]["rules.json"] = hashlib.sha256(rules).hexdigest()
    bad = _write(
        tmp_path / "bad.fdprofile",
        members,
        {"manifest.json": json.dumps(manifest).encode(), "rules.json": rules},
    )
    with pytest.raises(ProfileFormatError, match="malformed"):
        load_profile(bad)


def test_saved_profile_round_trips(good_profile: tuple[Path, dict[str, bytes]]) -> None:
    path, members = good_profile
    loaded = load_profile(path)
    assert set(loaded.manifest.member_hashes) == set(members) - {"manifest.json"}


def _profile_with_vectors(tmp_path: Path) -> tuple[LearningProfile, Path]:
    messy = pd.DataFrame({"id": range(20), "s": ["pend-ing"] * 10 + ["done"] * 10})
    clean = pd.DataFrame({"id": range(20), "s": ["pending"] * 10 + ["done"] * 10})
    profile = fd.learn(messy, clean, key="id", min_support=2)
    profile.examples = ExampleBank(
        examples=[ExamplePair("s", "odd", "even", "unexplained", 1, False)],
        vectors_path="examples_vectors.npz",
        embedding_model_id="test-model",
        masked=False,
    )
    profile.vectors = np.arange(8, dtype="float32").reshape(2, 4)
    path = tmp_path / "vectors.fdprofile"
    save_profile(profile, path)
    return profile, path


def test_profile_with_vectors_round_trips(tmp_path: Path) -> None:
    profile, path = _profile_with_vectors(tmp_path)
    loaded = load_profile(path)
    assert "examples_vectors.npz" in loaded.manifest.member_hashes
    assert loaded.profile_id == profile.profile_id
    assert loaded.vectors is not None
    assert np.allclose(loaded.vectors, profile.vectors)
    assert not any(math.isnan(v) for v in loaded.vectors.ravel())


def test_unhashed_vectors_member_fails_closed(tmp_path: Path) -> None:
    _, path = _profile_with_vectors(tmp_path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    del manifest["member_hashes"]["examples_vectors.npz"]
    bad = _write(
        tmp_path / "bad.fdprofile", members, {"manifest.json": json.dumps(manifest).encode()}
    )
    with pytest.raises(ProfileFormatError, match="examples_vectors.npz"):
        load_profile(bad)
