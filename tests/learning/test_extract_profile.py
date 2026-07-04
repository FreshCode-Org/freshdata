"""Extraction, holdout demotion, privacy masking, and .fdprofile IO tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
import freshdata.learning as fdl
from freshdata.learning import (
    LearningProfile,
    ProfileError,
    ProfileVersionError,
    learn,
    load_profile,
    save_profile,
)
from freshdata.learning.privacy import (
    derive_salt,
    detect_sensitive_columns,
    is_masked_token,
    mask_value,
)

N = 40


def _ids() -> list[str]:
    return [f"r{i}" for i in range(N)]


def _vocab_pair(vocab: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raws = (list(vocab) * (N // len(vocab) + 1))[:N]
    messy = pd.DataFrame({"id": _ids(), "col": raws})
    clean = pd.DataFrame({"id": _ids(), "col": [vocab[r] for r in raws]})
    return messy, clean


VOCAB = {"AA": "alpha", "BB": "beta", "CC": "gamma", "DD": "delta"}


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


class TestExtract:
    def test_repeated_vocabulary_becomes_value_map(self) -> None:
        messy, clean = _vocab_pair(VOCAB)
        profile = learn(messy, clean, key="id", min_support=2)
        entries = {e.raw_value: e.clean_value for e in profile.value_maps["col"].entries}
        assert entries == VOCAB
        for entry in profile.value_maps["col"].entries:
            assert entry.support >= 2
            assert entry.precision == 1.0

    def test_min_support_filters_rare_patterns(self) -> None:
        messy, clean = _vocab_pair(VOCAB)
        # A single-occurrence typo stays out of the literal map.
        messy.loc[0, "col"] = "singleton-typo"
        clean.loc[0, "col"] = "alpha"
        profile = learn(messy, clean, key="id", min_support=5)
        raws = {e.raw_value for e in profile.value_maps["col"].entries}
        assert "singleton-typo" not in raws

    def test_max_map_size_caps_entries(self) -> None:
        messy, clean = _vocab_pair(VOCAB)
        profile = learn(messy, clean, key="id", min_support=2, max_map_size=2)
        value_map = profile.value_maps["col"]
        assert len(value_map.entries) <= 2
        assert value_map.capped

    def test_imputation_never_learns_literal_nan_map(self) -> None:
        messy = pd.DataFrame({"id": _ids(), "v": [None if i % 2 else float(i) for i in range(N)]})
        clean = pd.DataFrame({"id": _ids(), "v": [4.0 if i % 2 else float(i) for i in range(N)]})
        profile = learn(messy, clean, key="id", min_support=2)
        for value_map in profile.value_maps.values():
            for entry in value_map.entries:
                assert not pd.isna(entry.raw_value)

    def test_protected_column_learns_nothing(self) -> None:
        messy, clean = _vocab_pair(VOCAB)
        profile = learn(
            messy,
            clean,
            key="id",
            context="col is critical. Never modify col.",
            min_support=2,
        )
        assert "col" not in profile.value_maps
        assert all(r.column != "col" for r in profile.rules)

    def test_embedded_memory_is_present(self) -> None:
        messy, clean = _vocab_pair(VOCAB)
        profile = learn(messy, clean, key="id", min_support=2)
        assert profile.memory is not None


# ---------------------------------------------------------------------------
# holdout validation / demotion
# ---------------------------------------------------------------------------


class TestHoldoutDemotion:
    def test_conflicting_mapping_is_demoted(self) -> None:
        # "xx" maps to two different clean values -> precision 0.5 < gate.
        messy = pd.DataFrame({"id": _ids(), "col": ["xx"] * N})
        clean = pd.DataFrame({"id": _ids(), "col": ["a" if i % 2 == 0 else "b" for i in range(N)]})
        profile = learn(messy, clean, key="id", min_support=2)
        assert profile.value_maps == {} or "col" not in profile.value_maps
        audit = profile.audit()
        rendered = audit.render()
        assert "demot" in rendered.lower() or "precision" in rendered.lower()

    def test_clean_mapping_is_not_demoted(self) -> None:
        messy, clean = _vocab_pair(VOCAB)
        profile = learn(messy, clean, key="id", min_support=2)
        assert "col" in profile.value_maps

    def test_learning_is_deterministic(self) -> None:
        messy, clean = _vocab_pair(VOCAB)
        p1 = learn(messy, clean, key="id", min_support=2, random_state=0)
        p2 = learn(messy, clean, key="id", min_support=2, random_state=0)
        e1 = [(e.raw_value, e.clean_value, e.support) for e in p1.value_maps["col"].entries]
        e2 = [(e.raw_value, e.clean_value, e.support) for e in p2.value_maps["col"].entries]
        assert e1 == e2

    def test_min_precision_zero_keeps_conflicts_out_anyway(self) -> None:
        # Even a permissive precision gate never stores a raw value mapping to
        # two clean targets — the map itself must stay a function.
        messy = pd.DataFrame({"id": _ids(), "col": ["xx"] * N})
        clean = pd.DataFrame({"id": _ids(), "col": ["a" if i % 2 == 0 else "b" for i in range(N)]})
        profile = learn(messy, clean, key="id", min_support=2, min_precision=0.0)
        entries = profile.value_maps.get("col")
        if entries is not None:
            raws = [e.raw_value for e in entries.entries]
            assert len(raws) == len(set(raws))


# ---------------------------------------------------------------------------
# privacy
# ---------------------------------------------------------------------------


EMAIL_MESSY = [" Judy@YAHOO.COM ", "ok@x.com", " Bob@GMAIL.COM ", "z@y.org"]
EMAIL_CLEAN = ["judy@yahoo.com", "ok@x.com", "bob@gmail.com", "z@y.org"]


def _email_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    messy = pd.DataFrame({"id": _ids(), "email": (EMAIL_MESSY * (N // 4))[:N]})
    clean = pd.DataFrame({"id": _ids(), "email": (EMAIL_CLEAN * (N // 4))[:N]})
    return messy, clean


class TestPrivacy:
    def test_detect_sensitive_columns(self) -> None:
        df = pd.DataFrame({"email": ["a@b.com", "c@d.org"], "x": [1, 2]})
        assert detect_sensitive_columns(df) == {"email": "email"}

    def test_mask_token_is_stable_and_recognizable(self) -> None:
        salt = derive_salt("seed")
        token = mask_value("a@b.com", salt=salt, semantic_type="email")
        assert is_masked_token(token)
        assert not is_masked_token("a@b.com")
        assert token == mask_value("a@b.com", salt=salt, semantic_type="email")

    def test_default_mask_stores_no_raw_literals(self, tmp_path: Path) -> None:
        messy, clean = _email_pair()
        profile = learn(messy, clean, key="id", min_support=2)
        entries = profile.value_maps["email"].entries
        assert all(e.masked and is_masked_token(str(e.raw_value)) for e in entries)
        assert not profile.manifest.contains_raw_values
        path = tmp_path / "masked.fdprofile"
        save_profile(profile, path)
        blob = path.read_bytes()
        with zipfile.ZipFile(path) as zf:
            members = b"".join(zf.read(name) for name in zf.namelist())
        for literal in ("Judy@YAHOO.COM", "judy@yahoo.com", "Bob@GMAIL.COM"):
            assert literal.encode() not in blob
            assert literal.encode() not in members

    def test_include_sensitive_requires_privacy_none(self) -> None:
        messy, clean = _email_pair()
        profile = learn(messy, clean, key="id", min_support=2, include_sensitive=True)
        # privacy stays "mask" -> literals still masked.
        assert all(e.masked for e in profile.value_maps["email"].entries)
        assert not profile.manifest.contains_raw_values

    def test_opt_in_raw_literals_flip_manifest_flag(self) -> None:
        messy, clean = _email_pair()
        profile = learn(
            messy, clean, key="id", min_support=2, privacy="none", include_sensitive=True
        )
        entries = profile.value_maps["email"].entries
        assert any(not e.masked for e in entries)
        assert profile.manifest.contains_raw_values

    def test_masked_entries_survive_roundtrip(self, tmp_path: Path) -> None:
        messy, clean = _email_pair()
        profile = learn(messy, clean, key="id", min_support=2)
        path = tmp_path / "roundtrip.fdprofile"
        save_profile(profile, path)
        loaded = load_profile(path)
        assert all(e.masked for e in loaded.value_maps["email"].entries)
        assert loaded.manifest.privacy_mode == "mask"


# ---------------------------------------------------------------------------
# .fdprofile IO
# ---------------------------------------------------------------------------


def _rezip(
    path: Path,
    out: Path,
    *,
    replace: dict | None = None,
    add: dict | None = None,
    drop: tuple = (),
) -> Path:
    """Rewrite a profile zip with mutated/added/dropped members."""
    replace = replace or {}
    add = add or {}
    with zipfile.ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    for name in drop:
        members.pop(name, None)
    members.update(replace)
    members.update(add)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)
    return out


class TestProfileIO:
    @pytest.fixture()
    def saved(self, tmp_path: Path) -> tuple[LearningProfile, Path]:
        messy, clean = _vocab_pair(VOCAB)
        profile = learn(messy, clean, key="id", min_support=2, dataset_id="io-suite")
        path = tmp_path / "io.fdprofile"
        save_profile(profile, path)
        return profile, path

    def test_roundtrip_preserves_content(self, saved) -> None:
        profile, path = saved
        loaded = load_profile(path)
        assert loaded.profile_id == profile.profile_id
        assert loaded.manifest.dataset_id == "io-suite"
        got = {e.raw_value: e.clean_value for e in loaded.value_maps["col"].entries}
        assert got == VOCAB

    def test_save_accepts_string_path(self, saved, tmp_path: Path) -> None:
        profile, _ = saved
        str_path = str(tmp_path / "strpath.fdprofile")
        profile.save(str_path)
        assert load_profile(str_path).profile_id == profile.profile_id

    def test_manifest_hashes_cover_all_members(self, saved) -> None:
        profile, path = saved
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        hashed = set(load_profile(path).manifest.member_hashes)
        assert hashed == names - {"manifest.json"}

    def test_tampered_member_fails_closed(self, saved, tmp_path: Path) -> None:
        _, path = saved
        with zipfile.ZipFile(path) as zf:
            rules = zf.read("rules.json")
        bad = _rezip(path, tmp_path / "tampered.fdprofile", replace={"rules.json": rules + b" "})
        with pytest.raises(ProfileError, match="hash"):
            load_profile(bad)

    def test_missing_required_member_fails_closed(self, saved, tmp_path: Path) -> None:
        _, path = saved
        bad = _rezip(path, tmp_path / "missing.fdprofile", drop=("rules.json",))
        with pytest.raises(ProfileError):
            load_profile(bad)

    def test_unsupported_major_version_is_rejected(self, saved, tmp_path: Path) -> None:
        _, path = saved
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        manifest["profile_version"] = "99.0"
        bad = _rezip(
            path,
            tmp_path / "future.fdprofile",
            replace={"manifest.json": json.dumps(manifest).encode()},
        )
        with pytest.raises(ProfileVersionError):
            load_profile(bad)

    def test_unknown_member_is_ignored(self, saved, tmp_path: Path) -> None:
        profile, path = saved
        odd = _rezip(path, tmp_path / "extra.fdprofile", add={"future_compartment.json": b"{}"})
        with pytest.warns(UserWarning, match="future_compartment"):
            loaded = load_profile(odd)
        assert loaded.profile_id == profile.profile_id

    def test_not_a_zip_fails_clearly(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.fdprofile"
        path.write_bytes(b"this is not a zip")
        with pytest.raises(ProfileError):
            load_profile(path)

    def test_summary_and_audit_render(self, saved) -> None:
        profile, _ = saved
        assert profile.profile_id in profile.summary()
        rendered = profile.audit().render()
        assert "value maps" in rendered or "rules" in rendered


# ---------------------------------------------------------------------------
# ExampleBank without the [semantic] extra
# ---------------------------------------------------------------------------


class TestExampleBankWithoutSemantic:
    def test_unexplained_low_support_lands_in_examples_only(self) -> None:
        messy, clean = _vocab_pair(VOCAB)
        messy.loc[0, "col"] = "zzqqk-unique"
        clean.loc[0, "col"] = "totally-other"
        profile = learn(messy, clean, key="id", min_support=5)
        raws = {e.raw_value for vm in profile.value_maps.values() for e in vm.entries}
        assert "zzqqk-unique" not in raws
        if profile.examples is not None:
            example_raws = {e.raw_value for e in profile.examples.examples}
            assert "zzqqk-unique" in example_raws

    def test_examples_do_not_block_replay_without_vectors(self) -> None:
        messy, clean = _vocab_pair(VOCAB)
        messy.loc[0, "col"] = "zzqqk-unique"
        clean.loc[0, "col"] = "totally-other"
        profile = learn(messy, clean, key="id", min_support=5)
        assert profile.vectors is None
        out = fd.clean(messy, profile=profile, semantic_mode="auto", verbose=False)
        assert "zzqqk-unique" in out["col"].tolist()  # suggest-only stays unapplied

    def test_lazy_top_level_exports(self) -> None:
        assert fd.learn is fdl.learn
        assert fd.load_profile is fdl.load_profile
        assert fd.save_profile is fdl.save_profile
        assert fd.LearningProfile is LearningProfile
        assert "learn" in dir(fd)
