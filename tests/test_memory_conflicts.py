"""Phase 16: learning/memory replay safety under meaning collisions.

The trap this file exists for: dataset A teaches freshdata that ``"M"`` means
``"male"``; dataset B uses ``"M"`` to mean *medium*. Replaying A's memory onto B
must not blindly rewrite the column.

``tests/test_cleaning_memory.py`` already covers the *date* version of a
memory/deterministic disagreement (``unsafe_ambiguous``, never mutated, held for
human review). The categorical one-token case is covered here, together with the
collision axes around it: same value / different column, same value / different
semantic type, stale memory, dataset-identity collisions, corrupted and foreign
payloads, and whether every memory-derived decision stays distinguishable in the
audit trail.

Tests whose docstring starts with ``FINDING`` pin **current** behaviour that the
lane reports as a defect or specification gap, not as desired behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata import CleaningMemory

CLEAN = {"return_report": True, "verbose": False}


def frame(values: list, column: str = "segment") -> pd.DataFrame:
    """A 3-column frame whose signature is stable across datasets.

    ``segment`` deliberately avoids the identifier-name tokens (``code``,
    ``key``, ...) that would otherwise veto every semantic proposal.
    """
    n = len(values)
    return pd.DataFrame({
        column: values,
        "amount": [float(i) for i in range(n)],
        "note": [f"n{i}" for i in range(n)],
    })


@pytest.fixture
def gender_memory() -> CleaningMemory:
    """Memory learned from a frame where ``segment`` holds genders."""
    learn_df = frame(["M", "F"] * 6)
    _, report = fd.clean(learn_df, semantic_mode="auto", **CLEAN)
    memory = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="hr")
    repairs = memory.value_patterns["semantic_repairs"]
    assert {(r["raw_value"], r["proposed_value"]) for r in repairs} == {
        ("M", "male"), ("F", "female"),
    }
    return memory


def sizes_frame() -> pd.DataFrame:
    """Dataset B: the same column name, but ``M`` means *medium*."""
    return frame(["S", "M", "L", "M"] * 3)


def semantic_actions(report) -> list:
    return [a for a in report.actions if a.step == "semantic"]


# --------------------------------------------------------------------------- #
# The key trap
# --------------------------------------------------------------------------- #


def test_one_token_categorical_replay_does_not_rewrite_a_column_that_changed_meaning(
    gender_memory: CleaningMemory,
) -> None:
    """A learned ``M -> male`` must not be replayed onto clothing sizes.

    This test was written to pin the defect: replay auto-applied the repair.

    Nothing in the retrieval path re-validates *why* the repair was learned.
    ``CategorySynonymExpert`` only proposed ``M -> male`` on dataset A because
    **every** value in that column was a gender synonym (``gender_like``); on
    dataset B (``S``/``M``/``L``) the deterministic expert correctly abstains.
    But an abstention is not a disagreement, so ``_merge_proposals`` passes the
    memory proposal through untouched, it keeps the full learned confidence
    (0.95), and ``risk="medium"`` clears the ``semantic_mode="auto"`` gate.

    The stored ``column_signature``/``value_signature`` on the memory record
    would not have helped either: both datasets profile as
    ``role="categorical", semantic_type=None, free_text=False``.

    The fix supplies the missing check: a context-dependent repair replayed with
    no deterministic corroboration in *this* frame is demoted to a
    review-required suggestion, so memory stays evidence rather than authority.
    """
    out, report = fd.clean(sizes_frame(), semantic_mode="auto", memory=gender_memory, **CLEAN)

    assert out["segment"].tolist() == sizes_frame()["segment"].tolist(), (
        "clothing sizes must survive a gender memory untouched"
    )
    assert not [a for a in semantic_actions(report) if a.status == "automatic"]
    held = [
        a for a in semantic_actions(report)
        if a.metadata.get("raw_value") == "M" and a.metadata.get("proposed_value") == "male"
    ]
    assert len(held) == 1
    assert held[0].status == "suggested"
    assert held[0].risk == "high"
    assert held[0].human_review is True

    # Control: without the memory, the deterministic layer leaves B alone.
    plain, plain_report = fd.clean(sizes_frame(), semantic_mode="auto", **CLEAN)
    assert plain["segment"].tolist() == sizes_frame()["segment"].tolist()
    assert not [a for a in semantic_actions(plain_report) if a.status == "automatic"]


def test_normalized_match_also_catches_the_lowercase_spelling(
    gender_memory: CleaningMemory,
) -> None:
    """Retrieval matches on the *normalized* value, so a learned ``"M"`` also
    reaches a dominant lowercase ``"m"``. That replay is context-dependent and
    uncorroborated here, so it is held for review rather than applied, while the
    genuinely *conflicting* value still takes the ``unsafe_ambiguous`` path."""
    df = frame(["m"] * 8 + ["M"] * 2 + ["S"] * 3 + ["L"] * 3)
    out, report = fd.clean(df, semantic_mode="auto", memory=gender_memory, **CLEAN)

    assert out["segment"].tolist()[:8] != ["male"] * 8, (
        "an uncorroborated gender replay must not rewrite the dominant spelling"
    )
    # The genuinely *conflicting* value is handled correctly, though:
    conflicts = [
        a for a in semantic_actions(report)
        if a.metadata.get("issue_type") == "unsafe_ambiguous"
    ]
    assert len(conflicts) == 1
    assert conflicts[0].metadata["raw_value"] == "M"
    assert conflicts[0].metadata["proposed_value"] is None
    assert conflicts[0].status == "suggested"
    assert conflicts[0].risk == "high"
    assert conflicts[0].human_review is True
    assert "M" in out["segment"].tolist()  # the conflicting value is never mutated


def test_learning_profile_replay_shows_the_same_one_token_trap() -> None:
    """FINDING (S2): the defect is in *replay*, not in ``CleaningMemory``.
    A ``.fdprofile`` value map learned from the same pair reproduces it through
    a different backend (``semantic:allowed_value_map:profile``)."""
    messy = frame(["M", "F"] * 12)
    clean = frame(["male", "female"] * 12)
    profile = fd.learn(messy, clean, dataset_id="hr", min_support=5)
    assert "segment" in profile.value_maps

    out, report = fd.clean(sizes_frame(), profile=profile, semantic_mode="auto", **CLEAN)
    assert out["segment"].tolist() == ["S", "male", "L", "male"] * 3
    applied = [a for a in semantic_actions(report) if a.status == "automatic"]
    assert [a.model_id for a in applied] == ["semantic:allowed_value_map:profile"]


# --------------------------------------------------------------------------- #
# Collision axes that ARE handled
# --------------------------------------------------------------------------- #


def test_same_value_in_a_differently_named_column_is_not_replayed(
    gender_memory: CleaningMemory,
) -> None:
    """Retrieval is keyed on the column name and gated by the signature match,
    so a renamed column blocks replay twice over."""
    renamed = frame(["S", "M", "L", "M"] * 3, column="tier")
    match = gender_memory.match(renamed)
    assert match.ok is False
    assert any("missing" in reason for reason in match.reasons)

    out, report = fd.clean(renamed, semantic_mode="auto", memory=gender_memory, **CLEAN)
    assert out["tier"].tolist() == renamed["tier"].tolist()
    assert semantic_actions(report) == []


def test_allowed_values_hint_blocks_the_collision(gender_memory: CleaningMemory) -> None:
    """Telling freshdata what ``segment`` now means is an effective mitigation:
    an explicit reference list moves the column to ``ReferenceExpert``, so the
    learned ``category_synonym`` repair no longer passes ``applies()``."""
    context = {"columns": {"segment": {"allowed_values": ["S", "M", "L"]}}}
    out, report = fd.clean(
        sizes_frame(), semantic_mode="auto", memory=gender_memory,
        semantic_context=context, **CLEAN,
    )
    assert out["segment"].tolist() == sizes_frame()["segment"].tolist()
    assert not [a for a in semantic_actions(report) if a.status == "automatic"]


def test_a_semantic_type_hint_does_not_revive_the_collision(
    gender_memory: CleaningMemory,
) -> None:
    """Declaring a semantic type must not re-enable the uncorroborated replay.

    The stored ``column_signature.semantic_type`` is still never compared
    against the live one at retrieval time -- that remains a gap -- but it no
    longer matters for safety here, because the repair is held for review on
    the absence of corroboration rather than on any signature check.
    """
    context = {"columns": {"segment": {"semantic_type": "category"}}}
    out, report = fd.clean(
        sizes_frame(), semantic_mode="auto", memory=gender_memory,
        semantic_context=context, **CLEAN,
    )
    assert out["segment"].tolist() == sizes_frame()["segment"].tolist()
    assert not [a for a in semantic_actions(report) if a.status == "automatic"]


def test_mutable_false_blocks_the_collision_and_audits_the_skip(
    gender_memory: CleaningMemory,
) -> None:
    context = {"columns": {"segment": {"mutable": False}}}
    out, report = fd.clean(
        sizes_frame(), semantic_mode="auto", memory=gender_memory,
        semantic_context=context, **CLEAN,
    )
    assert out["segment"].tolist() == sizes_frame()["segment"].tolist()
    skipped = semantic_actions(report)
    assert skipped and all(a.status == "skipped" for a in skipped)
    assert all(a.model_id.endswith(":memory") for a in skipped)


@pytest.mark.parametrize("mode", ("assist", "review"))
def test_non_auto_modes_never_apply_a_replayed_repair(
    gender_memory: CleaningMemory, mode: str
) -> None:
    out, report = fd.clean(sizes_frame(), semantic_mode=mode, memory=gender_memory, **CLEAN)
    assert out["segment"].tolist() == sizes_frame()["segment"].tolist()
    suggested = semantic_actions(report)
    assert suggested and all(a.status == "suggested" for a in suggested)
    assert all(a.human_review for a in suggested)


@pytest.mark.parametrize(
    "protection", ({"target_column": "segment"}, {"id_columns": ("segment",)},
                   {"preserve_columns": ("segment",)}),
)
def test_column_protection_survives_the_collision(
    gender_memory: CleaningMemory, protection: dict
) -> None:
    """Memory is evidence, not authority: target / id / preserve protection is
    never overridden by a replayed repair."""
    out, _ = fd.clean(
        sizes_frame(), semantic_mode="auto", memory=gender_memory, **protection, **CLEAN
    )
    assert out["segment"].tolist() == sizes_frame()["segment"].tolist()


def test_stale_role_in_memory_demotes_the_column_to_an_identifier(
    gender_memory: CleaningMemory,
) -> None:
    """A memory whose stored roles say ``segment`` is an id folds that into
    ``id_columns``, which then protects the column from its own replay."""
    stale = CleaningMemory.from_dict({**gender_memory.to_dict(), "roles": {"segment": "id"}})
    assert stale.config_overrides()["id_columns"] == ("segment",)
    out, _ = fd.clean(sizes_frame(), semantic_mode="auto", memory=stale, **CLEAN)
    assert out["segment"].tolist() == sizes_frame()["segment"].tolist()


# --------------------------------------------------------------------------- #
# Audit-trail distinguishability
# --------------------------------------------------------------------------- #


def test_every_memory_derived_decision_is_distinguishable_in_the_audit_trail(
    gender_memory: CleaningMemory,
) -> None:
    """All five signals hold on the colliding replay, so a reviewer can tell a
    memory-derived decision from a deterministic one without guessing.

    The provenance must survive the demotion: a replay that is held for review
    rather than applied is exactly the case a reviewer has to understand, so
    losing the memory markers there would be worse than losing them on an
    auto-applied one.
    """
    _, report = fd.clean(sizes_frame(), semantic_mode="auto", memory=gender_memory, **CLEAN)
    replayed = [
        a for a in semantic_actions(report)
        if a.metadata.get("backend") == "memory"
    ]
    assert len(replayed) == 1
    action = replayed[0]

    assert action.memory_influenced is True                      # signal 1
    assert action.status == "suggested"                          # signal 2: held, not applied
    assert action.human_review is True
    assert action.model_id == "semantic:category_synonym:memory"  # signal 3
    kinds = [e["kind"] for e in action.metadata["evidence"]]
    assert "memory_replay" in kinds                              # signal 4
    assert action.metadata["backend"] == "memory"                # signal 5
    assert "hr" in action.rationale

    # The report must round-trip to plain JSON with the provenance intact.
    payload = report.to_dict()
    json.dumps(payload)
    entry = next(a for a in payload["actions"] if a.get("model_id", "").endswith(":memory"))
    assert entry["memory_influenced"] is True
    assert entry["metadata"]["backend"] == "memory"


def test_non_semantic_replay_is_marked_approved_and_memory_influenced() -> None:
    """The ``status="approved"`` signal, on the pathway that actually sets it."""
    learn_df = pd.DataFrame({
        "amount": [1.0, 2.0, None, 4.0, 5.0],
        "name": ["x", "x", "y", None, "z"],
        "id": [1, 2, 3, 4, 5],
    })
    _, report = fd.clean(learn_df, **CLEAN)
    memory = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="crm")
    similar = pd.DataFrame({
        "amount": [10.0, None, 30.0, 40.0, 50.0],
        "name": ["a", "b", "b", "c", None],
        "id": [6, 7, 8, 9, 10],
    })
    _, replayed = fd.clean(similar, memory=memory, **CLEAN)
    summary = [a for a in replayed.actions if a.step == "memory"]
    assert len(summary) == 1
    assert summary[0].status == "approved"
    assert summary[0].memory_influenced is True
    assert all(
        a.status == "approved" for a in replayed.actions if a.memory_influenced
    )


# --------------------------------------------------------------------------- #
# Corrupted / foreign / colliding memory payloads
# --------------------------------------------------------------------------- #


def test_malformed_memory_file_raises_a_json_error(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        fd.load_cleaning_memory(str(path))


def test_missing_memory_file_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        fd.load_cleaning_memory(str(tmp_path / "nope.json"))


def test_foreign_payload_loads_permissively_and_is_ignored_on_replay(tmp_path) -> None:
    """FINDING (S4, spec gap): ``CleaningMemory`` JSON carries only
    ``freshdata_version`` — no schema version — and ``from_dict`` is a bare
    ``payload.get(...)`` per field with no type checking. A foreign JSON
    document therefore loads as an empty-but-valid memory (``dataset_id=""``,
    ``accepted`` left as whatever string was in the file).

    It is caught downstream — with no stored signature, ``match()`` fails and
    replay is refused with an explanatory warning — so nothing is corrupted,
    but the diagnostic points at data drift rather than at a bad file.
    """
    path = tmp_path / "foreign.json"
    path.write_text(json.dumps({"hello": "world", "accepted": "not-a-list"}), encoding="utf-8")
    memory = fd.load_cleaning_memory(str(path))

    assert memory.dataset_id == ""
    assert memory.signature == {}
    assert memory.accepted == "not-a-list"  # no type validation at all
    assert memory.match(sizes_frame()).ok is False

    out, report = fd.clean(sizes_frame(), semantic_mode="auto", memory=memory, **CLEAN)
    assert out["segment"].tolist() == sizes_frame()["segment"].tolist()
    assert any("was ignored" in w for w in report.warnings)


def test_memory_with_a_non_numeric_threshold_crashes_with_an_opaque_typeerror(
    tmp_path,
) -> None:
    """FINDING (S3): ``config_overrides()`` copies any value stored under an
    allowed threshold key straight into ``CleanConfig``, unvalidated. A
    hand-edited or foreign memory holding a string threshold takes down
    ``fd.clean`` with a bare ``TypeError`` from deep inside the missing-value
    band comparison, naming neither the memory nor the field.
    """
    payload = {
        "dataset_id": "x",
        "signature": {
            "columns": {"segment": "text", "amount": "float", "note": "text"},
            "n_cols": 3,
            "hash": "deadbeefdeadbeef",
        },
        "thresholds": {"missing_threshold_low": "not-a-number"},
    }
    path = tmp_path / "bad_threshold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    memory = fd.load_cleaning_memory(str(path))

    assert memory.match(sizes_frame()).ok is True
    assert memory.config_overrides()["missing_threshold_low"] == "not-a-number"
    with pytest.raises(TypeError, match="not supported between instances"):
        fd.clean(sizes_frame().assign(amount=[np.nan] + [1.0] * 11), memory=memory, **CLEAN)


def test_version_mismatch_is_recorded_but_never_checked(
    gender_memory: CleaningMemory,
) -> None:
    """FINDING (S4, spec gap): ``freshdata_version`` is written on learn and
    never read again. A memory claiming an ancient (or a future) version
    replays exactly like a current one, with no warning anywhere.

    Contrast ``freshdata.learning``, which ships ``ProfileManifest`` /
    ``ProfileVersionError`` for precisely this.
    """
    ancient = CleaningMemory.from_dict(
        {**gender_memory.to_dict(), "freshdata_version": "0.0.1"}
    )
    out, report = fd.clean(sizes_frame(), semantic_mode="auto", memory=ancient, **CLEAN)
    # The version is still never checked -- that is the finding. The values
    # survive only because the replay is uncorroborated here, not because the
    # stale version was noticed.
    assert out["segment"].tolist() == sizes_frame()["segment"].tolist()
    assert not [w for w in report.warnings if "version" in w.lower()]


def test_sqlite_dataset_id_collision_silently_overwrites(tmp_path) -> None:
    """FINDING (S3): the SQLite store is keyed on ``dataset_id`` alone with
    ``ON CONFLICT DO UPDATE``. Writing a memory for an unrelated schema under a
    dataset_id already in the store destroys the previous memory — different
    signature hash, different learned repairs — with no warning and no way to
    recover it.
    """
    store = str(tmp_path / "memories.sqlite")
    learn_df = frame(["M", "F"] * 4)
    _, report = fd.clean(learn_df, semantic_mode="auto", **CLEAN)
    first = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="shared")
    first.to_json(store)

    unrelated = pd.DataFrame({"totally": [1, 2, 3], "other": [4, 5, 6], "cols": [7, 8, 9]})
    second = fd.learn_cleaning_memory(unrelated, decisions=[], dataset_id="shared")
    second.to_json(store)  # no error, no warning

    loaded = fd.load_cleaning_memory(store, dataset_id="shared")
    assert loaded.signature["hash"] != first.signature["hash"]
    assert list(loaded.signature["columns"]) == ["totally", "other", "cols"]
    assert "semantic_repairs" not in loaded.value_patterns
    # ... and the diff API can only report the loss after the fact.
    assert first.diff(loaded)["signature_changed"] is True


def test_json_path_rejects_a_dataset_id_that_does_not_match(tmp_path) -> None:
    """The plain-JSON path *does* guard identity (the SQLite path cannot)."""
    memory = fd.learn_cleaning_memory(frame(["M", "F"] * 4), decisions=[], dataset_id="hr")
    path = tmp_path / "hr.json"
    memory.to_json(str(path))
    with pytest.raises(KeyError, match="finance"):
        fd.load_cleaning_memory(str(path), dataset_id="finance")
