"""High-risk configuration *combinations* (Phase 18).

Each option in this file is already covered on its own elsewhere. What is
tested here is what happens when two or three of them are set at once: does the
combination behave coherently, does one option silently cancel the other, and
is any genuine conflict reported rather than resolved in silence?

A combination that silently drops one option's effect is recorded as a finding
with an ``S2:``/``S3:`` comment on the test that pins it.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import freshdata as fd
from freshdata.api import _fold_profile
from freshdata.context import apply_policy_to_config
from freshdata.learning import learn

# ── fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def semantic_frame():
    """Wide enough (and repetitive enough) for the semantic layer to engage."""
    return pd.DataFrame(
        {
            "row_id": [f"R{i}" for i in range(12)],
            "email": [
                "a@x.com", "B@X.COM", "c@x.com", "d@x.com", "e@x.com", "f@x.com",
                "g@x.com", "h@x.com", "i@x.com", "j@x.com", "k@x.com", "l@x.com",
            ],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
                       90.0, 100.0, 110.0, 5000.0],
        }
    )


@pytest.fixture
def ledger():
    """A frame the finance domain pack recognises well enough to grade."""
    return pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3", "T4"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "debit": [100.0, 0.0, 50.0, 0.0],
            "credit": [0.0, 100.0, 0.0, 50.0],
            "amount": [100.0, 100.0, 50.0, 50.0],
        }
    )


def semantic_applied(frame):
    """True when the semantic layer normalised the mixed-case email."""
    return frame["email"].iloc[1] == "B@x.com"


# ── 1. domain + semantic_mode + context ─────────────────────────────────────────


def test_domain_semantic_and_context_all_stay_active_in_one_run(ledger):
    """All three layers run; none cancels another."""
    cleaned, report = fd.clean(
        ledger,
        domain="finance",
        semantic_mode="assist",
        context="Never modify amount. transaction_id is unique.",
        return_report=True,
        verbose=False,
    )
    steps = {action.step for action in report}
    assert any(step.startswith("domain:finance") for step in steps), "domain pack did not run"
    assert "context" in steps, "context policy was not applied"
    assert report.domain_trust_score is not None, "domain grading was skipped"
    # The context protection holds through the domain repair pass.
    pd.testing.assert_series_equal(cleaned["amount"], ledger["amount"])


def test_domain_defaults_to_conservative_but_context_protection_still_applies(ledger):
    """The domain default (``strategy="conservative"``) does not disable the policy."""
    _, with_context = fd.clean(
        ledger, domain="finance", context="Never modify debit.",
        return_report=True, verbose=False,
    )
    _, without_context = fd.clean(
        ledger, domain="finance", return_report=True, verbose=False,
    )
    assert "context" in {a.step for a in with_context}
    assert "context" not in {a.step for a in without_context}
    # Domain findings are identical either way — the policy adds, never subtracts.
    def domain_steps(report):
        return sorted(a.step for a in report if a.step.startswith("domain:"))

    assert domain_steps(with_context) == domain_steps(without_context)


# ── 2. engine="polars" + domain (and the rest of the execution keywords) ────────


def test_domain_silently_discards_engine_output_format_and_engine_config(ledger):
    """S2: ``domain=`` short-circuits the whole execution-layer dispatch.

    ``clean`` routes to the domain path *before* it looks at ``engine``,
    ``output_format``, ``engine_config`` or ``fallback_policy``. The run then
    executes on pandas and returns pandas, with no fallback event recorded, so
    ``output_format="polars"`` quietly returns the wrong handle type and an
    ``engine_config`` is discarded outright. Every other pandas-only feature
    (``contract=``/``memory=``/``profile=``/``context=``) raises a ``TypeError``
    on a native engine instead of doing this.
    """
    out, report = fd.clean(
        ledger, domain="finance", engine="polars", return_report=True, verbose=False
    )
    assert isinstance(out, pd.DataFrame), "ran on pandas despite engine='polars'"
    assert report.fallback_events == [], "the silent pandas execution is not even recorded"

    # ``output_format`` is meant to choose the returned handle type.
    assert isinstance(fd.clean(ledger, domain="finance", output_format="polars", verbose=False),
                      pd.DataFrame)

    # An explicit EngineConfig is dropped rather than refused.
    config = fd.EngineConfig(engine="polars", fallback_policy="error")
    assert isinstance(fd.clean(ledger, domain="finance", engine_config=config, verbose=False),
                      pd.DataFrame)


def test_domain_silently_defeats_the_strict_no_fallback_guarantee(ledger):
    """S2: ``fallback_policy="error"`` does not fire when ``domain=`` is set.

    ``fallback_policy="error"`` is documented as raising ``FallbackError``
    *before* any pandas materialization — "the strict out-of-core guarantee".
    Without a domain it does exactly that. With one, the same call runs to
    completion on pandas and returns a frame.
    """
    with pytest.raises(fd.FallbackError):
        fd.clean(ledger, engine="polars", fallback_policy="error", verbose=False)

    out = fd.clean(
        ledger, domain="finance", engine="polars", fallback_policy="error", verbose=False
    )
    assert isinstance(out, pd.DataFrame)


# ── 3. contract + a non-pandas engine ───────────────────────────────────────────


def test_contract_gate_is_refused_on_native_engines_and_honoured_on_pandas(ledger):
    """The conflict is reported, not resolved silently — the right behaviour."""
    contract = {"name": "ledger", "columns": [{"name": "transaction_id", "dtype": "object"}]}

    for engine in ("polars", "duckdb", "auto"):
        with pytest.raises(TypeError, match=r"contract= is only supported on the in-memory"):
            fd.clean(ledger, engine=engine, contract=contract, verbose=False)
    with pytest.raises(TypeError, match=r"contract= is only supported on the in-memory"):
        fd.clean(ledger, output_format="polars", contract=contract, verbose=False)

    polars = pytest.importorskip("polars")
    with pytest.raises(TypeError, match=r"contract= requires an in-memory pandas DataFrame"):
        fd.clean(polars.from_pandas(ledger), contract=contract, verbose=False)

    # On pandas the same contract is actually enforced.
    _, report = fd.clean(ledger, contract=contract, return_report=True, verbose=False)
    assert report.contract_violations is not None


# ── 4. profile + policy + memory ────────────────────────────────────────────────


@pytest.fixture
def learned_trio():
    messy = pd.DataFrame(
        {
            "state": ["CA ", "ca", "Calif.", "NY", "ny", "N.Y.", "CA", "NY"],
            "amount": [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0],
        }
    )
    cleaned = pd.DataFrame(
        {
            "state": ["CA", "CA", "CA", "NY", "NY", "NY", "CA", "NY"],
            "amount": [1.0, 2.0, 3.0, 4.5, 5.0, 6.0, 7.0, 8.0],
        }
    )
    profile = learn(messy, cleaned, dataset_id="ds1")
    _, report = fd.clean(messy, return_report=True, verbose=False)
    memory = fd.learn_cleaning_memory(messy, report, dataset_id="ds1")
    policy = fd.compile_context("Never modify state.", df=messy)
    return messy, profile, memory, policy


def test_policy_protection_outranks_both_profile_and_memory(learned_trio):
    """The documented precedence — "user options and policy always win"."""
    messy, profile, memory, policy = learned_trio

    baseline = fd.clean(messy, profile=profile, memory=memory,
                        semantic_mode="auto", verbose=False)
    guarded = fd.clean(messy, profile=profile, memory=memory, policy=policy,
                       semantic_mode="auto", verbose=False)

    # Without the policy the replay layers are free to touch ``state``; with it
    # the column is byte-identical to the input, whitespace and all.
    assert list(guarded["state"]) == list(messy["state"])
    assert isinstance(baseline, pd.DataFrame)


def test_profile_supplies_its_embedded_memory_only_when_none_was_passed(learned_trio):
    """``memory=`` is not silently replaced by the profile's embedded memory."""
    messy, profile, memory, _ = learned_trio
    assert profile.memory is not None, "fixture profile should embed a memory"

    _, _, adopted = _fold_profile(
        messy, profile, {}, None, engine="pandas", output_format="pandas", engine_config=None
    )
    assert adopted is profile.memory, "an absent memory= should adopt the profile's"

    _, _, kept = _fold_profile(
        messy, profile, {}, memory, engine="pandas", output_format="pandas", engine_config=None
    )
    assert kept is memory, "an explicit memory= must not be overridden by the profile's"


def test_profile_policy_and_memory_together_do_not_raise_or_warn(learned_trio):
    messy, profile, memory, policy = learned_trio
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, report = fd.clean(
            messy, profile=profile, memory=memory, policy=policy,
            return_report=True, verbose=False,
        )
    assert report is not None


# ── 5. target_column + semantic_mode ────────────────────────────────────────────


def test_target_column_withdraws_only_the_target_from_semantic_repair(semantic_frame):
    """Scoped, not global: the target is spared, other columns are still repaired."""
    unguarded = fd.clean(semantic_frame, semantic_mode="auto", verbose=False)
    assert semantic_applied(unguarded), "fixture no longer exercises the semantic layer"

    guarded, report = fd.clean(
        semantic_frame, semantic_mode="auto", target_column="email",
        return_report=True, verbose=False,
    )
    assert not semantic_applied(guarded)
    assert list(guarded["email"]) == list(semantic_frame["email"])
    # Protection is scoped to the target: the id guard still reports separately.
    assert any(a.column == "row_id" for a in report if a.step == "semantic")


@pytest.mark.parametrize("guard", ["target_column", "id_columns", "preserve_columns"])
def test_every_column_protection_channel_stops_semantic_repair(semantic_frame, guard):
    value = "email" if guard == "target_column" else ("email",)
    out = fd.clean(semantic_frame, semantic_mode="auto", verbose=False, **{guard: value})
    assert not semantic_applied(out), f"{guard} did not protect the column from semantic repair"


def test_sensitive_columns_masks_reporting_and_does_not_protect_values(semantic_frame):
    """Disproved suspicion: this is documented behaviour, not a silent override.

    ``sensitive_columns`` redacts values from report text; it is not a
    protection channel, so the semantic layer still repairs the column.
    """
    out, report = fd.clean(
        semantic_frame, semantic_mode="auto", sensitive_columns=("email",),
        return_report=True, verbose=False,
    )
    assert semantic_applied(out)
    assert not any("B@X.COM" in (a.description or "") for a in report)


# ── 6. id_columns + clean_text ──────────────────────────────────────────────────


def test_clean_text_has_no_id_column_channel_only_field_types():
    """SPECIFICATION GAP: ``fd.clean_text`` cannot be told which columns are ids.

    ``fd.clean``'s ``id_columns`` never reaches ``fd.clean_text``, which takes
    only ``columns``/``config``/``field_types``. With a lossy config an
    identifier column is mangled unless the caller *separately* remembers to
    pass ``field_types``. The two protection vocabularies do not meet.
    """
    ids = pd.DataFrame({"acct_id": ["  A-001  ", "A-002", "a-003"],
                        "note": ["  Hi  ", "YO", "x"]})
    lossy = fd.TextCleanConfig(case="lower", remove_punctuation=True)

    unguarded, _ = fd.clean_text(ids, config=lossy)
    assert list(unguarded["acct_id"]) == ["a001", "a002", "a003"]

    guarded, _ = fd.clean_text(ids, config=lossy, field_types={"acct_id": "identifier"})
    assert list(guarded["acct_id"]) == ["A-001", "A-002", "a-003"]

    assert "id_columns" not in (fd.clean_text.__doc__ or "")


@pytest.mark.parametrize("guard", ["id_columns", "preserve_columns", "sensitive_columns"])
def test_representation_level_case_folding_ignores_the_column_protections(guard):
    """SPECIFICATION GAP: ``string_case`` is a layer-1 repair with no opt-outs.

    ``id_columns`` ("never imputed; outliers ignored") and ``preserve_columns``
    ("must never be dropped") are both documented narrowly, so this is not a
    contradiction of the docs — but it does mean the obvious way to say "leave
    my identifiers alone" does not stop ``string_case`` from rewriting them.
    Only a context policy does (next test).
    """
    ids = pd.DataFrame({"acct_id": ["A-001", "A-002", "a-003"], "note": ["Hi", "YO", "x"]})
    out = fd.clean(ids, string_case="lower", verbose=False, **{guard: ("acct_id",)})
    assert list(out["acct_id"]) == ["a-001", "a-002", "a-003"]


def test_a_context_policy_does_stop_representation_level_case_folding():
    ids = pd.DataFrame({"acct_id": ["  A-001  ", "A-002", "a-003"], "note": ["Hi", "YO", "x"]})
    policy = fd.compile_context("Never modify acct_id.", df=ids)
    assert policy.protected_columns == ("acct_id",)

    out = fd.clean(ids, policy=policy, string_case="lower", verbose=False)
    assert list(out["acct_id"]) == list(ids["acct_id"]), "protection must be byte-identical"
    assert list(out["note"]) == ["hi", "yo", "x"], "other columns still fold"


# ── 7. strict=True + fallback_policy="error" ────────────────────────────────────


def test_context_refusal_outranks_both_strict_and_the_fallback_gate(semantic_frame):
    """Disproved suspicion: this ordering is stable, not input-dependent.

    I expected ``strict=True`` to compile the rules first and report an
    unresolved column (which is what happens on the pandas engine). On a native
    engine the context is refused *before* it is ever compiled, so the error is
    the same whether the rule text resolves or not — one deterministic message
    instead of two different ones depending on the user's wording.
    """
    for text in ("Never modify zzz.", "Never modify email.", "Blorble the wumpus."):
        with pytest.raises(TypeError, match=r"context=/policy= are only supported on the"):
            fd.clean(
                semantic_frame, context=text, strict=True,
                engine="polars", fallback_policy="error", verbose=False,
            )

    # On the pandas engine the strict compile is what fires.
    with pytest.raises(fd.PolicyError, match=r"unresolved column reference 'zzz'"):
        fd.clean(semantic_frame, context="Never modify zzz.", strict=True, verbose=False)


def test_strict_plus_fallback_error_still_raises_the_fallback_error_without_context(
    semantic_frame,
):
    with pytest.raises(fd.FallbackError, match=r"fell back to pandas"):
        fd.clean(
            semantic_frame, strict=True, engine="polars",
            fallback_policy="error", verbose=False,
        )


def test_strict_and_fallback_error_are_both_refused_together_with_a_context(semantic_frame):
    """A resolvable context on a native engine is a reported conflict, not a silent drop."""
    with pytest.raises(TypeError, match=r"context=/policy= are only supported on the"):
        fd.clean(
            semantic_frame, context="Never modify email.", strict=True,
            engine="polars", fallback_policy="error", verbose=False,
        )


# ── 8. streaming + semantic ─────────────────────────────────────────────────────


def test_streaming_runs_the_semantic_layer_rather_than_dropping_it(semantic_frame):
    cleaner = fd.StreamingCleaner(semantic_mode="auto")
    batch, report = cleaner.clean_batch(semantic_frame)
    assert semantic_applied(batch), "semantic_mode was silently dropped by StreamingCleaner"
    assert any(a.step == "semantic" for a in report)

    plain, _ = fd.StreamingCleaner().clean_batch(semantic_frame)
    assert not semantic_applied(plain)


def test_streaming_semantic_honours_the_column_protections(semantic_frame):
    cleaner = fd.StreamingCleaner(semantic_mode="auto", target_column="email")
    batch, _ = cleaner.clean_batch(semantic_frame)
    assert not semantic_applied(batch)


def test_streaming_reports_an_unknown_semantic_backend_like_clean_does(semantic_frame):
    cleaner = fd.StreamingCleaner(semantic_mode="auto", semantic_backends=("telepathy",))
    _, report = cleaner.clean_batch(semantic_frame)
    assert any("telepathy" in w for w in report.warnings)
    assert any(e["backend"] == "telepathy" for e in report.fallback_events)


def test_streaming_compiles_a_context_once_and_keeps_it_across_batches(semantic_frame):
    cleaner = fd.StreamingCleaner(context="Never modify email.")
    first, first_report = cleaner.clean_batch(semantic_frame.iloc[:6])
    second, second_report = cleaner.clean_batch(semantic_frame.iloc[6:])

    assert list(first["email"]) == list(semantic_frame["email"].iloc[:6])
    assert list(second["email"]) == list(semantic_frame["email"].iloc[6:])
    # The compiled policy is surfaced once, on the batch that first applied it.
    assert cleaner._policy is not None
    assert cleaner._policy.protected_columns == ("email",)
    assert first_report is not second_report


# ── 9. preserve_columns + a context-declared protected column ──────────────────


def test_preserve_columns_and_context_protection_are_unioned_not_overridden():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, None, 5.0, 6.0],
            "b": ["x", "y", None, "z", "w", "v"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0, 900.0],
        }
    )
    policy = fd.compile_context("Never modify c.", df=df)
    assert policy.protected_columns == ("c",)

    lowered = apply_policy_to_config(
        fd.CleanConfig(policy=policy, preserve_columns=("b",)), df=df
    )
    assert set(lowered.preserve_columns) == {"b", "c"}, "one protection list replaced the other"

    # And the policy's protection is stronger than preserve_columns alone: it is
    # also lowered into an immutable semantic hint.
    assert lowered.semantic_context["columns"]["c"]["mutable"] is False


def test_context_protection_survives_an_empty_preserve_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, None, 5.0, 6.0]})
    policy = fd.compile_context("Never modify a.", df=df)
    lowered = apply_policy_to_config(fd.CleanConfig(policy=policy, preserve_columns=()), df=df)
    assert lowered.preserve_columns == ("a",)


def test_preserve_columns_does_not_leak_into_the_policy_object():
    """The policy stays a faithful record of the *text*; folding happens in the config."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    policy = fd.compile_context("Never modify a.", df=df)
    apply_policy_to_config(fd.CleanConfig(policy=policy, preserve_columns=("b",)), df=df)
    assert policy.protected_columns == ("a",)
