"""#305: ``24:00`` in a date-less time column is suggested, never auto-applied.

Without a date, ``24:00`` (end of day) and ``00:00`` (start of day) are
different clock times, so the rewrite is held for review in every mode.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.semantic.canonical import TimeCanonicalExpert
from freshdata.semantic.context import build_semantic_context

_RATIONALE = (
    "24:00 marks end of day; in a time-only column 00:00 would read as start "
    "of day, so this is held for review"
)


def _shift_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shift_start": ["14:00", "15:00", "13:30", "16:00", "12:00"],
            "shift_end": ["22:00", "23:00", "21:30", "24:00", "20:00"],
        }
    )


def _time_actions(report: fd.CleanReport, column: str) -> list[fd.Action]:
    return [
        a
        for a in report.actions
        if a.step == "semantic"
        and a.column == column
        and a.metadata.get("expert") == "time_canonical"
    ]


@pytest.mark.parametrize("mode", ["auto", "review", "assist"])
def test_issue_305_repro_keeps_24_00_and_suggests(mode: str) -> None:
    df = _shift_frame()
    out, report = fd.clean(df, semantic_mode=mode, return_report=True, verbose=False)

    assert out["shift_end"].iloc[3] == "24:00"
    assert out["shift_end"].tolist() == df["shift_end"].tolist()
    actions = _time_actions(report, "shift_end")
    assert [a.status for a in actions] == ["suggested"]
    assert actions[0].rationale == _RATIONALE
    assert actions[0].metadata.get("raw_value") == "24:00"


@pytest.mark.parametrize("raw, proposed", [("24:00", "00:00"), ("24:00:00", "00:00:00")])
def test_proposal_scores_between_review_and_auto_thresholds(raw: str, proposed: str) -> None:
    df = pd.DataFrame({"end_time": ["08:00", "09:30", "10:15", "11:00", raw]})
    config = CleanConfig(semantic_mode="auto")
    ctx = build_semantic_context(df, config)
    info = ctx.columns["end_time"]
    expert = TimeCanonicalExpert()
    assert expert.applies(info)

    proposals = expert.propose(df["end_time"], info)
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.raw_value == raw
    assert proposal.proposed_value == proposed
    assert config.semantic_review_threshold <= proposal.confidence
    assert proposal.confidence < config.semantic_auto_threshold
    assert proposal.rationale == _RATIONALE
    assert "instant is unchanged" not in proposal.rationale


def test_seconds_form_is_kept_in_auto_mode() -> None:
    df = pd.DataFrame({"end_time": ["08:00:00", "09:30:00", "10:15:00", "11:00:00", "24:00:00"]})
    out, report = fd.clean(df, semantic_mode="auto", return_report=True, verbose=False)
    assert out["end_time"].iloc[4] == "24:00:00"
    assert [a.status for a in _time_actions(report, "end_time")] == ["suggested"]
