"""CLI tests for `freshdata learn`, `clean --profile`, and `profile audit|diff|merge`.

Driven through ``cli.main(argv)`` like the other enterprise CLI tests.
"""

from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from freshdata.enterprise import cli
from freshdata.learning import learn, save_profile


@pytest.fixture()
def csv_pair(orders_pair, tmp_path):
    messy, clean = orders_pair
    raw_path = tmp_path / "raw.csv"
    clean_path = tmp_path / "clean.csv"
    messy.to_csv(raw_path, index=False)
    clean.to_csv(clean_path, index=False)
    return raw_path, clean_path


@pytest.fixture()
def profile_path(orders_profile, tmp_path):
    path = tmp_path / "orders.fdprofile"
    save_profile(orders_profile, path)
    return path


@pytest.fixture()
def new_csv(new_batch, tmp_path):
    path = tmp_path / "new.csv"
    new_batch.to_csv(path, index=False)
    return path


class TestLearnCommand:
    def test_learn_writes_profile(self, csv_pair, tmp_path, capsys):
        raw_path, clean_path = csv_pair
        out = tmp_path / "out.fdprofile"
        code = cli.main(
            [
                "learn",
                str(raw_path),
                str(clean_path),
                "--key",
                "order_id",
                "--context",
                "order_id is a unique identifier. Never modify order_id.",
                "--min-support",
                "2",
                "-o",
                str(out),
            ]
        )
        assert code == 0
        assert out.exists()
        printed = capsys.readouterr().out
        assert "fdp-" in printed
        # No raw sensitive literals from the training data in the output.
        assert "asha@gmail.com" not in printed.lower()

    def test_learn_missing_output_flag_fails(self, csv_pair):
        raw_path, clean_path = csv_pair
        with pytest.raises(SystemExit):
            cli.main(["learn", str(raw_path), str(clean_path)])


class TestCleanWithProfileFlag:
    def test_clean_profile_replays(self, profile_path, new_csv, tmp_path, capsys):
        out = tmp_path / "cleaned.csv"
        code = cli.main(
            [
                "clean",
                str(new_csv),
                "--profile",
                str(profile_path),
                "--semantic-mode",
                "auto",
                "-o",
                str(out),
                "--quiet",
            ]
        )
        assert code == 0
        cleaned = pd.read_csv(out)
        assert list(cleaned["status"]) == ["delivered", "shipped", "shipped"]

    def test_clean_profile_drift_note(self, profile_path, tmp_path, capsys):
        drifted = tmp_path / "drifted.csv"
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(drifted, index=False)
        code = cli.main(["clean", str(drifted), "--profile", str(profile_path)])
        assert code == 0
        assert "not replayed" in capsys.readouterr().out

    def test_clean_profile_corrupt_exits_nonzero(self, tmp_path, new_csv, capsys):
        bad = tmp_path / "bad.fdprofile"
        bad.write_bytes(b"not a zip")
        code = cli.main(["clean", str(new_csv), "--profile", str(bad)])
        assert code == 2
        assert "error" in capsys.readouterr().out

    def test_clean_profile_non_pandas_engine_rejected(self, profile_path, new_csv, capsys):
        code = cli.main(
            [
                "clean",
                str(new_csv),
                "--profile",
                str(profile_path),
                "--engine",
                "polars",
            ]
        )
        assert code == 2
        assert "pandas engine" in capsys.readouterr().out


class TestProfileTools:
    def test_audit_renders(self, profile_path, capsys):
        code = cli.main(["profile", "audit", str(profile_path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "LearningProfile fdp-" in out
        assert "privacy" in out
        # Masked email literals never appear.
        assert "asha@gmail.com" not in out.lower()

    def test_audit_json(self, profile_path, capsys):
        code = cli.main(["profile", "audit", str(profile_path), "--json"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["privacy_mode"] == "mask"

    def test_audit_corrupt_hash_exits_nonzero(self, profile_path, tmp_path, capsys):
        tampered = tmp_path / "tampered.fdprofile"
        with zipfile.ZipFile(profile_path) as src, zipfile.ZipFile(tampered, "w") as dst:
            for name in src.namelist():
                data = src.read(name)
                if name == "rules.json":
                    data = data.replace(b"soft", b"SOFT", 1)
                dst.writestr(name, data)
        code = cli.main(["profile", "audit", str(tampered)])
        assert code == 2
        assert "error" in capsys.readouterr().out.lower()

    def test_diff_identical_exit_zero(self, profile_path, capsys):
        code = cli.main(["profile", "diff", str(profile_path), str(profile_path)])
        assert code == 0

    def test_diff_different_exit_one(self, profile_path, orders_pair, tmp_path, capsys):
        messy, clean = orders_pair
        clean = clean.copy()
        clean["status"] = ["returned" if s == "delivered" else s for s in clean["status"]]
        other = learn(messy, clean, key="order_id", dataset_id="o2", min_support=2)
        other_path = tmp_path / "other.fdprofile"
        save_profile(other, other_path)
        code = cli.main(["profile", "diff", str(profile_path), str(other_path)])
        assert code == 1
        assert "status" in capsys.readouterr().out

    def test_merge_writes_output(self, profile_path, tmp_path, capsys):
        merged = tmp_path / "merged.fdprofile"
        code = cli.main(
            [
                "profile",
                "merge",
                str(profile_path),
                str(profile_path),
                "-o",
                str(merged),
            ]
        )
        assert code == 0
        assert merged.exists()

    def test_merge_requires_output(self, profile_path, capsys):
        code = cli.main(["profile", "merge", str(profile_path), str(profile_path)])
        assert code == 2
        assert "-o" in capsys.readouterr().out

    def test_usage_errors(self, capsys):
        assert cli.main(["profile", "audit"]) == 2
        assert cli.main(["profile", "diff", "only-one.fdprofile"]) == 2


class TestBackCompat:
    def test_data_profiling_still_works(self, new_csv, capsys):
        code = cli.main(["profile", str(new_csv)])
        assert code == 0
        assert "order_id" in capsys.readouterr().out

    def test_extra_args_on_data_profiling_rejected(self, new_csv, capsys):
        code = cli.main(["profile", str(new_csv), "extra.csv"])
        assert code == 2
        assert "audit|diff|merge" in capsys.readouterr().out
