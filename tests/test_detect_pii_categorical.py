"""detect_pii / detection-driven anonymize scan categorical text on every pandas line (#280)."""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata._util import is_text_dtype
from freshdata.enterprise import PIIDetectionConfig, anonymize, detect_pii


def test_issue_poc_categorical_note_is_scanned_and_scrubbed():
    df = pd.DataFrame({"note": pd.Categorical(["mail a@b.com", "ssn 123-45-6789"])})
    assert detect_pii(df).columns_scanned == ("note",)
    out, rep = anonymize(df, detection_config=PIIDetectionConfig())
    assert out["note"].tolist() == ["mail <EMAIL>", "ssn <SSN>"]
    assert out["note"].dtype == object
    assert rep.entities_found == 2
    assert df["note"].dtype == "category"  # input untouched


def test_categorical_missing_values_stay_missing():
    df = pd.DataFrame({"note": pd.Categorical(["a@b.com", None, "plain"])})
    out, _ = anonymize(df, detection_config=PIIDetectionConfig())
    values = out["note"].tolist()
    assert values[0] == "<EMAIL>"
    assert pd.isna(values[1])
    assert values[2] == "plain"


def test_non_text_categorical_is_not_scanned():
    df = pd.DataFrame(
        {
            "n": pd.Categorical([4111111111111111, 5500000000000004]),
            "when": pd.Categorical(pd.to_datetime(["2020-01-01", "2021-01-01"])),
        }
    )
    assert detect_pii(df).columns_scanned == ()


def test_categorical_pan_column_is_masked_by_learn(tmp_path):
    cards = ["4111 1111 1111 1111", "5500 0000 0000 0004"]
    # The messy side is categorical: that is the frame detect_pii scans.
    messy = pd.DataFrame({"ref": pd.Categorical([c + " " for c in cards] * 10)})
    clean = pd.DataFrame({"ref": cards * 10})
    profile = fd.learn(messy, clean, min_support=2)
    assert profile.audit().sensitive_columns == {"ref": "payment_card"}
    path = tmp_path / "p.fdprofile"
    profile.save(path)
    with zipfile.ZipFile(path) as z:
        text = "".join(z.read(n).decode("utf-8") for n in z.namelist())
    assert [c for c in cards if c in text] == []


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.dtype(object), True),
        (pd.StringDtype(), True),
        (pd.CategoricalDtype(["a", "b"]), True),
        (pd.CategoricalDtype([1, 2]), False),
        (np.dtype("int64"), False),
        (np.dtype("float64"), False),
        (np.dtype("datetime64[ns]"), False),
        (pd.Int64Dtype(), False),
        (np.dtype(bool), False),
    ],
)
def test_is_text_dtype(dtype, expected):
    assert is_text_dtype(dtype) is expected


def test_is_text_dtype_arrow():
    pa = pytest.importorskip("pyarrow")
    arrow_dtype = getattr(pd, "ArrowDtype", None)
    if arrow_dtype is None:
        pytest.skip("pd.ArrowDtype not available")
    assert is_text_dtype(pd.StringDtype("pyarrow"))
    assert is_text_dtype(arrow_dtype(pa.string()))
    assert is_text_dtype(arrow_dtype(pa.large_string()))
    assert is_text_dtype(arrow_dtype(pa.dictionary(pa.int32(), pa.string())))
    assert not is_text_dtype(arrow_dtype(pa.int64()))
    assert not is_text_dtype(arrow_dtype(pa.dictionary(pa.int32(), pa.int64())))


def test_arrow_string_column_is_scanned():
    pa = pytest.importorskip("pyarrow")
    arrow_dtype = getattr(pd, "ArrowDtype", None)
    if arrow_dtype is None:
        pytest.skip("pd.ArrowDtype not available")
    df = pd.DataFrame({"note": pd.Series(["mail a@b.com", None], dtype=arrow_dtype(pa.string()))})
    rep = detect_pii(df)
    assert rep.columns_scanned == ("note",)
    assert [e.entity_type for e in rep.entities] == ["EMAIL"]
