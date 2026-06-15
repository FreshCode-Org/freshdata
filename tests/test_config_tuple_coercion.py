"""Single-string tuple config fields must not split into characters."""

import numpy as np
import pandas as pd

import freshdata as fd


def test_id_columns_accepts_single_string():
    cfg = fd.CleanConfig(id_columns="sku_num")
    assert cfg.id_columns == ("sku_num",)


def test_preserve_columns_accepts_single_string():
    cfg = fd.CleanConfig(preserve_columns="notes")
    assert cfg.preserve_columns == ("notes",)


def test_duplicate_subset_accepts_single_string():
    cfg = fd.CleanConfig(duplicate_subset="id")
    assert cfg.duplicate_subset == ("id",)


def test_extra_sentinels_accepts_single_string():
    cfg = fd.CleanConfig(extra_sentinels="missing")
    assert cfg.extra_sentinels == ("missing",)


def test_id_columns_string_protects_outliers_like_tuple():
    keys = [1] * 50 + [2] * 50 + [3] * 49 + [1000]
    df = pd.DataFrame({"sku_num": keys})
    isolate = {"drop_empty_rows": False, "drop_duplicates": False, "verbose": False}
    out_tuple = fd.clean(df, id_columns=("sku_num",), outlier_action="cap", **isolate)
    out_string = fd.clean(df, id_columns="sku_num", outlier_action="cap", **isolate)
    assert out_tuple["sku_num"].max() == out_string["sku_num"].max() == 1000


def test_preserve_columns_string_preserves_outliers_like_tuple():

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"mynotes": list(rng.normal(1, 1, 199)) + [1000]})
    isolate = {"drop_empty_rows": False, "drop_duplicates": False, "verbose": False}
    out_tuple = fd.clean(df, preserve_columns=("mynotes",), outlier_action="cap", **isolate)
    out_string = fd.clean(df, preserve_columns="mynotes", outlier_action="cap", **isolate)
    assert out_tuple["mynotes"].max() == out_string["mynotes"].max() == 1000


def test_duplicate_subset_string_dedupes_like_tuple():
    df = pd.DataFrame({"id": [1, 1, 2], "note": ["a", "b", "c"]})
    out_tuple = fd.clean(df, duplicate_subset=("id",), verbose=False)
    out_string = fd.clean(df, duplicate_subset="id", verbose=False)
    assert len(out_tuple) == len(out_string) == 2
