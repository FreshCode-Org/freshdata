import pandas as pd
import pytest

import freshdata as fd


def test_clean_returns_new_frame_and_never_mutates_input(messy):
    snapshot = messy.copy(deep=True)
    out = fd.clean(messy)
    assert out is not messy
    pd.testing.assert_frame_equal(messy, snapshot)


def test_clean_report_tuple(messy):
    out, report = fd.clean(messy, report=True)
    assert isinstance(out, pd.DataFrame)
    assert isinstance(report, fd.CleanReport)
    assert len(report) > 0


def test_clean_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        fd.clean([1, 2, 3])


def test_clean_rejects_series_with_helpful_message():
    with pytest.raises(TypeError, match="to_frame"):
        fd.clean(pd.Series([1, 2, 3]))


def test_unknown_option_raises_with_suggestion():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(TypeError, match="drop_duplicates"):
        fd.clean(df, drop_duplicate=True)  # missing trailing 's'


def test_legacy_return_report_still_works(messy):
    out, report = fd.clean(messy, return_report=True)
    assert isinstance(out, pd.DataFrame)
    assert isinstance(report, fd.CleanReport)


def test_domain_kwargs_still_route_to_domain_layer():
    df = pd.DataFrame({"a": [1, None, 3]})
    with pytest.raises(Exception, match="Unknown|unknown|domain"):
        fd.clean(df, domain="unknown_xyz")


def test_return_report_inside_mapping_config_is_rejected_as_config_field():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(TypeError, match="unknown option"):
        fd.clean(df, {"return_report": True})


def test_empty_frames_pass_through():
    no_rows = pd.DataFrame({"a": pd.Series([], dtype=object)})
    no_cols = pd.DataFrame(index=[0, 1])
    assert fd.clean(no_rows).shape[0] == 0
    assert fd.clean(no_cols).shape == (2, 0)
    assert fd.clean(pd.DataFrame()).empty


def test_already_clean_frame_is_untouched(already_clean):
    out, report = fd.clean(already_clean, report=True)
    pd.testing.assert_frame_equal(out, already_clean)
    assert not report  # falsy: nothing was changed


def test_config_object_plus_overrides(messy):
    config = fd.CleanConfig(drop_duplicates=False)
    out = fd.clean(messy, config=config)
    assert len(out) == 5  # duplicate row kept
    out2 = fd.clean(messy, config=config, drop_empty_columns=False)
    assert "empty" in out2.columns


def test_config_mapping_plus_overrides(messy):
    out, report = fd.clean(
        messy,
        {"drop_duplicates": False, "verbose": False},
        report=True,
        drop_empty_columns=False,
    )
    assert len(out) == 5
    assert "empty" in out.columns
    assert isinstance(report, fd.CleanReport)


def test_invalid_config_values_fail_fast():
    with pytest.raises(ValueError, match="impute"):
        fd.CleanConfig(impute="bogus")
    with pytest.raises(ValueError, match="numeric_threshold"):
        fd.CleanConfig(numeric_threshold=1.5)
    with pytest.raises(ValueError, match="outlier_method"):
        fd.CleanConfig(outlier_method="mad")


def test_cleaner_is_reusable_and_keeps_last_report(messy, already_clean):
    cleaner = fd.Cleaner(drop_duplicates=False)
    cleaner.clean(messy)
    first = cleaner.report_
    cleaner.clean(already_clean)
    assert cleaner.report_ is not first
    assert "drop_duplicates=False" in repr(cleaner)


def test_duplicate_column_labels_are_deduplicated(messy):
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    out = fd.clean(df)
    assert list(out.columns) == ["a", "a_2"]


def test_duplicate_column_labels_raise_when_renaming_disabled():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate column labels"):
        fd.clean(df, column_names=False)


def test_version_and_exports():
    assert fd.__version__
    for name in fd.__all__:
        assert getattr(fd, name, None) is not None


def test_legacy_top_level_helpers_remain_exported():
    for name in (
        "clean_csv",
        "clean_domain_file",
        "clean_timeseries",
        "compile_context",
        "infer_roles",
        "parse_domain",
        "validate",
    ):
        assert name in fd.__all__
        assert getattr(fd, name, None) is not None
