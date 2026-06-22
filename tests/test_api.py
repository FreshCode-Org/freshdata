import pandas as pd
import pytest

import freshdata as fd


def test_clean_returns_new_frame_and_never_mutates_input(messy):
    snapshot = messy.copy(deep=True)
    out = fd.clean(messy)
    assert out is not messy
    pd.testing.assert_frame_equal(messy, snapshot)


def test_clean_report_tuple(messy):
    out, report = fd.clean(messy, return_report=True)
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


def test_clean_csv_output_file(tmp_path):
    df = pd.DataFrame({"a": [1, None, 3]})

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "cleaned.csv"

    df.to_csv(input_path, index=False)

    fd.clean_csv(
        input_path,
        output_path=output_path,
        to_csv_kwargs={"sep": ";"},
    )

    loaded = pd.read_csv(output_path, sep=";")

    pd.testing.assert_frame_equal(loaded, pd.DataFrame({"a": [1.0, 3.0]}))


def test_clean_csv_return_report(tmp_path):
    df = pd.DataFrame({"a": [1, None, 3]})

    input_path = tmp_path / "input.csv"
    df.to_csv(input_path, index=False)

    cleaned, report = fd.clean_csv(
        input_path,
        return_report=True,
    )

    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(report, fd.CleanReport)


def test_clean_csv_read_csv_kwargs(tmp_path):
    input_path = tmp_path / "input.csv"

    input_path.write_text("a;b\n" "1;2\n" "3;4\n")

    cleaned = fd.clean_csv(
        input_path,
        read_csv_kwargs={"sep": ";"},
    )

    expected = pd.DataFrame(
        {
            "a": [1, 3],
            "b": [2, 4],
        }
    )

    pd.testing.assert_frame_equal(cleaned, expected)


def test_empty_frames_pass_through():
    no_rows = pd.DataFrame({"a": pd.Series([], dtype=object)})
    no_cols = pd.DataFrame(index=[0, 1])
    assert fd.clean(no_rows).shape[0] == 0
    assert fd.clean(no_cols).shape == (2, 0)
    assert fd.clean(pd.DataFrame()).empty


def test_already_clean_frame_is_untouched(already_clean):
    out, report = fd.clean(already_clean, return_report=True)
    pd.testing.assert_frame_equal(out, already_clean)
    assert not report  # falsy: nothing was changed


def test_config_object_plus_overrides(messy):
    config = fd.CleanConfig(drop_duplicates=False)
    out = fd.clean(messy, config=config)
    assert len(out) == 5  # duplicate row kept
    out2 = fd.clean(messy, config=config, drop_empty_columns=False)
    assert "empty" in out2.columns


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
