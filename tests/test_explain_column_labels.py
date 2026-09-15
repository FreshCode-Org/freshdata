"""explain_clean / infer_roles with non-string and duplicate column labels.

Regressions for #232 (parts 3 and 6) and #265 (part 3).
"""

from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

import freshdata as fd
from freshdata.explain import ExplainReport, _cell_changes


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _int_label_frame() -> pd.DataFrame:
    # Integer labels, as produced by ``pd.read_csv(header=None)``.
    return pd.DataFrame({0: [" a", "b ", "c", "d"], 1: [1.0, 2.0, 3.0, 4.0]})


def _multiindex_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [[1, " a", None], [2, "b", 3.0]],
        columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "z")]),
    )


def _mixed_label_frame() -> pd.DataFrame:
    return pd.DataFrame({0: [1.0, None, 3.0, 4.0], "name": ["a", "b", None, "d"]})


# -- #232 part 3: integer / tuple labels -------------------------------------


def test_explain_integer_labels_key_every_mapping_by_string():
    rep = fd.explain_clean(_int_label_frame(), verbose=False)
    for mapping in (rep.before_stats, rep.after_stats, rep.cell_changes):
        assert set(mapping) == {"0", "1"}
    assert rep.cell_changes["0"] == 2


def test_explain_integer_labels_to_frame_reports_changed_cells():
    frame = fd.explain_clean(_int_label_frame(), verbose=False).to_frame()
    row = frame.set_index("column").loc["0"]
    assert row["changed_cells"] == 2
    assert row["before_dtype"] == "object"
    assert row["after_dtype"] == "object"


def test_explain_integer_labels_html_shows_dtypes():
    html = fd.explain_clean(_int_label_frame(), verbose=False).to_html()
    assert "<td>object</td>" in html
    assert "<td>float64</td>" in html


def test_cell_changes_integer_labels_use_string_keys():
    before = pd.DataFrame({0: [1, 2], 1: ["a", "b"]})
    after = pd.DataFrame({0: [1, 9], 1: ["a", "b"], 2: [0, 0]})
    assert _cell_changes(before, after) == {"0": 1, "1": 0, "2": 2}


def test_explain_integer_labels_narratives_match_actions():
    df = pd.DataFrame({0: [1.0, None, 3.0, 4.0, None, 6.0], "b": [1, 2, 3, 4, 5, 6]})
    rep = fd.explain_clean(df, verbose=False)
    assert any(line.startswith("`0`:") for line in rep.narratives), rep.narratives


def test_explain_multiindex_labels_render_and_serialize():
    rep = fd.explain_clean(_multiindex_frame(), verbose=False)
    assert "('a', 'y')" in rep.cell_changes
    json.dumps(rep.to_dict())
    html = rep._repr_html_()
    assert html is not None
    assert html == rep.to_html()


def test_explain_to_dict_stringifies_non_string_keys():
    base = fd.explain_clean(pd.DataFrame({"a": [1, 2]}), verbose=False)
    rep = ExplainReport(
        strategy=base.strategy,
        rows_before=2,
        rows_after=2,
        cols_before=1,
        cols_after=1,
        before_stats={("a", "x"): {"dtype": "int64"}},
        after_stats={("a", "x"): {"dtype": "int64"}},
        cell_changes={("a", "x"): 1},
        actions_by_step={},
        narratives=[],
        report=base.report,
        roles=base.roles,
    )
    payload = json.loads(json.dumps(rep.to_dict()))
    assert payload["cell_changes"] == {"('a', 'x')": 1}
    assert rep.to_frame()["changed_cells"].tolist() == [1]


# -- #232 part 6: mixed int/str labels ---------------------------------------


def test_explain_clean_mixed_labels():
    rep = fd.explain_clean(_mixed_label_frame(), verbose=False)
    assert rep.to_frame()["column"].tolist() == ["0", "name"]
    assert rep.to_html()


def test_infer_roles_mixed_labels_keep_original_labels():
    df = _mixed_label_frame()
    roles = fd.infer_roles(df)
    assert roles["column"].tolist() == [0, "name"]
    for label in roles["column"]:
        assert df[label] is not None  # labels round-trip into the frame


def test_infer_roles_integer_label_semantic_hint_by_string_key():
    df = pd.DataFrame({0: ["a@b.com", "c@d.com"], 1: [1, 2]})
    roles = fd.infer_roles(df, semantic_context={"columns": {"0": {"semantic_type": "email"}}})
    assert roles.set_index("column").loc[0, "semantic_type"] == "email"


def test_infer_roles_multiindex_labels():
    roles = fd.infer_roles(_multiindex_frame())
    assert roles["column"].tolist() == [("a", "x"), ("a", "y"), ("b", "z")]


# -- #265 part 3: duplicate labels -------------------------------------------


def test_explain_clean_rejects_duplicate_labels():
    df = pd.DataFrame([[1.0, 2.0], [None, 4.0], [3.0, 5.0]], columns=["a", "a"])
    with pytest.raises(
        ValueError, match=r"explain_clean requires unique column labels; duplicated: \['a'\]"
    ):
        fd.explain_clean(df, verbose=False)


def test_infer_roles_rejects_duplicate_labels():
    df = pd.DataFrame([[1.0, 2.0], [None, 4.0], [3.0, 5.0]], columns=["a", "a"])
    with pytest.raises(
        ValueError, match=r"infer_roles requires unique column labels; duplicated: \['a'\]"
    ):
        fd.infer_roles(df)


def test_explain_clean_rejects_labels_with_same_string_form():
    df = pd.DataFrame({1: [1.0, None], "1": ["a", None]})
    with pytest.raises(ValueError, match=r"distinct string forms; colliding: \[\[1, '1'\]\]"):
        fd.explain_clean(df, verbose=False)


def test_infer_roles_accepts_labels_with_same_string_form():
    roles = fd.infer_roles(pd.DataFrame({1: [1.0, None], "1": ["a", None]}))
    assert sorted(map(repr, roles["column"])) == ["'1'", "1"]
