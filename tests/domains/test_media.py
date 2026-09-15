"""Conformance tests for the media (EIDR / DDEX) domain pack."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains import UnknownDomainError
from freshdata.domains.media import (
    AmbiguousMediaTypeError,
    MediaValidator,
    eidr_check_char,
    is_valid_eidr,
    is_valid_icpn,
)


def _violated(rep, rule_id: str) -> bool:
    return any(f["rule_id"] == rule_id and f["status"] == "violated" for f in rep.domain_findings)


def _make_eidr(payload20: str) -> str:
    groups = [payload20[i:i + 4] for i in range(0, 20, 4)]
    return "10.5240/" + "-".join(groups) + "-" + eidr_check_char(payload20)


EIDR_A = _make_eidr("0123456789ABCDEF0123")
EIDR_B = _make_eidr("FEDCBA9876543210ABCD")
EIDR_SERIES = _make_eidr("11112222333344445555")
UPC = "036000291452"
EAN = "4006381333931"


@pytest.fixture
def good_content() -> pd.DataFrame:
    return pd.DataFrame({
        "eidr_id": [EIDR_A, EIDR_B],
        "title": ["A Film", "An Episode"],
        "content_type": ["Movie", "Episode"],
        "release_date": ["2020-05-01", "2019"],
        "country_of_origin": ["US", "GB"],
        "language": ["en", "fr"],
        "runtime_seconds": [7200, 1500],
        "distributor_id": ["D1", "D2"],
        "series_eidr_id": [None, EIDR_SERIES],
        "season_number": [None, 1],
        "episode_number": [None, 5],
    })


@pytest.fixture
def good_release() -> pd.DataFrame:
    return pd.DataFrame({
        "release_id": ["R1", "R2"],
        "icpn": [UPC, EAN],
        "release_type": ["Album", "Single"],
        "title": ["Album One", "Single Two"],
        "language": ["en", "es"],
        "label_name": ["Label", "Label"],
        "artist_name": ["Artist A", "Artist B"],
        "party_id": ["PA1", "PA2"],
        "party_role": ["MainArtist", "Composer"],
        "territory": ["US", "Worldwide"],
        "release_date": ["2021-03-01", "2022-07-15"],
        "track_count": [12, 2],
    })


# -- pure check-digit functions -------------------------------------------

# Published EIDR IDs: the first four are real registry IDs, the rest are the examples
# printed in the EIDR ID Format document. Their check characters use ISO 7064 hybrid
# MOD 37,36, the system the spec names ("C is the ISO 7064 Mod 37,36 check character").
PUBLISHED_EIDRS = [
    "10.5240/7791-8534-2C23-9030-8610-5",
    "10.5240/B752-5B47-DBBE-E5D4-5A3F-N",
    "10.5240/0EF3-54F9-2642-0B49-6829-R",
    "10.5240/1489-49A2-3956-4B2D-FE16-5",
    "10.5240/F85A-E100-B068-5B8F-B1C8-T",
    "10.5240/3466-F12C-391A-D60B-206B-Y",
    "10.5240/CA51-02D0-3269-23C9-DB5A-E",
    "10.5240/7481-838B-59CA-63D0-B9A8-E",
    "10.5240/CE43-9B6A-2C41-35C3-42CA-V",
    "10.5240/823E-5DE9-0816-7BB5-A37F-X",
]


@pytest.mark.parametrize("eidr", PUBLISHED_EIDRS)
def test_eidr_published_ids_are_valid(eidr):
    assert is_valid_eidr(eidr)
    payload = eidr[len("10.5240/"):].replace("-", "")[:-1]
    assert eidr_check_char(payload) == eidr[-1]


def test_eidr_check_char_matches_published_example():
    # EIDR's canonical published identifier and its check character '5'.
    assert eidr_check_char("779185342C2390308610") == "5"
    assert eidr_check_char("F85AE100B0685B8FB1C8") == "T"


def test_eidr_old_mod_37_2_check_chars_are_rejected():
    # Regression (#259): the pure MOD 37-2 system gave '7' here and could emit '*';
    # neither is a valid EIDR check character.
    assert not is_valid_eidr("10.5240/7791-8534-2C23-9030-8004-7")
    assert eidr_check_char("779185342C2390308004") == "C"
    assert not is_valid_eidr("10.5240/7791-8534-2C23-9030-8610-*")


def test_eidr_check_char_is_alphanumeric():
    for payload in ("00000000000000000000", "ZZZZZZZZZZZZZZZZZZZZ", "0123456789ABCDEFGHIJ"):
        char = eidr_check_char(payload)
        assert len(char) == 1 and char.isalnum() and char == char.upper()


def test_eidr_roundtrip_and_tamper():
    eidr = _make_eidr("ABCDEF0123456789ABCD")
    assert is_valid_eidr(eidr)
    wrong_check = eidr[:-1] + ("0" if eidr[-1] != "0" else "1")
    assert not is_valid_eidr(wrong_check)


@pytest.mark.parametrize("value", [
    "10.5240/7791-8534-2C23-9030-8004",        # missing check group
    "10.5240/7791-8534-2C23-9030-8004-77",     # check group too long
    "10.5241/7791-8534-2C23-9030-8004-7",      # wrong DOI prefix
    "7791-8534-2C23-9030-8004-7",              # no prefix
    "not-an-eidr",
    None,
])
def test_eidr_rejects_malformed(value):
    assert not is_valid_eidr(value)


def test_icpn_known_values():
    assert is_valid_icpn(UPC)            # 12-digit UPC-A
    assert is_valid_icpn(EAN)            # 13-digit EAN-13
    assert is_valid_icpn("0-36000-29145-2")    # hyphens stripped first
    assert not is_valid_icpn("036000291459")   # bad check digit
    assert not is_valid_icpn("12345")          # wrong length
    assert not is_valid_icpn(None)


# -- happy paths -----------------------------------------------------------

def test_content_happy_path(good_content):
    out, rep = fd.clean(good_content, domain="media", media_type="content",
                        return_report=True, verbose=False)
    assert rep.domain == "media"
    assert rep.domain_trust_score >= 0.95
    assert not [f for f in rep.domain_findings
                if f["status"] == "violated" and f["severity"] == "error"]
    assert out.shape[0] == 2


def test_release_happy_path(good_release):
    out, rep = fd.clean(good_release, domain="media", media_type="release",
                        return_report=True, verbose=False)
    assert rep.domain_trust_score >= 0.95
    assert not [f for f in rep.domain_findings
                if f["status"] == "violated" and f["severity"] == "error"]
    assert out.shape[0] == 2


# -- required-field violations --------------------------------------------

def test_content_required_field_missing(good_content):
    validator = MediaValidator(media_type="content")
    actual = validator.detect_columns(good_content).actual("eidr_id")
    report = validator.validate(good_content.drop(columns=[actual]))
    assert "eidr_id" in report.mapping.unmapped_required
    assert any("MISSING_REQUIRED_FIELD" in r.message and r.violated for r in report.results)


def test_release_required_field_missing(good_release):
    validator = MediaValidator(media_type="release")
    actual = validator.detect_columns(good_release).actual("release_id")
    report = validator.validate(good_release.drop(columns=[actual]))
    assert "release_id" in report.mapping.unmapped_required
    assert any("MISSING_REQUIRED_FIELD" in r.message and r.violated for r in report.results)


# -- format violations -----------------------------------------------------

def test_content_format_violations(good_content):
    df = good_content.copy()
    df.loc[0, "eidr_id"] = "10.5240/0000-0000-0000-0000-0000-0"  # MD-C002: bad check char
    df.loc[1, "release_date"] = "May 2019"                       # MD-C004
    df["runtime_seconds"] = df["runtime_seconds"].astype(object)
    df.loc[0, "runtime_seconds"] = 12.5                          # MD-C007: not integer
    _, rep = fd.clean(df, domain="media", media_type="content",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-C002")
    assert _violated(rep, "MD-C004")
    assert _violated(rep, "MD-C007")


def test_release_format_violations(good_release):
    df = good_release.copy()
    df.loc[0, "icpn"] = "036000291459"     # MD-R002: bad check digit
    df.loc[1, "release_date"] = "soon"     # MD-R007
    df["track_count"] = df["track_count"].astype(object)
    df.loc[0, "track_count"] = 4.5         # MD-R008: not integer
    _, rep = fd.clean(df, domain="media", media_type="release",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-R002")
    assert _violated(rep, "MD-R007")
    assert _violated(rep, "MD-R008")


# -- reference violations --------------------------------------------------

def test_content_reference_violations(good_content):
    df = good_content.copy()
    df.loc[0, "content_type"] = "Hologram"   # MD-C003
    df.loc[1, "country_of_origin"] = "ZZ"    # MD-C005
    df.loc[0, "language"] = "zz"             # MD-C006
    _, rep = fd.clean(df, domain="media", media_type="content",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-C003")
    assert _violated(rep, "MD-C005")
    assert _violated(rep, "MD-C006")


def test_release_reference_violations(good_release):
    df = good_release.copy()
    df.loc[0, "release_type"] = "Hologram"   # MD-R003
    df.loc[1, "language"] = "zz"             # MD-R004
    df.loc[0, "party_role"] = "Sorcerer"     # MD-R005
    df.loc[1, "territory"] = "Mars"          # MD-R006
    _, rep = fd.clean(df, domain="media", media_type="release",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-R003")
    assert _violated(rep, "MD-R004")
    assert _violated(rep, "MD-R005")
    assert _violated(rep, "MD-R006")


# -- business / cross-field violations ------------------------------------

def test_content_business_violations(good_content):
    df = good_content.copy()
    df.loc[1, "series_eidr_id"] = None    # MD-C009: Episode without series
    _, rep = fd.clean(df, domain="media", media_type="content",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-C009")


def test_content_bad_series_eidr(good_content):
    df = good_content.copy()
    df.loc[1, "series_eidr_id"] = "10.5240/0000-0000-0000-0000-0000-0"   # MD-C010
    _, rep = fd.clean(df, domain="media", media_type="content",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-C010")


def test_content_season_episode_rules(good_content):
    df = good_content.copy()
    df.loc[1, "season_number"] = 0      # MD-C011: not positive
    _, rep = fd.clean(df, domain="media", media_type="content",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-C011")

    df2 = good_content.copy()
    df2.loc[0, "episode_number"] = 3    # episode present, season absent -> MD-C012
    _, rep2 = fd.clean(df2, domain="media", media_type="content",
                       return_report=True, verbose=False)
    assert _violated(rep2, "MD-C012")


def test_content_movie_with_episode_is_warning(good_content):
    df = good_content.copy()
    df.loc[0, "episode_number"] = 2   # a Movie should have no episode number
    _, rep = fd.clean(df, domain="media", media_type="content",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-C013")
    md013 = next(f for f in rep.domain_findings if f["rule_id"] == "MD-C013")
    assert md013["severity"] == "warning"


def test_release_business_violations(good_release):
    df = good_release.copy()
    df.loc[0, "party_id"] = None     # MD-R010: party_role without party_id
    _, rep = fd.clean(df, domain="media", media_type="release",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-R010")


def test_single_track_count_is_warning(good_release):
    df = good_release.copy()
    df.loc[1, "track_count"] = 9     # a Single with 9 tracks -> MD-R011 warning
    _, rep = fd.clean(df, domain="media", media_type="release",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-R011")


# -- semantic plausibility -------------------------------------------------

def test_content_runtime_too_long_is_warning(good_content):
    df = good_content.copy()
    df.loc[0, "runtime_seconds"] = 90000   # > 24h -> MD-C008
    _, rep = fd.clean(df, domain="media", media_type="content",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-C008")


def test_release_track_count_implausible_is_warning(good_release):
    df = good_release.copy()
    df.loc[0, "release_type"] = "Album"
    df.loc[0, "track_count"] = 600   # > 500 -> MD-R009
    _, rep = fd.clean(df, domain="media", media_type="release",
                      return_report=True, verbose=False)
    assert _violated(rep, "MD-R009")


# -- repair audit / ID safety ---------------------------------------------

def test_repairs_are_flag_only(good_content):
    df = good_content.copy()
    df.loc[0, "content_type"] = "Hologram"   # MD-C003 -> flagged, not repaired
    _, rep = fd.clean(df, domain="media", media_type="content",
                      return_report=True, verbose=False)
    flagged = [a for a in rep.domain_repairs if a["rule_id"] == "MD-C003"]
    assert flagged and all(a["status"] == "flagged" for a in flagged)
    assert all(a["status"] != "applied" for a in rep.domain_repairs)


def test_id_safety_null_eidr_never_filled(good_content):
    df = good_content.copy()
    df.loc[0, "eidr_id"] = None
    out, rep = fd.clean(df, domain="media", media_type="content",
                        return_report=True, verbose=False)
    assert pd.isna(out.loc[0, "eidr_id"])
    assert _violated(rep, "MD-C001")
    assert not [a for a in rep.domain_repairs
                if a["column"] == "eidr_id" and a["status"] == "applied"]


def test_id_safety_null_release_id_never_filled(good_release):
    df = good_release.copy()
    df.loc[1, "release_id"] = None
    out, rep = fd.clean(df, domain="media", media_type="release",
                        return_report=True, verbose=False)
    assert pd.isna(out.loc[1, "release_id"])
    assert _violated(rep, "MD-R001")


# -- sub-schema routing ----------------------------------------------------

def test_autodetect_content(good_content):
    validator = MediaValidator()
    validator.validate(good_content)
    assert validator.media_type == "content"


def test_autodetect_release_end_to_end(good_release):
    out, rep = fd.clean(good_release, domain="media", return_report=True, verbose=False)
    assert rep.domain_trust_score >= 0.95


def test_ambiguous_media_type_raises():
    df = pd.DataFrame({"title": ["x"], "language": ["en"], "release_date": ["2020-01-01"]})
    with pytest.raises(AmbiguousMediaTypeError):
        MediaValidator().validate(df)


# -- regression guards -----------------------------------------------------

def test_validation_never_mutates_input(good_release):
    before = good_release.copy()
    MediaValidator(media_type="release").validate(good_release)
    pd.testing.assert_frame_equal(good_release, before)


def test_unknown_domain_lists_available(good_release):
    with pytest.raises(UnknownDomainError) as exc:
        fd.clean(good_release, domain="unknown_xyz")
    assert "media" in exc.value.available


def test_standalone_import():
    assert MediaValidator(media_type="content").domain_name == "media"
