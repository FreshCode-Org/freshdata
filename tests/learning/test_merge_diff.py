"""Profile diff and merge: strategies, conflict handling, privacy strictness."""

from __future__ import annotations

import pytest

from freshdata.learning import learn, load_profile, save_profile
from freshdata.learning.merge import ProfileMergeError, diff_profiles, merge_profiles


def _status_map(profile):
    return {e.raw_value: e.clean_value for e in profile.value_maps["status"].entries}


@pytest.fixture()
def base_profile(orders_profile):
    return orders_profile


@pytest.fixture(scope="session")
def conflicting_profile(orders_pair):
    """Same corruption but 'Deliverd' now maps to 'returned' instead."""
    messy, clean = orders_pair
    clean = clean.copy()
    clean["status"] = ["returned" if s == "delivered" else s for s in clean["status"]]
    return learn(messy, clean, key="order_id", dataset_id="orders-v2", min_support=2)


@pytest.fixture(scope="session")
def raw_values_profile(orders_pair):
    messy, clean = orders_pair
    return learn(
        messy,
        clean,
        key="order_id",
        dataset_id="orders-raw",
        min_support=2,
        privacy="none",
        include_sensitive=True,
    )


class TestDiff:
    def test_self_diff_is_empty(self, base_profile):
        diff = diff_profiles(base_profile, base_profile)
        assert diff.is_empty

    def test_conflicting_value_maps_detected(self, base_profile, conflicting_profile):
        diff = diff_profiles(base_profile, conflicting_profile)
        assert not diff.is_empty
        assert any("status" in c for c in diff.conflicting_value_maps)

    def test_render_mentions_conflicts(self, base_profile, conflicting_profile):
        text = str(diff_profiles(base_profile, conflicting_profile))
        assert "status" in text

    def test_profile_diff_method(self, base_profile, conflicting_profile):
        diff = base_profile.diff(conflicting_profile)
        assert any("status" in c for c in diff.conflicting_value_maps)


class TestMergeStrategies:
    def test_union_drops_conflicts_keeps_rest(self, base_profile, conflicting_profile):
        merged = merge_profiles(base_profile, conflicting_profile, strategy="union_min_precision")
        status = _status_map(merged)
        # The conflicting raw value is dropped entirely...
        assert "Deliverd" not in status
        # ...while agreeing entries survive.
        assert status.get("SHIPPED") == "shipped"

    def test_prefer_self_keeps_self_mapping(self, base_profile, conflicting_profile):
        merged = merge_profiles(base_profile, conflicting_profile, strategy="prefer_self")
        assert _status_map(merged)["Deliverd"] == "delivered"

    def test_prefer_other_keeps_other_mapping(self, base_profile, conflicting_profile):
        merged = merge_profiles(base_profile, conflicting_profile, strategy="prefer_other")
        assert _status_map(merged)["Deliverd"] == "returned"

    def test_error_on_conflict_raises(self, base_profile, conflicting_profile):
        with pytest.raises(ProfileMergeError, match="status"):
            merge_profiles(base_profile, conflicting_profile, strategy="error_on_conflict")

    def test_error_on_conflict_ok_without_conflicts(self, base_profile):
        merged = merge_profiles(base_profile, base_profile, strategy="error_on_conflict")
        assert _status_map(merged) == _status_map(base_profile)

    def test_merge_method_delegates(self, base_profile, conflicting_profile):
        merged = base_profile.merge(conflicting_profile, strategy="prefer_self")
        assert _status_map(merged)["Deliverd"] == "delivered"

    def test_unknown_strategy_rejected(self, base_profile):
        with pytest.raises((ProfileMergeError, ValueError)):
            merge_profiles(base_profile, base_profile, strategy="bogus")


class TestMergedProfileIntegrity:
    def test_merged_id_differs_and_roundtrips(self, base_profile, conflicting_profile, tmp_path):
        merged = merge_profiles(base_profile, conflicting_profile, strategy="union_min_precision")
        assert merged.profile_id != base_profile.profile_id
        assert merged.profile_id != conflicting_profile.profile_id
        path = tmp_path / "merged.fdprofile"
        save_profile(merged, path)
        loaded = load_profile(path)
        assert loaded.profile_id == merged.profile_id
        assert _status_map(loaded) == _status_map(merged)

    def test_privacy_strictest_wins_in_union(self, base_profile, raw_values_profile):
        assert raw_values_profile.manifest.privacy_mode == "none"
        merged = merge_profiles(base_profile, raw_values_profile, strategy="union_min_precision")
        assert merged.manifest.privacy_mode == "mask"

    def test_prefer_other_takes_other_privacy(self, base_profile, raw_values_profile):
        merged = merge_profiles(base_profile, raw_values_profile, strategy="prefer_other")
        assert merged.manifest.privacy_mode == "none"
        assert merged.manifest.contains_raw_values is True
