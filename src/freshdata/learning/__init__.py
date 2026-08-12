"""Paired-data learning: ``fd.learn`` and reusable ``.fdprofile`` profiles.

Phase 4 of the AI-native semantic architecture.  Learning is deterministic,
offline, and model-free: it recognizes FreshData's own transforms in a
messy/clean training pair and packages the evidence — rules, value maps,
examples, and an embedded :class:`~freshdata.CleaningMemory` — into an
auditable profile that replays through the exact same policy gates as every
other proposal source.

See ``ARCHITECTURE.md`` for how this package fits into the overall cleaning flow.
"""

from __future__ import annotations

from .audit import ProfileAudit
from .merge import ProfileDiff, ProfileMergeError
from .profile import LearningProfile, learn, load_profile, save_profile
from .types import (
    ExampleBank,
    ExamplePair,
    ProfileError,
    ProfileFormatError,
    ProfileManifest,
    ProfileVersionError,
    ValueMap,
    ValueMapEntry,
)

__all__ = [
    "ExampleBank",
    "ExamplePair",
    "LearningProfile",
    "ProfileAudit",
    "ProfileDiff",
    "ProfileError",
    "ProfileFormatError",
    "ProfileManifest",
    "ProfileMergeError",
    "ProfileVersionError",
    "ValueMap",
    "ValueMapEntry",
    "learn",
    "load_profile",
    "save_profile",
]
