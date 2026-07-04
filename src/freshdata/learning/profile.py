"""`LearningProfile`, ``fd.learn`` and the ``.fdprofile`` archive format.

A ``.fdprofile`` is a plain zip archive with human-auditable JSON members::

    manifest.json          integrity header (hashes of every other member)
    rules.json             learned ColumnConstraints (Phase-1 vocabulary)
    value_maps.json        literal raw -> clean repairs with support/precision
    memory.json            embedded CleaningMemory (Phase-2 replay semantics)
    examples.json          example bank (evidence only, never replayed)
    examples_vectors.npz   optional float16 vectors for unexplained examples
    audit.json             demotions, notes, holdout metrics, provenance

``load_profile`` verifies every member hash, fails clearly on corrupt or
missing members and unsupported future versions, and ignores (with a
warning) unknown members written by future minor versions.  Full source rows
are never stored, and under the default ``privacy="mask"`` no raw sensitive
literal is stored anywhere in the archive.
"""

from __future__ import annotations

import io
import json
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ..context.types import ColumnConstraint, ContextPolicy, sha256_text
from ..memory import CleaningMemory
from ..repairplan import compute_frame_signature
from .audit import ProfileAudit, build_audit
from .types import (
    PROFILE_FORMAT_VERSION,
    ExampleBank,
    ProfileFormatError,
    ProfileManifest,
    ProfileVersionError,
    ValueMap,
)

__all__ = ["LearningProfile", "learn", "load_profile", "save_profile"]

_REQUIRED_MEMBERS = (
    "manifest.json",
    "rules.json",
    "value_maps.json",
    "memory.json",
    "examples.json",
    "audit.json",
)
_VECTORS_MEMBER = "examples_vectors.npz"

#: Fixed zip timestamp so identical learning inputs produce identical bytes.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _schema_hash(df: pd.DataFrame) -> str:
    schema = {str(c): str(df[c].dtype) for c in df.columns}
    return sha256_text(json.dumps(schema, sort_keys=True))


def _freshdata_version() -> str:
    try:
        from .. import __version__  # noqa: PLC0415 - avoid import at module load

        return str(__version__)
    except ImportError:  # pragma: no cover - defensive
        return "unknown"


@dataclass
class LearningProfile:
    """Reusable cleaning knowledge learned from one paired dataset.

    Note: the architecture spec asks for ``slots=True``; the repo supports
    Python 3.9, so the established no-slots convention is followed.
    """

    manifest: ProfileManifest
    rules: list[ColumnConstraint]
    value_maps: dict[str, ValueMap]
    examples: ExampleBank | None
    memory: CleaningMemory | None
    audit_info: ProfileAudit | None = None
    vectors: Any | None = field(default=None, repr=False)  # np.ndarray | None

    # -- identity ----------------------------------------------------------

    @property
    def profile_id(self) -> str:
        """Content-derived id: stable for identical learned artifacts."""
        payload = {
            "rules": [r.to_dict() for r in self.rules],
            "value_maps": {c: m.to_dict() for c, m in sorted(self.value_maps.items())},
            "examples": self.examples.to_dict() if self.examples is not None else None,
        }
        return "fdp-" + sha256_text(_canonical_json(payload).decode("utf-8"))[:12]

    @property
    def dataset_id(self) -> str:
        """The dataset the profile was learned from (mirrors CleaningMemory)."""
        return self.manifest.dataset_id or self.profile_id

    # -- persistence ---------------------------------------------------------

    def _members(self) -> dict[str, bytes]:
        members: dict[str, bytes] = {
            "rules.json": _canonical_json({"rules": [r.to_dict() for r in self.rules]}),
            "value_maps.json": _canonical_json(
                {"value_maps": {c: m.to_dict() for c, m in sorted(self.value_maps.items())}}
            ),
            "memory.json": _canonical_json(
                {"memory": self.memory.to_dict() if self.memory is not None else None}
            ),
            "examples.json": _canonical_json(
                {"examples": self.examples.to_dict() if self.examples is not None else None}
            ),
            "audit.json": _canonical_json(
                {"audit": self.audit_info.to_dict() if self.audit_info is not None else None}
            ),
        }
        if self.vectors is not None:
            import numpy as np  # noqa: PLC0415 - only when vectors exist

            buffer = io.BytesIO()
            np.savez_compressed(buffer, vectors=self.vectors.astype(np.float16))
            members[_VECTORS_MEMBER] = buffer.getvalue()
        return members

    def save(self, path: str | Path) -> None:
        """Write the profile as a ``.fdprofile`` zip archive."""
        members = self._members()
        member_hashes = {
            name: sha256_text(payload.decode("utf-8"))
            if name.endswith(".json")
            else _sha256_bytes(payload)
            for name, payload in members.items()
        }
        manifest = ProfileManifest(
            profile_version=self.manifest.profile_version,
            freshdata_version=self.manifest.freshdata_version,
            created_at=self.manifest.created_at,
            dataset_id=self.manifest.dataset_id,
            dataset_signature=self.manifest.dataset_signature,
            source_schema_hash=self.manifest.source_schema_hash,
            clean_schema_hash=self.manifest.clean_schema_hash,
            context_hash=self.manifest.context_hash,
            privacy_mode=self.manifest.privacy_mode,
            contains_raw_values=self.manifest.contains_raw_values,
            compartments=tuple(sorted(members)),
            member_hashes=member_hashes,
        )
        self.manifest = manifest
        manifest_bytes = _canonical_json(manifest.to_dict())

        target = Path(path)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in [("manifest.json", manifest_bytes), *sorted(members.items())]:
                info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload)

    @classmethod
    def load(cls, path: str | Path) -> LearningProfile:
        """Load and verify a ``.fdprofile`` archive."""
        target = Path(path)
        if not target.exists():
            raise ProfileFormatError(f"profile not found: {target}")
        try:
            with zipfile.ZipFile(target) as archive:
                names = set(archive.namelist())
                raw_members = {name: archive.read(name) for name in names}
        except zipfile.BadZipFile as exc:
            raise ProfileFormatError(f"{target} is not a valid .fdprofile archive: {exc}") from exc

        if "manifest.json" not in raw_members:
            raise ProfileFormatError(f"{target}: missing required member manifest.json")
        manifest = ProfileManifest.from_dict(
            json.loads(raw_members["manifest.json"].decode("utf-8"))
        )
        _check_version(manifest.profile_version, target)

        missing = [m for m in _REQUIRED_MEMBERS if m not in raw_members]
        if missing:
            raise ProfileFormatError(f"{target}: missing required member(s): {', '.join(missing)}")

        for name, expected in manifest.member_hashes.items():
            if name not in raw_members:
                raise ProfileFormatError(f"{target}: member {name} listed in manifest but absent")
            actual = (
                sha256_text(raw_members[name].decode("utf-8"))
                if name.endswith(".json")
                else _sha256_bytes(raw_members[name])
            )
            if actual != expected:
                raise ProfileFormatError(
                    f"{target}: hash mismatch for {name} (profile corrupt or tampered)"
                )

        known = set(_REQUIRED_MEMBERS) | {_VECTORS_MEMBER}
        for name in sorted(names - known):
            warnings.warn(
                f"{target}: ignoring unknown profile member {name!r} "
                "(written by a newer freshdata?)",
                UserWarning,
                stacklevel=2,
            )

        rules_payload = json.loads(raw_members["rules.json"].decode("utf-8"))
        maps_payload = json.loads(raw_members["value_maps.json"].decode("utf-8"))
        memory_payload = json.loads(raw_members["memory.json"].decode("utf-8"))
        examples_payload = json.loads(raw_members["examples.json"].decode("utf-8"))
        audit_payload = json.loads(raw_members["audit.json"].decode("utf-8"))

        vectors = None
        if _VECTORS_MEMBER in raw_members:
            try:
                import numpy as np  # noqa: PLC0415

                with np.load(io.BytesIO(raw_members[_VECTORS_MEMBER])) as data:
                    vectors = data["vectors"]
            except Exception:  # noqa: BLE001 - vectors are optional
                warnings.warn(
                    f"{target}: could not read {_VECTORS_MEMBER}; "
                    "example retrieval disabled for this profile",
                    UserWarning,
                    stacklevel=2,
                )

        memory_data = memory_payload.get("memory")
        examples_data = examples_payload.get("examples")
        audit_data = audit_payload.get("audit")
        return cls(
            manifest=manifest,
            rules=[ColumnConstraint.from_dict(r) for r in rules_payload.get("rules", [])],
            value_maps={
                column: ValueMap.from_dict(vm)
                for column, vm in maps_payload.get("value_maps", {}).items()
            },
            examples=ExampleBank.from_dict(examples_data) if examples_data else None,
            memory=CleaningMemory.from_dict(memory_data) if memory_data else None,
            audit_info=ProfileAudit.from_dict(audit_data) if audit_data else None,
            vectors=vectors,
        )

    # -- introspection -------------------------------------------------------

    def summary(self) -> str:
        entries = sum(len(m.entries) for m in self.value_maps.values())
        examples = len(self.examples.examples) if self.examples is not None else 0
        pieces = [
            f"LearningProfile {self.profile_id}",
            f"{len(self.rules)} rule(s)",
            f"{len(self.value_maps)} value map(s) / {entries} entrie(s)",
            f"{examples} example(s)",
            "memory embedded" if self.memory is not None else "no memory",
            f"privacy={self.manifest.privacy_mode}",
        ]
        if self.manifest.contains_raw_values:
            pieces.append("CONTAINS RAW SENSITIVE VALUES")
        return " | ".join(pieces)

    def audit(self) -> ProfileAudit:
        if self.audit_info is not None:
            return self.audit_info
        self.audit_info = build_audit(self)
        return self.audit_info

    def diff(self, other: LearningProfile) -> Any:
        from .merge import diff_profiles  # noqa: PLC0415 - avoid cycle

        return diff_profiles(self, other)

    def merge(
        self,
        other: LearningProfile,
        *,
        strategy: Literal[
            "union_min_precision", "prefer_self", "prefer_other", "error_on_conflict"
        ] = "union_min_precision",
    ) -> LearningProfile:
        from .merge import merge_profiles  # noqa: PLC0415 - avoid cycle

        return merge_profiles(self, other, strategy=strategy)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def _sha256_bytes(payload: bytes) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(payload).hexdigest()


def _check_version(version: str, target: Path) -> None:
    try:
        major = int(str(version).split(".", 1)[0])
    except ValueError as exc:
        raise ProfileFormatError(f"{target}: invalid profile_version {version!r}") from exc
    supported = int(PROFILE_FORMAT_VERSION.split(".", 1)[0])
    if major > supported:
        raise ProfileVersionError(
            f"{target}: profile_version {version} is newer than this freshdata "
            f"supports (<= {supported}.x); upgrade freshdata to load it"
        )


# ---------------------------------------------------------------------------
# fd.learn
# ---------------------------------------------------------------------------


def learn(
    messy_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    *,
    context: str | None = None,
    policy: ContextPolicy | None = None,
    key: str | Sequence[str] | None = None,
    dataset_id: str | None = None,
    privacy: Literal["mask", "none"] = "mask",
    include_sensitive: bool = False,
    min_support: int = 5,
    min_precision: float = 0.98,
    holdout_fraction: float = 0.2,
    random_state: int = 0,
    max_map_size: int = 2_000,
) -> LearningProfile:
    """Learn a reusable cleaning profile from a paired messy/clean dataset.

    The pipeline is deterministic and fully offline: align -> diff ->
    classify (against FreshData's own transforms) -> extract -> holdout
    validate.  Nothing is trained; low-support or low-precision patterns are
    demoted to examples, imputation never becomes a literal map, and raw
    sensitive literals are masked unless ``privacy="none"`` **and**
    ``include_sensitive=True``.
    """
    from .align import align_pair  # noqa: PLC0415 - keep import cost off fd import
    from .fit_eval import validate_and_demote  # noqa: PLC0415
    from .privacy import derive_salt, detect_sensitive_columns  # noqa: PLC0415

    if not isinstance(messy_df, pd.DataFrame) or not isinstance(clean_df, pd.DataFrame):
        raise TypeError("fd.learn requires pandas DataFrames for messy_df and clean_df")
    if messy_df.empty or clean_df.empty:
        raise ValueError("fd.learn requires non-empty messy_df and clean_df")
    if privacy not in {"mask", "none"}:
        raise ValueError(f"privacy must be 'mask' or 'none', got {privacy!r}")

    if context is not None and policy is None:
        from ..api import compile_context  # noqa: PLC0415 - avoid import cycle

        policy = compile_context(context, df=messy_df)

    protected = _policy_protected_columns(policy)
    keys = (key,) if isinstance(key, str) else tuple(key or ())
    learn_protected = protected | frozenset(keys)

    signature = compute_frame_signature(messy_df)
    dataset_signature = sha256_text(json.dumps(signature.to_dict(), sort_keys=True))
    salt = derive_salt(dataset_signature)
    sensitive = detect_sensitive_columns(messy_df)

    aligned = align_pair(messy_df, clean_df, key=key)
    resolved_dataset_id = dataset_id or f"learned-{dataset_signature[:10]}"

    final, demotions, holdout_metrics = validate_and_demote(
        aligned,
        extract_kwargs={
            "dataset_id": resolved_dataset_id,
            "sensitive": sensitive,
            "salt": salt,
            "privacy": privacy,
            "include_sensitive": include_sensitive,
            "protected": learn_protected,
            "min_support": min_support,
            "min_precision": min_precision,
            "max_map_size": max_map_size,
        },
        min_precision=min_precision,
        holdout_fraction=holdout_fraction,
        random_state=random_state,
    )

    bank, vectors = _build_example_bank(final.examples, privacy=privacy)

    contains_raw = (
        privacy == "none" and include_sensitive and _stores_sensitive_literals(final, sensitive)
    )
    if final.config_deltas:
        # Persist config deltas as dataset-level advisory rules so the profile
        # stays self-contained (rules.json is the single source of truth).
        final.rules.extend(_config_delta_rules(final.config_deltas))

    manifest = ProfileManifest(
        profile_version=PROFILE_FORMAT_VERSION,
        freshdata_version=_freshdata_version(),
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_id=resolved_dataset_id,
        dataset_signature=dataset_signature,
        source_schema_hash=_schema_hash(messy_df),
        clean_schema_hash=_schema_hash(clean_df),
        context_hash=sha256_text(context) if context is not None else None,
        privacy_mode=privacy,
        contains_raw_values=contains_raw,
        compartments=(),
        member_hashes={},
    )
    profile = LearningProfile(
        manifest=manifest,
        rules=final.rules,
        value_maps=final.value_maps,
        examples=bank,
        memory=final.memory,
        vectors=vectors,
    )
    profile.audit_info = build_audit(
        profile,
        sensitive_columns={c: t for c, t in sensitive.items() if c in set(messy_df.columns)},
        protection_candidates=final.protection_candidates,
        alignment={
            **aligned.alignment_report.to_dict(),
            "source_schema": {str(c): str(messy_df[c].dtype) for c in messy_df.columns},
        },
        holdout_metrics=holdout_metrics.to_dict(),
        demotions=demotions,
        notes=final.notes,
    )
    return profile


def _policy_protected_columns(policy: ContextPolicy | None) -> frozenset[str]:
    if policy is None:
        return frozenset()
    return frozenset(
        c.column for c in policy.constraints if c.column is not None and c.rule == "protected"
    )


def _stores_sensitive_literals(final: Any, sensitive: Mapping[str, str]) -> bool:
    for column, value_map in final.value_maps.items():
        if column in sensitive and any(not e.masked for e in value_map.entries):
            return True
    return any(e.column in sensitive and not e.masked for e in final.examples)


def _config_delta_rules(config_deltas: Mapping[str, Any]) -> list[ColumnConstraint]:
    from ..context.types import Provenance  # noqa: PLC0415

    rules = []
    for key in sorted(config_deltas):
        value = config_deltas[key]
        rules.append(
            ColumnConstraint(
                id=f"learned:config:{key}",
                column=None,
                resolved_from="",
                resolution_confidence=1.0,
                rule="config_delta",
                action="set_option",
                params={"option": key, "value": value, "learned": True},
                enforcement="soft",
                provenance=Provenance(
                    sentence=f"learned config delta {key}={value!r}",
                    tier=4,
                    parse_confidence=1.0,
                ),
            )
        )
    return rules


def _build_example_bank(
    examples: list[Any], *, privacy: str
) -> tuple[ExampleBank | None, Any | None]:
    """Assemble the example bank; vectors are best-effort and optional."""
    if not examples:
        return None, None
    bank = ExampleBank(
        examples=list(examples),
        vectors_path=None,
        embedding_model_id=None,
        masked=all(e.masked for e in examples) if examples else True,
    )
    encodable = [e for e in examples if not e.masked and isinstance(e.raw_value, str)]
    if not encodable:
        return bank, None
    try:  # [semantic] extra + available encoder are both optional
        from ..models.runtime import get_encoder  # noqa: PLC0415

        encoder = get_encoder()
    except Exception:  # noqa: BLE001 - any unavailability degrades silently
        return bank, None
    if encoder is None:
        return bank, None
    try:
        import numpy as np  # noqa: PLC0415

        texts = [str(e.raw_value) for e in encodable]
        vectors = np.asarray(encoder.encode_texts(texts), dtype="float16")
        bank.vectors_path = _VECTORS_MEMBER
        bank.embedding_model_id = getattr(encoder, "model_id", None) or getattr(
            encoder, "name", "unknown"
        )
    except Exception:  # noqa: BLE001 - encoding failure must not break learning
        return bank, None
    return bank, vectors


# ---------------------------------------------------------------------------
# Module-level save/load helpers (the public fd.save_profile / fd.load_profile)
# ---------------------------------------------------------------------------


def save_profile(profile: LearningProfile, path: str | Path) -> None:
    """Save ``profile`` to ``path`` (conventionally ``*.fdprofile``)."""
    if not isinstance(profile, LearningProfile):
        raise TypeError("save_profile expects a LearningProfile")
    profile.save(path)


def load_profile(path: str | Path) -> LearningProfile:
    """Load a ``.fdprofile`` archive, verifying member hashes."""
    return LearningProfile.load(path)
