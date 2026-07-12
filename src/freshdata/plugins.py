"""One plugin mechanism for the five FreshData extension points: semantic
**experts**, semantic **backends**, **validators**, entity-resolution
**comparators**, and report **exporters**.

Two ways to register, both landing in the same registry:

- **Entry points** (for installed packages)::

      [project.entry-points."freshdata.experts"]
      my_expert = "my_pkg:MyExpert"

- **Explicit registration** (for scripts/notebooks)::

      import freshdata as fd
      fd.register_expert(MyExpert())

Hard safety guarantees, enforced here so a plugin *cannot* opt out:

- **Plugins only propose or validate.** Experts/backends emit
  :class:`~freshdata.semantic.types.SemanticProposal`s that still flow through
  the policy gate (:func:`freshdata.semantic.policy.decide`) and the executor's
  byte-identity guard; validators return read-only findings. A plugin never
  touches the DataFrame directly.
- **Protected columns stay protected.** Plugin proposals go through the exact
  same gate + guard as built-ins, so a plugin can never change an id / target /
  ``preserve``-column value.
- **Declared risk is a ceiling.** A plugin declares ``max_risk``; any proposal
  scored above it is dropped before the gate ever sees it.
- **Network is opt-in.** A plugin with ``uses_network = True`` is registered but
  *inactive* unless explicitly allowed (``allow_network=True`` or
  ``FRESHDATA_ALLOW_NETWORK_PLUGINS=1``).
- **Failures degrade safely.** Every call into a plugin is wrapped: an exception
  disables that plugin for the run instead of failing the clean.
- **Output is schema-checked.** A plugin that returns the wrong type has its
  bad items dropped rather than corrupting the report.

See ``docs/plugins.md`` and ``examples/plugins/`` for authoring guidance, and
``freshdata.testing`` for the contract-test helpers.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from .findings import QualityFinding
    from .semantic.types import SemanticColumnInfo, SemanticContext, SemanticProposal

log = logging.getLogger("freshdata.plugins")

_RISK_RANK = {"low": 0, "medium": 1, "high": 2}
_ENTRY_POINT_GROUPS = {
    "expert": "freshdata.experts",
    "backend": "freshdata.backends",
    "validator": "freshdata.validators",
    "comparator": "freshdata.comparators",
    "exporter": "freshdata.exporters",
}

#: Built-in ER comparison kinds — plugin comparators may not shadow them.
#: Kept in sync with ``freshdata.enterprise.config._COMPARISON_KINDS``
#: (asserted by a test) so this module stays import-light.
_RESERVED_COMPARATOR_NAMES = frozenset(
    {
        "exact",
        "jaro_winkler",
        "levenshtein",
        "token_set",
        "numeric_distance",
        "date_distance",
        "phonetic",
        "metaphone",
        "custom_sql",
    }
)


# --------------------------------------------------------------------------- #
# metadata accessors (duck-typed, with safe defaults)                         #
# --------------------------------------------------------------------------- #


def _plugin_name(obj: object, fallback: str) -> str:
    name = getattr(obj, "name", None)
    return str(name) if isinstance(name, str) and name else fallback


def _uses_network(obj: object) -> bool:
    return bool(getattr(obj, "uses_network", False))


def _requires(obj: object) -> tuple[str, ...]:
    raw = getattr(obj, "requires", ())
    if isinstance(raw, str):
        return (raw,)
    try:
        return tuple(str(r) for r in raw)
    except TypeError:
        return ()


def _max_risk(obj: object) -> str:
    raw = getattr(obj, "max_risk", "high")
    return raw if raw in _RISK_RANK else "high"


def _semantic_types(obj: object) -> tuple[str, ...]:
    raw = getattr(obj, "semantic_types", ())
    if isinstance(raw, str):
        return (raw,)
    try:
        return tuple(str(t) for t in raw)
    except TypeError:
        return ()


def _missing_requirements(obj: object) -> list[str]:
    import importlib.util  # noqa: PLC0415

    missing = []
    for module in _requires(obj):
        top = module.split(".", 1)[0]
        try:
            if importlib.util.find_spec(top) is None:
                missing.append(module)
        except (ImportError, ValueError):
            missing.append(module)
    return missing


def _network_allowed(allow_network: bool) -> bool:
    return allow_network or os.environ.get("FRESHDATA_ALLOW_NETWORK_PLUGINS", "") == "1"


# --------------------------------------------------------------------------- #
# registry records                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class RegisteredPlugin:
    """One registered plugin and why it is (in)active."""

    name: str
    kind: str  # "expert" | "backend" | "validator"
    obj: Any  # a duck-typed plugin instance (expert / backend / validator)
    active: bool
    inactive_reason: str | None = None
    uses_network: bool = False
    max_risk: str = "high"
    semantic_types: tuple[str, ...] = ()
    source: str = "explicit"  # "explicit" | "entry_point"

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "active": self.active,
            "inactive_reason": self.inactive_reason,
            "uses_network": self.uses_network,
            "max_risk": self.max_risk,
            "semantic_types": list(self.semantic_types),
            "source": self.source,
        }


@dataclass
class _Registry:
    experts: dict[str, RegisteredPlugin] = field(default_factory=dict)
    backends: dict[str, RegisteredPlugin] = field(default_factory=dict)
    validators: dict[str, RegisteredPlugin] = field(default_factory=dict)
    comparators: dict[str, RegisteredPlugin] = field(default_factory=dict)
    exporters: dict[str, RegisteredPlugin] = field(default_factory=dict)
    _entry_points_loaded: bool = False

    def bucket(self, kind: str) -> dict[str, RegisteredPlugin]:
        return {"expert": self.experts, "backend": self.backends,
                "validator": self.validators, "comparator": self.comparators,
                "exporter": self.exporters}[kind]


_REGISTRY = _Registry()


def _register(kind: str, obj: object, *, allow_network: bool, source: str) -> RegisteredPlugin:
    name = _plugin_name(obj, fallback=type(obj).__name__)
    uses_network = _uses_network(obj)
    max_risk = _max_risk(obj)
    semantic_types = _semantic_types(obj)

    active = True
    reason: str | None = None
    missing = _missing_requirements(obj)
    if kind == "comparator" and name in _RESERVED_COMPARATOR_NAMES:
        active = False
        reason = "shadows a built-in comparison kind"
    elif missing:
        active = False
        reason = f"requires missing dependency: {', '.join(missing)}"
    elif uses_network and not _network_allowed(allow_network):
        active = False
        reason = (
            "network-using plugin disabled by default; pass allow_network=True "
            "or set FRESHDATA_ALLOW_NETWORK_PLUGINS=1 to enable"
        )

    record = RegisteredPlugin(
        name=name, kind=kind, obj=obj, active=active, inactive_reason=reason,
        uses_network=uses_network, max_risk=max_risk, semantic_types=semantic_types,
        source=source,
    )
    _REGISTRY.bucket(kind)[name] = record
    log.debug("registered %s plugin %r (active=%s)", kind, name, active)
    return record


# --------------------------------------------------------------------------- #
# public registration API (re-exported as fd.register_*)                      #
# --------------------------------------------------------------------------- #


def register_expert(expert: object, *, allow_network: bool = False) -> None:
    """Register a semantic *expert* (proposes value repairs per column)."""
    _register("expert", expert, allow_network=allow_network, source="explicit")


def register_backend(backend: object, *, allow_network: bool = False) -> None:
    """Register a semantic *backend* (proposes over a whole frame; named in
    ``semantic_backends``)."""
    _register("backend", backend, allow_network=allow_network, source="explicit")


def register_validator(validator: object, *, allow_network: bool = False) -> None:
    """Register a *validator* (read-only checks appended to ``fd.validate``)."""
    _register("validator", validator, allow_network=allow_network, source="explicit")


def register_comparator(comparator: object, *, allow_network: bool = False) -> None:
    """Register an entity-resolution *comparator*.

    Protocol: ``name`` (str) and ``__call__(a: str, b: str) -> float`` in
    ``[0, 1]``. Use the name as ``ComparisonLevel(kind=<name>)``. Built-in kind
    names are reserved and cannot be shadowed.
    """
    name = _plugin_name(comparator, fallback=type(comparator).__name__)
    if name in _RESERVED_COMPARATOR_NAMES:
        raise ValueError(
            f"comparator name {name!r} shadows a built-in comparison kind; "
            "pick a distinct name"
        )
    if not callable(comparator):
        raise TypeError("a comparator plugin must be callable: (a, b) -> float in [0, 1]")
    _register("comparator", comparator, allow_network=allow_network, source="explicit")


def register_exporter(exporter: object, *, allow_network: bool = False) -> None:
    """Register a report *exporter*.

    Protocol: ``name`` (str) and ``export(report) -> str | dict`` where
    *report* is a ``CleanReport`` / ``DriftReport`` (anything with
    ``to_dict()``). Consumed via ``fd.export(report, format=<name>)``.
    """
    export = getattr(exporter, "export", None)
    if not callable(export):
        raise TypeError("an exporter plugin must define export(report) -> str | dict")
    _register("exporter", exporter, allow_network=allow_network, source="explicit")


def clear_plugins(kind: str | None = None) -> None:
    """Remove registered plugins (all, or one ``kind``). Mainly for tests.

    Does not un-mark entry-point discovery: call with no argument in a fixture
    teardown to return to a clean slate, then re-discover on next access.
    """
    if kind is None:
        _REGISTRY.experts.clear()
        _REGISTRY.backends.clear()
        _REGISTRY.validators.clear()
        _REGISTRY.comparators.clear()
        _REGISTRY.exporters.clear()
        _REGISTRY._entry_points_loaded = False
    else:
        _REGISTRY.bucket(kind).clear()


def registered_plugins(kind: str | None = None) -> list[dict[str, Any]]:
    """Introspect every registered plugin (active and inactive)."""
    _ensure_entry_points()
    buckets = [kind] if kind else ["expert", "backend", "validator", "comparator", "exporter"]
    out: list[dict[str, Any]] = []
    for k in buckets:
        out.extend(rec.describe() for rec in _REGISTRY.bucket(k).values())
    return out


# --------------------------------------------------------------------------- #
# entry-point discovery                                                       #
# --------------------------------------------------------------------------- #


def _entry_points_for(group: str) -> list[Any]:
    """Return the entry points in *group*, across importlib.metadata versions."""
    from importlib.metadata import entry_points  # noqa: PLC0415

    try:
        return list(entry_points(group=group))  # Python 3.10+
    except TypeError:  # pragma: no cover - Python 3.9 dict API
        all_eps: Any = entry_points()
        getter = getattr(all_eps, "get", None)
        return list(getter(group, [])) if getter is not None else []


def _ensure_entry_points() -> None:
    if _REGISTRY._entry_points_loaded:
        return
    _REGISTRY._entry_points_loaded = True  # set first: never retry a bad group
    for kind, group in _ENTRY_POINT_GROUPS.items():
        for ep in _entry_points_for(group):
            try:
                factory = ep.load()
                obj = factory() if callable(factory) else factory
            except Exception as exc:  # noqa: BLE0001 - a broken plugin must not break import
                log.warning("freshdata plugin entry point %r failed to load: %s", ep.name, exc)
                continue
            _register(kind, obj, allow_network=False, source="entry_point")


# --------------------------------------------------------------------------- #
# safe adapters                                                               #
# --------------------------------------------------------------------------- #


def _cap_and_stamp(
    proposals: object, name: str, kind: str, max_risk: str,
) -> list[SemanticProposal]:
    """Validate plugin output, drop over-risk proposals, and stamp provenance."""
    from .semantic.types import SemanticProposal  # noqa: PLC0415

    if not isinstance(proposals, (list, tuple)):
        log.warning("plugin %s %r returned %s, not a list of proposals; ignored",
                    kind, name, type(proposals).__name__)
        return []
    cap = _RISK_RANK[max_risk]
    out: list[SemanticProposal] = []
    for p in proposals:
        if not isinstance(p, SemanticProposal):
            log.warning("plugin %s %r emitted a non-proposal %r; dropped", kind, name, type(p))
            continue
        if _RISK_RANK.get(p.risk, 2) > cap:
            log.warning("plugin %s %r emitted risk=%s above its declared max_risk=%s; dropped",
                        kind, name, p.risk, max_risk)
            continue
        provenance = {**(dict(p.provenance) if p.provenance else {}),
                      "plugin": name, "plugin_kind": kind}
        out.append(dataclasses.replace(p, backend=f"plugin:{name}", provenance=provenance))
    return out


class _SafeExpert:
    """Wrap a plugin expert so a bug or bad output can never break a clean."""

    def __init__(self, record: RegisteredPlugin):
        self._record = record
        self.name = record.name
        self.issue_type = getattr(record.obj, "issue_type", f"plugin:{record.name}")

    def applies(self, info: SemanticColumnInfo) -> bool:
        try:
            return bool(self._record.obj.applies(info))
        except Exception as exc:  # noqa: BLE0001 - isolate plugin failure
            log.warning("expert plugin %r.applies() failed: %s", self.name, exc)
            return False

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        try:
            raw = self._record.obj.propose(series, info)
        except Exception as exc:  # noqa: BLE0001 - isolate plugin failure
            log.warning("expert plugin %r.propose() failed: %s", self.name, exc)
            return []
        return _cap_and_stamp(raw, self.name, "expert", self._record.max_risk)


class _SafeBackend:
    """Wrap a plugin backend with the same isolation + provenance discipline."""

    def __init__(self, record: RegisteredPlugin):
        self._record = record
        self.name = record.name

    def warm_up(self) -> None:
        warm = getattr(self._record.obj, "warm_up", None)
        if warm is None:
            return
        from .semantic.backends.base import BackendUnavailable  # noqa: PLC0415

        try:
            warm()
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE0001 - a warm-up bug self-disables the backend
            raise BackendUnavailable(
                f"plugin backend {self.name!r} warm_up failed: {exc}") from exc

    def propose(self, df: pd.DataFrame, ctx: SemanticContext, budget: object) -> list:
        try:
            raw = self._record.obj.propose(df, ctx, budget)
        except Exception as exc:  # noqa: BLE0001 - isolate plugin failure
            log.warning("backend plugin %r.propose() failed: %s", self.name, exc)
            return []
        return _cap_and_stamp(raw, self.name, "backend", self._record.max_risk)


class _SafeValidator:
    """Wrap a plugin validator so it can only *append read-only findings*."""

    def __init__(self, record: RegisteredPlugin):
        self._record = record
        self.name = record.name

    def validate(self, df: pd.DataFrame, policy: object, ctx: object) -> list[QualityFinding]:
        from .findings import QualityFinding  # noqa: PLC0415

        try:
            raw = self._record.obj.validate(df, policy, ctx)
        except Exception as exc:  # noqa: BLE0001 - isolate plugin failure
            log.warning("validator plugin %r.validate() failed: %s", self.name, exc)
            return []
        if not isinstance(raw, (list, tuple)):
            log.warning("validator plugin %r returned %s, not findings; ignored",
                        self.name, type(raw).__name__)
            return []
        return [f for f in raw if isinstance(f, QualityFinding)]


class _SafeComparator:
    """Wrap a plugin comparator: exceptions skip the field, output is clamped
    to ``[0, 1]`` so a buggy plugin cannot distort the weighted score."""

    def __init__(self, record: RegisteredPlugin):
        self._record = record
        self.name = record.name

    def __call__(self, a: str, b: str) -> float | None:
        try:
            raw = self._record.obj(a, b)
        except Exception as exc:  # noqa: BLE001 - isolate plugin failure
            log.warning("comparator plugin %r failed: %s", self.name, exc)
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            log.warning(
                "comparator plugin %r returned %r, not a float; field skipped",
                self.name, raw,
            )
            return None
        return min(1.0, max(0.0, value))


class _SafeExporter:
    """Wrap a plugin exporter: exceptions raise a clear error (an export is an
    explicit user request, unlike an in-pipeline proposal)."""

    def __init__(self, record: RegisteredPlugin):
        self._record = record
        self.name = record.name

    def export(self, report: object) -> str | dict:
        result = self._record.obj.export(report)
        if not isinstance(result, (str, dict)):
            raise TypeError(
                f"exporter plugin {self.name!r} returned "
                f"{type(result).__name__}; expected str or dict"
            )
        return result


# --------------------------------------------------------------------------- #
# accessors used by the pipeline                                              #
# --------------------------------------------------------------------------- #


def active_experts() -> tuple[_SafeExpert, ...]:
    """Active plugin experts, wrapped for isolation (built-ins are separate)."""
    _ensure_entry_points()
    return tuple(_SafeExpert(rec) for rec in _REGISTRY.experts.values() if rec.active)


def active_backend_names() -> tuple[str, ...]:
    _ensure_entry_points()
    return tuple(name for name, rec in _REGISTRY.backends.items() if rec.active)


def get_active_backend(name: str) -> _SafeBackend | None:
    _ensure_entry_points()
    rec = _REGISTRY.backends.get(name)
    return _SafeBackend(rec) if rec is not None and rec.active else None


def known_backend_names() -> tuple[str, ...]:
    """Registered backend names (active or not) — for the ``strict`` name check."""
    _ensure_entry_points()
    return tuple(_REGISTRY.backends)


def active_validators() -> tuple[_SafeValidator, ...]:
    _ensure_entry_points()
    return tuple(_SafeValidator(rec) for rec in _REGISTRY.validators.values() if rec.active)


def get_active_comparator(name: str) -> _SafeComparator | None:
    """The comparator registered under *name*, or ``None`` (used by ER scoring)."""
    _ensure_entry_points()
    rec = _REGISTRY.comparators.get(name)
    return _SafeComparator(rec) if rec is not None and rec.active else None


def known_comparator_names() -> tuple[str, ...]:
    """Registered comparator names (active or not) — for config validation."""
    _ensure_entry_points()
    return tuple(_REGISTRY.comparators)


def get_active_exporter(name: str) -> _SafeExporter | None:
    _ensure_entry_points()
    rec = _REGISTRY.exporters.get(name)
    return _SafeExporter(rec) if rec is not None and rec.active else None


def active_exporter_names() -> tuple[str, ...]:
    _ensure_entry_points()
    return tuple(name for name, rec in _REGISTRY.exporters.items() if rec.active)


__all__ = [
    "RegisteredPlugin",
    "active_backend_names",
    "active_experts",
    "active_exporter_names",
    "active_validators",
    "clear_plugins",
    "get_active_backend",
    "get_active_comparator",
    "get_active_exporter",
    "known_backend_names",
    "known_comparator_names",
    "register_backend",
    "register_comparator",
    "register_expert",
    "register_exporter",
    "register_validator",
    "registered_plugins",
]
