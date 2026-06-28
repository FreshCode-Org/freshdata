"""Ready-made entity-resolution configs for common domains.

Each template bundles a tuned :class:`~freshdata.enterprise.config.EntityResolutionConfig`
(blocking rules + comparison levels) with sensible PII-redaction columns and a
default :class:`~freshdata.enterprise.entity_resolution.GoldenRecordPolicy`. They
are starting points, not guarantees — tune thresholds and weights to your data.

Blocking rules use only the pandas-evaluable SQL subset (``lower``/``upper``/
``trim``/``left``/``right``/``substr`` over equality predicates), so the
templates work on both the DuckDB and pandas backends. The default backend is
``pandas`` for zero-dependency use; pass ``backend="duckdb"`` to scale.

    from freshdata.enterprise import healthcare_template, resolve_entities

    # ``unique_id_column`` is a per-row identifier; the domain key (patient_id /
    # MRN) is a *comparison* field that legitimately repeats across duplicates.
    tpl = healthcare_template(unique_id_column="row_id")
    resolved, report = resolve_entities(
        df, config=tpl.config, redact_columns=tpl.redact_columns
    )
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .config import BlockingRule, ComparisonLevel, EntityResolutionConfig
from .entity_resolution import GoldenRecordPolicy


@dataclass(frozen=True)
class DomainTemplate:
    """A named entity-resolution starter kit."""

    name: str
    config: EntityResolutionConfig
    redact_columns: tuple[str, ...] = ()
    golden_policy: GoldenRecordPolicy | None = None


def _config(
    *,
    unique_id_column: str,
    backend: str,
    blocking: tuple[BlockingRule, ...],
    comparisons: tuple[ComparisonLevel, ...],
    match_threshold: float = 0.85,
    clerical_review_threshold: float = 0.65,
) -> EntityResolutionConfig:
    return EntityResolutionConfig(
        enabled=True,
        backend=backend,  # type: ignore[arg-type]
        unique_id_column=unique_id_column,
        blocking_rules=blocking,
        comparisons=comparisons,
        match_threshold=match_threshold,
        clerical_review_threshold=clerical_review_threshold,
    )


def education_template(
    *, unique_id_column: str = "id", backend: str = "pandas"
) -> DomainTemplate:
    """Students: ``student_id, email, dob, name, guardian_phone``."""
    config = _config(
        unique_id_column=unique_id_column,
        backend=backend,
        blocking=(
            BlockingRule("lower(l.email) = lower(r.email)", "same email"),
            BlockingRule(
                "l.dob = r.dob and substr(lower(l.name), 1, 3) = "
                "substr(lower(r.name), 1, 3)",
                "same DOB + name prefix",
            ),
            BlockingRule("l.guardian_phone = r.guardian_phone", "same guardian phone"),
        ),
        comparisons=(
            ComparisonLevel("student_id", "exact", weight=3.0),
            ComparisonLevel("email", "jaro_winkler", threshold=0.90, weight=2.5),
            ComparisonLevel("dob", "exact", weight=2.0),
            ComparisonLevel("name", "jaro_winkler", threshold=0.85, weight=2.0),
            ComparisonLevel("guardian_phone", "exact", weight=1.5),
        ),
    )
    return DomainTemplate(
        name="education",
        config=config,
        redact_columns=("email", "dob", "guardian_phone"),
        golden_policy=GoldenRecordPolicy(strategy="most_complete",
                                         id_column=unique_id_column),
    )


def healthcare_template(
    *, unique_id_column: str = "id", backend: str = "pandas"
) -> DomainTemplate:
    """Patients: ``patient_id (MRN), dob, phone, address, insurance_id``.

    Ships with PII redaction on by default (all identifying columns).
    """
    config = _config(
        unique_id_column=unique_id_column,
        backend=backend,
        blocking=(
            BlockingRule("l.patient_id = r.patient_id", "same MRN"),
            BlockingRule(
                "l.dob = r.dob and right(l.phone, 4) = right(r.phone, 4)",
                "same DOB + phone suffix",
            ),
            BlockingRule("l.insurance_id = r.insurance_id", "same insurance id"),
        ),
        comparisons=(
            ComparisonLevel("patient_id", "exact", weight=3.0),
            ComparisonLevel("dob", "exact", weight=2.5),
            ComparisonLevel("phone", "levenshtein", threshold=0.85, weight=1.5),
            ComparisonLevel("address", "jaro_winkler", threshold=0.80, weight=1.5),
            ComparisonLevel("insurance_id", "exact", weight=2.0),
        ),
        match_threshold=0.88,
    )
    return DomainTemplate(
        name="healthcare",
        config=config,
        redact_columns=("patient_id", "dob", "phone", "address", "insurance_id"),
        golden_policy=GoldenRecordPolicy(strategy="most_complete",
                                         id_column=unique_id_column),
    )


def retail_template(
    *, unique_id_column: str = "id", backend: str = "pandas"
) -> DomainTemplate:
    """Customers: ``customer_id, email, phone, loyalty_id, address``."""
    config = _config(
        unique_id_column=unique_id_column,
        backend=backend,
        blocking=(
            BlockingRule("lower(l.email) = lower(r.email)", "same email"),
            BlockingRule("l.phone = r.phone", "same phone"),
            BlockingRule("l.loyalty_id = r.loyalty_id", "same loyalty id"),
        ),
        comparisons=(
            ComparisonLevel("customer_id", "exact", weight=3.0),
            ComparisonLevel("email", "jaro_winkler", threshold=0.90, weight=2.5),
            ComparisonLevel("phone", "levenshtein", threshold=0.85, weight=1.5),
            ComparisonLevel("loyalty_id", "exact", weight=2.0),
            ComparisonLevel("address", "jaro_winkler", threshold=0.80, weight=1.0),
        ),
    )
    return DomainTemplate(
        name="retail",
        config=config,
        redact_columns=("email", "phone", "address"),
        golden_policy=GoldenRecordPolicy(strategy="most_complete",
                                         id_column=unique_id_column),
    )


def media_template(
    *, unique_id_column: str = "id", backend: str = "pandas"
) -> DomainTemplate:
    """Works: ``title/name, creator, release_date, external_ids``."""
    config = _config(
        unique_id_column=unique_id_column,
        backend=backend,
        blocking=(
            BlockingRule("l.external_ids = r.external_ids", "same external id"),
            BlockingRule(
                "substr(lower(l.title), 1, 5) = substr(lower(r.title), 1, 5) "
                "and l.release_date = r.release_date",
                "same title prefix + release date",
            ),
        ),
        comparisons=(
            ComparisonLevel("title", "jaro_winkler", threshold=0.85, weight=2.5),
            ComparisonLevel("creator", "jaro_winkler", threshold=0.85, weight=2.0),
            ComparisonLevel("release_date", "date_distance", threshold=2.0, weight=1.5),
            ComparisonLevel("external_ids", "exact", weight=3.0),
        ),
    )
    return DomainTemplate(
        name="media",
        config=config,
        redact_columns=(),
        golden_policy=GoldenRecordPolicy(strategy="most_recent",
                                         timestamp_column="release_date",
                                         id_column=unique_id_column),
    )


#: Registry of domain template factories.
TEMPLATES: dict[str, Callable[..., DomainTemplate]] = {
    "education": education_template,
    "healthcare": healthcare_template,
    "retail": retail_template,
    "media": media_template,
}


def get_template(name: str, **kwargs: object) -> DomainTemplate:
    """Build a domain template by name (``education``/``healthcare``/``retail``/``media``)."""
    try:
        factory = TEMPLATES[name]
    except KeyError:
        raise KeyError(
            f"unknown template {name!r}; choose from {sorted(TEMPLATES)}"
        ) from None
    return factory(**kwargs)


def with_overrides(template: DomainTemplate, **config_overrides: object) -> DomainTemplate:
    """Return a copy of *template* with its config fields overridden."""
    return replace(template, config=replace(template.config, **config_overrides))  # type: ignore[arg-type]
