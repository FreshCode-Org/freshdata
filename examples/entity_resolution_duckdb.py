"""Probabilistic entity resolution at scale with a DuckDB backend.

Deduplicate records using DuckDB-powered blocking (candidate-pair generation)
plus rule-weighted probabilistic scoring. Falls back to the pandas backend when
DuckDB is not installed. Run:

    python examples/entity_resolution_duckdb.py

DuckDB backend extra:

    pip install "freshdata-cleaner[entity-resolution]"
"""

import importlib.util

import pandas as pd

import freshdata as fd
from freshdata.enterprise import BlockingRule, ComparisonLevel, EntityResolutionConfig


def people() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": [
                "Jonathan Smith",
                "Jon Smith",
                "Johnny Smith",
                "Alice Brown",
                "Alicia Brown",
                "Robert King",
            ],
            "dob": [
                "1990-01-01", "1990-01-01", "1990-01-01",
                "1985-05-05", "1985-05-05", "1970-12-12",
            ],
            "email": [
                "jsmith@x.com", "jsmith@x.com", "jon@y.com",
                "alice@z.com", "alicia@z.com", "rking@q.com",
            ],
        }
    )


def main() -> None:
    backend = "duckdb"
    if importlib.util.find_spec("duckdb") is None:
        backend = "pandas"
        print("duckdb not installed -> using the pandas fallback backend\n")

    config = EntityResolutionConfig(
        enabled=True,
        backend=backend,
        unique_id_column="id",
        # Blocking caps the candidate space (no full cartesian product).
        blocking_rules=(
            BlockingRule(sql="lower(l.email) = lower(r.email)", description="same email"),
            BlockingRule(sql="l.dob = r.dob", description="same date of birth"),
        ),
        comparisons=(
            ComparisonLevel(column="name", kind="jaro_winkler", threshold=0.85, weight=3.0),
            ComparisonLevel(column="dob", kind="exact", weight=1.0),
            ComparisonLevel(column="email", kind="exact", weight=1.0),
        ),
        match_threshold=0.80,
        clerical_review_threshold=0.55,
        max_pairs=1_000_000,  # hard safety gate
    )

    resolved, report = fd.resolve_entities(people(), config=config)
    print(report.summary(), "\n")
    for cluster in report.clusters:
        print(
            f"cluster {cluster.cluster_id}: records {list(cluster.record_ids)} "
            f"(canonical={cluster.canonical_record_id}, confidence={cluster.confidence:.2f})"
        )
    print("\nresolved frame (with cluster_id):")
    print(resolved[["id", "name", "cluster_id"]].to_string(index=False))


if __name__ == "__main__":
    main()
