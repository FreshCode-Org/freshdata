"""Schema-drift & data-contract monitoring with persisted baselines.

Build a PII-safe baseline from a trusted dataset, persist it as JSON, then
monitor new batches for schema drift, distribution drift (KS / PSI), contract
violations, and a trust-score quality gate. Run:

    python examples/schema_drift_monitoring.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import freshdata as fd
from freshdata import ColumnContract, DataContract
from freshdata.enterprise import DriftConfig


def trusted() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": range(1000),
            "age": rng.normal(42, 9, 1000).round(1),
            "country": rng.choice(["US", "GB", "FR"], 1000, p=[0.6, 0.3, 0.1]),
            "plan": rng.choice(["free", "pro", "enterprise"], 1000, p=[0.7, 0.25, 0.05]),
        }
    )


def main() -> None:
    # A data contract declares the expectations new data must honour.
    contract = DataContract(
        name="customers",
        columns=(
            ColumnContract(name="customer_id", dtype="int64", unique=True, nullable=False),
            ColumnContract(name="age", dtype="float64", min_value=0, max_value=120),
            ColumnContract(name="country", allowed_values=("US", "GB", "FR")),
            ColumnContract(name="plan", allowed_values=("free", "pro", "enterprise")),
        ),
        trust_score_min=70.0,
    )

    # 1) Build + persist a baseline (no raw samples are stored by default).
    base = fd.build_baseline(trusted(), name="customers", contract=contract)
    path = Path(tempfile.gettempdir()) / "customers.baseline.json"
    fd.save_baseline(base, path)
    print(f"baseline written to {path} (PII-safe: raw values not stored)\n")

    # 2) A healthy new batch passes.
    healthy = trusted()
    report = fd.monitor_contract(healthy, baseline_path=path)
    print("== healthy batch ==")
    print(report.summary(), "\n")

    # 3) A drifted batch: age shifts up, France over-represented, a bad country.
    rng = np.random.default_rng(7)
    drifted = trusted()
    drifted["age"] = rng.normal(60, 9, len(drifted))  # distribution shift -> KS/PSI
    drifted["country"] = rng.choice(["US", "GB", "FR"], len(drifted), p=[0.2, 0.2, 0.6])
    drifted.loc[0, "country"] = "ZZ"  # contract.allowed_values violation

    report = fd.compare_to_baseline(drifted, fd.load_baseline(path), drift_config=DriftConfig())
    print("== drifted batch ==")
    print(report.summary())
    print(f"\npassed={report.passed}  errors={report.n_errors}  warnings={report.n_warnings}")

    # 4) Baseline-FREE schema diff: explain structural drift the moment a frame
    #    arrives, with no persisted snapshot. Here an upstream team renamed
    #    `country` -> `country_code`, dropped `plan`, and added `utm_source`,
    #    and `age` arrived as text. diff_schema explains all of it *before* any
    #    repair runs; on_unexpected / on_missing control the gate.
    incoming = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "age": ["42", "37", "51"],  # dtype drift: text, not float
            "country_code": ["US", "GB", "FR"],  # rename of `country`
            "utm_source": ["g", "fb", "g"],  # undeclared
        }
    )
    incoming.attrs["semantic_types"] = {"country_code": "country"}
    contract_sem = DataContract(
        name="customers",
        columns=(
            ColumnContract(name="customer_id", dtype="int64", unique=True, nullable=False),
            ColumnContract(name="age", dtype="float64"),
            ColumnContract(name="country", semantic_type="country"),
            ColumnContract(name="plan", allowed_values=("free", "pro", "enterprise")),
        ),
    )
    diff = fd.diff_schema(incoming, contract=contract_sem, on_unexpected="warn", on_missing="fail")
    print("\n== baseline-free schema diff ==")
    print(diff.summary())
    print("\ncategories:", diff.contract_results)
    print("\nfindings table:")
    print(diff.to_frame()[["check_id", "column", "level", "message"]].to_string(index=False))

    # The same diff can ride along with a suggested plan for the incoming data.
    plan = fd.suggest_plan(incoming, contract=contract_sem, on_missing="fail")
    print(f"\nplan carries schema_diff: passed={plan.schema_diff.passed}")


if __name__ == "__main__":
    main()
