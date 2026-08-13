"""Repair a DataFrame with freshdata, then validate it with GX Core.

Great Expectations is optional and remains separate from freshdata's core
dependencies. Install it before running this recipe:

    pip install great-expectations
    python examples/10_great_expectations_recipe.py
"""

import os

import pandas as pd

import freshdata as fd

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")

import great_expectations as gx  # noqa: E402


def make_checkpoint():
    """Build an in-memory checkpoint for the cleaned orders contract."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = {
        "globally": False,
        "metric_calculations": False,
    }
    data_source = context.data_sources.add_pandas(name="freshdata_recipe")
    data_asset = data_source.add_dataframe_asset(name="orders")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("whole_frame")

    suite = context.suites.add(gx.ExpectationSuite(name="clean_orders"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="order_amount",
            type_="float64",
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="active",
            type_="bool",
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="order_amount",
            min_value=0,
        )
    )

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="clean_orders_validation",
            data=batch_definition,
            suite=suite,
        )
    )
    return context.checkpoints.add(
        gx.Checkpoint(
            name="clean_orders_checkpoint",
            validation_definitions=[validation_definition],
        )
    )


def validate(checkpoint, frame: pd.DataFrame) -> bool:
    """Run the checkpoint against one in-memory DataFrame."""
    result = checkpoint.run(batch_parameters={"dataframe": frame})
    return bool(result.success)


def main() -> None:
    raw = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "order_amount": ["$19.99", "$5.00", "$8.25", "$12.40", "$7.10"],
            "active": ["true", "false", "true", "true", "false"],
        }
    )
    checkpoint = make_checkpoint()

    before = validate(checkpoint, raw)
    print(f"Before freshdata: checkpoint passed = {before}")

    cleaned, report = fd.clean(
        raw,
        id_columns=("customer_id",),
        return_report=True,
    )
    after = validate(checkpoint, cleaned)

    print(f"After freshdata:  checkpoint passed = {after}")
    print(report.summary())

    assert not before, "the raw string values should fail the typed contract"
    assert after, "the freshdata-cleaned values should pass the checkpoint"


if __name__ == "__main__":
    main()
