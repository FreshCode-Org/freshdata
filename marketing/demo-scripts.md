# FreshData Live Demo Scripts (30-Second & 2-Minute)

Scripts for live coding demonstrations at PyData meetups, YouTube/Loom walkthroughs, conference lightning talks, and developer tutorials.

---

## 30-Second Elevator Demo: "The One-Line Audit"

**Objective**: Hook the viewer by demonstrating immediate value on a dirty dataset in under 30 seconds.

### Script & Actions

1. **Speaker**:
   > *"Every time you ingest a raw CSV into pandas, you spend 20 minutes writing regex for whitespace, fixing currency signs, and replacing 'N/A' sentinels. Watch this."*

2. **Action (Terminal or IPython)**:
   ```python
   import pandas as pd
   import freshdata as fd

   df = pd.read_csv("examples/data/messy_customers.csv")
   cleaned, report = fd.clean(df, return_report=True)
   print(report.summary())
   ```

3. **Speaker**:
   > *"In one line, FreshData fixed the headers, parsed the dates, stripped padding, and flagged outliers. But notice this: it didn't touch customer IDs, and it gives you a complete audit log showing every cell it changed, why, and its confidence score."*

4. **Action**:
   ```python
   print(cleaned.head(3))
   ```

5. **Closing**:
   > *"Fast, explainable, and safe. `pip install freshdata-cleaner`."*

---

## 2-Minute Deep Dive Demo: "From Raw Data to ML-Ready with Safety Invariants"

**Objective**: Walk a technical developer through a full end-to-end workflow highlighting safety invariants and Polars support.

### Timeline

* **0:00 – 0:30: The Messy Input Problem**
  * Open Jupyter or VS Code with a raw dataset:
    * Uncleaned headers (`" Annual Income "`)
    * Dirty sentinels (`"N/A"`, `"-"`, `"missing"`)
    * Imbalanced outliers (`age = -5`, `age = 165`)
    * Target column (`churn`) with missing values
  * Speaker: *"If you run a blanket fillna or drop_duplicates, you leak target data and delete valid events."*

* **0:30 – 1:00: Clean with Target Protection & Review Plan**
  * Show the interactive clean call:
    ```python
    cleaned, report = fd.clean(df, target_column="churn", return_report=True)
    ```
  * Point out the summary output:
    * Shows rows before/after
    * Shows columns before/after
    * Flags age outliers in `age_outlier` boolean column rather than deleting rows.
    * Explains that `churn` was preserved because it's a target label.

* **1:00 – 1:30: Inspecting the Action Audit Log**
  * Show `report.actions`:
    ```python
    for a in report.actions:
        print(f"[{a.step}] {a.column}: {a.description} (risk={a.risk})")
    ```
  * Speaker: *"Every action has a risk score. If a decision has medium or high risk, FreshData explicitly flags it in `report.recommendations` for human review."*

* **1:30 – 2:00: Native Polars Execution**
  * Switch to Polars:
    ```python
    import polars as pl
    pl_df = pl.from_pandas(df)
    clean_pl = fd.clean(pl_df)
    assert isinstance(clean_pl, pl.DataFrame)
    ```
  * Speaker: *"Works identically on Polars DataFrames with zero pandas conversion overhead. Check out the open-source repository at github.com/FreshCode-Org/freshdata."*
