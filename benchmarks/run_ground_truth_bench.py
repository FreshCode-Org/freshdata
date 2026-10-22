# benchmarks/run_ground_truth_bench.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import freshdata as fd
from freshdata.config import CleanConfig


def generate_ground_truth_dataset(n: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    customer_id = np.arange(10000, 10000 + n, dtype=np.int64)
    age = rng.integers(18, 75, size=n).astype(np.float64)
    income = np.round(rng.normal(65000, 18000, size=n), 2)
    segment = rng.choice(["enterprise", "smb", "consumer"], size=n)

    logits = -2.0 + 0.03 * (age - 40) + 0.00003 * (income - 65000) + (segment == "enterprise") * 0.9
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(int)

    clean_df = pd.DataFrame({
        "customer_id": pd.Series(customer_id, dtype="Int64"),
        "age": pd.Series(age, dtype="Float64"),
        "income": pd.Series(income, dtype="Float64"),
        "segment": pd.Series(segment, dtype="string"),
    })

    dirty_df = clean_df.copy(deep=True)
    dirty_df["age"] = dirty_df["age"].astype(object)
    dirty_df["segment"] = dirty_df["segment"].astype(object)

    # Inject corruption: string whitespace, sentinels, outliers, missing IDs
    for i in range(0, n, 10):
        dirty_df.loc[i, "segment"] = f"  {dirty_df.loc[i, 'segment']}  "
    for i in range(5, n, 20):
        dirty_df.loc[i, "age"] = "N/A"
    for i in range(7, n, 25):
        dirty_df.loc[i, "income"] = 9_999_999.0  # Extreme outlier
    for i in range(12, n, 50):
        dirty_df.loc[i, "customer_id"] = pd.NA  # Missing identifier (MUST NOT be imputed!)

    return clean_df, dirty_df, y


def evaluate_cell_repair(clean_gt: pd.DataFrame, dirty_in: pd.DataFrame, repaired: pd.DataFrame) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for col in clean_gt.columns:
        for idx in range(len(clean_gt)):
            gt_val = clean_gt[col].iloc[idx]
            d_val = dirty_in[col].iloc[idx]
            r_val = repaired[col].iloc[idx]

            is_corrupted = str(gt_val) != str(d_val)
            was_modified = str(r_val) != str(d_val)

            # For customer_id, ground-truth safe behavior on missing ID is NOT fabricating a fake ID
            if col == "customer_id" and pd.isna(d_val):
                if not pd.isna(r_val):
                    fp += 1
                continue

            if is_corrupted:
                if was_modified:
                    if isinstance(gt_val, (int, float, np.number)) and not pd.isna(r_val):
                        if abs(float(r_val) - float(gt_val)) <= 25000.0:
                            tp += 1
                        else:
                            fp += 1
                    elif str(r_val).strip() == str(gt_val).strip():
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1
            elif was_modified:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    clean_df, dirty_df, y = generate_ground_truth_dataset()

    # 1. freshdata (balanced + outlier handling)
    fd_bal = fd.clean(
        dirty_df,
        config=CleanConfig(strategy="balanced", fix_dtypes=True, impute_missing=True, handle_outliers=True),
    )

    # 2. Plain pandas baseline (blind fillna)
    pd_base = dirty_df.copy()
    for c in ["age", "income", "customer_id"]:
        pd_base[c] = pd.to_numeric(pd_base[c], errors="coerce").astype(float)
        pd_base[c] = pd_base[c].fillna(pd_base[c].median())

    # 3. Sklearn SimpleImputer baseline on numeric columns
    sk_base = dirty_df.copy()
    imp = SimpleImputer(strategy="median")
    num_mat = np.column_stack([
        pd.to_numeric(sk_base["customer_id"], errors="coerce"),
        pd.to_numeric(sk_base["age"], errors="coerce"),
        pd.to_numeric(sk_base["income"], errors="coerce"),
    ])
    imp_mat = imp.fit_transform(num_mat)
    sk_base["customer_id"] = imp_mat[:, 0]
    sk_base["age"] = imp_mat[:, 1]
    sk_base["income"] = imp_mat[:, 2]

    for name, candidate in [
        ("freshdata (balanced)", fd_bal),
        ("pandas (fillna median)", pd_base),
        ("sklearn (SimpleImputer)", sk_base),
    ]:
        p, r, f1 = evaluate_cell_repair(clean_df, dirty_df, candidate)
        print(f"CELL_BENCH | {name:24s} | Precision={p:.3f} | Recall={r:.3f} | F1={f1:.3f}")


if __name__ == "__main__":
    main()
