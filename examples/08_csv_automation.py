"""Batch CSV cleaning automation with audit logs.

Creates a few messy CSVs in a temp folder, cleans each with a reusable Cleaner,
writes cleaned CSVs, and saves a per-file JSON audit report. Run:

    python examples/08_csv_automation.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import freshdata as fd


def seed_inbox(inbox: Path, n_files: int = 3, seed: int = 5) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_files):
        n = 50
        df = pd.DataFrame(
            {
                "ref": [f"R{i}{j:03d}" for j in range(n)],
                "value": rng.normal(100, 20, n),
                "label ": rng.choice(["a", "b", "N/A"], n),  # trailing space + sentinel
            }
        )
        df.loc[rng.choice(n, 5, replace=False), "value"] = np.nan
        df.to_csv(inbox / f"export_{i}.csv", index=False)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inbox, outbox, logs = root / "inbox", root / "clean", root / "logs"
        for d in (inbox, outbox, logs):
            d.mkdir()
        seed_inbox(inbox)

        cleaner = fd.Cleaner(id_columns=("ref",), strategy="balanced")
        for path in sorted(inbox.glob("*.csv")):
            cleaned = cleaner.clean(pd.read_csv(path))
            cleaned.to_csv(outbox / path.name, index=False)
            (logs / f"{path.stem}.json").write_text(
                json.dumps(cleaner.report_.to_dict(), indent=2, default=str)
            )
            print(f"{path.name}: {cleaner.report_.summary().splitlines()[0]}")

        print(f"\nCleaned {len(list(outbox.glob('*.csv')))} files; "
              f"{len(list(logs.glob('*.json')))} audit logs written.")


if __name__ == "__main__":
    main()
