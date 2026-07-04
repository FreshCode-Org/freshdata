"""Corrupt the seed corpus into labeled training pairs.

For each seed table a fixed, versioned corruption plan runs under an explicit
seed; the output is ``(messy frame, clean frame, labels.jsonl)`` where labels
are the machine-verifiable ground truth (see ``training/corruptors/base.py``).
Every label file is validated before it is written.

CLI::

    python -m training.datasets.build_corrupted [--seed 0] [--dev]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..common import BUILD_DIR, utc_now_iso, write_json, write_jsonl
from ..corruptors import compose, get_corruptor
from ..corruptors.base import apply_corruptor
from ..seed.synthetic import seed_tables
from .validators import validate_labels

DEFAULT_OUT = BUILD_DIR / "corrupted"

#: (corruptor, columns, share) plans per seed table. ``--dev`` keeps the
#: first few steps only so the dev pipeline stays fast.
PLANS: dict[str, list[tuple[str, tuple[str, ...] | None, float]]] = {
    "customers": [
        ("whitespace_insertion", ("full_name", "address", "city"), 0.25),
        ("casing_change", ("city", "country"), 0.25),
        ("sentinel_injection", ("address",), 0.08),
        ("empty_na_variants", ("state",), 0.08),
        ("email_at_whitespace", ("email",), 0.2),
        ("email_double_at", ("email",), 0.15),
        ("email_casing", ("email",), 0.15),
        ("email_punct_noise", ("email",), 0.1),
        ("phone_in_spacing", ("phone",), 0.2),
        ("phone_in_zero_prefix", ("phone",), 0.2),
        ("phone_in_plus91_format", ("phone",), 0.2),
        ("phone_hyphenation", ("phone",), 0.15),
        ("phone_unsafe_mutation", ("phone",), 0.05),
        ("allowed_value_case", ("status",), 0.3),
        ("allowed_value_whitespace", ("status",), 0.15),
        ("allowed_value_separator", ("status",), 0.1),
        ("edit_distance_typo", ("status",), 0.1),
        ("ambiguous_category", ("status",), 0.05),
        ("country_code_ambiguity", ("country",), 0.1),
        ("close_category_pair", ("country",), 0.08),
        ("date_format_shuffle", ("signup_date",), 0.25),
        ("date_dayfirst_ambiguity", ("signup_date",), 0.1),
        ("month_name_abbreviation", ("signup_date",), 0.15),
        ("invalid_date_phrase", ("signup_date",), 0.04),
        ("relative_date_phrase", ("signup_date",), 0.04),
        ("boolean_synonym_replacement", ("newsletter_opt_in",), 0.4),
    ],
    "transactions": [
        ("currency_formatting", ("monthly_revenue",), 0.3),
        ("thousands_separators", ("monthly_revenue",), 0.2),
        ("mixed_dtype_stringification", ("monthly_revenue",), 0.15),
        ("allowed_value_case", ("order_status",), 0.3),
        ("close_category_pair", ("order_status",), 0.08),
        ("date_format_shuffle", ("order_date",), 0.25),
        ("whitespace_insertion", ("notes",), 0.3),
        ("unit_suffix_insertion", ("monthly_revenue",), 0.05),
    ],
}

#: Frame-level steps applied after cell plans (traps, rows, headers).
FRAME_STEPS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "customers": [
        ("protected_column_trap", {"column": "national_id_like", "n": 3}),
        ("duplicate_row_injection", {"n_duplicates": 3}),
        ("duplicated_id_rows", {"id_column": "cust_id", "n": 2}),
    ],
    "transactions": [
        ("target_column_trap", {"column": "monthly_revenue", "n": 2}),
        ("duplicate_row_injection", {"n_duplicates": 2}),
    ],
}


def build_corrupted(
    out_dir: Path | str = DEFAULT_OUT, *, seed: int = 0, dev: bool = False
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"built_at": utc_now_iso(), "seed": seed, "dev": dev, "tables": {}}

    for name, clean in seed_tables(seed=seed).items():
        plan = PLANS.get(name, [])
        if dev:
            plan = plan[:8]
        steps = [(get_corruptor(cn), cols, share) for cn, cols, share in plan]
        messy, labels = compose(clean, steps, seed=seed)
        for corruptor_name, params in ([] if dev else FRAME_STEPS.get(name, [])):
            messy, extra = apply_corruptor(
                messy, get_corruptor(corruptor_name), seed=seed + 71, params=params,
            )
            labels.extend(extra)

        label_dicts = [label.to_dict() for label in labels]
        problems = validate_labels(label_dicts)
        if problems:
            raise SystemExit(f"{name}: invalid corruption labels:\n" + "\n".join(problems[:10]))

        _write_frame(out / f"{name}.clean", clean)
        _write_frame(out / f"{name}.messy", messy)
        write_jsonl(out / f"{name}.labels.jsonl", label_dicts)
        summary["tables"][name] = {
            "rows_clean": len(clean), "rows_messy": len(messy), "labels": len(label_dicts),
            "families": sorted({label["transform_family"] for label in label_dicts}),
        }

    write_json(out / "meta.json", summary)
    return summary


def _write_frame(stem: Path, frame: pd.DataFrame) -> Path:
    try:
        path = stem.with_suffix(".parquet")
        frame.to_parquet(path, index=False)
    except (ImportError, ValueError):
        path = stem.with_suffix(".csv")
        frame.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.datasets.build_corrupted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dev", action="store_true", help="small fast subset of corruptors")
    args = parser.parse_args(argv)
    summary = build_corrupted(args.out, seed=args.seed, dev=args.dev)
    for table, info in summary["tables"].items():
        print(f"{table}: {info['labels']} labels over {info['rows_messy']} rows", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
