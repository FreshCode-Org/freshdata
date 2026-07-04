"""Build the seed corpus into ``training/build/seed/``.

Runs the license gate first, then materializes every allowed source:
synthetic tables (parquet or csv), the hand-authored fixtures, and the
labeled context-sentence set. ``meta.json`` records provenance for each file.

CLI::

    python -m training.datasets.build_seed [--seed 0] [--out training/build/seed]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ..common import BUILD_DIR, TRAINING_ROOT, utc_now_iso, write_json, write_jsonl
from ..seed.synthetic import SYNTHETIC_SOURCE_ID, make_context_sentences, seed_tables
from .validators import check_licenses

DEFAULT_OUT = BUILD_DIR / "seed"


def build_seed(out_dir: Path | str = DEFAULT_OUT, *, seed: int = 0) -> dict[str, object]:
    problems = check_licenses()
    if problems:
        raise SystemExit("seed registry invalid:\n" + "\n".join(f"  {p}" for p in problems))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}

    for name, frame in seed_tables(seed=seed).items():
        try:
            path = out / f"{name}.parquet"
            frame.to_parquet(path, index=False)
        except (ImportError, ValueError):  # no parquet engine available
            path = out / f"{name}.csv"
            frame.to_csv(path, index=False)
        files[path.name] = {
            "source_id": SYNTHETIC_SOURCE_ID, "rows": len(frame), "synthetic": True}

    sentences = make_context_sentences(seed=seed)
    sentence_path = write_jsonl(out / "context_sentences.jsonl", sentences)
    files[sentence_path.name] = {
        "source_id": SYNTHETIC_SOURCE_ID, "rows": len(sentences), "synthetic": True,
    }

    handauthored = TRAINING_ROOT / "seed" / "sources" / "handauthored_ecommerce_v1.csv"
    shutil.copy(handauthored, out / handauthored.name)
    files[handauthored.name] = {"source_id": "handauthored_ecommerce_v1", "synthetic": True}

    meta = {"built_at": utc_now_iso(), "seed": seed, "files": files}
    write_json(out / "meta.json", meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.datasets.build_seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    meta = build_seed(args.out, seed=args.seed)
    print(f"seed corpus built: {len(meta['files'])} files -> {args.out}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
