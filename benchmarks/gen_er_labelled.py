"""Generate the labelled entity-resolution benchmark dataset (deterministic).

Each row carries ``entity_id`` ground truth. The generator plants the failure
modes a linkage system must survive, not just easy noisy copies:

* typos (swap / replace / drop a character);
* reordered names ("Ann van Dyke" vs "van Dyke, Ann");
* abbreviations (first initial, nicknames: William -> Bill);
* missing fields (blanked email / phone / dob);
* transliteration differences (Müller vs Mueller, José vs Jose);
* shared-household records — same surname + address + phone but *different*
  people (true non-matches with high similarity);
* deliberate near-collisions — distinct people with nearly identical names
  and emails (john.smith.83 vs john.smith.91).

Regenerate (byte-identical for a given seed/rows):

    python benchmarks/gen_er_labelled.py --rows 5000 --seed 7 \
        --out benchmarks/data/er_labelled_5k.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_FIRST = [
    "william", "elizabeth", "james", "katherine", "robert", "margaret",
    "michael", "jennifer", "david", "patricia", "jose", "sofia", "hans",
    "greta", "ann", "thomas", "charles", "susan", "daniel", "laura",
]
_NICK = {
    "william": "bill", "elizabeth": "liz", "james": "jim",
    "katherine": "kate", "robert": "bob", "margaret": "peggy",
    "michael": "mike", "jennifer": "jen", "david": "dave",
    "patricia": "trish", "thomas": "tom", "charles": "chuck",
    "susan": "sue", "daniel": "dan",
}
_LAST = [
    "smith", "johnson", "mueller", "garcia", "van dyke", "patel", "kim",
    "obrien", "nguyen", "silva", "kowalski", "jones", "brown", "murphy",
]
_TRANSLIT = {"mueller": "müller", "jose": "josé", "garcia": "garcía"}
_STREETS = ["oak st", "maple ave", "birch rd", "elm way", "cedar ln", "pine ct"]


def _typo(rng: np.random.Generator, s: str) -> str:
    if len(s) < 3:
        return s
    i = int(rng.integers(1, len(s) - 1))
    op = rng.random()
    if op < 0.34:  # swap adjacent
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    if op < 0.67:  # replace
        return s[:i] + chr(ord("a") + int(rng.integers(0, 26))) + s[i + 1 :]
    return s[:i] + s[i + 1 :]  # drop


def _corrupt(rng: np.random.Generator, rec: dict) -> tuple[dict, str]:
    """Return a corrupted copy of *rec* and the corruption kind."""
    out = dict(rec)
    kind = rng.choice(
        ["typo", "reorder", "abbrev", "missing", "translit", "case_noise"],
        p=[0.28, 0.16, 0.18, 0.18, 0.08, 0.12],
    )
    first, last = out["first_name"], out["last_name"]
    if kind == "typo":
        field = rng.choice(["first_name", "last_name", "email"])
        out[field] = _typo(rng, out[field])
    elif kind == "reorder":
        out["full_name"] = f"{last}, {first}"
    elif kind == "abbrev":
        if first in _NICK and rng.random() < 0.5:
            out["first_name"] = _NICK[first]
        else:
            out["first_name"] = first[0] + "."
    elif kind == "missing":
        for field in rng.choice(
            ["email", "phone", "dob"], size=int(rng.integers(1, 3)), replace=False
        ):
            out[field] = ""
    elif kind == "translit":
        out["last_name"] = _TRANSLIT.get(last, last.replace("o", "ø"))
    elif kind == "case_noise":
        out["email"] = out["email"].upper()
        out["first_name"] = f" {first.title()} "
    if kind != "reorder":
        out["full_name"] = f"{out['first_name']} {out['last_name']}"
    return out, str(kind)


def generate(rows: int, seed: int = 7) -> pd.DataFrame:
    """~55% unique entities, ~30% corrupted duplicates, ~8% household
    non-matches, ~7% near-collision non-matches."""
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    entity = 0

    def base_record(eid: int) -> dict:
        first = str(rng.choice(_FIRST))
        last = str(rng.choice(_LAST))
        street = f"{int(rng.integers(1, 200))} {rng.choice(_STREETS)}"
        return {
            "first_name": first,
            "last_name": last,
            "full_name": f"{first} {last}",
            "email": f"{first}.{last.replace(' ', '')}.{eid}@mail.test",
            "phone": f"555{int(rng.integers(1_000_000, 9_999_999))}",
            "dob": f"19{int(rng.integers(50, 99)):02d}-"
            f"{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}",
            "address": street,
            "entity_id": eid,
            "corruption": "none",
        }

    while len(records) < rows:
        roll = rng.random()
        if roll < 0.30 and records:
            # corrupted duplicate of an existing entity
            src = records[int(rng.integers(0, len(records)))]
            dup, kind = _corrupt(rng, src)
            dup["corruption"] = kind
            records.append(dup)
        elif roll < 0.38 and records:
            # shared household: same surname/address/phone, different person
            src = records[int(rng.integers(0, len(records)))]
            member = base_record(entity)
            entity += 1
            member["last_name"] = src["last_name"]
            member["full_name"] = f"{member['first_name']} {src['last_name']}"
            member["address"] = src["address"]
            member["phone"] = src["phone"]
            member["corruption"] = "household_nonmatch"
            records.append(member)
        elif roll < 0.45:
            # near-collision: two distinct people, nearly identical identifiers
            a = base_record(entity)
            entity += 1
            b = base_record(entity)
            entity += 1
            b["first_name"] = a["first_name"]
            b["last_name"] = a["last_name"]
            b["full_name"] = a["full_name"]
            b["email"] = a["email"].replace(f".{a['entity_id']}@", f".{b['entity_id']}@")
            a["corruption"] = "near_collision"
            b["corruption"] = "near_collision"
            records.extend([a, b])
        else:
            records.append(base_record(entity))
            entity += 1

    df = pd.DataFrame(records[:rows])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="benchmarks/data/er_labelled_5k.csv")
    args = ap.parse_args(argv)
    df = generate(args.rows, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_entities = df["entity_id"].nunique()
    print(
        f"wrote {args.out}: {len(df)} rows, {n_entities} entities, "
        f"corruption mix:\n{df['corruption'].value_counts().to_string()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
