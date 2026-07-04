"""Context / column corruptors: schema mangling, paraphrase hooks, traps.

Header corruptors emit labels with ``column`` = the *original* name and
``raw_value`` = the mangled alias (the "repair" is alias resolution). The two
trap corruptors label **any** mutation as a failure: a compliant cleaner must
leave those cells alone, so ``should_repair=False`` and auto-apply is banned.
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from .base import CorruptionLabel, Corruptor

#: Realistic ecommerce header aliases (clean name -> plausible wild forms).
ECOMMERCE_ALIASES: dict[str, tuple[str, ...]] = {
    "cust_id": ("CustomerID", "customer id", "CUST-ID", "cid"),
    "email": ("e-mail address", "customer_email", "Email ID", "mail"),
    "email_addr": ("e-mail", "EmailAddress", "customer email id"),
    "phone": ("mobile no.", "contact_number", "Phone#", "mob"),
    "mobile": ("phone_number", "Mobile No", "contact"),
    "status": ("cust_status", "Account Status", "state_flag"),
    "monthly_revenue": ("rev_m", "Monthly Rev.", "revenue(month)"),
    "order_date": ("OrderedOn", "date_of_order", "ord_dt"),
    "postal_code": ("PIN", "pincode", "zip"),
    "quantity": ("qty", "Qty.", "no_of_items"),
}

#: Deterministic Indian-English / Hinglish paraphrase hooks for context
#: sentences. Teacher models may *extend* this set (cached + reviewed); the
#: hooks themselves guarantee an offline floor.
HINGLISH_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "PROTECT": ("{col} kabhi mat badalna.", "{col} ko haath mat lagao.",
                "{col} bilkul change nahi karna."),
    "UNIQUE": ("{col} har row me alag hona chahiye.", "{col} repeat nahi hona chahiye."),
    "ALLOWED_VALUES": ("{col} sirf {values} ho sakta hai.",),
    "IMPUTE_IF": ("{col} khali ho to sirf {threshold}% pakka hone par bharna.",),
    "LOCALE_FORMAT": ("{col} Indian phone numbers hain.",),
}


def _snake_camel(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if "_" in v:
        parts = v.split("_")
        return rng.choice((
            "".join(p.title() for p in parts),                    # CamelCase
            parts[0] + "".join(p.title() for p in parts[1:]),     # camelCase
            " ".join(parts).title(),                              # Title Case words
            "-".join(parts),                                      # kebab-case
        ))
    if v != v.lower():
        return v.lower()
    return v.upper() if len(v) <= 4 else v.title()


def _rename_headers(
    df: pd.DataFrame, rng: random.Random, params: dict[str, Any],
    alias_of,
    corruptor_name: str,
) -> tuple[pd.DataFrame, list[CorruptionLabel]]:
    share = float(params.get("share", 0.5))
    mapping: dict[str, str] = {}
    labels: list[CorruptionLabel] = []
    for col in df.columns:
        if rng.random() >= share:
            continue
        alias = alias_of(str(col), rng)
        if not alias or alias == col or alias in df.columns:
            continue
        mapping[str(col)] = alias
        labels.append(CorruptionLabel(
            raw_value=alias, clean_value=str(col), column=str(col),
            transform_family="context_schema", params={"alias": alias},
            should_repair=True, should_auto_apply=True, risk="low",
            corruptor=corruptor_name, row=None,
        ))
    return df.rename(columns=mapping), labels


def _alias_rename_frame(df, rng, params):
    generic = {
        "email": ("contact_email", "e-mail"), "phone": ("tel", "phone no"),
        "name": ("full name", "customer_name"), "date": ("dt", "when"),
    }

    def alias_of(col: str, r: random.Random) -> str | None:
        for stem, options in generic.items():
            if stem in col.lower():
                return r.choice(options)
        return f"{col}_col"

    return _rename_headers(df, rng, params, alias_of, "column_alias_rename")


def _header_format_frame(df, rng, params):
    return _rename_headers(
        df, rng, params, lambda c, r: _snake_camel(c, r, params), "header_format_mangle",
    )


def _ecommerce_alias_frame(df, rng, params):
    def alias_of(col: str, r: random.Random) -> str | None:
        options = ECOMMERCE_ALIASES.get(col.lower())
        return r.choice(options) if options else None

    return _rename_headers(df, rng, params, alias_of, "ecommerce_alias_rename")


def _hinglish_paraphrase_frame(df, rng, params):
    """Context-sentence hook: emits paraphrase labels, leaves the frame alone.

    Params: ``intent``, ``column``, and optional slot values. The label's
    ``clean_value`` is the canonical English sentence the paraphrase must
    compile to; consumers feed these into the intent-head training set.
    """
    intent = str(params.get("intent", "PROTECT"))
    column = str(params.get("column", "monthly_revenue"))
    canonical = str(params.get("canonical", f"Never modify {column} values."))
    templates = HINGLISH_PARAPHRASES.get(intent, ())
    labels = []
    for template in templates:
        sentence = template.format(
            col=column,
            values=", ".join(params.get("values", ("active", "inactive", "pending"))),
            threshold=int(float(params.get("threshold", 0.95)) * 100),
        )
        labels.append(CorruptionLabel(
            raw_value=sentence, clean_value=canonical, column=column,
            transform_family="context_schema",
            params={"intent": intent, "language": "hinglish"},
            should_repair=True, should_auto_apply=True, risk="low",
            corruptor="hinglish_context_paraphrase", row=None,
        ))
    return df, labels


def _duplicated_id_rows_frame(df, rng, params):
    """Rows that share an ID but differ elsewhere — dedup-key bait."""
    id_column = str(params.get("id_column", df.columns[0]))
    n = int(params.get("n", 2))
    if df.empty or id_column not in df.columns:
        return df, []
    picks = [rng.randrange(len(df)) for _ in range(n)]
    extra = df.iloc[picks].copy()
    mutable = [
        c for c in df.columns
        if c != id_column and df[c].map(lambda v: isinstance(v, str)).all()
    ]
    for offset, (_, row) in enumerate(extra.iterrows()):
        if mutable:
            col = rng.choice(mutable)
            extra.iloc[offset, extra.columns.get_loc(col)] = f"{row[col]} (dup)"
    out = pd.concat([df, extra], ignore_index=True)
    labels = [
        CorruptionLabel(
            raw_value=str(df.iloc[p][id_column]), clean_value=None, column=id_column,
            transform_family="row_structure", params={"source_row": p, "id_column": id_column},
            should_repair=True, should_auto_apply=False, risk="medium", ambiguous=True,
            corruptor="duplicated_id_rows", row=len(df) + i,
        )
        for i, p in enumerate(picks)
    ]
    return out, labels


def _mutate_cells_trap(df, rng, params, *, corruptor_name: str, protected: bool):
    column = str(params.get("column", ""))
    if column not in df.columns or df.empty:
        return df, []
    n = min(int(params.get("n", 2)), len(df))
    positions = rng.sample(range(len(df)), n)
    labels = []
    for pos in positions:
        clean = df.iloc[pos][column]
        raw = rng.choice((f" {clean} ", f"{clean},00", str(clean).upper(), f"{clean}?"))
        if str(raw) == str(clean):
            raw = f" {clean} "
        df.iloc[pos, df.columns.get_loc(column)] = raw
        labels.append(CorruptionLabel(
            raw_value=raw, clean_value=clean, column=column,
            transform_family="trap", params={"trap": corruptor_name},
            should_repair=False, should_auto_apply=False, risk="high",
            protected=protected, ambiguous=False,
            corruptor=corruptor_name, row=pos,
        ))
    return df, labels


def _protected_trap_frame(df, rng, params):
    return _mutate_cells_trap(
        df, rng, params, corruptor_name="protected_column_trap", protected=True)


def _target_trap_frame(df, rng, params):
    return _mutate_cells_trap(
        df, rng, params, corruptor_name="target_column_trap", protected=False)


column_alias_rename = Corruptor(
    name="column_alias_rename", family="context_schema", kind="header",
    frame_fn=_alias_rename_frame)
header_format_mangle = Corruptor(
    name="header_format_mangle", family="context_schema", kind="header",
    frame_fn=_header_format_frame)
ecommerce_alias_rename = Corruptor(
    name="ecommerce_alias_rename", family="context_schema", kind="header",
    frame_fn=_ecommerce_alias_frame)
hinglish_context_paraphrase = Corruptor(
    name="hinglish_context_paraphrase", family="context_schema", kind="context",
    frame_fn=_hinglish_paraphrase_frame)
duplicated_id_rows = Corruptor(
    name="duplicated_id_rows", family="row_structure", kind="row",
    frame_fn=_duplicated_id_rows_frame, risk="medium",
    should_auto_apply=False, ambiguous=True)
protected_column_trap = Corruptor(
    name="protected_column_trap", family="trap", kind="row",
    frame_fn=_protected_trap_frame, risk="high",
    should_repair=False, should_auto_apply=False, protected=True)
target_column_trap = Corruptor(
    name="target_column_trap", family="trap", kind="row",
    frame_fn=_target_trap_frame, risk="high",
    should_repair=False, should_auto_apply=False)

CONTEXT_CORRUPTORS = (
    column_alias_rename,
    header_format_mangle,
    ecommerce_alias_rename,
    hinglish_context_paraphrase,
    duplicated_id_rows,
    protected_column_trap,
    target_column_trap,
)
