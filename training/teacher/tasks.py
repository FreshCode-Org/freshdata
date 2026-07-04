"""Teacher task harness: batched, masked, cached, compliance-gated.

Guarantees enforced here:

- **compliance first** — every run checks the provider ledger and aborts on
  anything but an explicit approval;
- **no full rows** — tasks accept only column names plus a bounded list of
  *masked* sample values, never DataFrame rows;
- **PII masking** — email locals, phone digits, and long identifiers are
  masked before any prompt is built (:func:`mask_pii`);
- **strict schemas** — invalid responses are dropped, not repaired;
- **cache everything** — prompt + response + provenance land in the audit
  cache before payloads are returned;
- **degrade safely** — a failed teacher call returns an empty batch so the
  pipeline can fall back to corruptor/hook data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import sha256_json, sha256_text
from . import compliance
from .cache import CacheKey, TeacherCache
from .clients import TeacherClient, TeacherUnavailable, default_client
from .schemas import SCHEMA_VERSION, SchemaError, validate_batch

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_EMAIL_RE = re.compile(r"([^@\s])[^@\s]*(@[^@\s]+)")
_DIGIT_RE = re.compile(r"\d")
# Opaque mixed alphanumeric identifiers (post digit-masking they still leak
# letter structure, so hash them). Pure digit runs keep their masked shape.
_LONG_TOKEN_RE = re.compile(r"(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*9)[A-Za-z0-9]{16,}")


def mask_pii(value: object) -> str:
    """Mask PII-shaped content while keeping the *shape* of the value.

    ``asha.voskette@example.com`` -> ``a***@example.com``; digits become
    ``9``; long opaque tokens are hashed to a stable placeholder.
    """
    text = str(value)
    text = _EMAIL_RE.sub(lambda m: f"{m.group(1)}***{m.group(2)}", text)
    text = _DIGIT_RE.sub("9", text)
    return _LONG_TOKEN_RE.sub(lambda m: f"tok_{sha256_text(m.group(0))[:8]}", text)


@dataclass(frozen=True)
class TeacherTaskSpec:
    """One allowed teacher task: purpose, prompt template, output schema."""

    name: str
    intended_use: str
    prompt_file: str
    schema: str
    max_batch: int = 32


TASKS: dict[str, TeacherTaskSpec] = {
    spec.name: spec
    for spec in (
        TeacherTaskSpec(
            "realism_direction", "realism_direction", "realism.txt", "RealismConfig"),
        TeacherTaskSpec(
            "column_role_labeling", "column_role_labeling",
            "column_role.txt", "ColumnRoleLabel"),
        TeacherTaskSpec(
            "context_paraphrase", "context_paraphrase",
            "paraphrase.txt", "ContextParaphraseBatch"),
        TeacherTaskSpec(
            "ambiguity_adjudication", "ambiguity_adjudication",
            "ambiguity.txt", "AmbiguityJudgment"),
        TeacherTaskSpec(
            "rationale_templates", "rationale_templates",
            "rationale.txt", "RationaleTemplateBatch"),
        TeacherTaskSpec(
            "red_teaming", "red_teaming", "redteam.txt", "RedTeamCaseBatch"),
    )
}


def _assert_no_full_rows(items: Sequence[dict[str, Any]]) -> None:
    """Refuse batches that smuggle whole records to the teacher."""
    for item in items:
        if len(item) > 6:
            raise ValueError(
                "teacher task items must be small (column + masked samples), "
                f"got {len(item)} fields — this looks like a full row"
            )
        samples = item.get("masked_samples", [])
        if isinstance(samples, list) and len(samples) > 10:
            raise ValueError("at most 10 masked sample values may be sent per item")


def build_prompt(spec: TeacherTaskSpec, items: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """Return ``(prompt_text, template_sha)`` for one batch."""
    template = (PROMPTS_DIR / spec.prompt_file).read_text(encoding="utf-8")
    payload = json.dumps(list(items), indent=2, sort_keys=True, ensure_ascii=False)
    return template.replace("{{ITEMS}}", payload), sha256_text(template)


def run_task(
    task_name: str,
    items: Sequence[dict[str, Any]],
    *,
    client: TeacherClient | None = None,
    cache: TeacherCache | None = None,
    ledger_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Run one teacher task batch and return validated payloads.

    Empty list means "teacher unavailable or response invalid" — callers must
    treat that as *no teacher data*, never as an error, so every consumer has
    an offline fallback path.
    """
    spec = TASKS.get(task_name)
    if spec is None:
        raise KeyError(f"unknown teacher task {task_name!r}; known: {sorted(TASKS)}")
    _assert_no_full_rows(items)

    client = client if client is not None else default_client()
    cache = cache if cache is not None else TeacherCache()
    ledger_kwargs = {"ledger_path": ledger_path} if ledger_path is not None else {}
    entry = compliance.require_approved(client.provider, spec.intended_use, **ledger_kwargs)

    payloads: list[dict[str, Any]] = []
    for start in range(0, len(items), spec.max_batch):
        batch = list(items[start:start + spec.max_batch])
        prompt, template_sha = build_prompt(spec, batch)
        key = CacheKey(
            provider=client.provider,
            model=client.model,
            prompt_template_sha256=template_sha,
            input_sha256=sha256_json(batch),
            schema_version=f"{spec.schema}:{SCHEMA_VERSION}",
        )
        cached = cache.get(key)
        if cached is not None:
            payloads.extend(cached.get("payloads", []))
            continue
        try:
            response = client.complete(prompt, schema=spec.schema)
            parsed = json.loads(response)
            if not isinstance(parsed, list):
                raise SchemaError(f"{spec.schema}: response must be a JSON array")
            validated = validate_batch(spec.schema, parsed)
        except (TeacherUnavailable, SchemaError, json.JSONDecodeError):
            # Degrade: no teacher data for this batch, pipeline continues.
            continue
        cache.put(
            key,
            prompt=prompt,
            response=response,
            payloads=validated,
            terms_snapshot_id=str(entry.get("terms_snapshot_id", "")),
        )
        payloads.extend(validated)
    return payloads


def column_role_items(
    columns: Sequence[str], samples: dict[str, Sequence[object]], *, max_samples: int = 8
) -> list[dict[str, Any]]:
    """Build masked column-role items from column names + sample values."""
    return [
        {
            "column_name": str(col),
            "masked_samples": [mask_pii(v) for v in list(samples.get(col, []))[:max_samples]],
        }
        for col in columns
    ]
