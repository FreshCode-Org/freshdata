# Repair plans: suggest, review, apply, undo

Phase 2 turns freshdata's semantic proposals into an **executable, reviewable
workflow**. Instead of trusting `fd.clean` end-to-end, you can look at exactly
what it wants to do, approve or reject each action, execute only what you
approved, and revert individual repairs afterwards — with a deterministic
audit hash for change control.

Everything below is deterministic, offline, and model-free.

## The ecommerce example, end to end

```python
import pandas as pd
import freshdata as fd

df = pd.DataFrame({
    "cust_id": ["C001", "C002", "C003", "C003"],
    "full_name": ["Asha Rao", "Ravi Kumar", "Neha Iyer", "Neha Iyer"],
    "email_addr": [" Asha@GMAIL.COM ", "ravi @ example.com", "neha@@shop.in", "neha@@shop.in"],
    "mobile": ["98765 43210", "+91 91234 56780", "09123456789", "09123456789"],
    "age": [25, None, None, 31],
    "monthly_revenue": ["1000", "2000", "3000", "3000"],
    "status": [" Active ", "INACTIVE", "pend-ing", "pend-ing"],
})

context = """
This is an ecommerce customer dataset.
CustomerID is unique.
Emails must be valid.
Phone numbers are Indian.
Allowed status values are active, inactive, pending.
Missing Age should be estimated only if confidence >95%.
Never modify revenue values.
"""

clean_df, report = fd.clean(df, context=context, semantic_mode="auto",
                            return_report=True)
```

What happens:

| column | behaviour |
| --- | --- |
| `email_addr` | ` Asha@GMAIL.COM ` → `Asha@gmail.com`, `ravi @ example.com` → `ravi@example.com`, `neha@@shop.in` → `neha@shop.in` (unambiguous mechanical repairs) |
| `mobile` | every safe form → canonical `+91XXXXXXXXXX`; `12345`-style values are flagged, never rewritten |
| `status` | ` Active ` → `active`, `INACTIVE` → `inactive`, `pend-ing` → `pending` (exact after separator/case normalization) |
| `age` | stays missing — every engine fill confidence (≤ 0.9) is below the required 0.95; the held-back fill is recorded as a *suggested* action |
| `monthly_revenue` | **byte-identical**, physically verified; a violation would raise `fd.ProtectedColumnError` |
| `cust_id` | duplicate `C003` is validated and reported (`fd.validate`), not silently dropped |

Every decision lands in the report with confidence, risk, rationale, status,
reversibility, and metadata (`issue_type`, `raw_value`, `proposed_value`,
`expert`, `evidence`, and `region` for phone repairs).

## Suggest → review → apply

```python
plan = fd.suggest_plan(df, context=context, semantic_mode="auto")
rp = plan.repair_plan            # fd.RepairPlan
print(rp.summary())
#   a1: [auto/approved] email_addr email_format ' Asha@GMAIL.COM ' -> 'Asha@gmail.com' ...
#   a7: [suggest/pending] status reference_value 'actve' -> 'active' ...

rp.approve("a7")                 # by action id
rp.approve("email_addr")         # by column
rp.reject("phone_format", reason="numbers come from the CRM")   # by kind
rp.override("a7", {"proposed_value": "inactive"})
rp.approve_all(max_risk="low")   # everything low-risk, not blocked/rejected

clean_df, report = fd.apply_plan(df, rp, keep_undo=True)
```

`apply_plan` executes **exactly** the approved actions:

- nothing is re-profiled or re-decided;
- rejected actions never run; blocked actions (protected columns, identifier
  vetoes) never run *even if approved*;
- the physical protected-column guard runs before the result is returned;
- your input frame is never mutated.

In `semantic_mode="auto"`, low-risk high-confidence actions arrive already
approved (they are what `fd.clean` would have applied). In `"review"` mode
everything non-trivial stays `pending`.

## Drift refusal

A plan remembers the frame it was built for (`FrameSignature`: row count,
column names+dtypes, content sample). Applying it to different data refuses
by default:

```python
fd.apply_plan(other_df, rp)                  # raises fd.PlanDriftError
fd.apply_plan(other_df, rp, allow_drift=True)  # applies; stale actions skipped + recorded
```

## Undo

```python
clean_df, report = fd.apply_plan(df, rp, keep_undo=True)
restored = report.revert(clean_df, action_ids=["a1"])   # undo one action
restored_all = report.revert(clean_df)                   # undo everything reversible
```

Old values are stored compactly (cell positions + the one raw value per
action). The log is capped by `undo_cell_limit` (default 100 000 cells);
actions that don't fit are honestly marked `reversible=False`. Row drops and
aggregations are not cell-reversible and are never claimed to be.

## Audit hash

```python
report.decisions_hash   # sha256 over action ids/kinds/columns/params,
                        # approvals/rejections, policy hash, thresholds
```

The hash is stable across runs for the same reviewed plan and changes when
any decision changes — suitable for change-control records. It also appears
in `report.to_dict()` / the CLI's `--report` JSON.

## Plans as files (CLI)

```bash
freshdata plan in.csv --context-file rules.txt --out plan.json
# ... review/edit plan.json, or pass --approve-all low ...
freshdata apply-plan in.csv --plan plan.json -o out.csv --report audit.json
```

`freshdata clean` also gained `--semantic-mode {off,assist,review,auto}`.

Plans serialize losslessly: `RepairPlan.to_json()` / `RepairPlan.from_json()`
round-trip actions, approval state, the compiled policy, and the frame
signature.

## What this is not

See [limitations.md](limitations.md): no model, no embeddings, no network —
only the deterministic tier-0 context language from Phase 1, and ambiguous
repairs are suggested or flagged, never auto-applied.
