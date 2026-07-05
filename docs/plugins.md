---
title: Writing plugins
description: >-
  Extend freshdata with custom semantic experts, semantic backends, and
  validators — via entry points or explicit registration — without ever
  bypassing the policy gate or protected-column guard.
keywords: freshdata plugins, custom expert, custom validator, entry points
---

# Writing plugins

FreshData has **one** plugin mechanism for three extension points:

| Kind | What it does | Runs in |
|------|--------------|---------|
| **expert** | proposes value repairs for **one column** (distinct value → repair) | `fd.clean(..., semantic_mode=...)` |
| **backend** | proposes over the **whole frame**; opted into by name | `fd.clean(..., semantic_backends=(...))` |
| **validator** | **read-only** checks that append findings | `fd.validate(...)` |

The hard rule: **a plugin can only propose or validate.** It never touches the
DataFrame. Expert/backend proposals flow through the same policy gate
(`freshdata.semantic.policy.decide`) and the executor's byte-identity guard as
the built-ins, so a plugin can never change a protected (id / target /
`preserve`) column, and never force an auto-apply. Validators are read-only.

## 1. Registering a plugin

Two ways, same registry.

**Explicit** (scripts, notebooks):

```python
import freshdata as fd
from my_pkg import MyExpert

fd.testing.expert_contract(MyExpert())   # verify the contract first
fd.register_expert(MyExpert())
```

`fd.register_backend(...)` and `fd.register_validator(...)` work the same way.

**Entry points** (installed packages — discovered automatically):

```toml
# your package's pyproject.toml
[project.entry-points."freshdata.experts"]
my_expert = "my_pkg:MyExpert"

[project.entry-points."freshdata.backends"]
my_backend = "my_pkg:MyBackend"

[project.entry-points."freshdata.validators"]
my_validator = "my_pkg:MyValidator"
```

The value points at a class (or any zero-arg factory) that FreshData
instantiates on first use. A broken entry point is logged and skipped — it can
never break `import freshdata`.

Introspect what's registered:

```python
fd.registered_plugins()          # every plugin, active and inactive
fd.registered_plugins("expert")  # just experts
```

## 2. The proposal / finding schema

- Experts and backends build proposals with
  `freshdata.semantic.scoring.make_proposal(...)`, which returns a
  `SemanticProposal` (scores confidence and derives risk for you).
- Validators build `freshdata.QualityFinding.create(...)` records.

Anything else you return is dropped with a warning — the registry validates the
output type so a mistake in a plugin can't corrupt the report.

## 3. Required metadata

Every plugin declares:

| Attribute | Meaning | Default if omitted |
|-----------|---------|--------------------|
| `name` | unique registry name (also the report attribution) | class name |
| `issue_type` (experts) | scoring/risk profile to reuse | — |
| `semantic_types` | column kinds the plugin claims (declaration) | `()` |
| `max_risk` | ceiling: `"low"` / `"medium"` / `"high"` | `"high"` |
| `uses_network` | does it make network calls? | `False` |
| `requires` | optional dependency module names | `()` |

Two of these are **enforced**, not advisory:

- **`max_risk` is a hard ceiling.** Any proposal scored above it is dropped
  before the gate sees it. Declare the truth.
- **`uses_network=True` disables the plugin by default.** A network-using
  plugin is registered but *inactive* until you opt in with
  `fd.register_expert(plugin, allow_network=True)` or
  `FRESHDATA_ALLOW_NETWORK_PLUGINS=1`. FreshData's own runtime never calls the
  network; a network plugin is your explicit choice, disclosed in
  `fd.registered_plugins()`.

If `requires` names a package that isn't installed, the plugin is registered but
inactive, with the reason recorded — no `ImportError` at clean time.

## 4. Lifecycle & safety

- **Failures degrade safely.** Every call into a plugin (`applies`, `propose`,
  `validate`, `warm_up`) is wrapped: an exception disables that plugin for the
  run and is logged — the clean still completes.
- **Provenance is automatic.** An applied plugin proposal shows up in the report
  with `model_id="semantic:<issue>:plugin:<name>"` and `metadata["plugin"]` set,
  so a plugin repair is never anonymous in the audit trail.
- **No mutation.** `propose`/`validate` must not modify the series/frame they are
  handed. The contract tests check this.

## 5. Contract tests

Run the matching contract test before you register — it raises `AssertionError`
with an actionable message on the first violation:

```python
fd.testing.expert_contract(MyExpert())
fd.testing.semantic_backend_contract(MyBackend())
fd.testing.validator_contract(MyValidator())
```

They verify the protocol methods and return types, well-formed metadata
(`max_risk` in range, `uses_network` a bool, iterable `requires`/`semantic_types`),
that the plugin does not mutate its input, that experts/backends only propose for
the column/frame they were given, and that nothing exceeds the declared
`max_risk`.

## 6. Full worked examples

Runnable, contract-passing examples live in the repo:

- [`examples/plugins/custom_expert/acronym_expert.py`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/plugins/custom_expert/acronym_expert.py) — expand acronyms in categorical columns
- [`examples/plugins/custom_backend/keyword_backend.py`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/plugins/custom_backend/keyword_backend.py) — strip a shared prefix frame-wide, budget-aware
- [`examples/plugins/custom_validator/min_rows_validator.py`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/plugins/custom_validator/min_rows_validator.py) — flag too-few-rows and constant columns

## 7. Security & privacy expectations

- Keep plugins **deterministic and offline** unless you truly need otherwise;
  network plugins are off by default for a reason.
- A plugin sees your data. Vet third-party plugins the same way you'd vet any
  dependency — they run in-process.
- Plugins **cannot** bypass the gate, the protected-column guard, the risk
  ceiling, or the no-mutation rule. If you find a way to, that's a bug — please
  report it.
