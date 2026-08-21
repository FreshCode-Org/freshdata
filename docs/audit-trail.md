---
title: Audit-trail JSON and schema
description: >-
  Export, persist, and validate FreshData CleanReport audit payloads with the
  JSON Schema shipped in every installation.
keywords: freshdata audit trail, CleanReport JSON Schema, data cleaning audit log
---

# Audit-trail JSON and schema

Every cleaning run can return a [`CleanReport`](api-reference.md) containing the
ordered actions, affected counts, rationale, risk, confidence, warnings, and
execution metadata. The report exposes the same stable payload as a dictionary
or JSON text:

```python
import freshdata as fd

cleaned, report = fd.clean(df, return_report=True)

payload = report.to_dict()
report.write_json("audit/run-2026-08-21.json", indent=2)
```

## Validate an exported report

FreshData ships a Draft 2020-12 JSON Schema inside the package. Load a fresh
copy with `CleanReport.to_json_schema()` and pass it to any compatible
validator:

```python
from jsonschema import validate

schema = fd.CleanReport.to_json_schema()
validate(instance=payload, schema=schema)
```

`jsonschema` is a development or application dependency, not a FreshData core
dependency. Install it separately when validation is part of your pipeline:

```bash
pip install jsonschema
```

The schema requires the core report summary and action fields, constrains known
enums such as action risk and status, and documents optional backend, streaming,
domain, provenance, contract, and profile-replay sections. The helper only reads
the bundled schema, so it works offline and does not add validation overhead to
normal cleaning runs.
