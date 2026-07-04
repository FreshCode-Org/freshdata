# Seed corpus

Inputs the training pipeline is allowed to learn from. Governance:

- **`registry.json`** — every source, with license, attribution, explicit
  `allowed_for_training`, and resolved `pii_risk` (`none | synthetic |
  review_required`). A `review_required` entry blocks the release pipeline
  until it carries an approving `legal_review` record.
- **`LICENSES.md`** — human-readable ledger; license texts in `licenses/`.
- **`sources/`** — small vendored files (hand-authored fixtures only).
- **`synthetic/`** — deterministic generators for PII-shaped fake data
  (emails, Indian phone numbers, names, addresses, postal codes, IDs,
  statuses, dates, revenue/currency strings, quantities with units,
  city/state/country). No real people, no production data; every record is
  tagged `synthetic=true`.

Validate with:

```bash
python -m training.datasets.validators --check-licenses
```

Never add: unclear-license data, share-alike data without recorded legal
approval, scraped private data, customer data, production logs, or real PII.
