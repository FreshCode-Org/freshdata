# Seed corpus license ledger

Every seed source used for training must appear in `registry.json` with an
explicit license, attribution, `allowed_for_training`, and a resolved PII risk.
`python -m training.datasets.validators --check-licenses` enforces this and is
part of `make training-release-artifacts`.

## Current sources

| source_id | license | PII risk | commercial use | notes |
|---|---|---|---|---|
| `synthetic_core_v1` | CC0-1.0 (generated in-repo) | synthetic | yes | Fully synthetic; no real people, no production data. |
| `handauthored_ecommerce_v1` | CC0-1.0 | none | yes | Hand-written fixture rows. |
| `context_golden_phase1_v1` | Apache-2.0 | none | yes | Phase-1 golden context corpus from this repository. |

## Allowed license classes

- Public domain / CC0 / PDDL
- CC-BY-4.0 (attribution recorded in the registry)
- Permissive software licenses (Apache-2.0, MIT, BSD) for in-repo fixtures
- OGL / government open data with explicit commercial-use permission
- Fully synthetic data generated in this repository

## Never allowed

- Unclear-license or license-missing datasets
- Share-alike licenses (CC-BY-SA, ODbL, GPL-family data) **unless** a recorded
  legal review approves the specific use (`legal_review.approved: true` on the
  registry entry)
- Scraped private data, customer data, production logs
- Any dataset containing real PII

License texts are vendored under `training/seed/licenses/`.
