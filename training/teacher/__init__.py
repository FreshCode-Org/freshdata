"""Development-time teacher harness.

Frontier teacher models are allowed **only** here, **only** for:

1. realism direction (corruption frequency/parameter tuning),
2. column-role labeling,
3. NL context paraphrase generation,
4. ambiguity adjudication,
5. rationale style templates,
6. benchmark red-teaming.

They are never called at runtime, never see full rows or real PII, never
generate messy/clean pairs from scratch, and their outputs are training data —
release-gating truth requires human verification (see ``review.py``).
"""
