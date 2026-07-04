"""CI-safe mini fixtures for CleanBench tiers T2 and T3.

Fixtures are generated in code (no data files), deterministic, and small
enough to run in every CI job. Each builder returns
``(truth, corrupted, kwargs)`` where *kwargs* are the exact arguments to pass
to ``freshdata.clean`` — so a benchmark run is literally
``fd.clean(corrupted, **kwargs)``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..corruptors import (
    allowed_value_casing,
    allowed_value_punctuation,
    email_at_spacing,
    email_double_at,
    phone_format_corruption,
    whitespace_corruption,
)

#: The Phase-2 reference context (mirrors the docs' ecommerce example).
ECOMMERCE_CONTEXT = """\
This is an ecommerce customer dataset.
CustomerID is unique.
Emails must be valid.
Phone numbers are Indian.
Allowed status values are active, inactive, pending.
Missing Age should be estimated only if confidence >95%.
Never modify revenue values.
"""

_FIRST = ("asha", "ravi", "neha", "vikram", "priya", "arjun", "meera", "kiran",
          "divya", "rohan", "sneha", "amit")
_DOMAINS = ("gmail.com", "example.com", "shop.in", "mail.org")
_STATUS = ("active", "inactive", "pending")


def _base_frame(n: int = 24) -> pd.DataFrame:
    rows = []
    for i in range(n):
        name = _FIRST[i % len(_FIRST)]
        rows.append(
            {
                "cust_id": f"C{i + 1:03d}",
                "email_addr": f"{name}{i}@{_DOMAINS[i % len(_DOMAINS)]}",
                "mobile": f"+91{9000000000 + i * 137}",
                "age": 20.0 + (i % 40),
                "monthly_revenue": str(1000 + 250 * (i % 8)),
                "status": _STATUS[i % len(_STATUS)],
            }
        )
    return pd.DataFrame(rows)


def make_t2_semantic_fixture(seed: int = 0):
    """T2: email / phone / reference-list value repairs.

    Only the three semantic columns are corrupted, so every other cell must
    survive untouched (that is what the false-modification gate measures).
    """
    truth = _base_frame()
    corrupted = email_double_at(truth, ["email_addr"], seed=seed, share=0.4)
    corrupted = email_at_spacing(corrupted, ["email_addr"], seed=seed + 1, share=0.3)
    corrupted = whitespace_corruption(corrupted, ["email_addr"], seed=seed + 2, share=0.3)
    corrupted = phone_format_corruption(corrupted, ["mobile"], seed=seed + 3, share=0.7)
    corrupted = allowed_value_casing(corrupted, ["status"], seed=seed + 4, share=0.5)
    corrupted = allowed_value_punctuation(corrupted, ["status"], seed=seed + 5, share=0.4)
    kwargs = {
        "context": ECOMMERCE_CONTEXT,
        "semantic_mode": "auto",
        "verbose": False,
        "drop_duplicates": False,
        "reset_index": False,
    }
    return truth, corrupted, kwargs


def make_t3_context_fixture(seed: int = 10):
    """T3: context compliance — protection, thresholds, allowed values, uniqueness.

    The revenue column is corrupted *on purpose*: a compliant cleaner must
    leave the corruption in place (protected means protected), and the
    duplicate ``cust_id`` must be surfaced by validation, not silently fixed.
    """
    truth = _base_frame()
    corrupted = truth.copy(deep=True)
    # Protected column: whitespace dirt that must NOT be repaired.
    corrupted.loc[0, "monthly_revenue"] = " 1000 "
    corrupted.loc[5, "monthly_revenue"] = "2,250"
    # Age gaps that must stay missing (threshold 0.95 beats every fill).
    corrupted.loc[[2, 7, 11], "age"] = None
    # Allowed-value dirt.
    corrupted = allowed_value_casing(corrupted, ["status"], seed=seed, share=0.5)
    # Uniqueness violation to be validated, never dropped.
    corrupted.loc[3, "cust_id"] = corrupted.loc[2, "cust_id"]
    kwargs = {
        "context": ECOMMERCE_CONTEXT,
        "semantic_mode": "auto",
        "verbose": False,
        "drop_duplicates": False,
        "reset_index": False,
    }
    return truth, corrupted, kwargs


class BigramStubEncoder:
    """Deterministic character-bigram encoder for the Phase-3 mini-suite.

    Model quality is out of scope for CleanBench's plumbing gates; what the
    Phase-3 fixtures need is a *reproducible* similarity signal where a
    distance-2 typo of an allowed value lands close to it and unrelated values
    land far apart. Character bigrams over sha256-hashed axes give exactly
    that with no model files and no randomness.
    """

    model_id = "fd-col-encoder-v1"
    model_sha256 = "cleanbench-bigram-stub"
    dim = 256

    def encode_texts(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            padded = f"^{str(text).strip().casefold()}$"
            grams = [padded[j : j + 2] for j in range(len(padded) - 1)] or [padded]
            vector = np.zeros(self.dim, dtype=np.float32)
            for gram in grams:
                digest = hashlib.sha256(gram.encode("utf-8")).digest()
                vector[digest[0]] += 1.0
                vector[128 + digest[1] % 128] += 1.0
            norm = float(np.linalg.norm(vector))
            out[i] = vector / norm if norm else vector
        return out


def make_phase3_embedding_fixture(seed: int = 20):
    """Phase 3: embedding-assisted reference repairs with calibrated confidence.

    Corruptions are chosen around the deterministic ReferenceExpert's
    edit-distance tolerance (1):

    - casing/punctuation dirt -> deterministic repairs (high raw confidence);
    - distance-2 typos of allowed status values -> only the embedding backend
      can rescue them, with calibrated confidence and model evidence;
    - one deliberately ambiguous value (``"nactive"`` is 1 edit from both
      ``active`` and ``inactive``) that must never be auto-applied by anyone;
    - protected ``monthly_revenue`` dirt that must survive byte-identical.
    """
    truth = _base_frame()
    corrupted = truth.copy(deep=True)
    # Deterministic-reach dirt for calibration spread.
    corrupted = allowed_value_casing(corrupted, ["status"], seed=seed, share=0.4)
    corrupted = allowed_value_punctuation(corrupted, ["status"], seed=seed + 1, share=0.3)
    corrupted = email_double_at(corrupted, ["email_addr"], seed=seed + 2, share=0.3)
    # Distance-2 typos (beyond ReferenceExpert's tolerance of 1) planted after
    # the random corruptors so nothing re-mangles them.
    corrupted.loc[0, "status"] = "activvee"  # truth: active
    corrupted.loc[2, "status"] = "penddingg"  # truth: pending
    # Ambiguous: 1 edit from both "active" and "inactive"; truth row is active.
    corrupted.loc[3, "status"] = "nactive"
    # Protected column dirt that a compliant cleaner must leave alone.
    corrupted.loc[1, "monthly_revenue"] = " 1250 "
    kwargs = {
        "context": ECOMMERCE_CONTEXT,
        "semantic_mode": "auto",
        "semantic_backends": ("deterministic", "memory", "embedding"),
        "verbose": False,
        "drop_duplicates": False,
        "reset_index": False,
    }
    return truth, corrupted, kwargs
