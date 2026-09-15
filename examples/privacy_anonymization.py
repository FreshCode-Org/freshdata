"""Stronger PII detection + reversible / format-preserving anonymization.

Demonstrates the dependency-free detector, reversible tokenization with a vault,
surrogate format-preserving anonymization, HIPAA/GDPR-tagged audit events, and a
k-anonymity check. Run:

    python examples/privacy_anonymization.py
"""

import os

import pandas as pd

import freshdata as fd
from freshdata.enterprise import (
    InMemoryTokenVault,
    MaskingRule,
    PIIDetectionConfig,
    detokenize_value,
    tokenize_value,
)

#: Keyed strategies read their secret from this environment variable. Without a
#: key they use a random per-call key, so output would change on every run.
KEY_ENV = "FRESHDATA_DEMO_KEY"


def main() -> None:
    # A throwaway demo key when none is set (or it is empty); use a real secret
    # in production.
    if not os.environ.get(KEY_ENV):
        os.environ[KEY_ENV] = "DEMO-KEY-DO-NOT-COMMIT"
    df = pd.DataFrame(
        {
            "patient_note": [
                "Patient MRN# 4451234 dob 1984-02-11, email j.doe@mail.com",
                "Contact 555-12-3456 (SSN), card 4111 1111 1111 1111",
            ],
            "email": ["j.doe@mail.com", "a.smith@mail.com"],
            "zip": ["90210", "90210"],
            "gender": ["M", "F"],
        }
    )

    # 1) Detect PII in free text (raw spans are redacted by default).
    scan = fd.detect_pii(df, config=PIIDetectionConfig())
    print("== detection ==")
    print(scan.summary(), "\n")

    # 2) Anonymize: tokenize the email column reversibly + scrub free text spans.
    rules = (
        MaskingRule(
            name="email_token",
            columns=("email",),
            strategy="tokenize",
            reversible=True,
            key_env=KEY_ENV,
            entity_types=("EMAIL",),
        ),
    )
    out, report = fd.anonymize(df, rules=rules, detection_config=PIIDetectionConfig())
    print("== anonymized frame ==")
    print(out[["patient_note", "email"]].to_string(index=False), "\n")
    print(report.summary())
    ev = report.events[0]
    print(f"sample event tags: hipaa={ev.hipaa_tag} gdpr={ev.gdpr_tag} risk={ev.risk_level}")
    print("report leaks raw PII? ", "j.doe@mail.com" in report.to_json(), "\n")

    # 3) Reversible tokenization round-trip via a vault.
    vault = InMemoryTokenVault()
    token = tokenize_value("j.doe@mail.com", vault, os.environ[KEY_ENV])
    print(f"token={token}  ->  detokenized={detokenize_value(token, vault)}\n")

    # 4) Surrogate format-preserving anonymization keeps the shape.
    fpe_rule = MaskingRule(
        name="ssn_fpe", columns=("zip",), strategy="surrogate", preserve_format=True,
        key_env=KEY_ENV,
    )
    masked, _ = fd.anonymize(df, rules=(fpe_rule,))
    print("surrogate zip:", masked["zip"].tolist(), "(same length, not crypto FPE)\n")

    # 5) k-anonymity over quasi-identifiers.
    k = fd.check_k_anonymity(df, ["zip", "gender"], k=2)
    print("== k-anonymity ==")
    print(k.summary())


if __name__ == "__main__":
    main()
