"""Cross-source entity linking with ``fd.link``.

Link customer records across two systems (CRM vs. billing), explain every
candidate match, and hand low-confidence pairs to a steward-review queue.
FreshData is the candidate-generation + preprocessing layer here — it formats
and explains matches; it does not replace a dedicated matcher (use the external
adapter hook to plug one in). Run:

    python examples/cross_source_link.py
"""

import pandas as pd

import freshdata as fd
from freshdata.enterprise import build_review_queue


def main() -> None:
    crm = pd.DataFrame(
        {
            "full_name": ["Acme Corp", "Globex LLC", "Initech Inc", "Stark Industries"],
            "zip": ["10001", "94105", "73301", "10002"],
        }
    )
    billing = pd.DataFrame(
        {
            "full_name": ["acme corporation", "Globex L.L.C.", "Umbrella Co", "Stark Ind."],
            "zip": ["10001", "94105", "60601", "10002"],
        }
    )

    # Fuzzy link: block on zip (cheap, exact), Jaro-Winkler compare the name.
    report = fd.link(
        crm, billing,
        keys=["full_name"],
        strategy="fuzzy",
        threshold=0.82,
        blocking="l.zip = r.zip",
    )
    print(report.summary())
    print("\ncandidate pairs (with per-field explanations):")
    print(report.to_frame()[["left_id", "right_id", "match_probability", "decision"]]
          .to_string(index=False))

    # Hand the borderline pairs to a steward-review queue.
    queue = build_review_queue(report)
    print(f"\nsteward review queue: {len(queue.items)} item(s) need a human decision")


if __name__ == "__main__":
    main()
