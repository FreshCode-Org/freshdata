"""Provenance-aware cleaning of a document-extracted table (``source_provenance=``).

FreshData is not a PDF parser — it is the post-extraction normalization and audit
layer. Pass the per-column provenance your extractor emitted and FreshData warns
when it coerces a *low-confidence* field, so a reviewer can tell a trustworthy
repair from one that silently "fixed" a mis-read cell. Run:

    python examples/document_provenance.py
"""

import pandas as pd

import freshdata as fd


def main() -> None:
    # A table lifted from a scanned invoice. 'amount' was read with low OCR
    # confidence; 'invoice_no' and 'vendor' came through clean.
    extracted = pd.DataFrame(
        {
            "invoice_no": ["INV-001", "INV-002", "INV-003"],
            "vendor": ["Acme Corp", "Globex LLC", "Initech Inc"],
            "amount": ["$1,200.00", "$3,400.00", "N/A"],  # currency text + a sentinel
        }
    )

    provenance = {
        "amount": {
            "parser_confidence": 0.52,
            "source_file": "invoice_batch_2024_05.pdf",
            "page": 7,
            "region": "line-items-table",
            "extracted_at": "2024-05-02T08:15:00Z",
        },
        "vendor": {"parser_confidence": 0.97, "source_file": "invoice_batch_2024_05.pdf"},
        "invoice_no": {"parser_confidence": 0.99, "source_file": "invoice_batch_2024_05.pdf"},
    }

    cleaned, report = fd.clean(
        extracted, source_provenance=provenance, return_report=True
    )

    print(report.summary())
    print("\nlow-confidence repairs flagged for review:")
    for col, meta in report.source_provenance.items():
        flag = "⚠ REVIEW" if meta["low_confidence_repair"] else "ok"
        conf = meta["parser_confidence"]
        print(f"  {col:<12} confidence={conf}  modified={meta['modified']}  -> {flag}")


if __name__ == "__main__":
    main()
