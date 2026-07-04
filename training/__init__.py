"""FreshData development-time training pipeline (Phase 5).

This package builds, evaluates, and packages FreshData's small local model
artifacts (``fd-col-encoder-v1``, ``fd-intent-v1``, ``calib-v1``). It is a
**developer tool**, not part of the library:

- it lives outside ``src/freshdata`` and is never shipped in the wheel;
- the runtime never imports it (guarded by tests);
- teacher models are allowed here only for development-time labeling,
  paraphrasing, ambiguity adjudication, red-teaming, and rationale templates —
  never at runtime, never per-cell, never as release-gating ground truth;
- ground truth comes from corruption metadata and human-verified labels.

Entry points (see ``training/Makefile`` / root ``Makefile``)::

    make training-seed
    make training-corrupt
    make training-teacher-labels
    make training-distill
    make training-eval
    make training-package-artifacts
    make training-dev-artifacts
    make training-release-artifacts
"""

from __future__ import annotations

__all__ = ["__phase__"]

__phase__ = 5
