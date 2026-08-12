"""Experimental freshdata features — APIs here may change between minor releases.

Modules in :mod:`freshdata.experimental` are shipped so users can try new
ideas early, but they sit outside the stability guarantees of the main
package: signatures, defaults, and report shapes may evolve based on
feedback. Everything here is offline-first and deterministic by default —
nothing in this package calls a network service unless you explicitly pass
a provider hook.

Current members
---------------
:mod:`freshdata.experimental.ai_copilot`
    Deterministic, privacy-first dataset analysis that produces an
    explainable cleaning plan and copy-ready freshdata code.

See ``ARCHITECTURE.md`` for how this package fits into the overall cleaning flow.
"""

from __future__ import annotations

__all__ = ["ai_copilot"]
