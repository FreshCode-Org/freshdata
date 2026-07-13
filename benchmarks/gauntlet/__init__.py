"""FreshData Validation Gauntlet.

Gold-labelled adversarial fixtures plus a harness that measures how
FreshData's validation surfaces (``fd.clean``, ``fd.validate_fields``,
``fd.clean_text``, domain packs, the semantic layer, PII detection) treat
each labelled cell: preserve, repair, flag, or route to review.

Unlike CleanBench (which scores whole-frame repair fidelity against a clean
oracle), the gauntlet scores *dispositions*: every injected defect carries the
disposition FreshData should choose, and every adversarial trap is a valid
value that must survive cleaning untouched.

Run ``python -m benchmarks.gauntlet run`` from the repo root.
"""

from .fixtures import FIXTURES, GauntletFixture, GoldCell, build_fixture
from .metrics import compute_metrics
from .runner import run_fixture, run_gauntlet

__all__ = [
    "FIXTURES",
    "GauntletFixture",
    "GoldCell",
    "build_fixture",
    "compute_metrics",
    "run_fixture",
    "run_gauntlet",
]
