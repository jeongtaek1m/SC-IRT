"""Repository-root resolution.

Every path in this package is derived from the repository root so the tree can
be cloned or moved without editing code. `SCIRT_REPRO` overrides the default,
which is the parent of the directory holding this file.
"""

import os

BASE = os.environ.get(
    "SCIRT_REPRO",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

DATA = os.path.join(BASE, "data")
MATRICES = os.path.join(DATA, "matrices")
FEATURES = os.path.join(DATA, "features")
INTERACT = os.path.join(DATA, "interact")
RESULTS = os.path.join(BASE, "results")


def ensure_results():
    """Create the results directory. Experiments call this before writing."""
    os.makedirs(RESULTS, exist_ok=True)
    return RESULTS


def result(name):
    """Absolute path of a generated artifact under results/."""
    return os.path.join(RESULTS, name)
