"""Feature loading. Only the kinematic auxiliary input ships with this release."""

import numpy as np

from . import paths

def load_st(name):
    """Load a `stats`-format descriptor npz into {route_id: float32 vector}."""
    d = np.load(f"{paths.FEATURES}/{name}.npz", allow_pickle=True)
    names = [str(x).replace("route_", "") for x in d["names"]]
    return {names[i]: d["stats"][i].astype(np.float32) for i in range(len(names))}
