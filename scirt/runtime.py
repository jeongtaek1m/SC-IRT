"""Pinned runtime: CPU + 1 thread + float32.

Published numbers are float32 optimisation results, so device and thread count
are part of the experiment (CPU vs CUDA disagree in the third decimal).
Override with SCIRT_DEVICE / SCIRT_THREADS. Details: REPRODUCIBILITY.md.
"""

import os

import numpy as np
import torch

#: Torch device every experiment runs on. Override with SCIRT_DEVICE=cuda to
#: reproduce a GPU run, but expect third-decimal differences from the shipped
#: reference outputs.
DEVICE = torch.device(os.environ.get("SCIRT_DEVICE", "cpu"))

#: Dtype of every tensor and every feature vector. Promotion to float64 changes
#: all reported values in the third decimal.
DTYPE = torch.float32

_THREADS = int(os.environ.get("SCIRT_THREADS", "1"))


def configure():
    """Pin threads and dtype. Idempotent; call once before any other library call."""
    torch.set_num_threads(_THREADS)
    torch.set_default_dtype(DTYPE)


def set_global_seeds(seed=0):
    """Seed the global numpy and torch generators.

    These are inert today: every parameter is zero-initialised, Adam is
    deterministic, and all sampling goes through local `RandomState` instances.
    They are kept because the moment any stochastic torch op enters the code
    their absence becomes a silent reproducibility bug.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
