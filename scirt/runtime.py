"""Pinned runtime configuration.

Every reported number in this package is a float32 optimisation result, so the
execution environment is part of the experiment. Two settings are pinned here
rather than discovered at run time:

* **Device.** The original scripts selected `cuda if torch.cuda.is_available()
  else cpu`, which makes every published number a property of the runner's
  hardware: CPU and CUDA float32 Adam reductions disagree by up to 2.1e-3 in the
  fitted difficulty vector. On a 219x16 panel the GPU also buys nothing. The
  device is therefore pinned to CPU unless `SCIRT_DEVICE` says otherwise.
* **Thread count.** Multi-threaded CPU reductions sum in nondeterministic order.
  One thread is slower in principle and not measurably so at this problem size.

See REPRODUCIBILITY.md for the measured cross-device deltas.
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


def describe():
    """One-line environment summary, printed into run logs for provenance."""
    return (
        f"device={DEVICE} threads={torch.get_num_threads()} "
        f"torch={torch.__version__} numpy={np.__version__}"
    )
