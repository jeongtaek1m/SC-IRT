"""Official-code baselines for Table 1 (UP): the published methods run through
THEIR OWN implementations on the ATDrive protocol.

Every wrapper in this package calls the original implementation (cloned under
OFFICIAL_ROOT; commit ids are written to results/up_official_provenance.json)
for its item-parameter fit, its item selection and its score estimate, and
adapts only the data plumbing. The protocol is the one of
experiments/run_up_frontier.py: 16 draws, 12 : 4 calibration : evaluation
planners, K_cal in {4, 8, 12}, budgets {30, 55, 110, 165}, and the bank = the
evaluation planner's recorded routes (210-220). Nothing is shared with
ATDrive's own calibration.

Contract (one object per method; see the wrappers for the per-method notes)

  fit(R, seed, workdir) -> model
      R : (K_cal, n_bank) float, the CALIBRATION planners' outcomes on the bank
          routes, 1 = pass, 0 = fail, NaN = no record. This is the only
          training data a method sees. `model` is a picklable dict.
  estimate(model, y, budgets, seed) -> {B: {'est': float, 'items': [int, ...],
                                             'variants': {name: float}}}
      y : (n_bank,) 0/1 outcomes of the EVALUATION planner. A method may read y
          ONLY at the items it selects: fixed-subset methods select first and
          then read, adaptive methods read one item at a time. 'est' is the
          method's own estimate of mean(y) (its native, published readout),
          'items' the bank indices it read, in the order it read them (len == B
          unless the method cannot fill the budget — then say so in 'note'),
          'variants' every other readout the official code offers (e.g.
          tinyBenchmarks: anchor-weighted, p-IRT, gp-IRT), keyed by name.
  stop(model, y, seed) -> {rule: {'est': float, 'items': [...]}}   (optional)
      for methods that publish their own stopping rule (ATLAS: SE <= tau).

Leakage rule: `y` is the evaluation planner's full outcome vector for
convenience; the audit checks that each wrapper indexes it only at 'items'.
`LeakGuard` wraps y so that any read outside the selected items raises.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

OFFICIAL_ROOT = Path('/data2/jeongtae/official_baselines')
OFFICIAL_PY = Path('/data2/jeongtae/envs/atdrive_official/bin/python')
OFFICIAL_RSCRIPT = Path('/data2/jeongtae/envs/r_metabench/bin/Rscript')
REPOS = {'tinyBenchmarks': 'felipemaiapolo/tinyBenchmarks', 'fluid-benchmarking': 'allenai/fluid-benchmarking',
         'AnchorPoints': 'rvivek3/AnchorPoints', 'disco-public': 'arubique/disco-public',
         'metabench': 'adkipnis/metabench', 'ATLAS': 'Peiyu-Georgia-Li/ATLAS'}


def provenance():
    """Commit id of every cloned official repository."""
    out = {}
    for d, gh in REPOS.items():
        p = OFFICIAL_ROOT / d
        try:
            h = subprocess.run(['git', '-C', str(p), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            h = None
        out[d] = {'github': gh, 'commit': h}
    return out


class LeakGuard:
    """A view of the evaluation planner's outcomes that records every index
    read and refuses reads outside an explicitly opened set."""

    def __init__(self, y):
        self._y = np.asarray(y, float)
        self.read = []
        self._open = None            # None = every single-index read is allowed and logged

    def __len__(self):
        return len(self._y)

    def __getitem__(self, i):
        idx = np.atleast_1d(np.asarray(i))
        if idx.dtype == bool:
            idx = np.where(idx)[0]
        for j in idx.tolist():
            if self._open is not None and j not in self._open:
                raise RuntimeError(f'leak: item {j} read outside the selected set')
            self.read.append(int(j))
        return self._y[i]

    def values_at(self, items):
        """Explicit read of a selected subset (fixed-subset methods)."""
        items = [int(i) for i in items]
        self.read.extend(items)
        return self._y[items]


class OfficialMethod:
    name = 'abstract'
    adaptive = False          # True: reads y one item at a time (prefix property holds)

    def fit(self, R, seed, workdir):
        raise NotImplementedError

    def estimate(self, model, y, budgets, seed):
        raise NotImplementedError

    def stop(self, model, y, seed):
        return {}


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, 'w'), indent=1, default=float)


def run_rscript(script, args, cwd):
    """Run an R script in the official R environment; raise with its stderr on failure."""
    r = subprocess.run([str(OFFICIAL_RSCRIPT), str(script)] + [str(a) for a in args],
                       capture_output=True, text=True, cwd=str(cwd))
    if r.returncode != 0:
        raise RuntimeError(f'Rscript {Path(script).name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}')
    return r.stdout
