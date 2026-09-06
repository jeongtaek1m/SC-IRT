"""ATLAS (Li et al., 2025) run through the OFFICIAL code,
/data2/jeongtae/official_baselines/ATLAS (Peiyu-Georgia-Li/ATLAS).

The whole method lives in R; this module is the plumbing that builds their input
files, starts their pipeline (experiments/official/atlas_cat.R, which parses the
function definitions out of the official scripts and evaluates them verbatim) and
serves the evaluation planner's outcomes one item at a time.

What is called (file:function)
  fit      scripts/01_fit_irt.r:96-97   mirt(dat, 1, itemtype = '3PL', method = 'EM',
           technical = list(NCYCLES = 100000)) on the calibration planners' responses,
           after their constant-column / constant-row cleaning (01:78-85);
           coef(model, simplify = TRUE)$items (01:103) is written as
           irt_item_parameters_combined.csv, the file 03 and 04 read;
           fscores(method = 'EAP', quadpts = 61) (01:101-102) is kept as a diagnostic.
  select   scripts/03_atlas_cat.r:142-280 run_atlas, sourced verbatim: item 1 drawn
           uniformly among |b - start_theta| < 0.5 (03:164-176), then
           catR::nextItem(criterion = 'MFI', method = 'EAP', randomesque = 5)
           (03:179-185); bank built by 03:63-113 prepare_item_bank (a = a1,
           b = -d/a1, c = g, d = u).
  ability  inside the same loop: catR::thetaEst(..., method = 'EAP') (03:206-208) and
           catR::semTheta(...) (03:211); the stopping test i >= min_items and
           se <= se_theta_stop (03:222).
  estimate scripts/04_pirt_accuracy.r:76-140 compute_pirt_accuracy with their
           04:61-73 compute_3pl_prob at the final theta and their 04:34-56
           prepare_item_parameters.

Readouts
  est          p-IRT accuracy (04_pirt_accuracy.r), i.e. the observed mean over the
               administered items plus the 3PL probabilities at the final EAP theta
               over the unadministered ones. Their denominator is the FITTED item set
               (ADAPTATION 8).
  variants     'sample_mean'    mean of the administered outcomes (04's avg_observed),
               'theta_eap'      the final EAP ability (03's final_theta),
               'theta_se'       the final semTheta standard error,
               'pirt_fullbank'  'est' rescaled to the full route bank (ours, ADAPTATION 8).
  stop()       their published rule (03:222 with min_items = 30, max_items = bank size):
               stop at se <= tau for the three tau of their pipeline
               (run_pipeline.sh:29 SE_VALUES = 0.1 0.2 0.3) -> 'tau0.1' / 'tau0.2' /
               'tau0.3', each with the same p-IRT readout.

Not used: scripts/02_wle_scoring.r. It produces the WLE ability of the FULL response
vector, which 03 only carries along as ground truth for its theta-recovery table
(03:329, 413-430); it is not an input to selection, to the stopping rule or to the
p-IRT estimate, and computing it here would read every item of y.

ADAPTATIONS
1. Single chunk (01_fit_irt.r:31-57 CONFIGS / :88-90 chunk selection, and the chunk
   concatenation their pipeline does before 03): their benchmarks have 600-5600 items
   and are fit in ~105-item chunks that are then combined into
   irt_item_parameters_combined.csv. A 210-220 route bank is one chunk, so the
   chunk_end plumbing is dropped and the single coef() table is written under the
   combined name that 03:62 and 04:28 read. No linking is involved either way.
2. Missing cells kept (01:75-76 `na.omit(data)` and `colSums(is.na(...)) == 0`): about
   1.3% of the calibration cells are "no record", scattered over the planners, and
   their row-wise na.omit deletes every planner that has any gap -- on average 2.1 of
   the 4 planners at K_cal = 4 (all 4 in 3 of the 64 cells), 3.4 of 8 and 5.0 of 12 --
   which would change the calibration panel size the protocol fixes. mirt's EM treats
   NA as missing, so those two lines are dropped and only the constant-column/row
   cleaning (01:78-85) is applied, evaluated on the non-NA entries of each column/row
   (with NA present their `length(unique(x)) == 1` can never fire).
3. M2 skipped (01:100 `m2 <- M2(model)`): a global fit statistic written to
   m2_<chunk>.csv and read by nothing in 03/04. For a 209-item 3PL fit on 12
   respondents it did not return within 300 s, which would dominate the 192-cell sweep.
   The other two outputs of 01 (fscores EAP, coef items) are kept.
4. score_response replaced (03:120-137): their version indexes a response matrix that
   holds every model's outcome on every item, so R would receive the evaluation
   planner's complete outcome vector. atlas_cat.R defines a same-signature
   score_response that writes "@ASK <item id>" on stdout and reads the outcome from
   stdin; this module answers each request from `y` (one LeakGuard read per
   administered item). run_atlas itself is sourced unchanged.
5. Single-model driver and seeding (03:344-395 makeCluster/registerDoParallel/foreach
   over the rows of the test response matrix, 03:367 `set.seed(123 + i)`): one cell =
   one evaluation planner, so the foreach body is executed once, directly, and their
   per-model seed is replaced by the harness cell seed, set immediately before each
   run_atlas call (this fixes the random first item, 03:175).
   That alone is not enough to make a run reproducible: catR 3.17's nextItem ends with
   `set.seed(NULL)` (line 370 of 372 of its body), i.e. it reseeds R's RNG from the
   clock on every call, so with their code as written two runs from the same seed
   select different items from item 3 on (verified: selftest_atlas.py would fail its
   determinism and prefix checks). atlas_cat.R therefore shadows nextItem with a
   function that seeds the stream from (cell seed, call index) and then calls
   catR::nextItem with the arguments unchanged -- the MFI criterion, the EAP method and
   the randomesque = 5 draw are theirs, only the stream feeding them is made
   reproducible. Without this the method has no prefix property and no repeatability.
6. Fixed budgets (03:376-379 `min_items = 30, max_items = 500, se_theta_stop`): the
   budget grid needs runs that never stop early, so budget B is one run_atlas call with
   max_items = B and se_theta_stop = 0 (their `se <= se_theta_stop` test can then never
   fire), keeping their min_items = 30. stop() uses the rule exactly as published:
   min_items = 30, se_theta_stop = tau (their three pipeline values,
   run_pipeline.sh:29), max_items = the bank size instead of their 500 -- with at most
   ~220 routes the bank is exhausted first, so the cap never binds.
7. Bank = the fitted items (03:62 reads the item-parameter file): items dropped by the
   constant-column cleaning of 01 have no (a, b, c) and can be neither selected nor
   scored. Over the 192 protocol cells that is on average 13.8 of ~217 items at
   K_cal = 12, 22.7 at K_cal = 8 and 52.4 (up to 90) at K_cal = 4; in 33 of the 64
   K_cal = 4 cells the fitted bank is smaller than B = 165 and the run returns fewer
   items than the budget (reported in 'note'). This is a property of their pipeline on
   a 4-respondent matrix, not a choice of the wrapper.
8. p-IRT denominator (04:92-98, 104, 126-130): their I_j is the item-parameter file,
   i.e. the fitted items, so their estimate targets the mean over the fitted bank while
   the protocol scores the mean over all 210-220 routes. 'est' is their number as it
   comes out of compute_pirt_accuracy; variants['pirt_fullbank'] rescales it,
   (n_fit * est + sum of the dropped items' constant calibration outcome) / n_bank,
   using 0.5 for an item with no calibration record at all (their 04:122 fallback).
   The rescaling is ours, not theirs.
9. compute_pirt_accuracy called directly (04:144-186 loops over the rows of the test
   response matrix and re-reads files per model): it is called once, with the
   selected-items csv run_atlas has just written (04:83-89) and response_matrix = NULL
   -- that argument is accepted at 04:76 and never used in the body.
10. BLAS threads pinned to 1 for the mirt fit (environment, not code): on this host mirt
   spreads a 12x209 EM over ~50 BLAS threads, which only adds contention across the 192
   cells and makes the arithmetic order machine-dependent. OMP/OPENBLAS/MKL_NUM_THREADS
   are set to 1 in the fit subprocess environment; the CAT stage is single-threaded R.
11. Warnings and convergence recorded (withCallingHandlers around the mirt call): the
   fit itself is unchanged; model['info'] carries mirt's warnings, `converged`, the EM
   iteration count and the (a, b, g) ranges, because a 3PL on 4 or 12 respondents is
   not expected to be well identified.

Cost / what the fit does on this data (their NCYCLES = 100000 is kept as it stands)
  K_cal = 12: the EM never converges -- it stops at the 100000-cycle cap after ~13 min
  with mirt's warning "EM cycles terminated after 100000 iterations" and discriminations
  from -74 to +78. K_cal = 4: converged in 1683 cycles / ~19 s (157 fitted items), with
  a1 up to 68 and EAP abilities pinned to a quadrature node. That is the finding, and it
  is why one harness cell costs ~13 min at K_cal >= 8 and seconds at K_cal = 4; the CAT
  stage itself is ~4 s for the four budgets and ~3 s for the three stopping rules.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .base import OFFICIAL_RSCRIPT, OfficialMethod

WRAPPER_R = Path(__file__).resolve().parent / 'atlas_cat.R'
MIN_ITEMS = 30                     # 03_atlas_cat.r:377 (their pipeline call)
STOP_TAUS = (0.1, 0.2, 0.3)
CAT_TIMEOUT_S = 3600


def _nums(v):
    """R vector from jsonlite -> list of floats (NA / null -> nan; scalars unboxed)."""
    if v is None:
        return []
    return [float('nan') if x is None else float(x) for x in np.atleast_1d(v).tolist()]


def _fit_env():
    """base.run_rscript's environment plus single-threaded BLAS (ADAPTATION 10)."""
    return {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


class Atlas(OfficialMethod):
    name = 'atlas'
    adaptive = True

    # ── fit: 01_fit_irt.r ────────────────────────────────────────────────────
    def fit(self, R, seed, workdir):
        R = np.asarray(R, float)
        K, n = R.shape
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        calib = workdir / 'calibration.csv'
        with open(calib, 'w') as f:
            f.write(','.join(f'X{j}' for j in range(n)) + '\n')
            for k in range(K):
                f.write(','.join('NA' if np.isnan(v) else str(int(v)) for v in R[k]) + '\n')

        r = subprocess.run([str(OFFICIAL_RSCRIPT), str(WRAPPER_R), 'fit', str(calib), str(workdir)],
                           capture_output=True, text=True, cwd=str(workdir), env=_fit_env())
        if r.returncode != 0:
            raise RuntimeError(f'atlas fit (01_fit_irt.r) failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}')
        info = json.load(open(workdir / 'fit_info.json'))

        item_ids = [int(i) for i in _nums(info['item_ids'])]
        dropped = [int(i) for i in _nums(info.get('dropped_constant_items'))]
        dvals = _nums(info.get('dropped_constant_values'))
        if len(item_ids) == 0:
            raise RuntimeError('atlas: every bank item was constant across the calibration planners')
        info.pop('item_ids', None)
        info.update({'K_cal': int(K), 'n_bank': int(n), 'stdout': r.stdout[-1500:]})
        return {'workdir': str(workdir), 'n_bank': int(n), 'item_ids': item_ids,
                'dropped_items': dropped, 'dropped_values': dvals, 'info': info}

    # ── estimate: 03 run_atlas at max_items = B, then 04 p-IRT ───────────────
    def estimate(self, model, y, budgets, seed):
        n_fit = len(model['item_ids'])
        cfgs = [{'name': f'B{int(B)}', 'max_items': min(int(B), n_fit),
                 'min_items': MIN_ITEMS, 'se_theta_stop': 0} for B in budgets]
        out = self._run_cat(model, y, cfgs, seed)
        return {int(B): self._readout(model, out[f'B{int(B)}'], int(B)) for B in budgets}

    # ── stop: their published SE rule ────────────────────────────────────────
    def stop(self, model, y, seed):
        n_fit = len(model['item_ids'])
        cfgs = [{'name': f'tau{t}', 'max_items': n_fit, 'min_items': MIN_ITEMS,
                 'se_theta_stop': t} for t in STOP_TAUS]
        out = self._run_cat(model, y, cfgs, seed)
        return {f'tau{t}': self._readout(model, out[f'tau{t}'], None) for t in STOP_TAUS}

    # ── plumbing ─────────────────────────────────────────────────────────────
    def _readout(self, model, res, budget):
        items = [int(i) for i in _nums(res['items'])]
        n_fit = int(res['n_all_items'])
        if res.get('pirt_accuracy') is None:
            # Not a plumbing failure: the official EAP ability came back NA (rendered as
            # null by toJSON(na = "null")), so 04's compute_3pl_prob produced NA for every
            # unobserved item and the p-IRT average is NA. Observed at K_cal = 4, where the
            # 3PL is unidentified (discriminations run to |a| > 60 and the EAP grid
            # saturates). The full-bank budget is unaffected because n_unobserved = 0 makes
            # avg_predicted a constant 0 in 04:126. Reported as a method-level failure of
            # ATLAS at this panel size rather than silently imputed.
            raise ValueError(
                f'official p-IRT is NA: EAP theta NA at budget {budget} '
                f'(n_items {len(items)}, n_fit {n_fit}, final_theta {res.get("final_theta")}, '
                f'final_se {res.get("final_se")}); 3PL unidentified at this K_cal')
        pirt = float(res['pirt_accuracy'])
        dv = np.asarray(model['dropped_values'], float)
        dropped_sum = float(np.where(np.isnan(dv), 0.5, dv).sum()) if dv.size else 0.0
        full = (n_fit * pirt + dropped_sum) / model['n_bank']              # ADAPTATION 8
        rec = {'est': pirt, 'items': items,
               'variants': {'sample_mean': float(res['avg_observed']),
                            'theta_eap': float(res['final_theta']),
                            'theta_se': float(res['final_se']),
                            'pirt_fullbank': float(full)}}
        if budget is not None and len(items) < budget:
            rec['note'] = (f'bank = {n_fit} fitted items ({model["n_bank"] - n_fit} dropped as constant '
                           f'across the calibration planners), budget {budget} cannot be filled')
        return rec

    def _run_cat(self, model, y, cfgs, seed):
        """One R session running `cfgs` run_atlas calls; answers its @ASK requests."""
        workdir = Path(model['workdir'])
        tag = '_'.join(c['name'] for c in cfgs)
        cfg_path, out_path = workdir / f'cfg_{tag}.json', workdir / f'cat_{tag}.json'
        json.dump(cfgs, open(cfg_path, 'w'))
        out_path.unlink(missing_ok=True)                 # never read a stale result
        p = subprocess.Popen([str(OFFICIAL_RSCRIPT), str(WRAPPER_R), 'cat', str(workdir),
                              str(int(seed)), str(cfg_path), str(out_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, cwd=str(workdir), env=_fit_env())
        log = []
        try:
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                if line.startswith('@ASK'):
                    j = int(line.split()[1].lstrip('X'))                   # ADAPTATION 4
                    try:
                        p.stdin.write(f'{int(y[j])}\n')
                        p.stdin.flush()
                    except BrokenPipeError:                                # R died mid-run
                        break
                else:
                    log.append(line)
            rc = p.wait(timeout=CAT_TIMEOUT_S)
        finally:
            if p.poll() is None:
                p.kill()
            for s in (p.stdin, p.stdout):
                try:
                    s.close()
                except Exception:
                    pass
        if rc != 0 or not out_path.exists():
            raise RuntimeError(f'atlas cat (03/04) failed (rc={rc}):\n{"".join(log)[-4000:]}')
        return json.load(open(out_path))


METHOD = Atlas()
