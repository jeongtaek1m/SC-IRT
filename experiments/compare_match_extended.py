#!/usr/bin/env python3
"""Does the difficulty-matching term's MAE_FR gain reach the extended SR estimate?

Pairs the 64 evaluations of `run_ups_full.py` between the encoder of record and its matching-term arm, for the
three encoder runs (`ATDRIVE_ENC_ARM=_match0.1 ATDRIVE_ENC_RUN=r`). What is compared is the whole evaluation,
selection included: the probe order of the canonical policy is chosen with the scene prior, so changing the
encoder changes which routes are executed as well as how the estimate is read out.

Checks before any comparison:
  - the two runs cover the same 64 (draw, planner) evaluations, without duplicates;
  - the truth each evaluation is scored against (n_C, n_T, SR_full, SR_C, SR_T) is identical, so only the
    estimate can differ;
  - the (policy, readout) combinations that touch no encoder quantity are identical to the last bit. Which
    combinations those are was measured, not assumed: under the canonical policy even `naive` moves, because
    the encoder changes the probe ORDER and therefore which outcomes are observed.

Statistic: for each evaluation the error difference (matching - record) is averaged over the three encoder
runs first, because the three share the same evaluations and are not three independent tests; the average is
then bootstrapped over the evaluation planners (the repo's cluster bootstrap). Per-run deltas and the spread
across runs are printed next to it, not folded into the interval.

    python experiments/compare_match_extended.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.metrics import paired_cluster_boot                                   # noqa: E402

REC = {0: 'results/ups_full.json', 1: 'results/ups_full_enc1/ups_full.json',
       2: 'results/ups_full_enc2/ups_full.json'}
MAT = {r: f'results/ups_full_match_enc{r}/ups_full.json' for r in range(3)}
POL, CAN, PRIOR = 'Delta-R1 on D', 'calC + sceneT', 'calC + priorT(marg)'
BP = (30, 55, 110)
# measured invariants: neither the probe order nor the readout of these reads the scene prior
INVARIANT = (('Random', 'naive'), ('Random', PRIOR), ('Random', 'calC + trueT'),
             ('Delta-R1 on D (scene-free)', 'naive'), ('Delta-R1 on D (scene-free)', PRIOR),
             ('Delta-R1 on D (scene-free)', 'calC + trueT'))


def load(path):
    recs = sorted(json.load(open(path)), key=lambda r: (r['seed'], r['js']))
    keys = [(r['seed'], r['js']) for r in recs]
    assert len(recs) == 64 and len(set(keys)) == 64, f'{path}: {len(recs)} records, {len(set(keys))} unique'
    return recs


def check_pair(rec_a, rec_b, tag):
    assert [(x['seed'], x['js']) for x in rec_a] == [(x['seed'], x['js']) for x in rec_b], f'{tag}: different evaluations'
    for x, y in zip(rec_a, rec_b):                            # the truth cannot depend on the encoder
        for f in ('nC', 'nT', 'SR_full', 'SR_C', 'SR_T'):
            assert abs(x[f] - y[f]) < 1e-12, f'{tag}: {f} differs on (draw {x["seed"]}, planner {x["js"]})'
    for pol, arm in INVARIANT:
        for budget in BP:
            for field in [f for f in ('sr', 'srT') if f in rec_a[0]['pol'][pol][str(budget)]['arm'][arm]]:
                va = [x['pol'][pol][str(budget)]['arm'][arm][field] for x in rec_a]
                vb = [x['pol'][pol][str(budget)]['arm'][arm][field] for x in rec_b]
                np.testing.assert_allclose(va, vb, rtol=0, atol=1e-12,
                                           err_msg=f'{tag}: {pol} / {arm} / B={budget} / {field} moved, which no encoder can do')


def err(recs, B, arm, target):
    key, truth = ('sr', 'SR_full') if target == 'full' else ('srT', 'SR_T')
    return np.array([abs(x['pol'][POL][str(B)]['arm'][arm][key] - x[truth]) for x in recs])


def main():
    A = {r: load(p) for r, p in REC.items()}
    Bm = {r: load(p) for r, p in MAT.items()}
    for r in range(3):
        check_pair(A[r], Bm[r], f'encoder run s{r}')
    print('checks passed: same 64 evaluations, same truth, six encoder-independent (policy, readout) '
          'combinations identical in all three runs\n')
    js = [x['js'] for x in A[0]]
    for target, label in (('full', 'extended benchmark SR-MAE (Y_B, Y_D)'), ('new', 'new-route block SR-MAE (Y_D)')):
        print(f'== {label} — probe {POL}, readout {CAN} ==')
        print(f'{"B":>4} {"record":>8} {"matching":>9} {"delta (3-run mean, planner-cluster bootstrap)":>48} {"per-run delta":>34}')
        for B in BP:
            ea = np.stack([err(A[r], B, CAN, target) for r in range(3)])
            eb = np.stack([err(Bm[r], B, CAN, target) for r in range(3)])
            d_eval = (eb - ea).mean(0)                         # average over encoder runs within evaluation
            d, lo, hi = paired_cluster_boot(list(eb.mean(0)), list(ea.mean(0)), js)
            per = (eb - ea).mean(1)
            print(f'{B:4d} {ea.mean():8.4f} {eb.mean():9.4f}   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]'
                  f'{"  (interval contains zero)" if lo < 0 < hi else "  (interval excludes zero)":>26}'
                  f'   ' + ' '.join(f'{x:+.4f}' for x in per) + f'  SD {per.std(ddof=1):.4f}')
        print(f'reference, the common difficulty prior (no encoder): ' +
              '  '.join(f'B{B}: {err(A[0], B, PRIOR, target).mean():.4f}' for B in BP) + '\n')


if __name__ == '__main__':
    main()
