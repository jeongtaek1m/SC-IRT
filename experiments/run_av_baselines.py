#!/usr/bin/env python3
"""The AV-testing baselines on the Table 1 protocol, one wrapper file per method under experiments/official/
(each file's docstring is its provenance chain: what is the paper's, what is the released code's, what is ours):

    official/fst.py      FST (Li et al., T-ITS 2025), re-implemented from the paper (no code)      fst_scene, fst_resp
    official/gp_port.py  GP adaptive sampling (Gong et al., T-ITS 2023), our port                   gp_scene, gp_resp
    official/mfgp.py     the same method through its OFFICIAL code (MFGPreliability)               gpo_scene
    official/ktcs.py     kernel test case sampling (Qian et al., Nat. Comm. 2026), from the Methods ktcs_scene
    official/dice.py     DICE (Farid et al., ICRA 2025), re-implemented end to end                  dice_desc, dice_mae, dice_full
    official/sim2val.py  Sim2Val (Luo et al., CoRL 2025) through its OFFICIAL package (NVlabs/sim2val) s2v_mean, s2v_vec

Every wrapper follows the run_up_official.py contract (fit / estimate / stop) with the bank rows `bi` as an
extra argument of fit, and passes the checks of official/av_common.selftest (official/selftest_<name>.py).
Same draws, K_cal subsamples, banks and budgets as run_up_frontier.py (official/data.py), so every cell is
paired with the ATDrive cell; --merge prints SR-MAE, the paired delta against ATDrive and the pairwise ranking
accuracy of every cell and writes results/up_avbase.json. A method whose estimate carries per-draw estimates
('ests': the random-order rows) is scored by the error AND the ranking accuracy averaged over the draws, as the
random rows of Table 1 (never by the mean of the draws' estimates, which would be an ensemble).

    $P experiments/run_av_baselines.py --methods fst_scene fst_resp --seeds 0 4     # shard
    OMP_NUM_THREADS=2 $P experiments/run_av_baselines.py --methods gpo_scene --seeds 0 2   # official code, CPU
    $P experiments/run_av_baselines.py --merge
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official.data import BGRID, KCALS, NDRAWS, protocol_cell, draw, panel     # noqa: E402
from official.av_common import cell_seed                                       # noqa: E402
from official import fst, gp_port, mfgp, ktcs, dice, sim2val                   # noqa: E402
from atdrive.splits import up_split                                            # noqa: E402

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
METHODS = {**fst.METHODS, **gp_port.METHODS, **mfgp.METHODS, **ktcs.METHODS, **dice.METHODS, **sim2val.METHODS}
ALL_METHODS = list(METHODS)


def run(methods, seeds, out_path):
    recs = json.load(open(out_path)) if out_path.exists() else []
    done = {(r['seed'], r['K'], r['slot'], r['method']) for r in recs}
    for seed in seeds:
        for Kc in KCALS:
            for slot in range(4):
                R, y, bi, SR = protocol_cell(seed, Kc, slot)
                for name in methods:
                    if (seed, Kc, slot, name) in done:
                        continue
                    t0 = time.time()
                    cs = cell_seed(seed, Kc, slot)
                    m = METHODS[name]
                    est = m.estimate(m.fit(R, cs, None, bi), y, list(BGRID), cs)
                    rec = {'seed': seed, 'K': Kc, 'slot': slot, 'method': name, 'SR': SR, 'n_bank': len(bi),
                           'fit_s': time.time() - t0,
                           'budgets': {str(B): {'est': v['est'], 'err': float(np.mean([abs(e - SR) for e in v['ests']])) if 'ests' in v else abs(v['est'] - SR),
                                                'n_items': len(v['items']), 'items': [int(i) for i in v['items']], 'note': v.get('note', ''),
                                                **({'ests': v['ests']} if 'ests' in v else {})}
                                       for B, v in est.items()}}
                    recs.append(rec)
                    json.dump(recs, open(out_path, 'w'))
                    print(f'seed {seed} K{Kc} slot {slot} {name:10} {rec["fit_s"]:6.1f}s  '
                          + ' '.join(f'B{B}:{v["err"]:.4f}' for B, v in rec['budgets'].items()), flush=True)
    return recs


def merge():
    recs = []
    for f in sorted(glob.glob(str(OUT / 'up_avbase_*_[0-9]*_[0-9]*.json'))):
        recs += json.load(open(f))
    ref = {(r['seed'], r['K'], r['js']): r for r in json.load(open(OUT / 'up_frontier.json'))}
    p = panel()
    T = {k: float(p.bank_rows(p.allr, k)[1].mean()) for k in range(p.J)}
    js_of = {s: draw(s)[0] for s in range(NDRAWS)}

    def rank_acc(seed, js, est):
        held, _ = up_split(seed, p.utypes, p.J)
        out = []
        for c in range(p.J):
            if c in held:
                continue
            st, se = np.sign(T[js] - T[c]), np.sign(est - T[c])
            out.append(0.5 if (st == 0 or se == 0) else float(st == se))
        return float(np.mean(out))

    table = {}
    for r in recs:
        js = js_of[r['seed']][r['slot']]
        for B, v in r['budgets'].items():
            t = table.setdefault(r['method'], {}).setdefault((r['K'], int(B)), {'err': [], 'd': [], 'acc': []})
            t['err'].append(v['err'])
            t['d'].append(v['err'] - ref[(r['seed'], r['K'], js)]['err']['ATDrive'][B])
            t['acc'].append(float(np.mean([rank_acc(r['seed'], js, e) for e in v['ests']])) if 'ests' in v   # random-draw rows:
                            else rank_acc(r['seed'], js, v['est']))                                  # per draw, then mean (no ensemble)
    out = {'methods': {}}
    print(f"{'method':10}{'K':>3}{'B':>5}{'n_eval':>7}{'SR-MAE':>8}{'RankAcc':>8}{'d vs ATDrive':>14}")
    for mname, Tm in table.items():
        out['methods'][mname] = {}
        for (K, B), t in sorted(Tm.items()):
            row = {'n_eval': len(t['err']), 'sr_mae': float(np.mean(t['err'])), 'rank_acc': float(np.mean(t['acc'])),
                   'd_atdrive': float(np.mean(t['d']))}
            out['methods'][mname][f'K{K}_B{B}'] = row
            print(f"{mname:10}{K:3d}{B:5d}{row['n_eval']:7d}{row['sr_mae']:8.4f}{row['rank_acc']:8.3f}{row['d_atdrive']:+14.4f}")
        cells = [v for (K, B), v in Tm.items() if B in (30, 55, 110)]
        print(f"{mname:10} macro over the 9 Table 1 cells: SR-MAE {np.mean([np.mean(c['err']) for c in cells]):.4f}")
    json.dump(out, open(OUT / 'up_avbase.json', 'w'), indent=1)
    print(f"wrote {OUT / 'up_avbase.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.merge:
        merge()
        return
    lo, hi = a.seeds if a.seeds else (0, NDRAWS)
    tag = '-'.join(a.methods)
    run(a.methods, range(lo, hi), OUT / f'up_avbase_{tag}_{lo}_{hi}.json')


if __name__ == '__main__':
    main()
