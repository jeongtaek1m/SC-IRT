"""Table-3A evaluation of ANY unseen-scene difficulty predictor.

Give per-draw predictions b_tilde for the held-out-type routes of the
unified split and get the paper's US metrics back — pooled cell AUROC
(against the planner-only null), Scene-MAE with relative reduction, and
rho_scene against observed failure rates — computed exactly as
experiments/run_us.py does for every row of Table 3A.

    preds[draw] = (route_ids, b_tilde)      # draw in 0..15, routes of block C

Ability theta_j for the 16 calibration planners comes from the canonical
1PL calibration of block A (`calibration.calibrate_dense`, it = 800), so
descriptor rows and encoder predictions are all scored on one
theta. Predictions may be a subset of the C routes of a draw; missing
routes are skipped (and counted).

The shipped encoder artifacts (data/encoder/relgraph_r2_s*.npz) are in
this format: keys draw{r}_rt / draw{r}_bt, plus draw{r}_sigma (the
residual SD learned on that draw's calibration block, used by run_ups.py).
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .b2d import Panel
from .splits import unified_split, R_DRAWS
from .calibration import calibrate_dense
from .curves import sig


def load_pred_npz(path):
    z = np.load(path, allow_pickle=True)
    return {r: ([str(x) for x in z[f'draw{r}_rt']], np.asarray(z[f'draw{r}_bt'], float))
            for r in range(R_DRAWS) if f'draw{r}_rt' in z}


class USEvaluator:
    """Caches the per-draw calibration so many predictors can be scored."""

    def __init__(self, panel=None, device='cuda'):
        self.panel = panel or Panel()
        self.Y0, self.MK = self.panel.dense()
        self.N = len(self.panel.allr)
        self.draws = {}
        for seed in range(R_DRAWS):
            hp, ht = unified_split(seed, self.panel.utypes, self.panel.J)
            cols = [c for c in range(self.panel.J) if c not in hp]
            tr = [i for i in range(self.N) if self.panel.sn[self.panel.allr[i]] not in ht]
            te = [i for i in range(self.N) if self.panel.sn[self.panel.allr[i]] in ht]
            _, th = calibrate_dense(self.Y0, self.MK, tr, cols, device=device)
            _, th0 = calibrate_dense(self.Y0, self.MK, tr, cols, device=device, freeze_b0=True)
            self.draws[seed] = dict(cols=cols, tr=tr, te=te, th=th, th0=th0)

    def _cells(self, i, cols):
        js = [c for c in cols if self.MK[i, c]]
        return js, [cols.index(c) for c in js], self.Y0[i, js]

    def null(self, scored=None):
        """Planner-only null (b = 0). `scored` = {draw: set(route index)}
        restricts the null to the routes a predictor actually covered, so
        the delta metrics stay like-for-like for partial predictions."""
        p, y, rp, ro = [], [], [], []
        for seed, d in self.draws.items():
            routes = d['te'] if scored is None else [i for i in d['te'] if i in scored.get(seed, ())]
            for i in routes:
                js, jj, ys = self._cells(i, d['cols'])
                ps = sig(d['th0'][jj])
                p += ps.tolist(); y += ys.tolist()
                rp.append(ps.mean()); ro.append(ys.mean())
        return dict(auroc=roc_auc_score(y, p), mae=float(np.mean(np.abs(np.array(rp) - np.array(ro)))))

    def evaluate(self, preds):
        """preds: {draw: (route_ids, b_tilde)} -> Table-3A metrics + per-draw rho."""
        idx = {r: i for i, r in enumerate(self.panel.allr)}
        p, y, rp, ro, bt, fl, rho_draw, missing = [], [], [], [], [], [], [], 0
        scored = {}
        for seed, d in self.draws.items():
            if seed not in preds:
                continue
            rts, bts = preds[seed]
            lut = dict(zip(rts, bts))
            te_set = set(d['te'])
            bt_d, fl_d = [], []
            for r, b in lut.items():
                i = idx.get(r)
                if i is None or i not in te_set:
                    missing += 1
                    continue
                js, jj, ys = self._cells(i, d['cols'])
                ps = sig(d['th'][jj] - b)
                p += ps.tolist(); y += ys.tolist()
                rp.append(ps.mean()); ro.append(ys.mean())
                bt_d.append(b); fl_d.append(1 - ys.mean())
                scored.setdefault(seed, set()).add(i)
            bt += bt_d; fl += fl_d
            if len(bt_d) > 2:
                rho_draw.append(spearmanr(bt_d, fl_d).correlation)
        nul = self.null(scored)
        auroc = roc_auc_score(y, p)
        mae = float(np.mean(np.abs(np.array(rp) - np.array(ro))))
        return dict(auroc=auroc, d_auroc=auroc - nul['auroc'], mae=mae,
                    rel_mae=1 - mae / nul['mae'], rho=float(spearmanr(bt, fl).correlation),
                    rho_per_draw=rho_draw, n_routes=len(bt), null=nul, skipped=missing)
