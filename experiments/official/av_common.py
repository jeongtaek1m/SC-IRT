"""Shared plumbing of the AV-testing baselines (fst.py, gp_port.py, mfgp.py, ktcs.py, dice.py): the protocol
cell seed, the surrogate matrix fill, feature standardisation, the route feature sets, and the self-test checks
every wrapper must pass. The wrapper contract is that of run_up_official.py (fit / estimate / stop) with one
extra argument, the bank rows `bi`, because these methods read route features."""
import numpy as np
import torch

from official.data import BGRID, draw
from atdrive.b2d import load_features

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GP_NINIT = 10


def cell_seed(seed, Kc, slot):
    return 100000 + 1000 * seed + 10 * Kc + slot                 # = run_up_official.cell_seed


def fill(R):
    """Surrogate response matrix (K_cal x bank), a missing cell filled with that surrogate's mean."""
    Rf = R.copy()
    for k in range(R.shape[0]):
        m = np.nanmean(R[k])
        Rf[k, np.isnan(Rf[k])] = m
    return Rf


def standardise(F):
    return (F - F.mean(0)) / (F.std(0) + 1e-6)


_FEATS = {}


def route_feats(bi, kind='desc'):
    """Route features of the bank rows: 'desc' = the 25-d kinematics + 48-d risk descriptors the US baselines use,
    'mae' = the 64-d masked-autoencoder embedding of experiments/dice_mae.py, 'pooled' = its pooled ego / track /
    road embeddings (192-d, the DICE head input)."""
    if kind not in _FEATS:
        _, _, calR = draw(0)                                     # the 220-route bank (no held-out types in UP)
        if kind == 'desc':
            ck, gt = load_features('eval_cmdkin_stats'), load_features('eval_gtrisk')
            _FEATS[kind] = np.stack([np.concatenate([ck[r], gt[r]]) for r in calR])
        else:
            f = load_features({'mae': 'eval_dice_mae', 'pooled': 'eval_dice_mae_pooled'}[kind])
            _FEATS[kind] = np.stack([f[r] for r in calR])
    return _FEATS[kind][bi]


def features_for(spec, R, bi):
    """'desc' / 'mae' / 'pooled' -> route features; 'resp' -> the surrogates' filled response profile."""
    return fill(R).T if spec == 'resp' else route_feats(bi, spec)


# ------------------------------------------------------------------ self-test checks -----------------------------
def selftest(method, name, cells=((0, 12, 0), (0, 4, 1)), adaptive=False, allow_repeats=False):
    """The checks every AV wrapper must pass on two protocol cells:
    BUDGET       every budget's items are in-range bank indices and there are B of them (repeats only where the
                 official loop allows them, allow_repeats);
    LEAK         estimate() again with every outcome OUTSIDE the routes the method read flipped -> identical est
                 (fixed designs read only their selected routes, all draws; an adaptive run reads the routes it
                 executed by n_max, and its budget-B readout only the first B);
    DETERMINISM  the same cell and seed twice -> identical items and est (est to 1e-6: multithreaded BLAS is not
                 bitwise reproducible across processes at the 1e-9 level);
    PREFIX       (adaptive only) the B = 30 items are the first 30 of the B = 165 items."""
    from official.data import protocol_cell
    import time
    for (seed, Kc, slot) in cells:
        R, y, bi, SR = protocol_cell(seed, Kc, slot)
        cs = cell_seed(seed, Kc, slot)
        t0 = time.time()
        model = method.fit(R, cs, None, bi)
        est = method.estimate(model, y, list(BGRID), cs)
        dt = time.time() - t0
        print(f'[{name}] cell s{seed} K{Kc} p{slot} SR {SR:.4f} {dt:5.1f}s ' + ' '.join(f'B{B}: {v["est"]:.4f} ({abs(v["est"] - SR):.4f})' for B, v in est.items()))
        read = set()
        for B, v in est.items():
            items = [int(i) for i in v['items']]
            assert all(0 <= i < len(y) for i in items), 'item out of range'
            assert len(items) == B, (B, len(items))
            assert allow_repeats or len(set(items)) == B, 'repeated item'
            for extra in v.get('items_draws', [v['items']]):
                read |= set(int(i) for i in extra)
        # LEAK
        y2 = y.copy()
        for i in range(len(y)):
            if i not in read:
                y2[i] = 1.0 - y2[i]
        est2 = method.estimate(method.fit(R, cs, None, bi), y2, list(BGRID), cs)
        for B in BGRID:
            assert abs(est2[B]['est'] - est[B]['est']) < 1e-6, f'LEAK at B={B}: {est[B]["est"]} vs {est2[B]["est"]}'
        if adaptive:
            for B in BGRID:
                y3 = y.copy()
                inside = set(int(i) for i in est[B]['items'])
                for i in range(len(y)):
                    if i not in inside:
                        y3[i] = 1.0 - y3[i]
                e3 = method.estimate(method.fit(R, cs, None, bi), y3, [B], cs)
                assert abs(e3[B]['est'] - est[B]['est']) < 1e-6, f'LEAK (budget prefix) at B={B}'
        # DETERMINISM
        est4 = method.estimate(method.fit(R, cs, None, bi), y, list(BGRID), cs)
        for B in BGRID:
            assert est4[B]['items'] == est[B]['items'] and abs(est4[B]['est'] - est[B]['est']) < 1e-6, f'DETERMINISM at B={B}'
        # PREFIX
        if adaptive:
            assert est[30]['items'] == est[165]['items'][:30], 'PREFIX'
        print(f'[{name}]   BUDGET / LEAK / DETERMINISM{" / PREFIX" if adaptive else ""} OK; routes read {len(read)}')
    print(f'[{name}] all checks passed')
