"""SC-IRT inside a real closed-loop evaluation (UP setting).

A new planner is evaluated on a calibrated bank: every route of the panel
is an item whose difficulty posterior comes from the panel's planners.
`LiveEvaluator` keeps the planner's posterior state, proposes the next
route(s) by Delta-R1, ingests each outcome as it arrives from the
simulator, reports the posterior-median success rate with its risk, and
stops when the calibrated risk c * R1 falls below the error target.

    ev = LiveEvaluator()                       # bank from data/matrices (all planners)
    ev.load_risk_scale('data/live/risk_scale.json')   # or ev.calibrate_risk(...) once
    while not ev.done(eps=0.03):
        rid = ev.next()[0]                     # route id to roll out
        passed = run_in_carla(rid)             # your simulator call
        ev.observe(rid, passed)
    print(ev.estimate())

Nothing here depends on the simulator; `tools/b2d_adaptive_eval.py` is the
Bench2Drive driver built on it.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .b2d import Panel, DATA
from .calibration import calibrate
from .curves import curves_from_posterior
from .bayes import Bank, State, track
from .acquisition import r1_pick, r1_scores, r1_traj, TIE_DECIMALS

RISK_T0 = 10


class LiveEvaluator:
    def __init__(self, exclude=(), routes=None, device='cuda', it=800, verbose=True):
        """exclude: planner names to drop from the calibration panel (e.g. the
        planner under evaluation if it is already in the matrix); routes: the
        bank as a subset of the panel's routes (default: all 220)."""
        self.panel = Panel()
        self.cols = [k for k, nm in enumerate(self.panel.names) if nm not in set(exclude)]
        self.routes = [r for r in self.panel.allr if r in set(routes)] if routes is not None else list(self.panel.allr)
        self.types = np.array([self.panel.sn[r] for r in self.routes])
        self.rid2i = {r: i for i, r in enumerate(self.routes)}
        self.device, self.it, self.verbose = device, it, verbose
        self.fit = calibrate(self.panel.Y, self.routes, self.cols, it=it, mode='1pl',
                             device=device, types=self.types)
        self.bank = Bank(curves_from_posterior(self.fit['W']), self.types, self.fit['sigma_g'])
        self.state = State(self.bank, np.zeros(len(self.routes)))
        self.obs = {}
        self.c = None
        if verbose:
            print(f'[scirt-live] bank {len(self.routes)} routes x {len(self.cols)} planners; '
                  f'sigma_b {self.fit["sigma_b"]}, sigma_g {self.fit["sigma_g"]}', flush=True)

    # --- fingerprint of the bank, used to key the cached risk scale ---------
    def fingerprint(self):
        h = hashlib.md5()
        h.update(','.join(self.panel.names[k] for k in self.cols).encode())
        h.update(','.join(self.routes).encode())
        h.update(str(self.it).encode())
        return h.hexdigest()[:12]

    def y_digest(self):
        """Digest of the responses the bank was calibrated on (a cache entry
        is valid only for the exact matrix it was computed from)."""
        cs = set(self.cols)
        return hashlib.md5(repr(sorted((r, k, v) for (r, k), v in self.panel.Y.items() if k in cs)).encode()).hexdigest()

    # --- acquisition ---------------------------------------------------------
    def remaining(self):
        return [i for i in range(len(self.routes)) if i not in self.state.S]

    def next(self, k=1, allowed=None):
        """Route ids to roll out next: the Delta-R1 argmin, or the k best
        candidates by expected branch risk when batching (no re-planning
        inside a batch). `allowed` restricts the candidates to those route ids."""
        rem = self.remaining()
        if allowed is not None:
            rem = [i for i in rem if self.routes[i] in allowed]
        if not rem:
            return []
        if k == 1:
            return [self.routes[r1_pick(self.state, rem)]]
        ev = np.round(r1_scores(self.state, rem), TIE_DECIMALS)
        return [self.routes[rem[j]] for j in np.argsort(ev, kind='stable')[:k]]

    def observe(self, route_id, passed):
        i = self.rid2i[str(route_id)]
        assert i not in self.state.S, f'route {route_id} already observed'
        self.state.add(i, 1.0 if passed else 0.0)
        self.obs[str(route_id)] = bool(passed)

    # --- readout -------------------------------------------------------------
    def estimate(self):
        if self.c is None:
            raise RuntimeError('no risk scale: call calibrate_risk() / load_risk_scale(), or set ev.c = 1.0 to use raw R1')
        sh, r1 = self.state.readout()
        return {'sr_hat': sh, 'r1': r1, 'risk': self.c * r1, 'c': self.c,
                'n_done': len(self.state.S), 'n_total': len(self.routes),
                'observed_sr': float(np.mean(list(self.obs.values()))) if self.obs else None,
                'types_covered': int(len(set(self.types[self.state.S]))) if self.state.S else 0}

    def done(self, eps, min_routes=RISK_T0):
        e = self.estimate()
        return e['n_done'] >= min_routes and e['risk'] <= eps

    # --- risk scale ----------------------------------------------------------
    def calibrate_risk(self, cache=None, quantile=0.9, T=110):
        """c = `quantile` of |S_hat_t - SR| / R1_t over leave-one-planner-out
        Delta-R1 trajectories on the calibration panel, t in [RISK_T0, T].
        Cached by bank fingerprint."""
        cache = Path(cache) if cache else DATA / 'live' / 'risk_scale.json'
        key, ymd5 = self.fingerprint(), self.y_digest()
        if cache.exists():
            d = json.load(open(cache))
            e = d.get(key)
            if e and e.get('y_md5') == ymd5 and e.get('quantile') == quantile and e.get('T') == T:
                self.c = float(e['c'])
                if self.verbose:
                    print(f'[scirt-live] risk scale c = {self.c:.3f} (cached, {e["n_planners"]} LOO planners)')
                return self.c
        ratios, per = [], {}
        for j in self.cols:
            csl = [k for k in self.cols if k != j]
            f = calibrate(self.panel.Y, self.routes, csl, it=self.it, mode='1pl',
                          device=self.device, sigma_b=self.fit['sigma_b'])
            bi, yy = self.panel.bank_rows(self.routes, j)
            bank = Bank(curves_from_posterior(f['W'][bi]), self.types[bi], self.fit['sigma_g'])
            S = r1_traj(bank, yy, T)
            Sh, R1 = track(bank, yy, S)
            act = np.abs(np.array(Sh[RISK_T0 - 1:]) - yy.mean())
            raw = np.array(R1[RISK_T0 - 1:])
            ok = raw > 1e-6
            ratios += list(act[ok] / raw[ok])
            per[self.panel.names[j]] = {'sr': float(yy.mean()), 'mae@30': float(abs(Sh[29] - yy.mean())),
                                        'mae@55': float(abs(Sh[54] - yy.mean())) if len(Sh) > 54 else None}
            if self.verbose:
                print(f'[scirt-live] LOO {self.panel.names[j]}: SR {yy.mean():.3f}  S_hat@30 {Sh[29]:.3f}  R1@30 {R1[29]:.4f}', flush=True)
        self.c = float(np.percentile(ratios, 100 * quantile))
        d = json.load(open(cache)) if cache.exists() else {}
        d[key] = {'c': self.c, 'quantile': quantile, 'T': T, 'n_planners': len(self.cols), 'y_md5': ymd5,
                  'planners': [self.panel.names[k] for k in self.cols], 'per_planner': per}
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(d, open(cache, 'w'), indent=1)
        if self.verbose:
            print(f'[scirt-live] risk scale c = {self.c:.3f} saved to {cache}')
        return self.c

    def load_risk_scale(self, cache=None):
        cache = Path(cache) if cache else DATA / 'live' / 'risk_scale.json'
        e = json.load(open(cache)).get(self.fingerprint())
        if not e or e.get('y_md5') != self.y_digest():
            raise KeyError('no cached risk scale for this bank / response matrix: run calibrate_risk()')
        self.c = float(e['c'])
        return self.c
