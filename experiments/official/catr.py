"""The two classical static information orders -- Total-Fisher (Birnbaum 1968
test information) and Marginal / posterior-weighted Fisher (van der Linden 1998
MPWI) -- run through the OFFICIAL catR 3.17 (Magis & Raiche, J. Stat. Software
48(8), 2012; the reference CAT implementation of the psychometrics community)
on an official mirt 1.47 2PL calibration.

catR and mirt are CRAN packages, not a cloned research repo, so "official" here
means the installed library in /data2/jeongtae/envs/r_metabench: every
algorithmic quantity is one of their exported functions, called from
experiments/official/catr_static.R.  Nothing of the item information, the
ability estimator or the response model is re-implemented in Python or R.

What is called (file:function)
  fit        mirt::mirt(dat, model, itemtype = '2PL', method = 'EM')
             -- .../r_metabench/lib/R/library/mirt, catr_static.R:102
             mirt::coef(model, simplify = TRUE)$items    catr_static.R:106
             mirt::fscores(model, method = 'EAP', quadpts = 61)   catr_static.R:107
             (the calibration planners' EAP abilities)
  select     catR::Ii(theta, itemBank)$Ii                catr_static.R:133
               total_fisher     = sum over the calibration planners' EAP
                                  abilities of Ii, ranked descending
             catR::MWI(itemBank, item, x = NULL, it.given = NULL, type = 'MPWI',
                       priorDist = 'norm', priorPar = c(0, 1))    catr_static.R:141
               marginal_fisher  = the step-0 MPWI, i.e. the integral of
                                  I_j(theta) * dnorm(theta, 0, 1), ranked descending
  estimate   catR::thetaEst(it.given, x, method = 'BM', priorDist = 'norm',
                            priorPar = c(0, 1))          catr_static.R:200
             catR::Pi(theta, itemBank[rest, ])$Pi        catr_static.R:202
             catR::semTheta(...)                         catr_static.R:205

'type' check (asked for explicitly): catR 3.17's signature is
MWI(itemBank, item, x, it.given, model = NULL, lower = -4, upper = 4, nqp = 33,
    type = 'MLWI', priorDist = 'norm', priorPar = c(0, 1), D = 1).
'MLWI' weights the information by the LIKELIHOOD of the responses so far, 'MPWI'
by likelihood * prior; ?MWI names van der Linden (1998) for the MPWI branch.  At
step 0 (x = NULL, it.given = NULL) the likelihood is identically 1, so MPWI is
exactly the prior-weighted information asked for -- verified against
catR:::integrate.catR(X, Ii(X) * dnorm(X, 0, 1)) on the same 33-point grid
(identical to 16 digits).  'MLWI' at step 0 would be the UNWEIGHTED integral over
(-4, 4), which is not the marginal-information order, so type = 'MPWI' is the
right argument.  x = NULL and it.given = NULL are both accepted by catR 3.17.

Readouts (both orders get all of them; 'est' is total_fisher's p-IRT)
  est        p-IRT plug-in of the total_fisher subset:
             (sum of the observed responses + sum of catR::Pi over the
              unadministered items at theta_BM) / n_bank
  items      the total_fisher order's first B bank indices (the cost of 'est')
  variants   'marginal_fisher'             the marginal order's p-IRT
             'total_fisher_theta'          catR::thetaEst BM ability
             'marginal_fisher_theta'
             'total_fisher_se'             catR::semTheta
             'marginal_fisher_se'
             'total_fisher_mean'           mean of the administered responses
             'marginal_fisher_mean'
  info       'order_total_fisher' / 'order_marginal_fisher' (the full bank
             orders), the item bank, the calibration EAP abilities, the mirt
             fit diagnostics, the tie counts, and the empirical-Bayes prior
             ('prior', 'prior_sd_d', 'prior_grid', 'prior_eb_loglik';
             ADAPTATION 1).
  stop()     {} -- these are static orders; catR's stopping rules belong to
             randomCAT, which is not part of either published selection rule.

Tie rule (explicit, as required).  With K_cal in {4, 8, 12} planners a bank of
210-220 routes has at most 2^K_cal distinct response patterns, so items share
item parameters and therefore share information exactly: over the nine cells
measured below a K_cal = 4 bank of 211-220 routes has only 20-26 DISTINCT
information values (196 of 219 tied at K_cal = 4 in the self-test cell (0, 4, 0),
122 of 215 at K_cal = 8 in (3, 8, 2), 55 of 219 at K_cal = 12 in (0, 12, 0)).
Both orders are produced by R's order(-info, seq_along(info))
(catr_static.R:info_order), i.e. descending information with exact ties broken by
the LOWEST bank index -- deterministic and device-independent.  The lite orders
this replaces now use the same convention (atdrive/baselines.py:
total_fisher_order / marginal_fisher_order round to TIE_DECIMALS and take the
lowest bank index), so the two agree on the tie rule and differ in the
calibration the information is computed from and, for the marginal row, in the
quadrature as well (61 points on (-3, 3) with renormalised N(0, 1) weights in
the lite version vs catR's 33 points on (-4, 4)), which can move that ranking
independently of the calibration.
This is a DEVIATION from catR's OWN tie convention: catR::nextItem breaks a tie
at random (`select <- ifelse(length(keep) == 1, keep, sample(keep, 1))`, in all
8 of its selection branches in the installed catR 3.17), so no deterministic
catR rule exists to inherit and the lowest-index rule is a choice made here.  It
is load-bearing and it is NOT neutral.  What matters is the estimate, so the
estimate itself was measured, by re-running estimate() with the ties permuted
uniformly inside each exact-tie group (catR's own convention), 12 draws per cell:
  * K_cal = 8 / 12: INERT.  93 and 164 of the ~215 information values are
    distinct, every budget boundary falls between tie groups, and all 12 draws
    reproduce the deterministic estimate bit-identically in cells (3, 8, 2) and
    (0, 12, 0) (sd 0.0000 at all four budgets).
  * K_cal = 4: LOAD-BEARING.  The B = 30 / 55 boundary falls INSIDE a tie group,
    so the lowest-index rule -- not catR -- picks most of the selected set.  Over
    six fresh cells ((1, 4, 0), (2, 4, 3), (5, 4, 1), (9, 4, 2), (13, 4, 0),
    (15, 4, 3)) the estimate moves with the permutation: mean within-cell sd
    0.0183 at B = 30 (0.0326 in the worst cell), 0.0128 at B = 55, 0.0074 at
    B = 110, 0.0040 at B = 165 -- at the smallest budget larger than ATDrive's
    own macro MAE (0.0262).  And the direction FAVOURS this baseline where it moves most: MAE
    to the cell's true SR is 0.0722 under the deterministic rule vs 0.0895 under
    catR's random rule at B = 30, 0.0711 vs 0.0738 at B = 55, 0.0336 vs 0.0390
    at B = 110, and 0.0178 vs 0.0150 at B = 165.
  The deterministic rule is KEPT -- averaging over tie permutations would read
  the union of the permuted subsets, i.e. more than B routes, and break the
  budget the row is priced at -- so the catR row is reported in the knowledge
  that at K_cal = 4 its small-budget error is ~0.017-0.02 BELOW what catR's own
  tie rule gives on average.  That understates rather than inflates ATDrive's
  margin, but it is a deviation, not a neutral implementation detail.
  (The earlier justification given here -- corr(bank index, y) = -0.065 on
  average, max |.| 0.165, and corr(bank index, calibration pass rate) -0.126 /
  0.238 over the 192 cells, both reproduced -- shows only that the bank index is
  not itself an easy / hard ordering.  It is not evidence that the rule is
  harmless; the sensitivity above is the measurement that matters.)

ADAPTATIONS
1. Item-parameter priors (catr_static.R:fit_one / eb_marginal_loglik).  The
   literal call mirt(dat, 1, itemtype = '2PL', method = 'EM') ABORTS on this
   protocol: mirt:::PrepData:74-76 raises "The following items have only one
   response category and cannot be estimated" for every route that all K_cal
   calibration planners pass (or all fail) -- 10 of 219 routes at K_cal = 12 and
   62 of 219 at K_cal = 4 (63 of the 64 K_cal = 4 cells have at least 30 such
   routes; min / median / max over them 28 / 52.5 / 90).  The two ways out both
   change something, and this is the smaller:
     (a) drop those routes, as the official ATLAS 01_fit_irt.r:78-85 does.  Then
         33 of the 64 K_cal = 4 cells have fewer than 165 calibrated routes, so
         the B = 165 budget cannot be filled, AND the p-IRT plug-in loses the
         all-pass / all-fail routes from its bank -- i.e. exactly the easiest and
         hardest routes -- which biases the SR estimate downwards by construction.
     (b) keep them and make them estimable with mirt's own documented PRIOR
         mechanism (?mirt.model: "PRIOR = (1-10, a1, lnorm, .2, .2)").
   (b) is used, uniformly over all J items:
       PRIOR = (1-J, a1, lnorm, 0, 0.5), (1-J, d, norm, 0, sigma_d).
   log a ~ N(0, 0.5^2) is ATDrive's own calibration prior (atdrive/calibration.py:19
   SIGMA_LOGA) and is fixed.  sigma_d is NOT fixed and is not tuned by hand: it is
   selected per cell by EMPIRICAL BAYES over ATDrive's own grid
   (atdrive/calibration.py:20 SIGMA_B_GRID = 0.5, 0.75, 1, 1.5, 2, 3) with
   ATDrive's own criterion (atdrive/curves.py:item_marginal_loglik) -- mirt is fitted
   once per grid value and the value maximising
       sum_j log int prod_k P(y_kj | a_hat_j, d) N(d; 0, sigma_d^2) dd
   at that fit's (a_hat, theta_hat_EAP) is kept, exactly as
   atdrive/calibration.py:65-78 _eb_fit selects sigma_b.  The baseline therefore
   gets the same prior-selection procedure and the same grid ATDrive gives itself,
   instead of a hand-picked constant.
   Two immaterial re-expressions in the R copy of the criterion
   (catr_static.R:eb_marginal_loglik): the integration grid is [-12, 12] rather
   than BG's [-10, 10] (same 0.025 spacing; d is the wider scale when a_hat < 1),
   and the log-sigmoid is exact instead of ATDrive's log(p + 1e-12).  On a matched
   synthetic case (K = 8, J = 40, a == 1, so d = -b) the two agree to 6 decimals
   for sigma <= 1.5 and differ by 1.7e-5 / 1.5e-2 nats at sigma = 2 / 3 out of
   ~160 -- far below the gaps between grid values, so the argmax is unaffected.
   This replaces an earlier fixed sigma_d = 3 (the loosest grid entry).  That
   constant was NOT neutral: a sweep of the sd alone gave, for cell
   (seed 0, K_cal 4, slot 0; SR 0.7397), B = 30 est 0.7485 / 0.6005 / 0.5516 /
   0.6730 at sd = 1 / 3 / 5 / 10, i.e. an order of magnitude in |est - SR|,
   while the K_cal = 12 cells barely move (0.7519 / 0.7400 / 0.7364 / 0.7344).
   The item ORDER is robust to the sd (top-30 overlap 30/30); the movement is in
   the p-IRT plug-in over the unadministered routes.  Fixing the loosest entry
   also pinned the baseline at a value ATDrive's own empirical Bayes never picks
   on this protocol (over the 192 cells of results/up_frontier_*.json it selects
   sigma_b = 1.5 in 144, 1.0 in 36, 2.0 in 12, and 0.5 / 0.75 / 3.0 never), which
   is why the "regularised no more tightly than ATDrive could pick" argument did
   not hold.  The EB rule here selects from the same set, but the check is a SPOT
   CHECK and it is not uniform: of the nine cells measured for the tie rule
   ((0, 4, 0), (1, 4, 0), (2, 4, 3), (5, 4, 1), (9, 4, 2), (13, 4, 0),
   (15, 4, 3), (0, 12, 0), (3, 8, 2)) eight select sigma_d = 1.5 -- an interior
   maximum, and the value ATDrive's own EB picks in 144 of its 192 cells --
   while (9, 4, 2) selects sigma_d = 0.5, the LOWER BOUNDARY of the grid, which
   ATDrive's own EB never picks over its 192 cells.  That is a consequence of the
   scale caveat below (the grid is applied to mirt's intercept d, so 0.5 on d is
   an induced b-prior sd of 0.5 / a_hat), not of a different criterion, and the
   selected value, the whole profile and the induced sd are recorded per cell
   (info['prior_sd_d'], info['prior_eb_loglik'],
   info['prior_sd_b_induced_min'/'_max']), so the distribution over all 192 cells
   is readable from the harness output rather than asserted here.
   Scale caveat: mirt's prior mechanism acts on the INTERCEPT d = -a * b, so the
   grid is applied on the d scale and the induced difficulty-prior sd is
   sigma_d / a_hat (about 1.6-2.8 in that cell, recorded as
   info['prior_sd_b_induced_min'/'_max']).  ATDrive's sigma_b grid is on b; the
   selection is over the same six values but on the scale mirt exposes.
   Cost: one mirt fit per grid value, i.e. 6 per cell -- 13-18 s of fit() per
   cell in the self-test, of which 11-13 s is mirt itself (~2 s per grid value),
   ~55-60 min for the 192-cell harness (info['n_prior_fits'], and
   info['fit_seconds'] is the sum over the grid).
   With the selected prior the EM converges in 13-21 cycles and every (a, b) is
   finite (a in ~[0.46, 1.34], |b| <= 5 in the six cells checked).  customK
   (ADAPTATION 2) makes mirt treat the single-category routes as dichotomous,
   and its max / min over the empty second
   category emits benign warnings, muffled and counted rather than printed: the
   only two texts seen are "no non-missing arguments to max; returning -Inf" and
   its min counterpart, 16 of them in the K_cal = 12 self-test cell (10 constant
   routes), 118 in the K_cal = 4 cell (62 constant routes), 50 in a K_cal = 8
   cell.  info['n_warnings'] (the SELECTED fit's count) and info['warnings']
   (unique texts) record them; they do not affect the estimates -- the fit
   converges and every parameter is finite.
   All of it is recorded: info['prior'], info['prior_sd_d'], info['prior_grid'],
   info['prior_eb_loglik'], info['prior_sd_b_induced_min'/'_max'],
   info['n_constant_items'], info['constant_items'].
2. technical = list(customK = rep(2L, J)) (catr_static.R:103).  mirt's
   single-category abort at PrepData:74 happens BEFORE any prior is applied; the
   only documented way past it is technical$customK (?mirt technical, used at
   PrepData:70-73), which declares every item dichotomous.  All cells are already
   0/1, so this changes no data -- without it adaptation 1 cannot even be reached.
3. Difficulty conversion (catr_static.R:31-41).  mirt reports the 2PL as
   a1 * theta + d; catR's itemBank is the 4PL (a, b, c, d) matrix (?MWI Details),
   so b = -d / a1, c = 0, d = 1.  This is the same conversion the official ATLAS
   03_atlas_cat.r:prepare_item_bank applies to a mirt fit; no algorithm is
   involved.
4. Ii over the calibration panel (catr_static.R:132-135).  catR has no
   "sum of information over a set of abilities" entry point -- Ii evaluates one
   ability at a time -- so the sum over the K_cal EAP abilities is done in R
   around K_cal calls to catR::Ii.  The information itself is entirely Ii's.
5. MWI is a per-item function (catr_static.R:139-143): catR::nextItem would apply
   it, but nextItem selects ONE item from a bank with an administered set and
   writes its own CAT loop; the static marginal order is the full ranking of the
   step-0 MPWI, so MWI is called once per item and the vector is ranked.  No part
   of the MPWI value is re-implemented.
6. p-IRT readout (catr_static.R:196-208).  catR's own report is the ability, not
   a success rate; the plug-in asked for is assembled in R from catR::thetaEst
   (BM = Bayes-modal / MAP with the N(0, 1) prior, catR's own default prior) and
   catR::Pi.  thetaEst's default range = c(-4, 4) and D = 1 are kept.
7. 'items' is the total_fisher subset only.  The marginal_fisher variant is a
   SECOND selection rule, not another readout of the same subset: it reads its
   own B routes, and the two subsets can differ a lot -- over the nine cells
   measured for the tie rule the overlap is 139-165 of the 165 and 0-30 of the 30
   (cell (15, 4, 3) shares none of its first 30 with total_fisher).
   Reporting one 'items' list per row is the harness's contract, so the primary
   order's cost is the one reported and the per-budget 'note' records the
   marginal order's own cost and overlap.  Each readout uses only its own subset;
   selftest_catr.py checks that separately (perturbing the marginal-only routes
   leaves 'est' bit-identical while marginal_fisher's own readout moves).  Note
   that base.LeakGuard.values_at LOGS the read set rather than enforcing it, so
   those perturbation tests -- not the guard -- are the evidence that y is read
   only at the selected items.
   Two consequences for the write-up, fixed here BEFORE any error was looked at:
     * the headline catR row is total_fisher (Birnbaum test information), the
       'est' field.  marginal_fisher is reported as its OWN row, never merged
       with it and never as a per-cell minimum of the two -- a best-of-two pick
       over two selection rules would inflate the baseline.
     * both rules cost exactly len(items) = min(B, n_bank) routes, never their
       union: items = order[:B] for each rule separately.  The smallest bank in
       the 192 protocol cells is 210 >= max budget 165, so len(items) == B always
       and the harness's single n_items column is the correct cost for BOTH rows
       (the 'note', which merge() drops, is then redundant rather than the only
       record of the marginal rule's cost).
8. No randomness anywhere: mirt's EM, fscores EAP, Ii, MWI, thetaEst and Pi are
   deterministic given the data.  `seed` is passed to R and used for set.seed()
   before the mirt call only, so that any future RNG use in mirt's start values
   would still be reproducible; it changes nothing today (asserted in the
   self-test's determinism check).
9. Bank routes with no calibration record at all would get parameters from the
   prior alone (a ~ 1, b ~ 0), which would rank them at the TOP of both orders.
   None occur in the 192 protocol cells (checked over every draw x K_cal x slot);
   fit() raises instead of inventing them.
"""
import csv
import json
from pathlib import Path

import numpy as np

from .base import LeakGuard, OfficialMethod, run_rscript

RSCRIPT = Path(__file__).resolve().parent / 'catr_static.R'
ORDERS = (('total_fisher', 'order_total_fisher'), ('marginal_fisher', 'order_marginal_fisher'))


class CatRStatic(OfficialMethod):
    name = 'catr'
    adaptive = False              # static prefix orders: order[:B] is the budget-B subset

    def fit(self, R, seed, workdir):
        R = np.asarray(R, float)
        K, n = R.shape
        if np.isnan(R).all(0).any():
            raise RuntimeError('catr: bank item(s) without any calibration record (ADAPTATION 9)')
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        calib = workdir / 'catr_calibration.csv'
        with open(calib, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([f'X{j}' for j in range(n)])
            for k in range(K):
                w.writerow(['NA' if np.isnan(v) else int(v) for v in R[k]])
        run_rscript(RSCRIPT, ['fit', calib, workdir, int(seed)], cwd=workdir)
        fit = json.load(open(workdir / 'catr_fit.json'))

        info = dict(fit['info'])
        info['order_total_fisher'] = [int(i) for i in fit['order_total_fisher']]
        info['order_marginal_fisher'] = [int(i) for i in fit['order_marginal_fisher']]
        return {'itemBank': [[float(v) for v in row] for row in fit['itemBank']],
                'orders': {name: [int(i) for i in fit[key]] for name, key in ORDERS},
                'theta_cal_eap': [float(t) for t in np.atleast_1d(fit['theta_cal_eap'])],
                'total_info': [float(v) for v in fit['total_info']],
                'marginal_info': [float(v) for v in fit['marginal_info']],
                'workdir': str(workdir), 'info': info}

    def estimate(self, model, y, budgets, seed):
        bank = model['itemBank']
        n = len(bank)
        orders = model['orders']
        workdir = Path(model['workdir'])
        guard = LeakGuard(y)

        admin, subsets = [], {}
        for name, _ in ORDERS:
            for B in budgets:
                items = [int(i) for i in orders[name][:int(B)]]
                x = guard.values_at(items)                       # the only read of y
                subsets[(name, int(B))] = items
                admin.append({'name': f'{name}_B{int(B)}', 'items': items,
                              'x': [int(v) for v in x]})
        payload = workdir / 'catr_admin.json'
        json.dump({'itemBank': bank, 'admin': admin}, open(payload, 'w'))
        out_path = workdir / 'catr_est.json'
        run_rscript(RSCRIPT, ['est', payload, out_path], cwd=workdir)
        out = json.load(open(out_path))

        res = {}
        for B in budgets:
            B = int(B)
            t, m = out[f'total_fisher_B{B}'], out[f'marginal_fisher_B{B}']
            items = subsets[('total_fisher', B)]
            ov = len(set(items) & set(subsets[('marginal_fisher', B)]))
            rec = {'est': float(t['pirt']), 'items': items,
                   'variants': {'marginal_fisher': float(m['pirt']),
                                'total_fisher_theta': float(t['theta']),
                                'marginal_fisher_theta': float(m['theta']),
                                'total_fisher_se': float(t['se']),
                                'marginal_fisher_se': float(m['se']),
                                'total_fisher_mean': float(t['sample_mean']),
                                'marginal_fisher_mean': float(m['sample_mean'])},
                   'note': f'marginal_fisher reads its own {len(subsets[("marginal_fisher", B)])} '
                           f'routes (overlap {ov}); ADAPTATION 7'}
            if len(items) < B:
                rec['note'] = f'bank has only {n} items; ' + rec['note']
            res[B] = rec
        return res

    def stop(self, model, y, seed):
        return {}                                                # static orders, no stopping rule


METHOD = CatRStatic()
