"""metabench (Kipnis et al., 2025) run through the OFFICIAL code,
/data2/jeongtae/official_baselines/metabench (adkipnis/metabench, commit abfeb64),
driven from experiments/official/metabench_reduce.R.  That R script evaluates the
top-level function definitions of the official files verbatim (`source.functions`, which
parses the file and eval()s only its `name <- function(...)` expressions) and supplies the
arguments they expect; no official code is re-typed.

All line numbers below were re-derived against the pinned clones:
  metabench  adkipnis/metabench          abfeb64cd8e6da1b753c8d133ba034f7ed168afb
  ATLAS      Peiyu-Georgia-Li/ATLAS      c0bc19b33aaa11703b8258377abb367533b8a824

What is called (file:function)
  fit        analysis/utils.R:76 run.mirt          -- mirt EM, 1 factor, itemtype '2PL',
             density 'Gaussian', TOL = 1e-4, technical$NCYCLES = 1000 (reduce.R's
             internal = FALSE branch, reduce.R:235-237).
  theta      analysis/utils.R:90 get.theta         -- mirt::fscores(method = 'MAP');
             for the evaluation planner with response.pattern = its subtest responses
             (the reduce.R:356 `get.theta(final$model, method=theta.type, resp=...)` call).
  item info  analysis/reduce.R:117 collect.item.info -- mirt::iteminfo(extract.item(model, i))
             on the sorted theta grid.
  bins       analysis/reduce.R:132 get.info.quantiles -- stats::quantile(theta.grid,
             probs = 0:steps/steps, type = 4), steps = B (ADAPTATION 3).
  select     analysis/reduce.R:154 select.items    -- one route per quantile bin, the
             largest item information in the bin that is >= threshold, single pass,
             each route removed from the pool once taken (threshold = 0, ADAPTATION 2).
  refit      analysis/utils.R:76 run.mirt on the selected columns (their step 3,
             reduce.R:235-237 via create.subtest, reduce.R:182).
  readout    analysis/reduce.R:94 make.df.score builds their (score, theta, ranks) frame;
             the model is reduce.R:105 `mgcv::gam(score ~ s(theta, bs = 'ad'))`
             (ADAPTATION 6).
  p-IRT      ATLAS scripts/04_pirt_accuracy.r:34 prepare_item_parameters, :61
             compute_3pl_prob, :76 compute_pirt_accuracy, called verbatim on csvs written
             in their layout (ADAPTATION 7).

Readouts
  est          the metabench score readout: GAM prediction of the full bank success rate
               from the evaluation planner's MAP ability on the refit subtest.  It is
               reduce.R:105 in every cell run so far; see ADAPTATION 6 for the fallback.
  variants     'sample_mean'  mean of the administered responses,
               'pirt'         ATLAS's p-IRT with the full-fit item parameters and the MAP
                              ability of the evaluation planner under that same full fit
                              restricted to the selected routes (scale-consistent),
               'pirt_refit'   the same formula with the refit subtest ability instead
                              (the literal reading of "p-IRT with the refit 2PL"; the two
                              2PL fits are only approximately on a common scale, so both
                              are reported),
               'wrapper_lin'  WRAPPER-DEFINED, NOT metabench code: `lm(score ~ theta)`,
                              i.e. the reduce.R:105 readout with the adaptive smooth
                              replaced by a straight line, reported so that the GAM's
                              curvature on 4-12 calibration points is readable.
                              metabench's own linear baseline is a different model --
                              meta.R:284 `lm(grand ~ ., data = data.train)`, the grand
                              score regressed on the ITEM RESPONSES (identically at
                              meta.R:319, disjoint.R:237, evaluate.clust.R:78) -- which is
                              rank-deficient here (4-12 respondents, 30-165 regressors)
                              and is therefore not run.  No `lm(score ~ theta)` exists
                              anywhere in the metabench repo.
  stop()       {} -- metabench publishes no stopping rule; reduce.R outputs one subtest.

Not adaptive: the quantile bins depend on B, so the B = 30 subset is NOT a prefix of the
B = 165 subset.  metabench is a fixed-subset method (select first, then read y).

ADAPTATIONS
1. Sourcing (analysis/reduce.R:16-38, ATLAS scripts/04_pirt_accuracy.r:6-31): both files
   are scripts -- reduce.R runs box::use/parse.args/here::i_am at the top level and
   readRDS of data/{BM}-sub-350-seed={seed}.rds at reduce.R:306-307, 04_pirt_accuracy.r
   parses argv and reads the ARC/GSM8K csvs -- so source()ing them executes a benchmark
   run we do not have.  metabench_reduce.R:source.functions parses each file and eval()s
   only its top-level function definitions, which is their code unchanged; analysis/utils.R
   is sourced the same way instead of through box::use(./utils[...]) because box's module
   loader requires the here::i_am project root of the clone.
2. Hyperparameters (reduce.R:42 default.hyperparams, :277-301 optimize.hyperparameters,
   :325-334 the LAMBDA branch): the rBayesianOptimization search needs a subject
   train/validation split (reduce.R:312 caret::createDataPartition(scores, p=0.1)) and
   3PL/4PL fits.  With K_cal in {4, 8, 12} respondents a 10% validation split is 0-1
   planners and 3PL/4PL are not identified, so the search is skipped and their
   `default.hyperparams` (reduce.R:42) are used verbatim: model.type = 1L ('2PL'),
   theta.type = 1L ('MAP'), threshold = 0.0.  Recorded in model['info'].
3. Budget (reduce.R:222 `get.info.quantiles(info.items, theta.grid, steps=N_QUANT)`):
   metabench outputs a subset size, it does not take one.  A fixed budget B is obtained
   by asking for B quantile bins (steps = B) with threshold 0, so select.items returns at
   most one route per bin and at most B routes; the realised count is in info['n_items']
   and, when it is short of B, in the budget's 'note'.
4. Theta grid (reduce.R:205-210): grid.type = 2 (`seq(range(theta), length.out = N_QUANT)`)
   rather than their default grid.type = 1 (the subjects' own thetas, reduce.R:42).
   grid.type = 1 gives collect.item.info exactly K_cal rows, so at most K_cal quantile bins
   can be non-empty and select.items can never return more than K_cal <= 12 routes -- the
   budget grid would be unreachable.  N_QUANT = 500 here, the top of their legal range
   (reduce.R:24 `N_QUANT = seq(100, 500, 1)`), chosen for grid resolution: with B = 165
   quantile bins the number of grid points per bin is what makes the within-bin argmax of
   item information well defined.  DEVIATION: their default is 250 (reduce.R:20
   `defaults = c("arc", 0.005, 250, 1)`), and this is a change of default, not just a
   choice inside the legal range.  Sensitivity, measured on cells (0, 12, 0) and (0, 4, 0):
   250 also fills every bin (n_items 30/55/110/165 at K_cal = 12, and 30/55/110/157 at
   K_cal = 4, the same 157-route cap as at 500), so reachability is not the reason; what
   changes is which route wins each bin -- the selections overlap 28/30, 55/55, 110/110,
   165/165 at K_cal = 12 and 29/30, 51/55, 107/110, 157/157 at K_cal = 4.  The effect is
   therefore small but nonzero, and 500 rather than 250 is a wrapper choice.
5. Item pre-selection (analysis/preprocess.R:154 `items$sd <- apply(data, 2, sd)` and
   :162-164 step 0 "item answers must vary", sd <= 0.01 excluded): applied, because mirt
   refuses items with a single response category (10 of 219 routes at K_cal = 12, 62 at
   K_cal = 4).  Their steps 1 and 2 -- preprocess.R:168-171 "too easy" (diff > 0.95 * 3/4,
   the 4-option multiple-choice guess rate) and preprocess.R:175-177 "part-whole
   correlation" with a per-benchmark limit -- are NOT applied: both constants are
   properties of LLM multiple-choice items.  The dropped routes stay in the bank (the
   estimand is the success rate over all 210-220 routes), they are only ineligible for the
   subtest.  Counted in info['n_fitted'] / info['dropped_low_variance'].
   DEVIATION: the wrapper's filter is `sd(x, na.rm = TRUE)` with an extra `!is.na(sds)`
   guard (metabench_reduce.R:53-56).  preprocess.R:154 has no na.rm because their response
   matrix is complete; with the NA cells of ADAPTATION 9 their line would return NA and the
   comparison would fail.  The `!is.na(sds)` guard additionally drops any route with no
   record at all from the eligible pool (it stays in the bank).
6. Score readout (reduce.R:102-112 get.score.table): their function fits the GAM and
   returns a table of predictions, not the model, so the wrapper issues their own
   modelling line (reduce.R:105 `mgcv::gam(score ~ s(theta, bs='ad'), data=df.train)`) on
   the frame their make.df.score (reduce.R:94) builds and keeps the fitted object to
   predict the new subject.  The GAM does NOT error on 4-12 calibration planners: it warns
   ("there is *no* information about some basis coefficients", "basis dimension is larger
   than number of unique covariates") and fits a degenerate adaptive smooth.  info['readout']
   has been 'gam' at every budget in every cell run.  The wrapper nevertheless carries a
   WRAPPER-DEFINED fallback -- `lm(score ~ theta)`, not metabench code (see 'wrapper_lin'
   above) -- in case mgcv ever does error; it has never triggered, and which one ran is in
   info['readout'] per budget.  Their `score` is a percentage (reduce.R:311
   `scores/max.points.orig*100`); here it is the success rate itself, a fixed rescaling
   that the GAM absorbs.
7. p-IRT (ATLAS scripts/04_pirt_accuracy.r:76-140 compute_pirt_accuracy): the function
   takes a selected-items csv path (item_id, score, theta) and an item-parameter frame and
   ignores its model_name / response_matrix arguments, so the wrapper writes those two
   csvs in their layout and calls it unchanged.  Two of their own imputation branches fire
   on this data, both because of how the wrapper fills the item-parameter table:
     (a) routes dropped by ADAPTATION 5 have no 2PL parameters; they are written with
         a1 = d = NA and g = 0.5, so their compute_3pl_prob invalid-parameter branch
         (04_pirt_accuracy.r:63-65 `if (is.na(a) || is.na(b) || is.na(c) || a <= 0)
         return(c)`) supplies the 0.5 that compute_pirt_accuracy itself uses for unmatched
         items (04_pirt_accuracy.r:122).  At K_cal = 4 that is ~28% of the bank, which is
         why 'pirt' is a variant and not 'est'.
     (b) fitted routes are written with g = 0 (metabench_reduce.R:121), so the SAME
         `a <= 0` branch returns p = 0, not 0.5, for every unobserved route whose 2PL
         discrimination is non-positive.  Negative discriminations are common here
         (10 of 209 fitted routes in cell (0, 12, 0), 7 of 157 in cell (0, 4, 0), 23 of 187
         in cell (3, 8, 2); info['a_min'] is negative in every cell run), so this biases
         'pirt' downward.  It is ATLAS's own
         branch, so no code change is made; the counts are reported instead, in
         info['n_a_nonpositive'] and per budget in the result's 'pirt_impute'
         {'p_zero': routes hitting (b), 'p_half': routes hitting (a)}.
8. Seeding: mirt EM, fscores MAP, the quantile bins and the GAM are all deterministic; the
   harness `seed` is passed to R and set with set.seed() before anything runs, and is
   recorded, but nothing in the pipeline consumes it.  reduce.R's own `seed` argument
   selects the benchmark data split (reduce.R:306-307), which does not exist here.
9. NA cells: run.mirt is given the calibration matrix with NA at unrecorded (route,
   planner) pairs; mirt's EM handles missing responses natively, so no imputation.  The
   consequence for the variance filter is in ADAPTATION 5.
"""
import json
from pathlib import Path

import numpy as np

from .base import LeakGuard, OfficialMethod, run_rscript

RSCRIPT = Path(__file__).resolve().parent / 'metabench_reduce.R'
BUDGETS = (30, 55, 110, 165)          # experiments/official/data.py:BGRID
N_QUANT = 500                         # ADAPTATION 4 (their default is 250, reduce.R:20)
MODEL_TYPE = '2PL'                    # ADAPTATION 2 (reduce.R:42 default.hyperparams)
THETA_TYPE = 'MAP'
THRESHOLD = 0.0
NCYCLES = 1000                        # reduce.R:235 internal = FALSE
TOL = 1e-4                            # analysis/utils.R:76 run.mirt default


class Metabench(OfficialMethod):
    name = 'metabench'
    adaptive = False

    def fit(self, R, seed, workdir):
        R = np.asarray(R, float)
        K, n = R.shape
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        bank_items = [f'i{j}' for j in range(n)]

        with open(wd / 'cal.csv', 'w') as f:                         # ADAPTATION 9
            f.write(','.join(bank_items) + '\n')
            for k in range(K):
                f.write(','.join('NA' if np.isnan(v) else str(int(v)) for v in R[k]) + '\n')
        with open(wd / 'scores.csv', 'w') as f:                      # calibration planners' bank SR
            f.write('score\n')
            for k in range(K):
                f.write(f'{np.nanmean(R[k]):.12f}\n')
        cfg = {'seed': int(seed), 'budgets': [int(b) for b in BUDGETS], 'n_quant': N_QUANT,
               'model_type': MODEL_TYPE, 'theta_type': THETA_TYPE, 'threshold': THRESHOLD,
               'ncycles': NCYCLES, 'tol': TOL, 'bank_items': bank_items}
        json.dump(cfg, open(wd / 'config.json', 'w'))

        run_rscript(RSCRIPT, ['fit', str(wd)], cwd=wd)
        out = json.load(open(wd / 'fit.json'))
        pos = {name: j for j, name in enumerate(bank_items)}
        budgets = {int(B): {'items': [pos[i] for i in np.atleast_1d(v['items']).tolist()],
                            'names': np.atleast_1d(v['items']).tolist(), 'readout': v['readout']}
                   for B, v in out['budgets'].items()}
        info = dict(out['info'])
        info['n_items'] = {str(B): len(v['items']) for B, v in budgets.items()}
        info['readout'] = {str(B): v['readout'] for B, v in budgets.items()}
        info['seed'] = int(seed)                                     # ADAPTATION 8
        return {'workdir': str(wd), 'budgets': budgets, 'n_bank': n, 'info': info}

    def estimate(self, model, y, budgets, seed):
        wd = Path(model['workdir'])
        fitted = model['budgets']
        missing = [int(B) for B in budgets if int(B) not in fitted]
        if missing:
            raise RuntimeError(f'metabench: subsets were selected for {sorted(fitted)}, not for {missing}')
        guard = LeakGuard(y)
        resp = {}
        for B in budgets:
            B = int(B)
            resp[str(B)] = [float(v) for v in guard.values_at(fitted[B]['items'])]
        json.dump(resp, open(wd / 'resp.json', 'w'))
        run_rscript(RSCRIPT, ['estimate', str(wd)], cwd=wd)
        out = json.load(open(wd / 'est.json'))

        res = {}
        for B in budgets:
            B = int(B)
            v = out['budgets'][str(B)]
            items = fitted[B]['items']
            rec = {'est': float(v['est']), 'items': items,
                   'variants': {'sample_mean': float(v['sample_mean']), 'pirt': float(v['pirt']),
                                'pirt_refit': float(v['pirt_refit']),
                                'wrapper_lin': float(v['wrapper_lin'])},
                   # ADAPTATION 7: which of ATLAS's imputation branches the unobserved
                   # routes fall into (p = 0 for a1 <= 0, p = 0.5 for NA parameters).
                   'pirt_impute': {'p_zero': int(v['pirt_p_zero']),
                                   'p_half': int(v['pirt_p_half'])}}
            if len(items) < B:
                rec['note'] = (f'select.items filled {len(items)} of {B} quantile bins '
                               f'({model["info"]["n_fitted"]} of {model["n_bank"]} routes eligible)')
            res[B] = rec
        return res

    def stop(self, model, y, seed):
        return {}


METHOD = Metabench()
