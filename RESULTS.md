# Results of record

Every number here is printed (and asserted, `anchors OK`) by the script named
in each section. Conventions: K = planners, S = scenes, K_cal = calibration
panel size, B = number of routes rolled out, SR-MAE = |SR_hat - SR| averaged
over 64 planner evaluations per cell (16 draws x 4 evaluation planners).
`*` marks a paired-bootstrap 95% CI vs ATDrive that excludes zero; the
bootstrap resamples the 16 unique planners (the same planners recur across
draws; n_eff = 16 planner clusters behind every 64-evaluation cell). Stars are uncorrected
paired-bootstrap 95% intervals; no multiplicity correction is applied across cells. Acquisition ties
(routes of one type with identical posteriors are exchangeable) are broken
by bank order — a documented, deterministic choice. Its sensitivity was measured on the K_cal = 12
cell by breaking exact ties uniformly at random (3 seeds): B = 30 moves from .0477 to .034-.041,
B = 55 from .0231 to .0226-.0265, B = 110 from .0160 to .0141-.0156; 4-34 of the first 55 routes
change. The one Table 1 cell a baseline wins outright (K12 B30, Fluid .0404) is inside that
tie-break range. K_cal = 4 / 8 were not measured and can move more (PROTOCOL section 4).

**Estimand.** SR is the planner's success rate over the routes the simulator completed for
it — the routes with a recorded outcome in the response matrix (210-220 of the 220 Bench2Drive
routes; 3,482 of 3,520 cells observed) — and the bank on which routes are administered, every
estimator, and the true value it is scored against all use that same route set. The 38 missing
cells (TCP-traj 10, LEAD-tfv6 9, VAD 7, UniAD-Tiny 5, ORION 4, MindDrive-3B 2, SimLingo-IVL2-1B 1)
lie on 25 routes concentrated in HighwayExit, MergerIntoSlowTraffic, EnterActorFlow and
InterurbanAdvancedActorFlow; those routes have an observed fail rate of .588 against .475 for
complete routes (full-panel Rasch b_hat +0.51 vs -0.10), so missingness is not at random.
Calibration drops missing cells from the likelihood, i.e. assumes MAR. Under the alternative
convention pass/220 with unrecorded routes counted as failures (the published Bench2Drive number
for TCP-traj, VAD, UniAD-Tiny), SR_220 = SR_rec x n_rec/220 for the truth and the estimate alike,
so every error in this document scales by n_rec/220 in [0.955, 1]: the published SR-MAE is an
upper bound and no cell moves by more than 4.6% (K12 B55 .0231 -> .0227; eps = .05 .0207 -> .0204).

**The UP bank is the whole benchmark.** A new planner is placed on all 220
Bench2Drive routes of all 44 scenario types, and the item bank is calibrated
from the 12 calibration planners on those same routes; B is how many of the
220 are actually executed. Only US and UPS hold scenario types out (36 : 8),
because only they need routes with no calibrated difficulty.

## Table 1 — UP at fixed budgets (`run_up_frontier.py`)

Unseen-planner SR reconstruction on the 16 x 220 Bench2Drive panel (one
planner per model family, PROTOCOL section 1), random 12:4 planner split,
B routes rolled out (30/55/110/165 = 5 x {6, 11, 22, 33} = 14/25/50/75% of
the benchmark). Each method uses its native readout.

| method | K4 B30 | B55 | B110 | B165 | K8 B30 | B55 | B110 | B165 | K12 B30 | B55 | B110 | B165 | macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random (IRT-free) | .0623* | .0397 | .0245 | .0144* | .0623* | .0397 | .0245* | .0144* | .0623* | .0397* | .0245* | .0144* | .0352 |
| Random + IRT | .0536 | .0368 | .0232 | .0138 | .0536 | .0362 | .0232 | .0133* | .0534 | .0362* | .0227* | .0132* | .0316 |
| Random-strat + IRT | .0594* | .0353 | .0210 | .0122 | .0568 | .0331 | .0194 | .0113* | .0565 | .0332* | .0195 | .0112 | .0307 |
| DISCO-sel + IRT | .0566 | .0376 | .0264 | .0137* | .0536 | .0473* | .0271* | .0135* | .0593 | .0423* | .0254* | .0114* | .0345 |
| AnchorPoints | .1290* | .1290* | .1290* | .1290* | .0556 | .0499* | .0392* | .0397* | .0601 | .0449* | .0281* | .0176* | .0709 |
| Total-Fisher | .0512 | .0459* | .0306* | .0198* | .0646* | .0393 | .0221 | .0117 | .0581 | .0430* | .0222 | .0136* | .0352 |
| Marginal-Fisher | .0580 | .0497* | .0309* | .0211* | .0596 | .0413 | .0244 | .0109 | .0576 | .0392* | .0236* | .0133* | .0358 |
| tinyBenchmarks-lite | .0660* | .0488* | .0243 | .0154* | .0516 | .0369 | .0212 | .0122* | .0451 | .0331* | .0204 | .0121 | .0323 |
| metabench-lite | .0728* | .0452* | .0308* | .0204* | .0501 | .0420 | .0276* | .0152* | .0629 | .0388* | .0247* | .0154* | .0372 |
| Fluid-style | .0541 | .0387 | .0312* | .0165* | .0499 | .0364 | .0210 | .0105 | .0404 | .0281 | .0243* | .0119* | .0303 |
| ATLAS-style | .0514 | .0410* | .0277* | .0160* | .0428 | .0320 | .0211 | .0099 | .0437 | .0314 | .0227* | .0121* | .0293 |
| **ATDrive** | **.0450** | **.0332** | .0223 | **.0116** | .0448 | .0337 | .0202 | **.0082** | .0477 | **.0231** | **.0160** | **.0081** | **.0262** |

The three random-policy rows are the expected error over five independent
orders per evaluation; every other row is deterministic. ATLAS-style is
ATLAS as its own system — its 3PL (guessing c and sigma_b profiled on the
calibration block), EAP ability, top-5 randomesque Fisher selection and
p-IRT readout — read at a fixed budget: the prefix of its
`run_system_comparison.py` trajectory on the same draws, planners and banks
(`results/syscmp_table.json`, `K*|ATLAS fixed B=*`; its own stopping rule is
in the complete-system section). AnchorPoints is
degenerate at K_cal = 4: four binary responses admit only 14-23 distinct
route patterns, the correlation distance collapses and the same anchor
estimate comes out at every budget.

Reading. ATDrive has the lowest error in 7 of 12 cells and the lowest macro
average by a clear margin (.0262 against .0293 for ATLAS-style, .0303 for
Fluid-style, .0307 for type-stratified Random and .0316 for Random + IRT).
The five cells it does not win are ties inside the intervals: K4 B110 and
K8 B55/B110, where the type-stratified order is lower by .001-.002, K8 B30,
where ATLAS-style is lower by .002, and K12 B30, where Fluid is lower by
.007. The margin grows with the budget and with the
calibration panel — at K_cal = 12 ATDrive is .0231 / .0160 / .0081 at
B = 55 / 110 / 165 while the best baseline sits at .0281 / .0195 / .0112 —
because a better-calibrated bank makes the Delta-R1 score sharper.

**What each budget buys** (`run_adaptive.py --merge`). SR-MAE is an average;
the decision a user makes needs the distribution. Over the same 64
evaluations, ATDrive's estimate and the pairwise ranking of the four
evaluation planners within a draw (the 6 pairs of estimates ordered as their
true SRs; PROTOCOL section 7):

| K_cal | B | SR-MAE | within 1 pt | within 2 pt | within 3 pt | pairwise rank correct |
|---|---|---|---|---|---|---|
| 4 | 30 | .0450 | 20% | 27% | 38% | 91.7% |
| 4 | 55 | .0332 | 17% | 33% | 55% | 92.7% |
| 4 | 110 | .0223 | 25% | 50% | 70% | 93.8% |
| 4 | 165 | .0116 | 52% | 84% | 98% | 99.0% |
| 8 | 30 | .0448 | 11% | 31% | 44% | 93.8% |
| 8 | 55 | .0337 | 19% | 36% | 56% | 90.6% |
| 8 | 110 | .0202 | 20% | 55% | 80% | 95.8% |
| 8 | 165 | .0082 | 70% | 94% | 97% | 100.0% |
| 12 | 30 | .0477 | 11% | 22% | 36% | 89.6% |
| 12 | 55 | .0231 | 31% | 48% | 70% | 95.8% |
| 12 | 110 | .0160 | 36% | 70% | 84% | 97.9% |
| 12 | 165 | .0081 | 72% | 94% | 98% | 100.0% |

Reading. The two questions separate. Ordering planners is nearly settled at
14% of the cost: 90-94% of the within-draw pairs are already in the right
order at B = 30. Reporting an SR number is not: at B = 30 only 36-44% of
estimates land within 3 SR points, which is the resolution a leaderboard
entry needs, and that only reaches 70-84% at B = 110 and 97-98% at B = 165.
The low budgets are therefore a ranking regime, not a reporting regime, and
the stopping rule below never selects one — it spends 70-129 routes.

## Table 1 through the baselines' OWN code (`run_up_official.py`)

Every published baseline of Table 1 re-run through its official implementation
(`experiments/official/`, one wrapper per method, repository commit ids in
`results/up_official.json`), on the same protocol: the same 16 draws, the same
K_cal subsamples, the same banks and budgets as `run_up_frontier.py`, so each
cell is paired with the ATDrive cell at the evaluation level. 1,344 cells ran;
2 failed (ATLAS at K_cal = 4, below). Every wrapper passes a leak test — flipping
the outcome of every item the method did not select leaves its output bit-identical
— determinism, and the budget / prefix semantics its own code defines.

SR-MAE, each method's native (published) readout; the last column is the best of
the readouts its own code offers, macro over the 12 cells:

| method | K4 B30 | B55 | B110 | B165 | K8 B30 | B55 | B110 | B165 | K12 B30 | B55 | B110 | B165 | macro | best own readout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tinyBenchmarks | 0.0627 | 0.0396 | 0.0294 | 0.0137 | 0.0594 | 0.0446 | 0.0258 | 0.0171 | 0.0483 | 0.0389 | 0.0282 | 0.0162 | 0.0353 | pirt 0.0349 |
| metabench | 0.1875 | 0.1532 | 0.2406 | 0.1280 | 0.1479 | 0.1742 | 0.0845 | 0.1105 | 0.0785 | 0.0855 | 0.0860 | 0.0794 | 0.1297 | sample_mean 0.0467 |
| Fluid | 0.0835 | 0.0622 | 0.0448 | 0.0234 | 0.0580 | 0.0432 | 0.0254 | 0.0148 | 0.0626 | 0.0375 | 0.0261 | 0.0134 | 0.0412 | sample_mean 0.0596 |
| AnchorPoints | 0.1459 | 0.1116 | 0.1131 | 0.1081 | 0.0841 | 0.0551 | 0.0463 | 0.0300 | 0.0758 | 0.0478 | 0.0321 | 0.0203 | 0.0725 | anchors_unweighted 0.0428 |
| DISCO | 0.1438 | 0.1469 | 0.1473 | 0.1429 | 0.1381 | 0.1335 | 0.1285 | 0.1061 | 0.1248 | 0.0997 | 0.0899 | 0.0884 | 0.1242 | jsd_gpirt 0.0489 |
| catR Total-Fisher | 0.0447 | 0.0412 | 0.0267 | 0.0144 | 0.0614 | 0.0420 | 0.0257 | 0.0128 | 0.0611 | 0.0426 | 0.0246 | 0.0127 | 0.0342 | marginal_fisher 0.0351 |
| ATLAS | 0.1098 | 0.0924 | 0.0550 | 0.0332 | 0.0898 | 0.0677 | 0.0440 | 0.0216 | 0.0782 | 0.0657 | 0.0411 | 0.0168 | 0.0596 | sample_mean 0.0509 |
| **ATDrive** (`run_up_frontier.py`) | **0.0450** | **0.0332** | **0.0223** | **0.0116** | **0.0448** | **0.0337** | **0.0202** | **0.0082** | **0.0477** | **0.0231** | **0.0160** | **0.0081** | **.0262** | — |

ATDrive is lower in 12 of 12 cells against every method except catR, where it is
lower in 11 of 12 (K4 B30: .0450 vs .0447, a .0003 loss). Mean paired differences
against ATDrive: tinyBenchmarks +.0092, catR +.0080, Fluid +.0151, ATLAS +.0334,
AnchorPoints +.0464, DISCO +.0980, metabench +.1035.

Reading. Two facts matter more than the ordering. First, **our re-implementations
were not strawmen — they were, if anything, generous**: metabench .0372 -> .1297,
DISCO .0345 -> .1242, Fluid .0303 -> .0412, AnchorPoints .0709 -> .0725 and
tinyBenchmarks .0323 -> .0353 when the methods run their own code; only catR
improves on our Total-Fisher (.0352 -> .0342). Second, the native readouts of metabench and DISCO collapse
here for a reason that is about our setting, not their code: metabench's readout is
a GAM fitted on the calibration respondents and DISCO's headline estimator is a
random forest over model signatures, and neither is estimable from 4-12 respondents
(DISCO's headline predictor becomes nearly constant: SR .135 with estimates in
.35-.48). Their best own readouts (metabench sample mean .0467, DISCO jsd + gp-IRT
.0489) are the fair comparison and are still well above ATDrive's .0262. The
paper should quote both.

Method-level findings the official code exposes, which our re-implementations hid:

- **ATLAS's 3PL is unidentified at this panel size.** At K_cal = 12 the EM runs to
  its 100,000-cycle ceiling with discriminations between -74 and +78; at K_cal = 4
  it "converges" with a1 up to 68 and every EAP ability pinned to one grid point,
  so all three SE thresholds fire at the 30-item minimum. In 2 of 192 cells the EAP
  ability comes back NA and the official p-IRT script therefore returns NA — recorded
  as a method-level failure, not imputed. Its own stopping rule
  (SE <= tau, minimum 30 items) gives, at K_cal = 12: tau = .1 -> 61.4 routes and
  SR-MAE .0680, tau = .2 -> 38.7 and .0744, tau = .3 -> 32.2 and .0769. That is a
  different machine from the "ATLAS-style" row of the complete-system table, which is
  our re-implementation.
- **Three of the seven are not reproducible as published.** catR's `nextItem` ends
  with `set.seed(NULL)`, so the official ATLAS loop is not reproducible under a seed
  and two budgets share no prefix (our wrapper seeds the stream deterministically and
  says so). AnchorPoints' `fasterpam` call is unseeded. Fluid's selection and MAP have
  no randomness at all, so a seed changes nothing and there is no seed-to-seed spread
  to average.
- **AnchorPoints at K_cal = 4 is degenerate for a structural reason**: 48.6% of the
  correlation matrix is NaN (constant rows), and the bank has only 14-23 distinct
  response patterns, so the medoids cannot separate more than that many routes however
  large the budget.

### Pairwise ranking accuracy at fixed budgets — the Table 1 companion (`run_up_frontier.py --merge`, `run_up_official.py --merge`)

The paper's Eq. 10 = the pairwise accuracy of Kocmi et al. (WMT 2021): for each
new planner, the fraction of the 12 calibration-pool planners of its draw for
which sgn(SR_hat - SR_j) = sgn(SR_true - SR_j), the true SR over the recorded
routes; sgn(0) = 0, so a tie on one side only scores 0 (an earlier revision
gave such ties 1/2). No true SR is tied on this panel, so the rule matters only
when an estimate equals a pool SR exactly: the IRT-free random mean (189 of
its 46,080 pairs) and AnchorPoints' anchor-weighted readout (19 of 9,216), which
lose .001-.005 per cell; every other row has no tie. 64 evaluations per cell
(ATLAS at K_cal = 4: 62). The estimates are exactly the ones Table 1 scores —
`est` in `up_frontier.json` (stored since this run; the errors are unchanged
to the last digit and the anchors hold) and the official shards' `est` —
and the random rows average the accuracy over their random orders as their
error column averages the error. In rank units the same number is
|r_hat - r| = 12 x (1 - accuracy) (the identity of the ranking-quality
section), so .96 is half a rank position and .98 a quarter.

Table 1 rows (shared 2PL calibration + p-IRT readout for the
re-implementations; the random references; ATDrive):

| method | K4 B30 | B55 | B110 | B165 | K8 B30 | B55 | B110 | B165 | K12 B30 | B55 | B110 | B165 | macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random (IRT-free) | .919 | .951 | .965 | .986 | .919 | .951 | .965 | .986 | .919 | .951 | .965 | .986 | .955 |
| Random + IRT | .934 | .953 | .970 | .986 | .936 | .952 | .971 | .987 | .932 | .952 | .971 | .988 | .961 |
| Random-strat + IRT | .915 | .953 | .973 | .986 | .923 | .957 | .977 | .988 | .922 | .957 | .976 | .988 | .959 |
| DISCO | .953 | .957 | .977 | .991 | .949 | .958 | .969 | .990 | .938 | .967 | .974 | .995 | .968 |
| AnchorPoints | .868 | .868 | .868 | .868 | .940 | .952 | .957 | .957 | .921 | .945 | .964 | .983 | .924 |
| Total-Fisher | .947 | .943 | .973 | .979 | .927 | .952 | .974 | .992 | .938 | .944 | .973 | .988 | .961 |
| Marginal-Fisher | .938 | .930 | .964 | .979 | .917 | .941 | .965 | .993 | .936 | .948 | .967 | .988 | .956 |
| tinyBenchmarks | .935 | .953 | .977 | .984 | .936 | .954 | .984 | .992 | .927 | .965 | .977 | .993 | .965 |
| metabench | .924 | .953 | .969 | .978 | .935 | .941 | .966 | .988 | .924 | .956 | .971 | .983 | .957 |
| Fluid | .952 | .957 | .966 | .984 | .934 | .949 | .980 | .993 | .949 | .964 | .975 | .999 | .967 |
| **ATDrive** | **.958** | **.966** | **.977** | .990 | **.953** | **.962** | .979 | **.997** | **.957** | **.982** | **.986** | **.999** | **.975** |

The official-code rows (each method's own fit, selection and native readout,
as in the Table 1-official section):

| method (own code) | K4 B30 | B55 | B110 | B165 | K8 B30 | B55 | B110 | B165 | K12 B30 | B55 | B110 | B165 | macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AnchorPoints | .829 | .884 | .858 | .861 | .914 | .935 | .940 | .965 | .893 | .936 | .954 | .977 | .912 |
| DISCO | .811 | .822 | .794 | .792 | .826 | .833 | .836 | .879 | .848 | .867 | .895 | .888 | .841 |
| tinyBenchmarks | .923 | .941 | .961 | .986 | .947 | .954 | .969 | .975 | .948 | .948 | .958 | .979 | .957 |
| metabench | .724 | .773 | .741 | .841 | .833 | .844 | .891 | .863 | .897 | .883 | .884 | .896 | .839 |
| catR | .956 | .948 | .973 | .987 | .932 | .952 | .970 | .995 | .936 | .952 | .973 | .988 | .963 |
| Fluid | .891 | .917 | .941 | .966 | .936 | .949 | .974 | .986 | .936 | .961 | .973 | .992 | .952 |
| ATLAS | .859 | .875 | .934 | .952 | .866 | .913 | .948 | .973 | .915 | .936 | .958 | .977 | .925 |
| **ATDrive** | **.958** | **.966** | **.977** | **.990** | **.953** | **.962** | **.979** | **.997** | **.957** | **.982** | **.986** | **.999** | **.975** |

Reading. ATDrive orders the new planner best in every one of the 12 cells
against the official-code baselines; against the Table 1 rows it is highest
outright in 8 of 12, tied at K4 B110 (DISCO, tinyBenchmarks .977) and K12
B165 (Fluid .999), and below at K4 B165 (DISCO .991 vs .990) and K8 B110
(tinyBenchmarks .984, Fluid .980 vs .979) — no paired test is run on this
metric. The margins are small
in rank units: at K12 B55 ATDrive's .982 is a quarter of a position, the best
random reference (.957) and the best official baseline (Fluid .961) half a
position. The metric separates methods mostly at B = 30, where the random
references sit at .915-.936 and the official-code readouts of AnchorPoints,
DISCO, metabench and ATLAS at .72-.92, following their SR bias in the
Table 1-official section; the same selection rules under the shared
calibration and readout (upper table) order at .92-.95 there. Fluid's own
code, whose SR-MAE is the closest to ATDrive's in Table 1-official, orders at
.89-.94 at B = 30 against ATDrive's .95-.96.

## Table 1 — AV-testing baselines re-implemented from their papers (`run_av_baselines.py`)

Five efficient-testing methods of the driving literature on the Table 1
protocol: the same 16 draws, K_cal subsamples, banks and budgets, every cell
paired with the ATDrive cell of `up_frontier.json`. FST has no public code (a
sweep of the paper texts, the authors' GitHub accounts and the Feng group's
release organisation found none) and is ported from the paper; the GP method
has official code (github.com/umbrellagong/MFGPreliability, named in the paper
itself) and is run both through that code and as our port. 64 evaluations per cell, `results/up_avbase.json`; * =
the paired cluster bootstrap over the evaluation planners (as in Table 1)
excludes zero against ATDrive.

- **FST** (Li, He, Yang, Hu, Zhang, Feng, "Few-shot testing of autonomous
  vehicles with scenario similarity learning", IEEE T-ITS 2025): a FIXED test
  set of n = B routes and aggregation weights, chosen before the new planner is
  seen, from the K_cal calibration planners as the surrogate vehicle set. The
  cross-attention similarity network (MLP features, reciprocal-L2 attention,
  softmax over the selected routes for every route of the bank) gives a
  selected route the similarity mass of the bank it collects as its weight
  (paper Eq. 13-16); the loss is the max over surrogates of |weighted estimate
  - true mean| (Eq. 17 / 20) plus the paper's fluctuation term with w_M = 1,
  the value of its experiments (under the paper's query-wise normalisation the
  term equals the surrogate's own error, so it is the mean surrogate error
  here); training sets are drawn from k-means clusters of the surrogate
  performance (the paper's critical distribution P_c); one network per cell
  serves every budget (the paper reuses one network for n = 5, 10, 20), then
  the set is optimised per budget. Deviation: the paper moves continuous
  scenario coordinates by gradient descent; the bank is discrete, so the set is
  optimised by best-improvement swap search from the best of 32 P_c draws.
  Readout = the weighted observed outcomes. "scene descriptor" feeds the
  network the route descriptor (the paper's scenario state; the 25-d kinematics
  + 48-d risk descriptors of Table 3A), "response profile" the surrogates'
  responses instead.
- **GP adaptive sampling** (Gong, Feng, Pan, "An adaptive multi-fidelity
  sampling framework for safety analysis of connected and automated vehicles",
  IEEE T-ITS 2023), the single-fidelity method: GP regression on the new
  planner's outcomes over a route feature space; the next route is the one
  whose execution most reduces the variance bound of the accident rate (paper
  Eq. 8-13: a hypothetical sample at the current mean leaves the mean and
  shrinks the covariance in closed form, no refit); RBF kernel, hyperparameters
  by marginal likelihood at every step, 10 random routes first. Deviations: the
  outcome is binary (regression on {0, 1}, delta = .5, constant prior mean; an
  unexecuted route's success probability is that of its predictive outcome —
  latent plus noise variance, the latent alone would call a constant-plus-noise
  fit certain); the candidates are the unexecuted routes (argmax in place of
  continuous optimisation); executed routes enter the estimate with their
  outcome and leave the bound (the paper's f is noise-free, where the two
  coincide). Same two feature spaces.
- **Kernel test case sampling** (Qian, Xu, Xing, Guo, "Test case sampling
  optimization for safety validation of automated driving systems", Nature
  Communications 17:3114, 2026): a FIXED subset of M = B routes from the route
  descriptors alone, no surrogate planner. Step 1, coverage: importance weights
  w = softmax over the cases of a one-layer network trained to minimise the
  information potential sum_{i != j} w_i w_j K(x_i, x_j), then Pareto-order
  sampling (U_i ~ U(0, 1), Q_i = U_i / (1 - U_i) x (1 - w_i) / w_i, the M
  smallest Q; nested over the budgets for one draw of U; five draws, the error
  averaged per draw as for the random rows). Step 2, representativeness:
  distribution-alignment weights lambda = argmin 1/2 lambda' K_zz lambda -
  lambda' Kbar with Kbar = (1/N) 1' K_xz, sum lambda = 1, lambda >= 0 (a convex
  QP). Readout sum_j lambda_j y_j — the paper's accident-rate estimate with unit
  exposure and no correction factor. RBF kernel with the median-distance
  bandwidth (the paper states none). Its Code Ocean capsule (10.24433/CO.9203840.v1)
  was not reachable, so this is from the paper's Methods.
- **DICE** (Farid, Schleede, Huang, Heckman, "Foundation models for rapid
  autonomy validation", ICRA 2025), re-implemented end to end at the bank's
  scale, because the paper's own artefacts — a 34M-parameter masked
  autoencoder pre-trained on 14 million proprietary Zoox snippets, a collision
  head trained on the simulation outcomes of previous software versions, an
  800,000-scenario validation set — are unreleased (no code, no weights, no
  data; "patent pending" on the first author's page). (i) Pre-training
  (`experiments/dice_mae.py`): a masked autoencoder with the paper's recipe on
  the bank's own 2,656 RelGraph windows — ego / agent-track / lane-polyline
  tokens (Bench2Drive has no traffic signals), mask ratio .5 to a zero vector
  (Sec. III-C), a 4-layer transformer, L1 reconstruction on every valid token
  with equal type weights (Sec. III-E), 154k parameters, 64 s on one GPU, no
  label read; the ego-token mean is the 64-d scenario embedding (Sec. VI-A).
  (ii) The difficulty head (Sec. IV-B / VI-A): backbone frozen, the pooled ego
  / track / road embeddings concatenated, an MLP to a scalar, binary
  cross-entropy on the simulation outcomes of previous software versions —
  here every (route, calibration planner) cell with a response is one example,
  as the paper adds a log once per software version, trained per cell on the
  K_cal calibration planners only; its output is the difficulty d (Sec. IV-C).
  (iii) Algorithm 2: concat(z, d) clustered by k-means into M = 10 groups (the
  paper leaves M and the weight of d open: z standardised, d scaled to the
  variance of the whole z block), a cluster drawn with probability
  proportional to K_0 + mean(d) with K_0 = 1 (the value the paper discusses), a
  route uniformly inside it without replacement, nested over the budgets, five
  draws, the error averaged per draw; readout = the paper's stratified count,
  the failures found in a cluster scaled by the inverse of its sampled share
  (an unsampled cluster contributes none), SR = 1 - failures / N. This is
  `dice_full`; two ablations of the port replace the head by the calibration
  planners' failure rate (`dice_mae`) and, additionally, the embedding by the
  route descriptor (`dice_desc`). The paper reports no estimation error — its
  metric is the share of collisions found on its own data — so there is no
  number of its own to reproduce.
- **Sim2Val** (Luo, Yang, Watson, Sharma, Veer, Schmerling, Pavone, "Sim2Val:
  leveraging correlation across test platforms for variance-reduced metric
  estimation", CoRL 2025), through its OFFICIAL package (github.com/NVlabs/sim2val,
  `control_variates_estimator`, unmodified; `experiments/official/sim2val.py`):
  control variates. With n paired samples (the costly metric F and a cheap
  surrogate G of it) and k unpaired surrogate samples, mu = mean_paired(F -
  beta G) + beta mean_unpaired(G), beta = k / (k + n) Var(G)^-1 Cov(G, F), plus
  their variance estimate. In the protocol the costly platform is the new
  planner's execution and the cheap platform the historical response panel:
  G_i = the K_cal calibration planners' mean success on route i (`s2v_mean`)
  or their K_cal individual responses, a missing cell filled by that
  planner's mean (`s2v_vec`, the official matrix version) — the surrogate
  definition is ours, the paper's surrogates are nuPlan open-loop metrics or a
  trained metric correlator. Routes are drawn i.i.d. (the paper's assumption:
  a random order nested over the budgets, five draws, the error and the
  ranking accuracy averaged per draw); the method has no adaptive selection
  and no stopping. The estimate is clipped to [0, 1] (4 of 3,840 draw-budget
  estimates of the matrix version, none of the scalar one) and the plain mean
  replaces it when the official code raises (a singular Var(G): 1 of 3,840,
  matrix version only).
- **Not run: adaptive importance sampling** (O'Kelly, Sinha, Namkoong, Duchi,
  Tedrake, NeurIPS 2018). Its substance is a generative traffic model whose
  proposal distribution the cross-entropy method tilts toward failures, with new
  scenarios drawn from it and a likelihood-ratio-corrected estimate of a rare
  failure probability (10^-3 and below). A fixed 220-route benchmark has no
  generative model to tilt, and its SR of .4-.9 is not a rare event: the
  variance-optimal sampling design for a Bernoulli mean in that range is nearly
  uniform, so what survives the transfer is unequal-probability route sampling
  with a Horvitz-Thompson correction — a survey-sampling estimator, not the
  method. It is cited, not compared.

SR-MAE (mean seconds per cell: FST 11-13, GP 4-10, official GP 19, KTCS 7, DICE-style < 1, Sim2Val < 1):

| method | K4 B30 | K4 B55 | K4 B110 | K4 B165 | K8 B30 | K8 B55 | K8 B110 | K8 B165 | K12 B30 | K12 B55 | K12 B110 | K12 B165 | macro (9) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FST, scene descriptor (faithful) | .0856* | .0558* | .0357* | .0168* | .0611* | .0449* | .0266* | .0135* | .0617 | .0403* | .0243* | .0146* | .0485 |
| FST, response profile | .0986* | .0673* | .0337* | .0155* | .0651* | .0399 | .0292* | .0182* | .0700* | .0493* | .0234* | .0169* | .0529 |
| GP adaptive, scene descriptor (faithful) | .0536 | .0410 | .0254 | .0120 | .0466 | .0397 | .0254 | .0108 | .0495 | .0455* | .0241* | .0116* | .0390 |
| GP adaptive, response profile | .0848* | .0680* | .0443* | .0188* | .0614* | .0534* | .0294* | .0177* | .0591 | .0457* | .0287* | .0171* | .0527 |
| GP adaptive, OFFICIAL code, scene descriptor | .1542* | .1244* | .0977* | .0924* | .1354* | .1133* | .1005* | .0917* | .1340* | .1165* | .0933* | .0832* | .1188 |
| Kernel test case sampling, scene descriptor | .1182* | .0800* | .0457* | .0225* | .1149* | .0835* | .0441* | .0219* | .1170* | .0751* | .0440* | .0226* | .0803 |
| DICE, re-implemented end to end (MAE + head + Algorithm 2) | .0877* | .0548* | .0293* | .0168* | .0835* | .0519* | .0297* | .0164* | .0849* | .0565* | .0301* | .0162* | .0565 |
| DICE, head replaced by the surrogate failure rate | .0857* | .0552* | .0307* | .0172* | .0772* | .0522* | .0288* | .0159* | .0808* | .0561* | .0283* | .0150* | .0550 |
| Sim2Val, OFFICIAL code, surrogate = calibration mean success | .0579 | .0378 | .0226 | .0131 | .0571 | .0419 | .0217 | .0118 | .0551 | .0382 | .0209 | .0126 | .0392 |
| Sim2Val, OFFICIAL code, surrogate = the K_cal response vector | .0631 | .0386 | .0225 | .0132 | .0693 | .0435 | .0224 | .0123 | .0778 | .0439 | .0223 | .0126 | .0448 |
| **ATDrive** (Table 1) | **.0450** | **.0332** | **.0223** | **.0116** | **.0448** | **.0337** | **.0202** | **.0082** | **.0477** | **.0231** | **.0160** | **.0081** | **.0318** |

Paired delta against ATDrive, GP adaptive with the scene descriptor: K4
+.0086 [-.0051, +.0243] / +.0078 [-.0049, +.0217] / +.0030 [-.0045, +.0113] /
+.0004 [-.0026, +.0036] at B = 30 / 55 / 110 / 165, K8 +.0019 / +.0060 / +.0052 /
+.0026 (all intervals include zero), K12 +.0018 [-.0153, +.0195] / +.0224
[+.0117, +.0334] / +.0081 [+.0007, +.0162] / +.0035 [+.0002, +.0066]. FST with
the scene descriptor: +.0406 [+.0236, +.0596] at K4 B30 down to +.0052 at K4
B165, +.0164 / +.0112 / +.0064 / +.0054 at K8, +.0140 [-.0007, +.0304] /
+.0173 / +.0083 / +.0064 at K12. Sim2Val with the mean-success surrogate: +.0129
/ +.0047 / +.0003 / +.0014 at K4, +.0123 / +.0082 / +.0015 / +.0037 at K8, +.0074
/ +.0152 / +.0049 / +.0044 at K12 (ATDrive lower in all twelve cells). Pairwise
ranking accuracy (the Table 1 companion metric, mean over the nine cells; for
the random-draw rows the accuracy of EACH draw averaged, never the accuracy of
the draws' mean estimate, which would be an ensemble): FST 0.940 / 0.935, GP
0.962 / 0.941 (scene / response), the official GP code 0.868, kernel test
case sampling 0.905, DICE 0.933 (end to end; 0.931 / 0.935 for the two
ablations), Sim2Val 0.954 / 0.946, ATDrive .969. (An earlier revision scored the
random-draw rows on the mean of their five estimates — DICE .971, KTCS .945 —
which is an ensemble and overstated them; the per-draw numbers above replace
it.)

The official code (`run_gp_official`: `DiscreteInputs` over the bank with
weights 1/N, `AcqIVR_FP`, `OptimalDesign.seq_sampling(discrete=True)` and its
`failure_probability` readout, unmodified; the wrapper only draws the initial
routes from the bank, resolves the grid index the discrete branch hands to the
acquisition, and uses an isotropic RBF + constant + white kernel) is far behind
our port of the same algorithm (.1188 vs .0390, every cell's interval excluding
zero, +.075 to +.109 against ATDrive). Two behaviours of the released code
account for it, neither of which matters on its own continuous, noise-free
cut-in problem: its readout thresholds the posterior MEAN at the limit (a route
is a failure iff its predicted mean is below 0) without substituting the
observed outcome of an executed route, so with the white-noise term the
executed outcomes are shrunk toward the constant mean; and nothing excludes an
executed route from re-selection, so 3-4 of the first 30 and 31-45 of the 165
selections are repeats (budget spent on routes already run). Our port is the
paper's algorithm with the predictive success probability, the observed
outcome for executed routes and the unexecuted-route candidate set, and is the
number the comparison rests on.

Reading. The GP method on the route descriptor and Sim2Val's official control
variates on the calibration mean are the strongest of the five (macro .0390
and .0392 over the nine Table 1 cells; ATDrive .0318, the random references
.0371-.0422, tinyBenchmarks' own code .0419). Sim2Val is what a random order
plus an unbiased variance-reduced readout gives: it beats the IRT-free random
mean (.0422) by the control variate alone, matches our GP port without any
adaptive selection, and is behind ATDrive in every cell — by .0003-.0015 at
B = 110 and by .007-.015 at B = 30 / 55, i.e. the adaptive choice of routes is
what ATDrive adds on top of a good readout. On the GP method: ATDrive is lower in all twelve
cells, but the paired interval excludes zero only at K_cal = 12 with B >= 55;
at K_cal = 4 and 8 the two are within noise. That method sees the same route
descriptor the ATDrive encoder uses only for NEW routes, and no calibration
response at all — its adaptive route choice on a feature-space surrogate does
most of the work, and it is the closest competitor to ATDrive's own adaptive
rule in the whole comparison. The fixed-design FST sits at .0485 (ATDrive lower
in all twelve cells, eleven with the interval excluding zero), between
tinyBenchmarks and Fluid's own code; the min-max fit to the K_cal surrogates is
exact on them (the paper's Theorem 2) and the question is only how the fixed
set transfers to the new planner — the same failure mode as AnchorPoints,
milder. Feeding either method the surrogates' response profile instead of the
descriptor is worse (.0527 / .0529): a 4-12-dimensional binary feature space
gives an RBF kernel or a similarity net little to work with. Two facts frame the
comparison: both ports use a route descriptor that ATDrive's planner-evaluation
setting never touches (it uses the calibration responses only), and at 165 of
about 215 routes every method converges (.011-.019).

Every method of this section is one wrapper file under `experiments/official/`
(`fst.py`, `gp_port.py`, `mfgp.py`, `ktcs.py`, `dice.py`, `sim2val.py`, sharing
`av_common.py`) with the fit / estimate / stop contract of the official
wrappers and a self-test (`selftest_<name>.py`: budget, leak, determinism,
prefix); `run_av_baselines.py` only drives them. The DICE head is pinned to one
BLAS thread so that its clustering does not depend on the machine's thread
count (the earlier end-to-end number, .0538, was a two-thread run of the same
code).

Kernel test case sampling (.0803) is behind UNIFORM random sampling with the
plain mean at every budget (.115-.118 vs .0623 at B = 30, .075-.084 vs .0397
at B = 55, .044-.046 vs .0245 at B = 110, .022-.023 vs .0144 at B = 165). Two reasons, both structural: the alignment weights
lambda concentrate the estimate on the few routes that stand for the
descriptor distribution, which with binary outcomes cuts the effective sample
far below M; and representativeness in descriptor space is not
representativeness in outcome space — the descriptor's difficulty signal is
weak (Table 3A, rho .50-.55 for the best stacks), so a subset that mirrors the
descriptor distribution mirrors the planner's success rate no better than a
uniform draw does. The method is built for a 300,000-case naturalistic pool
where a representative 118 is the point; on a 220-route benchmark whose SR is
the target it has nothing to add.

DICE (.0565 end to end; .0550 / .0547 for the ablations) is also behind
uniform random sampling at every budget (.078-.088 vs .0623 at B = 30,
.052-.057 vs .0397 at B = 55, .028-.031 vs .0245 at B = 110, .015-.017 vs
.0144 at B = 165), and the three variants give the same number: the head
reproduces the calibration planners' failure rate it is trained on (Spearman
.96-.99 with it in every cell), so the end-to-end pipeline and its ablations
feed Algorithm 2 the same difficulty. The scheme trades estimation variance for
exposure to hard clusters by design: the difficulty tilt (at most 2 x between
clusters at K_0 = 1) buys nothing for an SR of .4-.9, while the stratified
count over ten k-means clusters, several of them a handful of routes, has a
larger variance than the plain mean of a uniform draw. Its pairwise ranking
accuracy is nevertheless as high as ATDrive's (.971 vs .969): the
stratified estimate is noisy but unbiased, and an unbiased noise of .05
rarely crosses the .1-.2 gaps between planners. As a difficulty signal the
bank-trained MAE embedding is real but weaker than the hand-crafted
descriptor (a 10-fold ridge probe of the failure rate over the whole bank:
Spearman +.48 vs +.70), which is why the DICE rows coincide. The
Zoox paper's argument — find more collisions per simulated mile while
covering every cluster — is about a 800,000-scenario pool with rare
collisions, not about the error of a benchmark score.

## Provenance chain of every baseline row

What each baseline row is made of, stage by stage, so that a reader can see
which stage is the cited paper's procedure, which is executed through the
authors' released code, and which is ours. Tags: **P** = the paper's own
procedure, **C** = executed through the authors' released code (commit ids in
`results/up_official.json` and the wrapper docstrings), **O** = ours (an
addition, a substitution forced by the data, or a design choice the paper
leaves open). The wrapper docstrings (`experiments/official/*.py`,
`experiments/run_av_baselines.py`, `experiments/us_official/`) carry the
line-level version of this table. This is a record of sources and
constructions; it is not a re-execution of the papers' own experiments.

Planner evaluation (Table 1; every row on the same 16 draws x K_cal x B):

| row | input | calibration / model | selection | estimator (readout) | target | verdict |
|---|---|---|---|---|---|---|
| tinyBenchmarks | binary matrix, missing cells omitted (O, data) | py-irt multidim 2PL, their fork, D from their validation split (P, C) | K-means anchors on the IRT embedding (P, C) | gp-IRT (P, C) | benchmark score = SR (P) | exact |
| metabench | same | mirt 2PL, their utils.R (P, C) | their quantile-bin information selection, threshold 0 (P, C; O: data) | their GAM score readout (P, C) | full score = SR (P) | exact |
| ATLAS | same; single fit chunk (O) | mirt 3PL (P, C) | catR MFI + randomesque + EAP, stop SE <= tau (P, C) | their p-IRT accuracy script, denominator = fitted items (P, C); full-bank rescale as a variant (O) | accuracy = SR (P) | exact |
| Fluid | same | their 2PL fit script (P, C) | their MFI selection at the MAP ability (P, C) | ability theta (P, C); **SR by a p-IRT plug-in (O)** | paper: theta; here: SR (O) | adapted: our readout |
| DISCO | binary outcomes where the paper takes class probabilities (O, data) | py-irt 2PL via their experiments.py (P, C) | PDS ranking (P, C), two-valued on binary input | their headline signature random-forest metamodel on the source models (P, C; 4-12 training points) | accuracy = SR (P) | adapted: input |
| AnchorPoints | binary votes where the paper takes class probabilities (O, data); K_cal 4-8 below the paper's N >= 10 | correlation clustering, their function (P, C) | PAM medoids, num_medoids = B (P, C) | APW cluster-size-weighted vote (P, C) | accuracy = SR (P) | adapted: input |
| GP adaptive, official code | route descriptor as the scenario coordinates, f = outcome - 1/2, uniform p (O) | their sklearn GPR; isotropic RBF kernel (O; the paper's is 2-d ARD) | their AcqIVR_FP on the discrete grid, the grid index resolved in the wrapper (P, C) | their failure_probability, posterior mean < 0 (P, C) | accident rate = 1 - SR (P) | adapted: input, kernel |
| GP adaptive, our port | same input | our GP with the same acquisition (P; O impl.) | unexecuted-route argmax (O) | predictive success probability + observed outcomes (O) | SR | our port |
| FST | surrogates = the K_cal calibration planners (P role); route descriptor as the scenario state (O) | cross-attention similarity net, min-max surrogate loss, P_c draws (P; O impl., no code) | fixed set by discrete swap search (O; the paper: gradient descent on continuous coordinates) | weighted observed outcomes (P) | performance index = SR (P) | re-implemented |
| KTCS | route descriptor (O) | information-potential weights, Pareto-order sampling, QP alignment weights (P; O impl., capsule unreachable) | fixed set (P) | lambda-weighted outcomes, unit exposure (P; O: exposure) | accident rate -> SR (P role) | re-implemented |
| DICE | our route tensors (O, data) | masked autoencoder with the paper's recipe at our scale; their head on calibration outcomes (P; O impl., no code) | Algorithm 2, M = 10 and K_0 = 1 (P; M, K_0 O) | their stratified count (P, mentioned but not evaluated by the paper) | collision rate -> SR | re-implemented |
| Sim2Val | new-planner outcomes on i.i.d. random routes (P: i.i.d.); surrogate = the calibration planners' mean success or response vector (O) | none (P) | random order, 5 draws (P: i.i.d.; O: draws) | their control_variates_estimator, official package (P, C); clipped to [0, 1] (O) | mean of the costly metric = SR (P) | adapted: surrogate |
| Random references | — | ATDrive's own calibration (O) | random / type-stratified (O) | plain mean or ATDrive's readout (O) | SR | ours |
| catR (static MFI order) | — | — | — | — | — | our construct: excluded (a library, not a published method; the same object is the Fisher ablation) |

Route difficulty (Table 3A / Table V; every row through the same two-stage
Ridge readout, which is ours):

| row | input | difficulty / criticality computation | readout | verdict |
|---|---|---|---|---|
| Min-TTC | reference rollout (O, data) | TTC per Hayward 1972 / Westhofen 2023 Sec. 5.2.1, min over time and actors per their Sec. 5.2 (P; O impl. from the definition) | Ridge -> difficulty (O) | literature computation + our readout |
| EDRF-based risk field | rollout; the single realised future replaces the multimodal predictor (O, substitution of the model's input) | DRP / EDRF / IR / F equations and constants (P; O impl., no code); six route statistics (O) | Ridge (O) | EDRF-based features, not EDRF |
| Agent-JEPA | the bank's 10 Hz rollouts in 5 s windows (O, data) | the official minDrive-JEPA tokenizer, trainer and surprise score, retrained (P, C); route aggregation = mean over windows (O); a re-implementation on expert clips kept as a second row (O impl.) | Ridge (O) | literature method through its code + our readout |
| REEval | the 73-d rollout descriptor as the item's content embedding (O; the paper embeds question text with an LLM) | amortized Rasch calibration z = w . e + b fitted jointly with the response likelihood, the notebook's L-BFGS code verbatim (P, C) | the amortized model's own P and z (P, C); no Ridge | literature method through its code + our features |
| Traffic entropy | rollout converted to the model's format (O) | SMART model through the official CAT-K code and checkpoint (C); the entropy as a difficulty descriptor (O) | Ridge (O) | our construct on a released model: not a literature baseline |
| in-house rows | rollout | ours | Ridge (O) | ours |
| ATDrive encoder | rollout | ours (Section Method) | — | ours |

REEval's amortized calibration (Truong et al. 2025) is a Table 3A row under
the same rule, not a Table 1 row: z = w . e + b fitted jointly with the Rasch
likelihood by their notebook code, with our 73-d route descriptor as the
content embedding, read out by the amortized model itself
(`experiments/us_official/reeval_amortized.py`): AUROC 0.725 / scene-MAE
0.210 / rho +0.456, between the single descriptors and the Ridge
readout of the same descriptor (.758 / .175 / +.533).

## Route-level discrimination at fixed budget (`run_route_discrimination.py`)

Table 1 scores one aggregate per evaluation. This diagnostic asks how well
the posterior predicts the routes it did *not* buy: after B rollouts the
ATDrive posterior gives P(y_s = 1 | D_B) for every bank route, the B
administered routes are removed, and the rest are scored against the
planner's true outcomes with AUROC and Brier. All five bank orders run under
the common ATDrive readout, so only the purchased routes differ; the ATDrive
order is Table 1's stored Delta-R1 selection (reuse verified by recomputing
`r1_traj` on three records, 165 / 165 routes each). 16 draws x K_cal in
{4, 8, 12} x 4 held-out planners = 192 evaluations, 64 per cell. An
evaluation whose unobserved outcomes are all one class has no AUROC and is
dropped from the AUROC average only: 36 of 3,840 scorings (0.9%), all at
B >= 110, all all-failure residuals (the order bought every success),
17 for ATDrive and 19 for Fluid, on the two weakest planners (SR .135 and
.155).

AUROC on each order's own unobserved routes (`*` = paired planner-cluster
95% CI vs ATDrive excludes zero):

| order | K4 B30 | B55 | B110 | B165 | K8 B30 | B55 | B110 | B165 | K12 B30 | B55 | B110 | B165 | macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ATDrive | .7319 | .7467 | .7541 | .7105 | .7707 | .7886 | .8021 | .7976 | .7842 | .8011 | .8049 | .8281 | .7767 |
| Fluid | .7226* | .7299* | .7389 | .7414 | .7555* | .7630* | .7860 | .8009 | .7730* | .7826* | .7986 | .8156 | .7673 |
| metabench | .7215* | .7301* | .7329 | .7419 | .7565* | .7678* | .7702 | .7875 | .7646* | .7670* | .7787 | .8015 | .7600 |
| Random | .7241 | .7376 | .7596 | .7673 | .7571* | .7686* | .7856 | .8008 | .7695* | .7805* | .7957 | .8110 | .7715 |
| Random-strat | .7219* | .7437 | .7692 | .7746* | .7560* | .7752 | .7964 | .8086 | .7674* | .7844* | .8040 | .8181 | .7766 |

Macro over the 12 cells, with the two controls that decide how to read it.
The zero-rollout row scores the same residual sets with the posterior
predictive before any rollout (calibration only, no evaluation-planner
data), so it cannot differ between orders through posterior quality:

| statistic | ATDrive | Fluid | metabench | Random | Random-strat |
|---|---|---|---|---|---|
| AUROC, own residual set | .7767 | .7673 | .7600 | .7715 | .7766 |
| Brier, own residual set | .1341 | .1452 | .1791 | .1717 | .1687 |
| Brier, zero-rollout predictor, same residual sets | .1815 | .1861 | .2258 | .2241 | .2238 |
| AUROC, zero-rollout predictor, same residual sets | .7471 | .7336 | .7201 | .7319 | .7342 |
| SR-MAE, common readout (Table 2 machine) | .0262 | .0284 | .0334 | .0334 | .0288 |
| SD of predicted p on the residual set | 0.1784 | 0.1575 | 0.1692 | 0.1821 | 0.1830 |

| control | result |
|---|---|
| ATDrive - Random-strat AUROC, pooled over the 751 paired evaluations | +.0009, 95% CI [-.0189, +.0194]; ATDrive ahead in 9 of 12 cells; the .0001 macro gap comes from one cell, K4 B165 (Random-strat - ATDrive +.0670 [+.0058, +.1392] on 59 records), without which the macro is .7827 vs .7768; on the records where all five orders have an AUROC the macro is .7772 vs .7763 |
| Zero-rollout predictor vs SR-MAE | macro Brier order identical to the SR-MAE order (rho +1.00) with a between-order spread as wide as the real posteriors' (.0443 vs .0450); AUROC macro rho +0.90 (real posteriors +0.70) |
| Skill over the zero-rollout predictor, Random-strat - ATDrive | Brier skill +.0076 [+.0020, +.0146]; AUROC gain +.0119 [+.0009, +.0232] (Random - ATDrive: +.0049 [-.0012, +.0118] and +.0096 [-.0012, +.0204]; Fluid - ATDrive: -.0065 [-.0081, -.0051] and +.0017 [-.0037, +.0083]) |
| Common evaluation set (routes no order administered; B = 30 and 55 only, 112.3 and 60.9 routes, both classes present in 100% / 99.5% of evaluations) | AUROC ATDrive .7666, Fluid .7591, metabench .7597, Random .7647, Random-strat .7651; Random-strat - ATDrive -.0014 [-.0074, +.0056], Random -.0018 [-.0086, +.0051], Fluid -.0075 [-.0109, -.0038], metabench -.0069 [-.0119, -.0016]; Brier .1600 / .1620 / .1644 / .1619 / .1617, spread across orders .0182 -> .0044; per-cell Spearman with SR-MAE falls from +0.55 to +0.12 (AUROC) and +0.92 to +0.65 (Brier) |
| Rank agreement with SR-MAE, own residual sets | mean per-cell Spearman: AUROC +0.57 (common readout) / +0.72 (native Table 1); Brier +0.76 / +0.73. Same cell winner (common readout): AUROC 7 of 12, Brier 6 of 12. Identical 5-way order: AUROC 1 of 12, Brier 2 of 12. Brier's macro order matches SR-MAE only by placing Random (.0334) below metabench (.0334), a gap below the fourth decimal this document reads as a tie |

Reading. AUROC and SR-MAE agree on the extremes (ATDrive best, metabench
worst) and correlate positively but loosely. Route-level AUROC cannot
resolve ATDrive from the type-stratified order: the pooled difference is
+.0009 with an interval two hundred times wider, so the two are unresolved,
not tied. Brier's apparent agreement with SR-MAE is not evidence about the
posteriors: a predictor that has seen no evaluation-planner data reproduces
the Brier ranking exactly on the same residual sets, and once that baseline
is subtracted the type-stratified order is ahead of ATDrive on both scores.
Both route-level scores are dominated by which routes an order leaves
behind; with the evaluation set held fixed at B = 30 and 55 the five orders
barely separate. The residual-flatness explanation is a correlation only
and fails where it was invoked: at K4 B165 Fluid's residual set is flatter
than ATDrive's (SD .1178 vs .1336) yet its AUROC is higher (.7414 vs .7105).
Neither route-level score should be read as a method comparison without
the common-set or skill-corrected version beside it.

## Table 2 — risk-target stopping (`run_tau_calibration.py`, `run_adaptive.py`)

The configuration a user of the tool is actually in: 12 calibration
planners, a new planner, the whole 220-route benchmark, and no budget
chosen in advance. Each order stops at the first t with c * R1_t <= eps,
where R1 is the posterior L1 risk of the reported SR under the common
readout and c is that order's calibration-fixed risk scale
(leave-one-planner-out on the calibration panel, 90th percentile of
realised / predicted error over t in [10, 220]; never selected on
evaluation planners; ATDrive: c = 1.97 / 2.12 / 1.95 at K_cal = 4 / 8 / 12).
eps is an *error* target. No trajectory is censored: every order may run to
route 220, and none did (cap 0% everywhere). Columns: mean rollouts spent
and the same as a fraction of the benchmark, SR-MAE at the stop, the
calibration gap (mean realised error minus mean c * R1 at the stop; negative =
conservative), and the paired delta of the SR-MAE vs ATDrive.

| K_cal | eps | method | rollouts | of 220 | SR-MAE | gap | d vs ATDrive |
|---|---|---|---|---|---|---|---|
| 4 | .05 | **ATDrive** | **83.5** | **38%** | .0272 | -.022 | — |
| | | Fluid | 98.7 | 45% | .0285 | -.021 | +.0013 |
| | | metabench | 109.7 | 50% | .0300 | -.020 | +.0028 |
| | | Random | 98.7 | 45% | .0267 | -.023 | -.0005 |
| | | Random-strat | 87.9 | 40% | .0219 | -.028 | -.0053 |
| 4 | .03 | **ATDrive** | **128.9** | **59%** | .0164 | -.013 | — |
| | | Fluid | 141.9 | 65% | .0194 | -.010 | +.0030 |
| | | metabench | 158.2 | 72% | .0156 | -.014 | -.0008 |
| | | Random | 150.5 | 68% | .0154 | -.014 | -.0009 |
| | | Random-strat | 139.9 | 64% | .0130 | -.017 | -.0033 |
| 8 | .05 | **ATDrive** | **79.0** | **36%** | .0281 | -.022 | — |
| | | Fluid | 98.8 | 45% | .0216 | -.028 | -.0065 |
| | | metabench | 117.6 | 53% | .0184 | -.031 | -.0097* |
| | | Random | 93.3 | 42% | .0260 | -.024 | -.0021 |
| | | Random-strat | 87.7 | 40% | .0208 | -.029 | -.0073 |
| 8 | .03 | **ATDrive** | **124.9** | **57%** | .0174 | -.013 | — |
| | | Fluid | 142.0 | 65% | .0172 | -.013 | -.0002 |
| | | metabench | 165.6 | 75% | .0127 | -.017 | -.0046 |
| | | Random | 146.2 | 66% | .0165 | -.013 | -.0008 |
| | | Random-strat | 140.3 | 64% | .0122 | -.018 | -.0051* |
| 12 | .05 | **ATDrive** | **70.3** | **32%** | **.0207** | -.029 | — |
| | | Fluid | 88.4 | 40% | .0283 | -.021 | +.0076* |
| | | metabench | 110.0 | 50% | .0207 | -.029 | +.0001 |
| | | Random | 91.2 | 41% | .0294 | -.020 | +.0087* |
| | | Random-strat | 87.4 | 40% | .0221 | -.028 | +.0015 |
| 12 | .03 | **ATDrive** | **114.8** | **52%** | .0138 | -.016 | — |
| | | Fluid | 132.0 | 60% | .0190 | -.011 | +.0052* |
| | | metabench | 159.5 | 73% | .0147 | -.015 | +.0009 |
| | | Random | 144.1 | 65% | .0177 | -.012 | +.0039 |
| | | Random-strat | 140.8 | 64% | .0112 | -.019 | -.0026 |

Reading. For an error target of .05, ATDrive stops after 70-84 of the 220
routes (32-38% of the benchmark); the same target costs
Fluid 88-99, Random 91-99, the type-stratified order 87-88 and metabench
110-118 routes. For .03 it needs 115-129 routes (52-59%) against 132-166 for
the others. The saving is the acquisition, not the scale: every order
carries its own leave-one-planner-out c, and the orders that keep going
arrive at a similar error. Only at K_cal = 12 does the error difference
reach significance in ATDrive's favour (-.009 vs Random, -.008 vs Fluid at
eps = .05, -.005 vs Fluid at eps = .03); at K_cal = 8 two orders that keep
going arrive lower — the type-stratified order at eps = .03 (-.005*) for 15
more routes and metabench at eps = .05 (-.010*) for 39 more routes. The negative gaps
say the scaled risk over-states the realised error at the stop, as a 90th-percentile scale (about
twice the median ratio) must; c at K_cal = 4 rests on four LOO trajectories per draw and ranges
1.25-3.42 across draws (medians 1.97 / 2.12 / 1.95). The raw R1 tracks the
realised error: pooled over the LOO tracks, the decile means of raw R1 and
of |SR_hat - SR| agree within .005 at K_cal = 8 / 12 and within .012 at
K_cal = 4 (K_cal = 8: .0025 / .0021
in the lowest decile, .0147 / .0159 in the middle, .0601 / .0652 in the
highest).

**Matched cost (appendix).** The earlier rule — tau_hat chosen so that the
LOO mean rollouts hit a target — is kept as the cost-matched comparison. On
a tau sweep, ATDrive's adaptive stop matches its own fixed-budget curve at
the same mean cost within .005 in every cell (K_cal = 12, tau = .040: 37.5
rollouts, .0344 adaptive vs .0391 fixed), so the stopping rule spends the
budget as well as a fixed budget of the same mean length: at matched mean cost the adaptive stop's
error equals the fixed-budget error (paired deltas -.0047 to +.0034; the stop is a budget-selection
device that converts an SR-unit error target into a stopping time, not an accuracy gain).

## K_cal = 15 leave-one-planner-out supplement (`ATDRIVE_UP_LOO=1`, `results/loo15/`)

The 12 : 4 draws cap the calibration panel at 12 planners. To see the UP
tables with every other planner in the panel, the same scripts run under
`ATDRIVE_UP_LOO=1 ATDRIVE_KCALS=15 ATDRIVE_RESULTS_DIR=results/loo15`: 16
folds, fold k holds planner k out and calibrates on the other 15, everything
else (bank, budgets, readouts, risk scales, the leave-one-planner-out c inside
each fold) unchanged. Each cell is therefore 16 evaluations — one per
planner — against 64 in the tables of record, and the paired intervals are
about twice as wide. The anchors are not asserted in this mode. This is a
supplement: the protocol of record stays the 12 : 4 draws.

Fixed budgets (`run_up_frontier.py`; the ATLAS-style row is the fixed-B
prefix of its `run_system_comparison.py` trajectory as in Table 1):

| method | B30 | B55 | B110 | B165 | macro |
|---|---|---|---|---|---|
| Random (IRT-free) | .0585 | .0458 | .0262 | .0152 | .0364 |
| Random + IRT | .0394 | .0353 | .0222 | .0133 | .0276 |
| Random-strat + IRT | .0614 | .0376 | .0214 | .0113 | .0329 |
| DISCO-sel + IRT | .0598 | .0460 | .0259 | .0096 | .0353 |
| AnchorPoints | .0536 | .0502 | .0344 | .0190 | .0393 |
| Total-Fisher | .0512 | .0374 | .0208 | .0142 | .0309 |
| Marginal-Fisher | .0592 | .0347 | .0234 | .0125 | .0324 |
| tinyBenchmarks-lite | .0453 | .0334 | .0203 | .0089 | .0270 |
| metabench-lite | .0617 | .0344 | .0221 | .0152 | .0334 |
| Fluid-style | **.0249** | **.0221** | .0190 | .0111 | **.0193** |
| ATLAS-style | .0321 | .0298 | .0223 | .0116 | .0240 |
| **ATDrive** | .0573 | .0258 | **.0149** | **.0061** | .0260 |

Risk-target stopping (`run_tau_calibration.py` / `run_adaptive.py`; c medians
ATDrive 2.02, Fluid 2.08, metabench 2.18, Random 1.88, Random-strat 1.96):

| eps | method | rollouts | SR-MAE | d vs ATDrive |
|---|---|---|---|---|
| .05 | **ATDrive** | **71.8** | .0233 | — |
| | Fluid | 84.9 | .0200 | -.0034 [-.0125, +.0061] |
| | metabench | 109.8 | .0203 | -.0030 [-.0169, +.0113] |
| | Random | 88.1 | .0222 | -.0011 [-.0114, +.0097] |
| | Random-strat | 86.3 | .0211 | -.0022 [-.0108, +.0062] |
| .03 | **ATDrive** | **116.6** | **.0119** | — |
| | Fluid | 129.8 | .0193 | +.0074 [+.0015, +.0139] |
| | metabench | 158.8 | .0177 | +.0057 [-.0013, +.0129] |
| | Random | 142.6 | .0156 | +.0037 [-.0059, +.0142] |
| | Random-strat | 141.4 | .0128 | +.0009 [-.0050, +.0061] |

Complete systems at their own stop (`run_system_comparison.py`,
`run_ranking_quality.py`; RankAcc = insertion accuracy on the 15-planner
leaderboard):

| row | routes | SR-MAE | RankAcc |
|---|---|---|---|
| ATLAS-style SE <= 0.1 (exhausts the bank) | 217.6 | .0000 | 1.000 |
| ATLAS-style SE <= 0.2 | 50.0 | .0291 | .962 |
| ATLAS-style SE <= 0.3 | 30.0 | .0321 | .954 |
| Fluid-style fixed B = 100 | 100.0 | .0223 | .971 |
| **ATDrive eps = .05** | 71.8 | .0233 | .983 |
| ATDrive eps = .03 | 116.6 | .0119 | .983 |
| ATDrive fixed B = 30 / 55 / 110 / 165 | — | .0573 / .0258 / .0149 / .0061 | .938 / .979 / .988 / 1.000 |

Component ablation (`run_ablation.py`, paired vs full):

| B | full | w/o b-uncertainty | w/o testlet | w/o risk acquisition |
|---|---|---|---|---|
| 30 | .0573 | .0528 (-.0045) | .0631 (+.0058) | .0441 (-.0133) |
| 55 | .0258 | .0268 (+.0011) | .0486 (+.0229*) | .0407 (+.0150) |
| 110 | .0149 | .0153 (+.0004) | .0240 (+.0090*) | .0198 (+.0048) |
| 165 | .0061 | .0051 (-.0010*) | .0086 (+.0025) | .0134 (+.0073*) |

Selection orders under the common ATDrive readout (`results/loo15/adaptive.json`
and `cat_objective.json`, B = 30 / 55 / 110 / 165): Delta-R1 .0573 / .0258 /
.0149 / .0061, the Fluid order .0254 / .0264 / .0161 / .0096, 1PL Fisher
.0523 / .0296 / .0181 / .0073, theta-EIG .0502 / .0336 / .0137 / .0066, three
random permutations at B = 30 .0394 / .0441 / .0686. The one-IRT factorial at
about 70 routes: C - B (Fisher vs Delta-R1, fixed length) -.0012 [-.0185, +.0149],
G - B (theta-EIG vs Delta-R1) -.0070 [-.0145, +.0010], B - A (fixed length vs
risk stop) +.0024 [-.0040, +.0080], E - A (ability-SD stop vs risk stop)
+.0064 [-.0015, +.0137].

Reading. With every other planner in the panel the picture splits by budget.
At B >= 110 and at the stricter stop ATDrive is clearly ahead: .0149 / .0061
against the next-best .0190 / .0089 (Fluid / tinyBenchmarks), and at eps = .03
it reaches .0119 in 116.6 routes where Fluid needs 129.8 for .0193 (+.0074,
interval excluding zero). At eps = .05 it stops first (71.8 vs 85-110 routes)
with an error inside the others' intervals, and its ranking accuracy at the
stop (.983) is the highest of the systems. At B = 30 it is the worst of the
adaptive orders: .0573 against Fluid .0249 (the Fluid order read by ATDrive's
own readout gives .0254, so this is the order, not the readout), the
ATLAS-style system .0321 and even two of three random permutations. The error
at B = 30 is a shrinkage of the extreme planners toward the panel middle —
the signed error correlates -.80 with the planner's true SR (-.49 at K_cal =
12), the strongest planner is under-estimated by .04 and the weakest ones
over-estimated by .05-.13 — and it is shared by every order that reads the
1PL posterior (Fisher -.81, theta-EIG -.53, Random -.69). Removing the
testlet or the difficulty posterior does not repair it (+.0058 / -.0045,
both inside their intervals); the Fluid order, whose 2PL Fisher rule moves
with the MAP ability, does not show it. The advantage of Delta-R1 over
Fisher and theta-EIG at matched cost that the 12 : 4 factorial establishes
(+.0062* / +.0031*) is not resolved at n = 16 (-.0012 / -.0070, intervals
containing zero). The macro over the four budgets (.0260 vs Fluid's .0193)
is therefore the wrong summary for this panel size: the low-budget cell
dominates it, and the honest statement is that at K_cal = 15 ATDrive's
selection wins from 55 routes on and loses below.

## Adaptive policies under one IRT (`run_cat_objective.py`, `run_policy_matrix.py`)

Complete policies (selection rule x stopping rule) scored on the saved
`results/cat_objective.json` trajectories with the ATDrive IRT, bank and
posterior-median readout held fixed. 192 evaluations plus 384
leave-one-planner-out calibration tracks. Thresholds are published (ATLAS's
tau and 30-route minimum, Fluid's n_max = 100, our eps) or fixed on the
calibration planners only; the two starred rows are our constructions,
matched on the leave-one-out tracks to ATDrive's cost at eps = .05. Fisher
here is the 1PL information at the ATDrive posterior ability
(`atdrive.acquisition.fisher_pick`), so the ATLAS-style and Fluid-style
rows share one selection rule and differ in where they stop; they are not
the Table 1 / Table 2 Fluid rows. The Random order is `run_cat_objective.py`'s
draw (seeded with the 0-3 index of the held-out planner), a different
permutation from Table 2's Random row. IES = (SR-MAE / SR-MAE_ref) x
(routes / 55) with the reference the uniform random order read at 55 routes
(PROTOCOL section 7; reference SR-MAE .0402 / .0375 / .0393 at K_cal = 4 / 8 / 12).

| policy | K4 routes | K4 SR-MAE | K4 IES | K8 routes | K8 SR-MAE | K8 IES | K12 routes | K12 SR-MAE | K12 IES |
|---|---|---|---|---|---|---|---|---|---|
| ATDrive Delta-R1 + c R1 <= .05 | 83.5 | .0272 | 1.02 | 79.0 | .0281 | 1.07 | 70.3 | .0207 | 0.67 |
| ATDrive Delta-R1 + c R1 <= .03 | 128.9 | .0164 | 0.95 | 124.9 | .0174 | 1.05 | 114.8 | .0138 | 0.73 |
| ATLAS-style Fisher + SE <= 0.1 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 |
| ATLAS-style Fisher + SE <= 0.2 | 212.2 | .0015 | 0.15 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 |
| ATLAS-style Fisher + SE <= 0.3 | 100.8 | .0289 | 1.32 | 103.2 | .0228 | 1.14 | 99.9 | .0199 | 0.92 |
| ATLAS-style Fisher + SE <= tau_m * | 79.5 | .0332 | 1.19 | 80.4 | .0325 | 1.27 | 68.9 | .0269 | 0.86 |
| Fluid-style Fisher + fixed B = 100 | 100.0 | .0263 | 1.19 | 100.0 | .0270 | 1.31 | 100.0 | .0203 | 0.94 |
| Fluid-style Fisher + fixed B = match * | 82.9 | .0329 | 1.23 | 80.0 | .0313 | 1.21 | 69.7 | .0270 | 0.87 |
| theta-EIG + SE <= 0.1 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 |
| theta-EIG + SE <= 0.2 | 212.1 | .0017 | 0.17 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 |
| theta-EIG + SE <= 0.3 | 97.0 | .0274 | 1.20 | 96.5 | .0263 | 1.23 | 93.8 | .0217 | 0.94 |
| Random order + c R1 <= .05 | 110.4 | .0195 | 0.97 | 95.1 | .0217 | 1.00 | 112.3 | .0199 | 1.03 |
| Random order + c R1 <= .03 | 160.5 | .0104 | 0.76 | 147.5 | .0123 | 0.88 | 162.0 | .0113 | 0.85 |

ATLAS's tau = 0.1, and tau = 0.2 at K_cal >= 8, are unreachable on this
ability scale: after the whole bank the ATDrive posterior SD of theta only
falls to 0.233 / 0.244 / 0.242, so those rows exhaust the benchmark and
report SR-MAE 0 by construction (SE here is ATDrive's posterior SD, not
ATLAS's 1 / sqrt(sum I + 1); on its own scale ATLAS tau = 0.2 stops at 66.4
routes at K_cal = 8, see the complete-system comparison below).

The factorial that carries the claim. Pooled over K_cal (192 evaluations,
clusters = the 16 planner ids), every arm at 76-78 routes; B = match is the
same integer budget per draw for arms B, C, G, so those arms administer
identical lengths record by record; tau_m for arm E is matched on the
Delta-R1 track's own ability SD:

| arm | selection | stopping | routes | SR-MAE |
|---|---|---|---|---|
| A | Delta-R1 | c R1 <= .05 (ATDrive) | 77.6 | .0253 |
| B | Delta-R1 | fixed B = match | 77.5 | .0242 |
| C | Fisher | fixed B = match | 77.5 | .0304 |
| E | Delta-R1 | SE <= tau_m | 76.2 | .0243 |
| F | Fisher | SE <= tau_m | 76.3 | .0309 |
| G | theta-EIG | fixed B = match | 77.5 | .0274 |

| contrast | what changes | Delta SR-MAE [95% CI] |
|---|---|---|
| C - B | selection only (Fisher vs Delta-R1), identical fixed budget | +.0062 [+.0006, +.0123] |
| F - E | selection only (Fisher vs Delta-R1), SE stop at matched cost | +.0066 [+.0011, +.0128] |
| G - B | selection only (theta-EIG vs Delta-R1), identical fixed budget | +.0031 [+.0003, +.0062] |
| B - A | stopping only (fixed length vs risk stop), Delta-R1 fixed | -.0011 [-.0037, +.0012] |
| E - A | stopping only (ability-SD stop vs risk stop), Delta-R1 fixed | -.0010 [-.0046, +.0023] |
| F - C | stopping only (ability-SD stop vs fixed length), Fisher fixed | +.0005 [-.0022, +.0030] |
| C - A | both rules swapped at once (the only equal-cost row of the table above) | +.0051 [-.0004, +.0112] |
| Fisher - Delta-R1 at exactly fixed budgets | no stopping rule | +.0103 [+.0014, +.0197] at B = 30; +.0100 [+.0019, +.0192] at 55; +.0081 [+.0024, +.0144] at 70; +.0063 [+.0012, +.0118] at 78; +.0057 [+.0008, +.0109] at 80 |

Reading. With the IRT, the bank and the readout held fixed, the selection
rule is what separates these policies and the stopping rule is not:
swapping Fisher for Delta-R1 at an identical budget costs about .006 SR-MAE
with an interval that excludes zero, at every fixed budget in the 30-80
route operating range and under either stopping rule, while swapping the
stopping rule with the selection held fixed moves the error by .001 or less
with intervals that include zero. The mixed contrast C - A is that
selection effect partly cancelled by the stopping effect and blurred by
pairing an adaptive-length arm with a fixed-length one, which is why it
alone does not clear zero. Two nulls follow. ATDrive's adaptive risk stop
is not shown to beat spending its own mean budget as a fixed length (.0253
at 77.6 routes vs .0242 at 77.5). Every non-matched delta in the policy
table is a cost difference: the Random order under our risk stop reaches a
lower SR-MAE than ATDrive eps = .05 in all three cells while spending
+32% / +20% / +60% more routes, and on the cost-adjusted IES it is ahead of
ATDrive at K_cal = 4 and 8 (0.97 vs 1.02, 1.00 vs 1.07) and behind at
K_cal = 12 (1.03 vs 0.67). The ATLAS-style tau_m row at K_cal = 4 spends
79.5 routes against ATDrive's 83.5 and is not cost-matched there.

## Complete-system comparison (`run_system_comparison.py`)

Each published method run as its own system on the protocol split
(PROTOCOL section 8): ATLAS-style (3PL with a bank-wide guessing constant,
EAP ability, top-5 randomesque Fisher, SE <= tau after a 30-item minimum,
p-IRT readout), Fluid-style (2PL, Newton-MAP ability, greedy Fisher, fixed
length or a precision stop built from its own MAP standard error), ATDrive
as in Table 2. Nothing is shared between the rows except the response
matrix, the split and the metric. 192 evaluations; `*` = paired
planner-cluster 95% CI vs ATDrive eps = .05 excludes zero; IES against the
uniform random order read at 55 routes with each system's own readout.

| K_cal | row | rollouts | of 220 | SR-MAE | IES | d vs ATDrive eps = .05 |
|---|---|---|---|---|---|---|
| 4 | ATLAS tau = 0.1 | 217.4 | 99% | .0000 | 0.00 | -.0272* |
|  | ATLAS tau = 0.2 | 77.7 | 35% | .0370 | 1.33 | +.0098* |
|  | ATLAS tau = 0.3 | 32.5 | 15% | .0514 | 0.77 | +.0242* |
|  | Fluid fixed B = 100 | 100.0 | 45% | .0303 | 1.40 | +.0031 |
|  | Fluid fixed B = match * | 82.9 | 38% | .0317 | 1.21 | +.0045 |
|  | Fluid SE <= delta* * | 3.2 | 1% | .1406 | 0.21 | +.1134* |
|  | **ATDrive eps = .05** | **83.5** | 38% | .0272 | 1.02 | — |
|  | ATDrive eps = .03 | 128.9 | 59% | .0164 | 0.95 | -.0108* |
| 8 | ATLAS tau = 0.1 | 217.4 | 99% | .0000 | 0.00 | -.0281* |
|  | ATLAS tau = 0.2 | 66.4 | 30% | .0311 | 1.02 | +.0030 |
|  | ATLAS tau = 0.3 | 30.0 | 14% | .0424 | 0.63 | +.0143* |
|  | Fluid fixed B = 100 | 100.0 | 45% | .0229 | 1.13 | -.0052 |
|  | Fluid fixed B = match * | 80.0 | 36% | .0273 | 1.07 | -.0008 |
|  | Fluid SE <= delta* * | 8.8 | 4% | .0852 | 0.37 | +.0571* |
|  | **ATDrive eps = .05** | **79.0** | 36% | .0281 | 1.05 | — |
|  | ATDrive eps = .03 | 124.9 | 57% | .0174 | 1.02 | -.0107* |
| 12 | ATLAS tau = 0.1 | 217.4 | 99% | .0000 | 0.00 | -.0207* |
|  | ATLAS tau = 0.2 | 56.9 | 26% | .0314 | 0.85 | +.0107 |
|  | ATLAS tau = 0.3 | 30.0 | 14% | .0437 | 0.62 | +.0230* |
|  | Fluid fixed B = 100 | 100.0 | 45% | .0258 | 1.21 | +.0051 |
|  | Fluid fixed B = match * | 69.7 | 32% | .0267 | 0.87 | +.0060 |
|  | Fluid SE <= delta* * | 19.0 | 9% | .0543 | 0.48 | +.0337* |
|  | **ATDrive eps = .05** | **70.3** | 32% | .0207 | 0.67 | — |
|  | ATDrive eps = .03 | 114.8 | 52% | .0138 | 0.74 | -.0069* |

Reading. This is not an equal-budget comparison: every system stops where
its own rule stops, so most rows differ from ATDrive in cost as well as in
error, and only the starred Fluid B = match row spends what ATDrive spends
(there it is within .006 of ATDrive, inside the intervals at every K_cal).
ATLAS at tau = 0.1 exhausts the bank — every route is observed, its SR-MAE
of 0 is the success rate read off the benchmark, and its IES of 0 says
nothing about ATLAS; only tau = 0.3 (and tau = 0.2 at K_cal >= 8) gives a
genuine stop, at 30-78 routes and a higher error than ATDrive's stop.
Fluid's own precision stop halts after 3-19 routes and is far off. IES
prices the cost in but rewards an early stop with a large error (ATLAS
tau = 0.3 and Fluid's precision stop have the lowest IES of the genuine
stops because they halt at 3-33 routes), so it is read with the SR-MAE
column beside it.

**The same two systems through their own code** (`ATDRIVE_OFFICIAL_ORDERS=1
... --merge`, `results/syscmp_official_table.json`): the ATLAS-style and
Fluid-style rows above are our re-implementations. Read from the per-cell
records of `results/up_official.json` — their fit, their selection, their
stopping rule and their p-IRT readout on the same protocol cells, paired to the
same ATDrive evaluations — the published systems behave differently, and worse:

| K_cal | row (official code) | rollouts | of 220 | SR-MAE | d vs ATDrive eps = .05 |
|---|---|---|---|---|---|
| 4 | ATLAS tau = 0.1 | 36.4 | 17% | .1026 | +.0760* |
| | ATLAS tau = 0.2 | 34.1 | 15% | .1093 | +.0827* |
| | ATLAS tau = 0.3 | 31.3 | 14% | .1109 | +.0842* |
| | Fluid fixed B = 55 | 55.0 | 25% | .0622 | +.0350* |
| | Fluid fixed B = 110 | 110.0 | 50% | .0448 | +.0176* |
| 8 | ATLAS tau = 0.1 | 47.9 | 22% | .0759 | +.0478* |
| | ATLAS tau = 0.2 | 34.0 | 15% | .0856 | +.0575* |
| | ATLAS tau = 0.3 | 31.4 | 14% | .0874 | +.0593* |
| | Fluid fixed B = 55 | 55.0 | 25% | .0432 | +.0151* |
| | Fluid fixed B = 110 | 110.0 | 50% | .0254 | -.0027 |
| 12 | ATLAS tau = 0.1 | 61.4 | 28% | .0680 | +.0473* |
| | ATLAS tau = 0.2 | 38.7 | 18% | .0744 | +.0537* |
| | ATLAS tau = 0.3 | 32.2 | 15% | .0769 | +.0563* |
| | Fluid fixed B = 55 | 55.0 | 25% | .0375 | +.0169* |
| | Fluid fixed B = 110 | 110.0 | 50% | .0261 | +.0054 |

The official ATLAS stopping rule never exhausts the bank (its SE <= tau
criterion is met after 31-61 routes at every tau, where our re-implementation
at tau = 0.1 runs to route 217) and its p-IRT readout leaves an error of
.068-.111, four to five times ATDrive's at its eps = .05 stop; two K_cal = 4
cells failed in the official code (EAP ability undefined, n = 62 of 64). Fluid
publishes no stopping rule, so its own system is a fixed budget: at 110 routes
its official code ties ATDrive's 70-84-route stop at K_cal >= 8 (-.0027 /
+.0054, intervals containing zero) and is behind at K_cal = 4 and at 55 routes.
No IES is given for these rows because the official runs have no random
reference read with their own readout.

### Ranking quality of the complete-system trajectories (`run_ranking_quality.py`)

Pure scoring of the saved `results/syscmp.json` trajectories at every
system's own stop plus the ATDrive order at fixed budgets; 192 evaluations,
no refit. Three ranking readings (PROTOCOL section 7). *Insertion accuracy*
places one held-out planner's estimate among the 12 calibration planners of
its draw at their published success rates (fraction of the 12 comparisons
ordered as the truth; |Delta rank| = |r_hat - r| on the 13-rung ranking; no
true SR is tied on this panel, closest pair .0091). *Pairwise rank correct*
is the metric of Table 1: the four held-out planners of a draw ordered
against each other, both sides estimated. The two are different numbers on
the same rows (e.g. K4 B30: 95.8% insertion vs 91.7% pairwise).

| row | routes K4 / K8 / K12 | SR-MAE (macro) | insertion accuracy (macro) | \|Delta rank\| (macro) | pairwise rank correct K4 / K8 / K12 |
|---|---|---|---|---|---|
| ATLAS tau = 0.1 (exhausts the bank; not a ranking result) | 217.4 | .0000 | .0000 | .000 | 100.0 / 100.0 / 100.0 |
| ATLAS tau = 0.2 | 77.7 / 66.4 / 56.9 | .0331 | .9640 | .432 | 93.8 / 93.8 / 92.7 |
| ATLAS tau = 0.3 | 32.5 / 30.0 / 30 | .0458 | .9484 | .620 | 88.5 / 93.8 / 91.7 |
| Fluid fixed B = 100 | 100 | .0264 | .9709 | .349 | 92.7 / 95.8 / 95.8 |
| Fluid fixed B = match | 82.9 / 80.0 / 69.7 | .0286 | .9666 | .401 | 91.7 / 95.8 / 96.9 |
| Fluid SE <= delta* | 3.2 / 8.8 / 19 | .0934 | .8859 | 1.370 | 66.1 / 84.4 / 90.1 |
| **ATDrive eps = .05** | 83.5 / 79.0 / 70.3 | .0253 | .9718 | .339 | 92.7 / 91.7 / 95.8 |
| ATDrive eps = .03 | 128.9 / 124.9 / 114.8 | .0158 | .9848 | .182 | 94.8 / 96.9 / 97.9 |
| ATDrive B = 30 | 30 | .0458 | .9562 | .526 | 91.7 / 93.8 / 89.6 |
| ATDrive B = 55 | 55 | .0300 | .9701 | .359 | 92.7 / 90.6 / 95.8 |
| ATDrive B = 110 | 110 | .0195 | .9805 | .234 | 93.8 / 95.8 / 97.9 |
| ATDrive B = 165 | 165 | .0093 | .9952 | .057 | 99.0 / 100.0 / 100.0 |

At its published eps = .05 stop ATDrive places the new planner at exactly
the right rung in 70 / 64 / 75% of evaluations (macro 69.8%) and within one
rung in 96.9 / 95.3 / 98.4% (macro 96.9%). Fluid's SE <= delta* stop is off
by 2.20 rungs at K_cal = 4 (3.2 rollouts) and 1.37 on macro (10.3 rollouts).

Does ranking separate anything SR-MAE does not? 33 paired comparisons
against ATDrive eps = .05, cluster bootstrap over the 16 planner ids:

| reading | both separate | SR-MAE only | ranking only | both tie |
|---|---|---|---|---|
| insertion (12 published SRs) | 18 | 5-6 | 0 | 9-10 |
| co-estimated, all four held-out planners at their own estimates on the 16-planner board | 15 | 8 | 2 (K8 Fluid B = 100 +.0156 [+.0020, +.0304]; K12 ATDrive B = 110 +.0104 [+.0023, +.0187]) | 8 |
| pairwise rank correct (Table 1 definition, within draw) | 15 | 8 | 3 (K8 Fluid B = 100 +.0417 [+.0051, +.0773]; K8 Fluid B = match +.0417 [+.0051, +.0773]; K12 ATDrive B = 110 +.0208 [+.0050, +.0398]) | 7 |

Reading. Under insertion scoring the answer is no, and that answer is
designed in: with the incumbents entered at their exact published SRs,
|Delta rank| = 12 x (1 - insertion accuracy) identically (maximum deviation
8.9e-16), so both are a deterministic coarsening of the SR error that
SR-MAE averages and cannot separate what that statistic ties. The count of
separations SR-MAE makes and ranking loses is 5 or 6 depending on the
bootstrap draw (K12 ATLAS tau = 0.2 sits on the boundary, +.0107
[-.0001, +.0217]). With both sides estimated the identity no longer holds
and 2-3 cells separate that SR-MAE ties; the cells at K_cal = 8 go against
ATDrive's eps = .05 stop, and one of them is cost-matched (Fluid B = match,
80.0 routes vs 79.0; pairwise rank correct 95.8% vs 91.7%). Under seed
clustering those intervals narrow to touching zero, so the defensible
statement is "not established as ties", not "Fluid ranks better". On this
panel the rank metric is a worse test statistic than SR-MAE for
single-planner placement: the 12-rung leaderboard has a median adjacent-SR
gap of .0318 against an estimator error of .02-.03, so the rank is exact in
1,419 of 2,112 evaluations (excluding the bank-exhausting ATLAS tau = 0.1
row) and moves only when the error is large (mean |err| .0202 where the
rank is exact vs .0619 where it moves). What it adds is the failure rate
above, which the SR scale hides.

## Full-system ablation (`run_system_ablation.py`)

ATDrive is two IRT pieces (the exact difficulty posterior, the planner x type
testlet) and two CAT pieces (the Delta-R1 acquisition, the LOO-calibrated
risk scale c). Each is switched off alone and scored twice: at the fixed
budget B = 55, and under the risk-target rule. Removing c means stopping on
the raw R1 (c = 1), which changes nothing at a fixed budget.

| K_cal | arm | B55 SR-MAE | eps=.05 roll / MAE | eps=.03 roll / MAE |
|---|---|---|---|---|
| 4 | **ATDrive (full)** | .0332 | **83.5** / .0272 | **128.9** / .0164 |
| | w/o b posterior | .0333 (+.0001) | 87.7 / .0266 | 131.2 / .0183 |
| | w/o testlet | .0477 (+.0145*) | 96.3 / .0283 | 139.2 / .0174 |
| | w/o Delta-R1 acquisition | .0405 (+.0074) | 98.7 / .0267 | 150.5 / .0154 |
| | w/o LOO calibration of c | .0332 (=) | 29.0 / .0423 | 63.7 / .0299 |
| 8 | **ATDrive (full)** | .0337 | **79.0** / .0281 | **124.9** / .0174 |
| | w/o b posterior | .0344 (+.0007) | 78.4 / .0263 | 123.3 / .0162 |
| | w/o testlet | .0383 (+.0046) | 102.3 / .0273 | 144.7 / .0147 |
| | w/o Delta-R1 acquisition | .0385 (+.0048) | 93.3 / .0260 | 146.2 / .0165 |
| | w/o LOO calibration of c | .0337 (=) | 26.8 / .0501 | 58.7 / .0365 |
| 12 | **ATDrive (full)** | .0231 | **70.3** / .0207 | **114.8** / .0138 |
| | w/o b posterior | .0270 (+.0039) | 72.1 / .0254 | 116.4 / .0136 |
| | w/o testlet | .0414 (+.0184*) | 94.4 / .0255 | 136.9 / .0169 |
| | w/o Delta-R1 acquisition | .0393 (+.0162*) | 91.2 / .0294 | 144.1 / .0177 |
| | w/o LOO calibration of c | .0231 (=) | 25.9 / .0535 | 57.4 / .0251 |

Reading. The four pieces fail in different ways, which is why all four are
in the method. Dropping the LOO calibration is the one that breaks the
calibration: the raw risk stops after 26-29 routes, less than half the cost,
but the realised error at that stop is .042-.054 against the .05 target it
claims to have met, so the saving is not real. Dropping the testlet or the
acquisition keeps the error at the target but pays for it in routes, 13-24
more at eps = .05 and 10-30 more at eps = .03, and costs .005-.018 SR-MAE at
a fixed budget. The difficulty posterior is the smallest piece: it is
neutral at K_cal = 4 / 8 and worth .004 SR-MAE at K_cal = 12, where the bank
is sharp enough for the shape of the posterior to matter.

## Component ablation at fixed budgets (`run_ablation.py`)

The same three switchable components across the whole budget grid; paired
deltas vs full:

| cell | full | w/o b-uncertainty | w/o testlet | w/o risk acquisition |
|---|---|---|---|---|
| K4 B30 | .0450 | .0454 (+.0004) | .0595 (+.0145) | .0551 (+.0101) |
| K4 B55 | .0332 | .0333 (+.0001) | .0477 (+.0145*) | .0405 (+.0074) |
| K4 B110 | .0223 | .0238 (+.0014) | .0241 (+.0017) | .0243 (+.0020) |
| K4 B165 | .0116 | .0107 (-.0009) | .0126 (+.0010) | .0134 (+.0017) |
| K8 B30 | .0448 | .0398 (-.0050) | .0585 (+.0137) | .0567 (+.0119) |
| K8 B55 | .0337 | .0344 (+.0007) | .0383 (+.0046) | .0385 (+.0048) |
| K8 B110 | .0202 | .0193 (-.0009) | .0252 (+.0050) | .0239 (+.0036) |
| K8 B165 | .0082 | .0074 (-.0008*) | .0101 (+.0019) | .0133 (+.0051*) |
| K12 B30 | .0477 | .0467 (-.0010) | .0595 (+.0118) | .0577 (+.0100) |
| K12 B55 | .0231 | .0270 (+.0039) | .0414 (+.0184*) | .0393 (+.0162*) |
| K12 B110 | .0160 | .0161 (+.0000) | .0238 (+.0078*) | .0244 (+.0084*) |
| K12 B165 | .0081 | .0076 (-.0006*) | .0110 (+.0029*) | .0134 (+.0052*) |

Reading. The testlet and the acquisition carry the fixed-budget error, most
clearly at K_cal = 12 where both are significant at every budget from 55 up
(+.003 to +.018). The difficulty posterior is neutral for the point
estimate — every cell is inside +-.005 and the two significant ones favour
the point curves by .0006-.0008 at B = 165 — because its job is the risk,
not the estimate.

## What the testlet does (`run_ablation.py`, `run_tau_calibration.py` / `run_adaptive.py` with `ATDRIVE_NO_TESTLET=1`)

The same ATDrive with sigma_g fixed to 0 (routes of one type conditionally
independent), everything else unchanged:

| | with testlet | without (sigma_g = 0) |
|---|---|---|
| ATDrive B30 / B55 / B110 / B165 at K4 | .0450 / .0332 / .0223 / .0116 | .0595 / .0477* / .0241 / .0126 |
| at K8 | .0448 / .0337 / .0202 / .0082 | .0585 / .0383 / .0252 / .0101 |
| at K12 | .0477 / .0231 / .0160 / .0081 | .0595 / .0414* / .0238* / .0110* |
| raw R1 vs realised error, LOO deciles 1 / 5 / 10 at K4 | .0030/.0029, .0157/.0174, .0627/.0749 | .0032/.0032, .0159/.0197, .0631/.0749 |
| at K8 | .0025/.0021, .0147/.0159, .0601/.0652 | .0025/.0024, .0145/.0216, .0594/.0762 |
| at K12 | .0022/.0019, .0140/.0144, .0585/.0579 | .0022/.0021, .0134/.0190, .0568/.0640 |
| risk scale c (ATDrive) at K4 / 8 / 12 | 1.97 / 2.12 / 1.95 | 2.30 / 2.88 / 2.77 |
| eps = .05: rollouts, SR-MAE at K4 | 83.5, .0272 | 96.3, .0283 |
| at K8 | 79.0, .0281 | 102.3, .0273 |
| at K12 | 70.3, .0207 | 94.4, .0255 |
| eps = .03: rollouts at K4 / 8 / 12 | 128.9 / 124.9 / 114.8 | 139.2 / 144.7 / 136.9 |

Reading. Without the dependence structure the posterior L1 risk under-states
the realised error in the middle decile by 24-49% (K8 middle decile .0145
vs .0216; |err| / raw R1 - 1 at K_cal = 4 / 8 / 12) and in the top decile by
13-28%, so the calibration has to inflate
it (c 2.3-2.9 instead of 1.9-2.1), the error target of .05 costs 94-102
routes instead of 70-84, and the fixed-budget error rises in every cell. The
testlet is not a small-panel patch: it is what makes R1 a risk. The grouping
it uses is the benchmark's own scenario-type annotation, entered only as
"these routes share an offset"; no difficulty or feature is read from it
(PROTOCOL section 1).

## Table 3A — US: unseen scenes (`run_us.py`)

Predict scene difficulty (and per-cell outcomes) for the 8 evaluation
scenario types from the scene alone; pooled over 16 draws (640 route
evaluations). Descriptor rows are scored through a two-stage Ridge plug-in
fitted on the calibration types; the encoder row is the out-of-fold
prediction of the encoder of record (trained per draw on the 36 calibration
types of the 12 calibration planners). Planner-only null: AUROC .699 /
scene-MAE .214.

Provenance of the descriptor rows (the readout — Ridge from the descriptor
to the calibrated difficulty, then the plug-in — is OURS for every row; what
differs is where the descriptor itself comes from; `experiments/us_official/`,
`results/us_official_provenance/`):
- *literature criticality / difficulty computations*: Min-TTC (the metric of
  Hayward 1972 as reviewed by Westhofen et al. 2023, minimum over time and
  actors per their Sec. 5.2, computed from the definition on the reference
  rollout); EDRF-based risk field (the equations and constants of Jiang et al.
  2024, with the single realised future of the rollout in place of the
  multimodal predictor — a substitution of the model's core input, so the row
  is "EDRF-based features", not EDRF; six route statistics); Agent-JEPA through the
  OFFICIAL code (minDrive-JEPA, github.com/hellojais/mindrive-jepa, unmodified:
  its tokenizer on our 10 Hz rollouts cut into 5 s windows, its trainer with the
  default config on one GPU, best.pt by its best-validation rule, its surprise =
  latent L2; `experiments/us_official/jepa_official_b2d.py`; a route = the mean
  surprise over its windows, ours) and our earlier re-implementation of Sec. 3
  on Bench2Drive expert clips (the paper-literal best-validation checkpoint and
  the full 50-epoch schedule); REEval's amortized calibration (Truong et al.
  2025, github.com/sangttruong/reeval: the Rasch difficulty of an item is
  z = w . e + b from its content embedding e, fitted jointly with the response
  likelihood by the notebook's L-BFGS code, copied verbatim; the 73-d rollout
  descriptor stands in for the paper's LLM text embedding, and the row is read
  out by the amortized model itself, not by our Ridge plug-in;
  `experiments/us_official/reeval_amortized.py`, `results/us_reeval_amortized.json`).
- *our descriptors*: Traffic entropy (the mean next-token entropy of the SMART
  traffic model run through the official CAT-K code and checkpoint; neither
  paper proposes any quantity of the model as a difficulty measure, so the
  descriptor is ours and the papers are cited as the source of the model
  only), Route geometry, Agent density + kin., the in-house traffic risk stack
  (the earlier "Risk field" row: an uncalibrated potential field plus SSM
  columns), Kinematics (cmdkin) and Hand-crafted risk (cmdkin + gtrisk). All
  six read the PDM-Lite reference rollout, so they are probe-rollout
  descriptors rather than scene-only ones.
The earlier shipped rows (Min-TTC .692 from the in-house ssm_min_ttc column,
Risk field .705 = the in-house stack, Traffic entropy .706 from a
mismatched-vocabulary SMART load, Agent-JEPA .696 from an epoch-0
checkpoint) are superseded by the rebuilt rows below. The official JEPA
model, retrained on the bank's 5,445 windows (best validation loss at epoch
6 of 50, then rising), ranks the routes at the null level (AUROC .699,
rho +.005): its surprise carries no planner-difficulty signal on this bank,
which is also what the fully trained re-implementation found; the +.151 of
the re-implementation's epoch-1 checkpoint is the route-length artefact
documented in `results/us_official_provenance/`, not a JEPA signal.

THE ENCODER OF RECORD HAS NO LANE GRAPH. It is RelGraph R2-noLane: the same
R2Net with the whole map side of the graph removed before any tensor is built
— no lane tokens, no lane geometry or lane_feat, no lane-lane edges, no
agent-lane candidates and no ego-route relation — leaving ego, command and
agents. The lane-carrying R2 that earlier releases shipped as canonical is now
a control and heads the control block below.

| difficulty source | AUROC | scene-MAE | rho(b_tilde, fail rate) |
|---|---|---|---|
| Min-TTC (Hayward 1972 / Westhofen 2023; 1-d) | .713 | .213 (+0.2%) | +.217 |
| EDRF-based risk field (Jiang 2024; single realised future; 6-d) | .695 | .217 (-1.6%) | +.099 |
| Agent-JEPA, OFFICIAL code retrained on the bank (mean surprise) | .699 | .215 (-0.5%) | +.005 |
| Agent-JEPA, official code ([mean, max, p90] over windows) | .697 | .216 (-1.0%) | -.019 |
| Agent-JEPA (our re-impl.; paper-literal best-val ckpt) | .711 | .205 (+4.1%) | +.151 |
| Agent-JEPA (our re-impl.; full 50-epoch schedule) | .699 | .216 (-1.0%) | -.003 |
| REEval amortized calibration (Truong 2025; official notebook fit, own readout) | 0.725 | 0.210 (+1.8%) | +0.456 |
| Traffic entropy (ours, on SMART / CAT-K) | .704 | .213 (+0.2%) | +.072 |
| Route geometry (ours) | .719 | .201 (+6.1%) | +.268 |
| Agent density + kin. (ours) | .715 | .212 (+0.9%) | +.220 |
| In-house traffic risk stack (ours; the former "Risk field") | .705 | .213 (+0.3%) | +.091 |
| Kinematics (cmdkin, 25d; ours) | .752 | .180 (+15.6%) | +.497 |
| Hand-crafted risk (cmdkin+gtrisk, 73d; ours) | .758 | .175 (+18.0%) | +.533 |
| **ATDrive: RelGraph R2-noLane scene encoder (3 runs)** | **.761 +- .006** | **.181 +- .007** | **+.545 +- .024** |
| Oracle (response-calibrated) | .870 | .037 | +.995 |

Reading. The learned encoder and the two hand-crafted stacks clear every
single-descriptor row, literature or ours, by +.04-.07 AUROC (16-18 points of
scene-MAE for the hand-crafted stacks, 15 for the encoder); between them the lane-free
encoder is slightly ahead on AUROC (.761 vs .758) and on rank correlation
(R2-noLane minus hand-crafted risk: Delta rho +.012 +- .024 across runs, i.e.
inside its own run-to-run noise) and still behind on scene-MAE (.181 vs .175).
The encoder buys no difficulty signal beyond well-chosen rollout descriptors on
this bank; what it offers is the same signal from the raw scene graph without
feature engineering. Note the direction of the earlier record: the
lane-carrying encoder was BEHIND the hand-crafted stack on rank correlation
(Delta rho -.043 +- .016); dropping the lane graph closed that gap. The oracle
(.870) is the ceiling of any scene-only predictor: roughly half the
difficulty variance is not visible from the scene. (Earlier versions
reported a descriptor stack that included the scenario-definition
parameters; it was removed because those values are the benchmark's own
construction parameters, not observable scene content.)

### Table 3A(b) — structural controls of the encoder (`run_us.py`)

Same architecture, recipe, seeds and calibration as the encoder of record;
the first row keeps the whole lane graph, the next three keep it and damage
one relation, the fifth keeps it and removes an ego channel instead, and the
last removes that channel from the encoder of record itself (three runs each,
Delta rho paired by seed against the lane-free R2-noLane runs):

| variant | AUROC | scene-MAE | rho | Delta rho vs R2-noLane |
|---|---|---|---|---|
| R2, lane graph kept (canonical in earlier releases) | .751 +- .003 | .192 +- .003 | +.490 +- .016 | -.055 +- .031 |
| R2 without the ego-route relation | .756 +- .002 | .184 +- .005 | +.520 +- .014 | -.025 +- .012 |
| R2, route correspondence shuffled | .754 +- .007 | .189 +- .009 | +.512 +- .024 | -.033 +- .024 |
| R2, agent-lane correspondence shuffled | .753 +- .006 | .191 +- .007 | +.501 +- .035 | -.044 +- .043 |
| R2, ego-speed channel removed (channel control) | .753 +- .005 | .191 +- .005 | +.500 +- .020 | -.045 +- .033 |
| R2-noLane, ego-speed channel removed (channel control of the record) | .762 +- .006 | .181 +- .006 | +.547 +- .026 | +.003 +- .013 |

Reading. The lane side of the graph is not inert, it is harmful, and the
controls order themselves by how much of it survives: keeping it whole costs
-.055 rho against the lane-free encoder, dropping the ego-route relation
recovers half of that (-.025), and shuffling the route or agent-lane
correspondence sits between (-.033 / -.044, both within their own seed
spread). Deleting the map side outright beats every one of these variants on
all three metrics, and beats the lane-carrying model itself in all three runs
on all three metrics, which is why the lane-free model is the encoder of
record: a strictly simpler model with the better numbers. What the
encoder learns is carried by the ego and agent tracks, and not by the ego
speed: removing the ego-speed channel changes nothing on this bank, from the
lane-carrying model (-.045 vs the lane-free encoder, i.e. the lane graph's own
cost, and +.010 against the model it ablates) and from the encoder of record
itself (+.003 +- .013, every metric within .002 of the record). The speed
channel matters only on nuPlan (next section). The paper's claim for the encoder is the learned-from-raw-tracks
difficulty prior and its transport to UPS, not the graph structure — and the
one place the lane graph does pay is the UPS per-cell NLL (Table 3B).

### Recipe control — two-stage training of the encoder of record

The encoder of record is trained single-stage: 30 epochs on the calibration
block, no selection. The two-stage recipe the training code also offers
(`--early-stop`: stage 1 holds 30 routes / 6 types of the calibration block
out as an inner validation set and picks the epoch e* of lowest inner-val
NLL; stage 2 refits from the same initial weights on the whole calibration
block for e* + 1 epochs) was run on the same lane-free graph, same seeds,
same draws (`relgraph_e16sel/es/r2nolane_b2d_s{0,1,2}.npz`, scored with
`eval_us_predictions.py` after `build_data.export_relgraph`):

| recipe | AUROC | scene-MAE | rho | paired delta vs single-stage (3 runs) |
|---|---|---|---|---|
| single-stage, 30 epochs (record) | .761 +- .006 | .181 +- .007 | +.545 +- .024 | — |
| two-stage, inner-val e* then refit | .747 +- .002 | .190 +- .002 | +.464 +- .023 | AUROC -.015 +- .004, MAE +.010 +- .008, rho -.081 +- .010 |

The two-stage recipe is worse on all three metrics in every run. The selected
epoch is unstable across draws (e* median 12, range 1-29 over the 48
draw x run pairs; 17% of them stop at e* <= 5), because the inner validation
set is 30 routes of 6 types: its NLL curve is too noisy to select on, and the
refit then under-trains where e* was small. The single-stage recipe stays
the recipe of record. The `--proper-init` variant (re-seed the weights after
the Rasch fit) is bit-identical to `--early-stop` in the current code — the
Rasch fit no longer consumes the torch RNG — so it is not a separate arm.

### Recipe control — a difficulty-matching term in the encoder objective

The encoder of record is trained on the marginal likelihood alone,
-log integral N(b | f_phi(x), sigma_r^2) prod_j Bern(y_ij | sigmoid(theta_hat_j - b)) db
+ 0.05 (log sigma_r)^2, with theta_hat fixed from the calibration block. An
ablation arm (`--match lambda` of the training code, 2026-09-11) adds
lambda x mean_i (f_phi(x_i) - b_hat_i)^2 over the training routes of the batch,
where b_hat_i is the Rasch point difficulty of the same fit that supplies
theta_hat — the encoder is asked to regress the calibration point estimates
directly on top of explaining the responses. Same lane-free graph, seeds,
draws and single-stage 30-epoch recipe as the record
(`relgraph_e16sel/r2nolane_match{0.1,1}_b2d_s{0,1,2}.npz`, exported to
`data/encoder/relgraph_r2nolane_match{0.1,1}_s*.npz`, scored by `run_us.py`
in the control block; deltas paired by seed):

| objective | AUROC | scene-MAE | rho | paired delta vs record (3 runs) |
|---|---|---|---|---|
| marginal likelihood only (record) | .761 +- .006 | .181 +- .007 | +.545 +- .024 | — |
| + matching term, lambda = 0.1 | .763 +- .007 | .177 +- .008 | +.540 +- .032 | AUROC +.001 +- .001, MAE -.004 +- .001, rho -.005 +- .009 |
| + matching term, lambda = 1 | .758 +- .002 | .180 +- .001 | +.518 +- .003 | AUROC -.003 +- .005, MAE -.001 +- .006, rho -.027 +- .022 |

Reading. At lambda = 0.1 the term changes the ranking metrics by nothing
(AUROC +.001, rho -.005, both inside the seed spread, rho lower in two runs of
three) and lowers scene-MAE by .004 in all three runs — a calibration-level
gain of 2% of the null MAE, from pulling the predictions onto the scale of the
point estimates, not a difficulty signal the likelihood had missed. At
lambda = 1 the rank correlation drops in all three runs (-.027, paired) and the
run-to-run spread collapses (SD of rho .003 against .024): with the squared
term dominant the encoder becomes a regressor onto the block-A point
difficulties, which are estimates from 12 planners per route and already
enter the objective through the responses themselves. The marginal-likelihood
objective stays the recipe of record; the arms are kept as controls, not
candidates. On UPS (Table 3B, control priors 3 and 4) they behave like the
other control priors: block-SR MAE unchanged, per-cell NLL .008 (lambda = 0.1)
and .002-.004 (lambda = 1) better than the record.

### Initialisation control — track-SSL pre-training of the encoder of record (`--ssl`)

The one remaining "does self-supervision help" question, asked on the record's
own encoder rather than on frozen features: before the IRT loss, the encoder is
pre-trained on the stage's TRAINING routes to reconstruct hidden agent-track
segments (30% of the live agents per window, a contiguous 4-step segment of
[dx, dy, cos, sin, v], decoded from [agent embedding ; window vector ; step
embedding], MSE). The pretext runs through the window encoder only, so it
pre-trains that path (agent MLP, window ego query, readout attention, z_out;
51,456 of the model's 90,177 parameters); the route-level ego branch and the
difficulty head are not in the pretext's graph and keep their initial weights;
the epoch is chosen by the reconstruction loss on an inner 10% validation split
of the training routes (max 120, patience 6; chosen epochs 41-118, mean 77),
then the
same 30-epoch IRT recipe as the record with the same seeds and batch order. The
only difference from the record is therefore the initialisation.

| initialisation | AUROC | scene-MAE | rho | paired delta vs record (3 runs) |
|---|---|---|---|---|
| random (record) | .761 +- .006 | .181 +- .007 | +.545 +- .024 | — |
| track-SSL | .756 +- .009 | .183 +- .009 | +.528 +- .036 | AUROC -.006 +- .003 (3 of 3 lower), MAE +.002 +- .002, rho -.017 +- .013 (2 of 3 lower) |

No gain: the SSL-initialised encoder is inside the seed spread on scene-MAE and
slightly below the record on AUROC in every run. On 180 training routes the
reconstruction pretext does not find a representation the response likelihood
would not find itself. Together with the block below it closes the question for
this bank: neither frozen foundation features nor self-supervised initialisation
of the track encoder improves on training it on the responses.

### Foundation-model encoder ablation — frozen DINOv3 / SMART features with a light head (`run_us.py` control block)

Asked whether the route encoder could be replaced by frozen foundation-model
features, two frozen sources were attached to the harness with the SAME loss,
draws, seeds, standardisation rule (training routes only) and 30-epoch recipe as
the encoder of record, and scored on the unified split (Table 3A metrics):
- visual: DINOv3 ViT-S/16 (timm `vit_small_patch16_dinov3.lvd1689m`, frozen,
  256 x 256) on every 5th 10-Hz frame of the PDM-Lite reference rollout's
  rgb_front camera (`experiments/visual_features.py`; a first run used ViT-L
  and the three cameras), the CLS token (384-d per frame, 1024-d for ViT-L);
- motion: the SMART-tiny traffic model with CAT-K's WOMD pre-training checkpoint
  (the frozen model of the traffic-entropy row, official code, strict load) run
  over 9.1-s windows every 2 s of the same rollout; per window the agent hidden
  state feeding its next-token head, pooled [mean over valid (agent, step) cells
  ; the ego agent's mean] (256-d; `experiments/motion_fm_features.py`). The model
  reads agent tracks AND the map; its features here are a difficulty input of
  OUR predictor, not a difficulty method of that paper.
Branches (`--visual`, `--motion-fm`): per frame / window a shared MLP (in -> 64 ->
64) then the record's [mean, max, softmin, std] route pooling, concatenated to
the head; `--visual-only` drops the R2-noLane track branch; `--ego` adds the
record's route-level ego branch (speed, acceleration, yaw rate, command) to a
no-track arm. Window-level variants: `--visual-window` (per track window its 12
frames -> projection + learned positions -> bidirectional Transformer -> mean,
fused with z_w, [mean, max] over windows, own head; a 0-layer version is the
temporal ablation) and `--fuse-window` (per window one 64-d token per source +
modality embeddings -> 1-layer 4-head fusion Transformer -> mean -> the record's
route pooling). Three seeds each; Delta rho paired by seed. Planner-only null
AUROC .0.699 / scene-MAE .0.214; kinematics-only Ridge .752 / .180 / +.497.

| arm (front camera, ViT-S unless stated) | AUROC | scene-MAE | rho | Delta rho vs record |
|---|---|---|---|---|
| **encoder of record: RelGraph R2-noLane (tracks, trained)** | **.761 +- .006** | **.181 +- .007** | **+.545 +- .024** | — |
| tracks + front-camera visual | .750 +- .005 | .187 +- .004 | +.477 +- .015 | -.068 +- .012 |
| tracks + front visual + SMART/CAT-K (three branches) | .742 +- .004 | .189 +- .004 | +.445 +- .017 | -.099 +- .019 |
| front visual + SMART/CAT-K + ego status, no track branch | .745 +- .003 | .188 +- .001 | +.445 +- .005 | -.100 +- .029 |
| SMART/CAT-K + ego status, no track branch | .736 +- .003 | .197 +- .003 | +.441 +- .015 | -.104 +- .038 |
| front visual + ego status, no track branch | .741 +- .002 | .194 +- .000 | +.430 +- .014 | -.115 +- .010 |
| front visual + SMART/CAT-K, no track branch, no ego | .713 +- .003 | .213 +- .003 | +.321 +- .017 | -.224 +- .010 |
| front visual only | .702 +- .011 | .225 +- .010 | +.268 +- .036 | -.276 +- .028 |
| SMART/CAT-K only | .689 +- .004 | .228 +- .003 | +.222 +- .016 | -.323 +- .034 |
| window token fusion [visual, track] | .745 +- .005 | .190 +- .002 | +.452 +- .012 | -.093 +- .012 |
| window token fusion [visual, track, SMART] | .736 +- .003 | .195 +- .001 | +.434 +- .022 | -.111 +- .021 |
| window token fusion [visual, SMART] + ego status, no track branch | .727 +- .004 | .203 +- .002 | +.388 +- .020 | -.157 +- .027 |
| window-level visual-temporal fusion (2-layer Transformer, 128-d) | .691 +- .001 | .229 +- .002 | +.178 +- .009 | -.367 +- .029 |
| window-level fusion, 1-layer 64-d Transformer | .692 +- .008 | .227 +- .006 | +.212 +- .032 | -.332 +- .013 |
| window-level fusion, no Transformer (temporal ablation) | .694 +- .007 | .228 +- .004 | +.205 +- .037 | -.340 +- .057 |
| tracks + three-camera DINOv3 ViT-L visual (first run) | .732 +- .005 | .202 +- .004 | +.402 +- .017 | -.143 +- .008 |
| three-camera DINOv3 ViT-L visual only (first run) | .707 +- .009 | .219 +- .007 | +.284 +- .042 | -.261 +- .022 |

The same arms with the difficulty-matching term (lambda = 0.1) added to the loss:

| arm + matching term | AUROC | scene-MAE | rho | Delta rho vs record |
|---|---|---|---|---|
| tracks + front visual | .756 +- .003 | .180 +- .003 | +.481 +- .010 | -.064 +- .016 |
| three branches | .752 +- .004 | .181 +- .005 | +.462 +- .020 | -.083 +- .026 |
| front visual + SMART + ego | .750 +- .000 | .182 +- .002 | +.445 +- .007 | -.100 +- .025 |
| SMART/CAT-K only | .701 +- .003 | .218 +- .002 | +.231 +- .013 | -.314 +- .022 |
| front visual + SMART, no ego | .723 +- .005 | .201 +- .004 | +.315 +- .024 | -.230 +- .007 |
| window-level visual-temporal | .701 +- .003 | .219 +- .002 | +.175 +- .016 | -.370 +- .027 |
| tracks + three-camera ViT-L (first run) | .746 +- .005 | .188 +- .003 | +.421 +- .025 | -.124 +- .010 |
| three-camera ViT-L only (first run) | .722 +- .007 | .205 +- .005 | +.285 +- .041 | -.260 +- .019 |

Reading. No frozen-feature arm reaches the trained track encoder: the best
(tracks + front visual) is .011 AUROC and .07 rho below the record, and every
arm without the track branch sits between the null and the kinematics-only
Ridge. Frozen features alone are at the null (visual .702, SMART .689); adding
the explicit ego status lifts them to .736-.745, which is the ego signal, not
the features (ego kinematics through a Ridge reach .752 with no encoder).
Attaching a frozen branch to the trained encoder lowers it (.761 -> .750 ->
.742): on 180 training routes a 384-d (let alone 3072-d) frozen input adds
dimensions the likelihood cannot select against. The window-level fusion fails
independently of its temporal Transformer (.691 with two layers, .694 with
none), so the structure, not the Transformer, is what does not fit; the
two-token fusion is a bounded loss (.745) but still below the route-level
concatenation of the same sources. The matching term repeats what it did on the
record: +.00-.02 AUROC, no change of order. The encoder of record stays; this
block is the evidence that a learned track encoder is not replaceable here by
frozen foundation features plus a light head, and that what those features
carried was mostly the ego state.
Feature caches: `/data2/jeongtae/visual_feats/dinov3s_2hz` (and `dinov3_2hz` for ViT-L),
`/data2/jeongtae/motion_feats/smart_catk_2s`; the harness records the cache's model
/ checkpoint identity in every output npz (`feature_provenance`) and refuses a
mixed cache. Earlier three-camera, camera-pooled and PCA variants (single seeds)
were superseded by the front-camera ViT-S runs and are not kept.

### Attention maps of the encoder of record (`experiments/viz_encoder_attention.py`)

The lane-free encoder has one attention: per window the ego-motion + command
query over the agent embeddings (Section Method). `r2_graph.py --dump-attn`
retrains one draw with the record recipe (predictions bitwise identical to the
record file) and saves, for the held-out routes, that attention per window and
the saliency ||d f_phi / d z_w|| of every window; `fig_encoder_attention.py`
draws the peak-saliency window of chosen routes (BEV, agents coloured by
attention), `pick_attention_examples.py` ranks candidates. Over the 490
held-out windows of draw 0 the attention is diffuse: median entropy 94% of
uniform, median top weight .26 over a median of 9 live agents. The four routes
of `results/figs/fig_encoder_attention.pdf` are cherry-picked (candidates in
`results/figs/encoder_attention_candidates.txt`): in those the attended agents
are the scenario's actors (the flow vehicles at a merge, the cross traffic of a
left turn, the crossing pedestrians, the cyclist on a right-turn corner).
`results/figs/encoder_architecture.pdf` and `encoder_learning_example.png`
(the per-route likelihood, the learned Gaussian and the objective as a function
of its mean, on real routes) document the model and its training.

## nuPlan val14 zero-shot retrieval (`run_nuplan_zeroshot.py`)

A scene encoder trained on Bench2Drive difficulty ranks 584 nuPlan val14
scenarios zero-shot — a SUBSET of the official split: val14 has 1,118 scenario
tokens spanning 328 val log databases, and the 584 are exactly the val14
tokens contained in the 218 log databases that the auxiliary simulation
server held when the planners were run (the scenario list was fixed before
any simulation; one further token in those logs was left out of the handoff
list; no scenario was removed by a simulation failure or a filter, 6,421 of
the 6,424 planner x scenario cells are scored). Every rate below is relative
to this subset. The score is the drop in planner performance on
the predicted-hard top-q%, Delta M_CLS = M(all 584) - M(top-q%), on the
11-planner closed-loop score (6,421 finite cells, 218 logs; the matrix is
not binary, so M_CLS is the primary metric; M_CLS(full) = .7808, base
failure rate 17.85% with failure = CLS < .5, which reproduces the stored
binary matrix cell for cell). Enrichment of the failure rate is reported
beside it. Three encoder arms — NLe, the encoder of record, which has no lane
graph (the ablation is applied to the Bench2Drive source graph and to the
nuPlan target graph alike); C0e, the lane-carrying encoder earlier releases
shipped; and A2e, lane-carrying with the ego speed removed from both ego paths
— three training seeds each, are trained on the 16-planner panel of record
(`b2d_e2e16sel`) and each tested against label-shuffled encoders trained under
the same ablation, so the NLe-vs-C0e comparison is what the lane graph cost on
transfer. The difficulty target is the repo calibration (`atdrive.calibration.calibrate_dense`, as
for the shipped in-domain encoders), not the frozen driver's own MAP fit
(torch, 400 iterations, fixed L2 penalties) — a second difference from the
driver's native §21 arms, recorded here because they are not directly
comparable to those.

The null is matched to the arm statistic and to its variance structure. An
arm's three seeds share one labeling, so the exchangeable unit under the
null is one labeling with three training seeds: the null family is 20
fixed label permutations x 3 training seeds (60 encoders per arm), the arm
mean is compared with the 20 per-permutation three-seed means — T95 is
their 95th percentile, the verdict is the exact count p = (#{means >=
arm} + 1) / 21 (floor .048, clears = p <= .05) — and a one-way variance
decomposition of the 60 runs gives SD_perm (between labelings) and
SD_train (between training seeds under one labeling), with z = (arm - null
mean) / sqrt(SD_perm^2 + SD_train^2 / 3) and its Gaussian tail p for
resolution below the floor. The oracle ranks by the response-calibrated
difficulty b_ref (in sample) and random q% subsets give the floor.

| q | arm | Delta M_CLS | shuffle mean (60) | T95 | p (20 labelings) | z (Gaussian p) | enrichment | T95 | p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 5% | oracle b_ref | +.4486 | — | — | — | — | 3.619 | — | — | ceiling |
| 5% | random 5% (3,000 draws) | +.0005 | — | — | — | — | 1.003 | — | — | floor |
| 5% | **NLe, lane-free (record)** | **+.2351** | +.0842 | +.1871 | **.048 (0/20)** | +1.64 (.050) | **2.278** | 1.969 | .048 | **clears** |
| 5% | A2e, lane kept, -speed | +.2275 | +.0885 | +.1972 | .048 (0/20) | +1.73 (.042) | 2.202 | 2.057 | .048 | clears |
| 5% | C0e, lane kept, speed kept | +.1914 | +.0791 | +.2052 | .143 (2/20) | +1.40 (.081) | 1.998 | 2.102 | .143 | does not clear |
| 10% | oracle b_ref | +.3631 | — | — | — | — | 3.057 | — | — | ceiling |
| 10% | random 10% | -.0001 | — | — | — | — | 1.000 | — | — | floor |
| 10% | **NLe, lane-free (record)** | **+.1973** | +.0674 | +.1582 | **.048 (0/20)** | +1.60 (.055) | **2.023** | 1.786 | .048 | **clears** |
| 10% | A2e, lane kept, -speed | +.1590 | +.0680 | +.1598 | .095 (1/20) | +1.28 (.101) | 1.842 | 1.836 | .095 | marginal |
| 10% | C0e, lane kept, speed kept | +.1376 | +.0638 | +.1515 | .190 (3/20) | +1.07 (.141) | 1.728 | 1.800 | .190 | does not clear |

Variance components of the null at q = 5%: SD_perm .082 / .067 / .068 and
SD_train .071 / .076 / .073 for the lane-free / speed-kept / -speed families —
under one shuffled labeling the three training seeds spread as much as the
labelings do. The arms' own training-seed SD is .011 (NLe), .059 (C0e) and
.001 (A2e). At q = 5% the single-run 95th percentile of the 60 shuffles is
+.2171 for the lane-free family and +.2205 for both lane-carrying ones; all
three NLe seeds and all three A2e seeds exceed their own, C0e one of three. At
q = 10% all three NLe seeds still exceed theirs (+.1846) while neither
lane-carrying arm has a single seed above its own (+.1807 / +.1868). The arms
recover 52.4% / 54.3% (NLe), 50.7% / 43.8% (A2e) and 42.7% / 37.9% (C0e) of the
oracle ceiling at q = 5% / 10%; the label-shuffled encoders already recover
18-20%.

Threshold-free description (scene sampling only). Paired cluster bootstrap
over the 218 logs of the arm-minus-shuffle contrast (top-q re-selected
inside each resample): NLe - shuffles +.1509 [+.0874, +.2239] at q = 5% and
+.1299 [+.0910, +.1739] at q = 10%; A2e - shuffles +.1390 [+.0757, +.2045]
and +.0910 [+.0385, +.1414]; C0e - shuffles +.1123 [+.0434, +.1857] and
+.0738 [+.0280, +.1154]. This covers the scene-sampling uncertainty
of a fixed set of encoders, not the labeling / training-seed variation,
which is the matched null's job.

Whole-panel Spearman of predicted difficulty with the observed failure
rate: NLe +.425 (seeds SD .009), A2e +.302 (.036) and C0e +.249 (.043)
against per-labeling null means of +.076, +.007 and +.001 (SD_perm .193 /
.167 / .126, SD_train .143 / .123 / .148); one labeling of 20 reaches each
arm (p .095; z +1.66 / +1.62 / +1.63, Gaussian p .048 / .052 / .052) —
marginal for all three, the lane-free arm included.

Reading. On the panel of record, against a null that carries the arm's
own variance structure, the lane-free encoder of record retrieves genuinely
harder scenes at BOTH budgets: no shuffled labeling of 20 reaches its drop at
q = 5% or at q = 10% (p .048 both, z 1.6), and all three of its seeds exceed
the single-run 95th percentile of its own 60 shuffles at both q. Among the
lane-carrying controls the -speed arm clears only at q = 5% (p .048) and is
marginal at q = 10% (p .095), and the speed-kept arm does not clear at either
(p .14 / .19) — dropping the lane graph buys more on transfer than dropping
the speed channel did, and buys it at both budgets. The whole-panel rank
correlation stays marginal for all three arms (p .095), the lane-free arm's
+.425 included: the null's per-labeling spread (SD_perm .19) is wide enough
that 20 labelings cannot resolve it. Two corrections against the previous record: (i) the
earlier arms and nulls were trained on the pre-selection e2e16 matrix
(MindDrive, SimLingo-IVL35-1B and UniAD-Base in place of Drive-pi0-Base,
Hydra-NeXt and PGS), whose calibrated difficulty correlates .943 with the
panel of record; everything here is retrained on the record. (ii) The
earlier null averaged three single-seed shuffles that carried three
different permutations, which divides the permutation variance by three
as well and understated the threshold (that reading, "A2e clears at both
q and both arms clear on the rank correlation", is withdrawn); the frozen
driver ties the permutation seed to the training seed, so the permutation-
fixed family was run through a wrapper that only swaps the seed handed to
the shuffle. Top-q retrieval stays a low-power statistic — per-run shuffle
drops span -.16 to +.24 — and 20 labelings floor the achievable p at .048.

## nuPlan Val14 zero-shot retrieval on the full split (`ATDRIVE_NUPLAN_BUNDLE=data/nuplan/val14_full_zeroshot.npz`)

The section above scores the 584-token log-availability subset. This
section repeats the test on the WHOLE official Val14 split: 1,118 scenario
tokens in all 328 val log databases, 10 planners (IDM, PDM-Closed,
PDM-Hybrid, PDM-Open, GC-PGP, UrbanDriver, PLUTO, DTPP, DiffusionPlanner,
FlowPlanner; STR2 has no full-split run and is dropped), one closed-loop
run per planner on the main server (`irt_v14full_*` for the first seven,
`k12_val14_*` for the last three), 11,141 of the 11,180 planner x scenario
cells finite, base failure rate 19.74% (failure = CLS < .5), M_CLS(full) =
.7596. These runs are not the runs behind the 584 panel: on the 584 overlap
the stochastic planners disagree with the record run (failure labels flip
on 17% of the GC-PGP cells, 4% PLUTO, 2% DTPP, 0.3% IDM), so the full split
uses the one main-server run per planner throughout and its numbers on the
584 tokens differ from the section above. Output
`results/nuplan_zeroshot_full.json`; the panel sources (token lists, the
10 x 1,118 response matrix, the scenario / log / map table) ship in
`data/nuplan/full_val14/`.

The scene tensors of the 534 tokens absent from the 584 were built with
copies of the record pipeline (`encoder/nuplan/full_val14/`); rebuilding the
584 with the same copies reproduces the record tensors, logged-ego file and
relgraph windows bit for bit. The arm is the encoder of record NLe (R2-noLane)
trained on the 16 x 220 Bench2Drive panel of record with the repo
calibration, three training seeds, against its permutation-fixed null C4nl
(20 labelings x 3 seeds); the driver wrapper swaps only the nuPlan file
constants, and on the 584 overlap it reproduces the record predictions to
5e-7 (nuPlan responses enter the score and the oracle b_ref, never the
training). The lane-carrying controls C0e / A2e were not re-run on the full
split. b_ref is the Rasch calibration of the 10 x 1,118 matrix (in sample).
Same statistic, null, verdict rule and bootstraps as above; the paired log
bootstrap resamples the 328 logs; random = 3,000 q% draws.

| q | arm | Delta M_CLS | shuffle mean (60) | T95 | p (20 labelings) | z (Gaussian p) | enrichment | T95 | p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 5% | oracle b_ref | +.4548 | — | — | — | — | 3.458 | — | — | ceiling |
| 5% | random 5% (3,000 draws) | +.0001 | — | — | — | — | 1.001 | — | — | floor |
| 5% | **NLe, lane-free (record)** | **+.1544** | +.0566 | +.1349 | **.048 (0/20)** | +1.28 (.101) | **1.736** | 1.598 | .048 | **clears** |
| 10% | oracle b_ref | +.3713 | — | — | — | — | 2.939 | — | — | ceiling |
| 10% | random 10% | +.0002 | — | — | — | — | 1.001 | — | — | floor |
| 10% | **NLe, lane-free (record)** | +.1364 | +.0541 | +.1322 | .095 (1/20) | +1.14 (.126) | 1.617 | 1.598 | .095 | marginal |

Variance components of the null: SD_perm .071 / SD_train .050 at q = 5%,
.064 / .056 at q = 10%; the arm's own training-seed SD is .008 / .004. The
single-run 95th percentile of the 60 shuffles is +.1492 at q = 5% (all three
NLe seeds above it) and +.1504 at q = 10% (none of the three above it). The
arm recovers 34.0% / 36.7% of the oracle ceiling; the shuffled encoders
recover 12-15%. Paired cluster bootstrap over the 328 logs of the
arm-minus-shuffle contrast: +.0978 [+.0549, +.1479] at q = 5% and +.0823
[+.0580, +.1147] at q = 10%. Whole-panel Spearman of predicted difficulty
with the observed failure rate: +.393 (seeds SD .007) against a
per-labeling null mean of +.070 (SD_perm .175, SD_train .124); one labeling
of 20 reaches it (p .095; z +1.71, Gaussian p .044), single-run 95th
percentile +.391 with two of three seeds above it.

Where the signal sits. Scoring the same predictions and the same
10-planner labels inside each half of the split (top-q re-selected within
the half, the 20 labelings re-scored the same way):

| tokens | n | base fail | q | Delta M_CLS (NLe) | enrichment | shuffle mean | T95 | labelings >= arm | Spearman NLe / shuffle |
|---|---|---|---|---|---|---|---|---|---|
| the 584 of the record panel | 584 | .1932 | 5% / 10% | +.2377 / +.2038 | 2.171 / 1.959 | +.0844 / +.0684 | +.1885 / +.1618 | 0/20 / 0/20 | +.419 / +.078 |
| the 534 added tokens | 534 | .2021 | 5% / 10% | +.0729 / +.0766 | 1.313 / 1.338 | +.0332 / +.0378 | +.0962 / +.1019 | 8/20 / 6/20 | +.362 / +.062 |

The retrieval signal is concentrated on the 584 tokens of the record panel
(their re-scored enrichment under the main-server labels, 2.17 / 1.96, is
close to the record panel's 2.28 / 2.02) and is weak on the 534 added
tokens, where the arm's top-q drop is inside the shuffle distribution at
both budgets even though its whole-half rank correlation (+.362) is not.
The added tokens shift the map mix: the 584 are 487 Las Vegas / 66
Pittsburgh / 30 Boston / 1 Singapore, the 534 are 334 / 68 / 38 / 94
Singapore (left-hand traffic; the Bench2Drive source is right-hand). Per-map
Spearman on the full split: Las Vegas +.295 (821 tokens), Boston +.298 (68),
Pittsburgh +.261 (134), Singapore +.111 (95); inside the 534 the Las Vegas
and Pittsburgh tokens also correlate less than their 584 counterparts (+.223
vs +.351, +.162 vs +.352). Of the retrieved top-5% / top-10% on the full
split, 44% / 39% are 584 tokens (52% of the panel). The pipeline is not the
cause: the 584 tensors rebuild identically and the predictions match the
record; the 534 tokens are the part of Val14 the encoder transfers to
less well.

Reading. On the full official split the lane-free encoder of record
clears the matched null at top-5% (no labeling of 20 reaches its drop, all
three seeds above the single-run 95th percentile, enrichment 1.74) and is
marginal at top-10% (one labeling of 20, p .095, no seed above the
single-run 95th percentile). The paired bootstrap over logs excludes zero at
both budgets, and the whole-panel rank correlation is again marginal
(p .095). Against the 584-token result the effect is smaller (enrichment
1.74 / 1.62 vs 2.28 / 2.02) and the top-10% verdict drops from clears to
marginal; the difference is the 534 added tokens, above. The paper's nuPlan
table reports this full-split run.

## Table 3B — UPS: unseen planner x unseen scenes (`run_ups.py`)

Predict an unseen planner's behaviour on unseen scenario types with zero
rollouts on the target block: probe the planner on B calibration-type
routes, transport the ability posterior through the difficulty prior
N(b_tilde_s, sigma^2) of the encoder of record — RelGraph R2-noLane, which
has no lane graph — with the testlet prior on the (unobserved) evaluation
types. The MAE scores the posterior median of the block-D success rate, the
NLL scores the per-cell posterior predictive. 64 evaluations, encoder run s0
(across the three encoder runs the SD of every MAE cell is .001-.003). The canonical probe rule is Delta-R1 on the block-D success
rate — the UP acquisition with its risk evaluated on the target block;
theta-EIG is ATDrive's own ablation of it (posterior-variance acquisition on the
ability). No published method is run here: none of the Table 1 baselines has a
transport step, so UPS reports ATDrive, its ablation and the two floors only.
naive = the planner's success rate on the probed routes, used as-is for block D.

| probe policy | B30 MAE | NLL | B55 MAE | NLL | B110 MAE | NLL |
|---|---|---|---|---|---|---|
| naive (no IRT) | .1007 | .6414 | .0900 | .6262 | .0865 | .6203 |
| Random | .1140 | .6136 | .1052 | .6056 | .0961 | .5993 |
| theta-EIG (abl.) | .0919 | .5978 | .0943 | .5973 | .0924 | .5983 |
| **ATDrive (Delta-R1 on D)** | .0922 | **.5974** | **.0920** | **.5963** | .0926 | .5985 |

Reading. The transport is a per-cell result, not a block-SR result: it
lowers the predictive NLL of the unseen cells from .62-.64 to .596-.599
at every budget, but its block-SR error sits on a floor of about .092
from B = 30 on — the floor of the scene prior (rho about .55), which more
probes cannot lower — while the planner's own success rate on the probed
calibration routes reaches .090 / .087 at B = 55 / 110. The naive error
is the calibration-vs-evaluation gap itself: |SR_A - SR_D| averages .085
over the 64 evaluations of this panel, and the transported estimate's
floor is not below that gap here. The
block-SR claim for UPS is therefore withdrawn on this panel; what the
transport delivers is the per-cell predictive. Among probe rules the
target-aligned Delta-R1 is ahead of Random by .022* / .013* / .004 and
indistinguishable from its ablation (theta-EIG -.000 / +.002* / -.000); the
probe placement matters at low budgets, and only through the prior.

**Control prior 1 — the lane graph kept** (`results/ups_lane.json`: the
encoder earlier releases shipped as canonical, driving both the transport and
the Delta-R1 probe rule):

| probe policy | B30 MAE | NLL | B55 MAE | NLL | B110 MAE | NLL |
|---|---|---|---|---|---|---|
| naive (no IRT) | .1007 | .6414 | .0900 | .6262 | .0865 | .6203 |
| Random | .1129 | .6021 | .1057 | .5932 | .0985 | .5872 |
| theta-EIG (abl.) | .0966 | .5868 | .0949 | .5852 | .0953 | .5865 |
| ATDrive (Delta-R1 on D) | .0950 | .5857 | .0936 | .5848 | .0950 | .5863 |

THIS IS THE ONE PLACE THE LANE-FREE ENCODER IS WORSE. Keeping the lane graph
costs .003 / .002 / .002 of block-SR MAE (paired delta of the Delta-R1 row,
lane kept minus lane-free: +.0029 [-.0076, +.0146] / +.0015 [-.0085, +.0121] /
+.0024 [-.0077, +.0137], all containing zero) but buys .012 / .012 / .012 of
per-cell NLL (.5857 / .5848 / .5863 against .5974 / .5963 / .5985). The
block-SR difference is inside the noise; the NLL difference is a consistent
shift at every budget and under every probe rule, and it is the reason the
lane-carrying model is kept as a control rather than dropped. US, UPS block-SR
and the nuPlan transfer all favour the lane-free encoder; the UPS per-cell
predictive is the exception.

**Control prior 2 — the speed channel removed from the encoder of record**
(`results/ups_nospeed.json`: R2-noLane with the ego-speed channel zeroed,
i.e. a lane-free speed ablation, three runs; the Delta-R1 probe rule runs on
the same prior):

| probe policy | B30 MAE | NLL | B55 MAE | NLL | B110 MAE | NLL |
|---|---|---|---|---|---|---|
| naive (no IRT) | .1007 | .6414 | .0900 | .6262 | .0865 | .6203 |
| Random | .1126 | .6056 | .1030 | .5974 | .0962 | .5913 |
| theta-EIG (abl.) | .0916 | .5901 | .0906 | .5883 | .0931 | .5905 |
| ATDrive (Delta-R1 on D) | .0889 | .5887 | .0890 | .5872 | .0930 | .5905 |

The paired delta of its Delta-R1 row against the encoder of record is
-.0033 [-.0109, +.0044] / -.0031 [-.0098, +.0038] / +.0004 [-.0076, +.0084]:
indistinguishable on the block-SR scale, .009 / .009 / .008 better on the NLL.
Speed is not what the transport lives on; the reading above does not change.

**Control priors 3 and 4 — the difficulty-matching arms of the encoder of
record** (`results/ups_match0.1.json`, `results/ups_match1.json`: R2-noLane
trained with lambda x mean (f_phi(x) - b_hat)^2 added to its objective, see the
recipe control of Table 3A; three runs each, run s0 in the table, the Delta-R1
probe rule running on the same prior):

| probe policy, lambda = 0.1 | B30 MAE | NLL | B55 MAE | NLL | B110 MAE | NLL |
|---|---|---|---|---|---|---|
| naive (no IRT) | .1007 | .6414 | .0900 | .6262 | .0865 | .6203 |
| Random | .1113 | .6057 | .1023 | .5977 | .0949 | .5914 |
| theta-EIG (abl.) | .0897 | .5903 | .0917 | .5891 | .0919 | .5904 |
| ATDrive (Delta-R1 on D) | .0884 | .5889 | .0908 | .5888 | .0918 | .5903 |

| probe policy, lambda = 1 | B30 MAE | NLL | B55 MAE | NLL | B110 MAE | NLL |
|---|---|---|---|---|---|---|
| naive (no IRT) | .1007 | .6414 | .0900 | .6262 | .0865 | .6203 |
| Random | .1124 | .6110 | .1028 | .6022 | .0950 | .5969 |
| theta-EIG (abl.) | .0909 | .5958 | .0891 | .5943 | .0893 | .5945 |
| ATDrive (Delta-R1 on D) | .0886 | .5951 | .0886 | .5941 | .0891 | .5944 |

Both arms sit where the other two control priors sit. On the block-SR scale
the paired delta of the Delta-R1 row against the encoder of record contains
zero at every budget (lambda = 0.1: -.0038 [-.0084, +.0013] / -.0012 [-.0068,
+.0053] / -.0009 [-.0061, +.0056]; lambda = 1: -.0036 [-.0115, +.0032] /
-.0034 [-.0096, +.0031] / -.0035 [-.0105, +.0031]; across-run SD of the cells
.001-.003), and the .09 floor does not move. On the per-cell NLL the
lambda = 0.1 arm is .008 better at every budget (.5889 / .5888 / .5903 against
.5974 / .5963 / .5985), the size of the speed control's shift (.009) and below
the lane control's (.012); the lambda = 1 arm is .002-.004 better. The prior
width is not what moves: every one of the five priors learns sigma = .652-.655.
The predictions' spread does differ (SD of b_tilde over the held-out routes:
encoder of record 1.05-1.07, matching arms .84-.93, lane control 1.14-1.20),
but the NLL ordering does not follow it either. What the four control priors
say together is that each of them beats the encoder of record on the per-cell
NLL by .002-.012 and none of them on the block-SR MAE; the matching term adds
nothing to UPS beyond that, and the reading of Table 3B does not change.

## UPS retargeted to the full 220-route SR (`run_ups_full.py`)

Table 3B predicts only the 40-route target block. Here the estimand is the
held-out planner's success rate on the whole benchmark, I = S_t u (C \ S_t)
u T: the probed calibrated routes are used as observed, the remaining
calibrated routes are inferred from their response-calibrated difficulty
posteriors, and the 40 held-out-type routes from the scene-conditioned
prior N(b_tilde_s, sigma^2) of the encoder of record — RelGraph R2-noLane,
which has no lane graph. 64 evaluations, 36 : 8 type split, K_cal = 12,
encoder run s0; per evaluation |C| = 177.9 calibrated + |T| = 39.5 target
routes, so T is 18.2% of the benchmark; mean |SR_C - SR_T| = .0848. The
readout arms differ only in the difficulty model of the unobserved routes:
naive (the probed success rate for everything), calC + priorT(marg)
(T from b ~ N(0, sigma_b^2), encoder off), calC + priorT(const) (T from one
constant b = mean b_tilde, encoder level kept, per-route signal removed),
calC + sceneT (canonical), and two oracles that reveal C or T. Probe
orders: Random, Delta-R1 on D (the canonical UPS rule, acquisition driven
by the scene prior), Delta-R1 on D with a scene-free target bank
(encoder-free acquisition), and Delta-R1 on the full bank. The
target-block diagnostic reproduces Table 3B element-wise. That diagnostic
does not by itself identify the prior — the lane-carrying and lane-free
target cells agree to within its .003 anchor tolerance — so every record
also carries the name and content md5 of the prior npz, and the script
refuses to score records made with any other one.

Full-benchmark SR-MAE:

| probe order | arm | B = 30 | B = 55 | B = 110 |
|---|---|---|---|---|
| Random | naive | .0698 | .0455 | .0273 |
| Random | calC + priorT(marg) | .0603 | .0422 | .0269 |
| Random | calC + priorT(const) | .0614 | .0429 | .0269 |
| Random | calC + sceneT | .0624 | .0444 | .0277 |
| Random | trueC + sceneT | .0207 | .0191 | .0174 |
| Random | calC + trueT | .0483 | .0313 | .0179 |
| Delta-R1 on D | naive | .0560 | .0594 | .0428 |
| Delta-R1 on D | calC + priorT(marg) | .0457 | .0319 | .0208 |
| Delta-R1 on D | calC + priorT(const) | .0444 | .0299 | .0213 |
| **Delta-R1 on D** | **calC + sceneT (canonical)** | **.0448** | **.0300** | **.0209** |
| Delta-R1 on D | trueC + sceneT | .0167 | .0167 | .0168 |
| Delta-R1 on D | calC + trueT | .0384 | .0220 | .0097 |
| Delta-R1 on D, scene-free bank | calC + priorT(marg) (encoder-free end to end) | .0454 | .0326 | .0205 |
| Delta-R1 on D, scene-free bank | calC + sceneT | .0447 | .0310 | .0207 |
| Delta-R1 on full I | calC + priorT(marg) | .0407 | .0243 | .0201 |
| Delta-R1 on full I | calC + sceneT | .0393 | .0245 | .0197 |
| Delta-R1 on full I | trueC + sceneT | .0160 | .0158 | .0168 |
| Delta-R1 on full I | calC + trueT | .0333 | .0190 | .0105 |

Encoder-run stability of these cells (the table above and the paper's Table IV use
encoder run s0; `ATDRIVE_ENC_RUN=1,2` repeats the 64 evaluations with runs s1 and
s2, `results/ups_full_enc{1,2}/`): full-benchmark SR-MAE of the canonical arm
.0448 / .0464 / .0447 at B = 30 (mean .0453 +- .0009), .0300 / .0299 / .0298 at
B = 55 (.0299 +- .0001), .0209 / .0193 / .0195 at B = 110 (.0199 +- .0009); the
common-prior arm .0459 +- .0006 / .0328 +- .0008 / .0204 +- .0004 (its probe order
is ATDrive's, so it moves with the encoder too); the naive arm is encoder-free
(.0698 / .0455 / .0273 in every run). AUROC on the new-route cells: .744 / .756 /
.766 at B = 30 (.755 +- .011), .748 / .758 / .767 at B = 55 (.758 +- .009), .747 /
.758 / .767 at B = 110 (.757 +- .010) against the common prior's .709 / .707 /
.706 (+- .001). Run s0, the one in the paper, is the lowest of the three on
AUROC and in the middle on SR-MAE.

What the scene encoder buys (+ = the scene prior is worse; paired
planner-cluster bootstrap):

| probe order | contrast | B = 30 | B = 55 | B = 110 |
|---|---|---|---|---|
| Random | sceneT vs priorT(marg) | +.0021 [-.0007, +.0048] | +.0022 [-.0012, +.0057] | +.0008 [-.0016, +.0032] |
| Delta-R1 on D | sceneT vs priorT(marg) | -.0010 [-.0026, +.0007] | -.0018 [-.0039, +.0004] | +.0001 [-.0021, +.0027] |
| Delta-R1 on D | sceneT vs priorT(const) | +.0004 [-.0009, +.0017] | +.0002 [-.0012, +.0015] | -.0004 [-.0016, +.0007] |
| Delta-R1 on full I | sceneT vs priorT(marg) | -.0014 [-.0042, +.0015] | +.0002 [-.0028, +.0034] | -.0004 [-.0031, +.0023] |
| encoder-free acquisition + readout vs the canonical cell | | +.0006 [-.0018, +.0029] | +.0026 [-.0004, +.0056] | -.0004 [-.0030, +.0020] |
| acquisition channel alone (scene-free vs scene-driven order, both read out scene-free) | | -.0004 [-.0022, +.0015] | +.0008 [-.0004, +.0022] | -.0002 [-.0006, +.0001] |
| Delta-R1 on D | headroom: sceneT vs trueT | +.0063 [+.0013, +.0114] | +.0080 [+.0043, +.0119] | +.0112 [+.0082, +.0146] |
| naive on its own Random order vs the canonical cell | | +.0250 [+.0044, +.0460] | +.0155 [+.0044, +.0258] | +.0063 [-.0001, +.0119] |

Where the error lives (posterior-mean parts, canonical order): the C-part
error falls .0386 -> .0220 -> .0097 from B = 30 to 110 while the T-part
error is flat at .0167 / .0167 / .0168 (Random order: .0487 / .0316 / .0180
and .0206 / .0191 / .0174); the two parts reproduce the median-based
SR-MAE to within .0007. Pooled AUROC of the per-cell predictive on the T
cells is .744-.750 for the scene prior against .705-.712 scene-free under
the Delta-R1 orders (.727-.745 vs .690-.702 under Random); the
within-evaluation AUROC of .666 is the raw RelGraph out-of-fold ranking
(predicted p is a monotone function of -b_tilde_s within an evaluation) and
is identical at every budget and under every probe order.

Reading. With 180 of 220 routes response-calibrated, the scene encoder
contributes nothing measurable to the full-benchmark SR: swapping the
per-route RelGraph prior for a scene-free prior moves the SR-MAE by at most
.0022 and all nine sceneT-vs-priorT(marg) intervals contain zero, and
removing the encoder from the acquisition as well leaves the error unchanged
(largest move .0026, interval containing zero). Exactly one of the 24
scene-vs-scene-free readout cells (4 probe orders x 2 scene-free arms x 3
budgets) excludes zero, and it goes the wrong way for the encoder — Random
order, priorT(const), B = 55: +.0015 [+.0002, +.0028], the scene prior worse
by a thousandth and a half. That null rests on the
planner-cluster intervals (half-width about .003) alone: Table 3B's
across-run SD of .001-.003 is in target-block units and enters the full SR
scaled by n_T / N = .182, i.e. .0002-.0006, so the deltas are above the
encoder's run-to-run noise, not below it. The test is weakly powered: the
scene-free-minus-scene intervals reach .0021-.0039 on the side that would
favour the encoder against a trueT oracle headroom of .0063-.0112, so a
contribution above roughly a fifth to a half of the available T information
is excluded and a smaller one is not. Against the naive baseline on its own
random probe order the model is better at B = 30 and 55 and tied at
B = 110. The T-part error is a floor no probe budget lowers — it is the
calibration-vs-target block gap of .0848 — and at B = 110 the 18% of routes
the scene prior has to guess carry more of the error than the 82% the
responses cover. Arithmetically the T channel of this error is Table 3B's
already-withdrawn block-SR null multiplied by .182; the encoder still ranks
unseen routes within a planner (AUROC .75 vs .71 scene-free), which is the
per-cell result Table 3B reports.

Against the superseded lane-carrying record this section barely moves. The
canonical cell improves by .0019 / .0019 / .0002 (.0467 / .0319 / .0211 ->
.0448 / .0300 / .0209) and the pooled T-cell AUROC falls from .760 to .744 —
the same split verdict as Table 3B, the lane-free encoder better on the SR
readout and the lane-carrying one on the per-cell predictive. The
scene-prior null came out the same under either encoder, which makes it a
property of the estimand — 180 of 220 routes are already
response-calibrated — rather than of the encoder filling the other 18%.

## Readout drop-in (`run_readout_dropin.py`)

Analysis, not a headline: every baseline's selected subset re-scored with
the ATDrive readout (exact posteriors, testlet, posterior median). SR-MAE:

| selector (subset) | K4 B30 | B55 | B110 | B165 | K12 B30 | B55 | B110 | B165 |
|---|---|---|---|---|---|---|---|---|
| Fluid | .0536 | .0362 | .0247 | .0135 | .0410 | .0290 | .0225 | .0113 |
| Total-Fisher | .0489 | .0417 | .0250 | .0163 | .0494 | .0365 | .0207 | .0147 |
| metabench | .0671 | .0428 | .0237 | .0157 | .0574 | .0308 | .0232 | .0131 |
| tinyBenchmarks | .0698 | .0514 | .0240 | .0139 | .0463 | .0334 | .0205 | .0108 |
| AnchorPoints | .0687 | .0493 | .0313 | .0166 | .0550 | .0396 | .0217 | .0104 |
| Random | .0551 | .0405 | .0243 | .0134 | .0577 | .0393 | .0244 | .0134 |
| ATDrive (own order, Table 1) | .0450 | .0332 | .0223 | .0116 | .0477 | .0231 | .0160 | .0081 |

Reading. Under one readout the remaining differences are selection. At
K_cal = 12 the ATDrive order beats every re-scored subset at B = 55 / 110 by
.005-.017 and at B = 165 by .002-.007; at K_cal = 4 it leads the non-Fluid
subsets by .004-.025 at B = 30 / 55 and by .001-.009 at B = 110 / 165, and
the Fluid subset by .002-.009 at every budget. The readout itself is a
drop-in improvement for selectors whose native readout is a plug-in on a
scarce panel (compare the native Table 1 rows: AnchorPoints .1290 -> .0687 at
K4 B30, metabench .0728 -> .0671 at K4 B30 and .0629 -> .0574 at K12 B30);
the one native readout that is already better than the posterior median is
tinyBenchmarks' anchor-weighted estimate at K4 B30 / B55 (.0660 / .0488
native vs .0698 / .0514 re-scored).

## Model adequacy (`run_model_adequacy.py`)

Appendix. On the UP calibration block (12 calibration planners x 220
routes per draw) 10% of the observed cells are held out and predicted by
the 1PL, 2PL and 3PL fits of the rest; 16 draws. Held-out cell NLL:
1PL .5284 +- .0110, 2PL .5358 +- .0123, 3PL .5323 +- .0104 (2PL - 1PL
+.0074 +- .0029 across draws); the split-half reliability of the fitted
log-discrimination across two random 6 / 6-planner halves is +.062 +- .022.
Neither richer model predicts held-out responses better than the Rasch
model and the discrimination it would add is not reproducible across
halves of the panel, so the 1PL is the adequate model here, not a
simplification.

## Inside a real evaluation (`tools/b2d_adaptive_eval.py`)

The same posterior drives an actual Bench2Drive run: the bank is the full
22 x 220 panel (excluding the planner under test if it is in it), the risk
scale c is fixed by leave-one-planner-out on that bank (c = 2.36 for the
22-planner bank; 2.55 / 2.32 / 2.16 for the three held-out banks below),
and the driver picks one route at a time, runs it through the leaderboard
evaluator, reads the outcome and stops at c * R1 <= eps — there is no
route cap; the run ends when the risk target is met or the bank is
exhausted. Simulated from the matrix (`--dry-run`, planner held out of the
bank, eps = .03; the bank is the planner's recorded routes, 211-220 of
220):

| planner | true SR | routes run | stopped by | SR_hat | abs err | types covered |
|---|---|---|---|---|---|---|
| VAD | .155 | 106 | risk (c * R1 = .0296) | .170 | .015 | 37 / 44 |
| HiP-AD | .664 | 134 | risk (c * R1 = .0298) | .656 | .007 | 42 / 44 |
| LEAD-tfv6 | .777 | 99 | risk (c * R1 = .0300) | .814 | .036 | 38 / 44 |

Every run stopped on its own: half the bank for VAD and LEAD-tfv6, 61% for
HiP-AD, with errors of .015 and .007 for VAD and HiP-AD. The LEAD-tfv6
stop (99 routes) missed its .03 target with a realised error of .036. Such
misses are not rare: in Table 2 the realised error exceeds the target in
6-12 of the 64 evaluations per cell (9-19%, K_cal and eps dependent); c is the
90th percentile of the calibration planners' pooled error ratios, not a
coverage guarantee for the new planner's stop. The weakest planner remains the hardest case: an all-fail
record only bounds theta from above, so its estimate leans on the prior
early (SR_hat .22 after 40 routes) and settles as the acquisition finds
the routes it can pass.
