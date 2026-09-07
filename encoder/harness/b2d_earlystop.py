#!/usr/bin/env python3
"""B2D CHECKPOINT SELECTION BY INNER-VALIDATION NLL — the ONE implementation all B2D arms share.

TERMINOLOGY.  All 30 epochs are always run and the best one is chosen afterwards, so nothing is
"stopped early" and no compute is saved.  The name of the thing is **checkpoint selection by
inner-validation NLL**.  The CLI flag is still spelled `--early-stop` because it is already in
three shell drivers; the flag name is the only place the old word survives.

WHY.  NAVSIM already selects its checkpoint on an inner-validation split carved out of the
training block (r2_graph.run_navsim: `if nll < best[0]: best = (nll, pred_on(te), ep)` then
`pred[te] = best[1]`).  B2D had nothing: it ran a fixed 30 epochs and read the LAST one.  The
b2d_innerval_diag.py run measured what that costs — inner-val NLL bottoms at median epoch 5.5
(mean 9.2, min 2, max 24) while train NLL keeps falling 0.5543 -> 0.4734 and inner-val climbs
0.5633 -> 0.5717.  B2D trains ~25 epochs past its optimum with no mechanism to stop.

WHAT THIS MODULE IS.  Every B2D loop (r0_ego, r1_map, hrel's MLP, r2_graph and the arms that
reuse it) calls THESE functions, so no arm can end up with a subtly different selection rule.
A rule that differed between arms would invalidate the very comparison the exercise exists to
make, so the rule lives in exactly one place.

WHAT IT DOES NOT CHANGE.  The statistical model is untouched:
    b~ = f(x);  b = b~ + eps,  eps ~ N(0, sigma^2);  Gauss-Hermite marginalised cell
    likelihood;  theta by rasch() on training columns only.
Only WHICH CHECKPOINT is read, and which theta each stage is fitted under.


────────────────────────────────────────────────────────────────────────────────
THE TWO-STAGE NESTED RULE  (TwoStage below).  This is the whole point of the module.
────────────────────────────────────────────────────────────────────────────────
The single-stage version of this analysis changed TWO things at once relative to the frozen
run: the epoch count AND the training set (it trained on A_train only, 30 of the 36 outer
training types).  A difference in the held-out result could then come from either, and the two
could not be separated.  The question this analysis exists to answer is narrower —
**was training for too long the problem, holding the training data fixed?**  So:

    stage 1   theta_inner = rasch(A_train responses)
              train on A_train, evaluate inner-val NLL after every epoch, e* = argmin
    stage 2   theta_outer = rasch(FULL A responses)
              re-initialise the model with the SAME seed, train on FULL A for exactly e*+1
              epochs (epoch indices 0..e*), no inner-val, nothing selected
    stage 3   forward the held-out block C exactly ONCE, at the very end

After stage 2 the ONLY difference from the frozen arm is 30 -> e*+1 epochs.  The training set,
theta, the feature standardisation and every derived statistic are refitted on the full outer
training block, exactly as the frozen run fits them.

e* IS SELECTED PER DRAW PER SEED.  That matches how NAVSIM already behaves: it selects inside
each run, per fold.  There is no pooling of e* across draws or seeds.

*** THE APPROXIMATION, STATED RATHER THAN HIDDEN. ***  e* is chosen under theta_inner but
stage 2 trains under theta_outer.  The two thetas are close but not equal (their max abs
difference is measured and written to the output npz as `es_theta_delta`), so e* is an epoch
count transferred across a slightly different objective, not the argmin of the stage-2
objective.  This is inherent to nested selection — the outer objective cannot be consulted
without spending the inner-val data twice — and it is the standard construction, but it is an
approximation and the report must say so.

STAGE 1's WEIGHTS ARE THROWN AWAY.  Stage 1 yields one integer, e*, and nothing else.  That is
what makes stage 2 a clean re-fit rather than a fine-tune, and it is why Selector no longer
snapshots model state: there is no checkpoint to restore, only an epoch number to reuse.


THE INNER-VAL SPLIT IS BY SCENARIO TYPE, AND IS THE SAME FOR EVERY ARM AND SEED.
B2D windows overlap (consecutive windows share 11 of 12 timesteps) and a route contributes
many windows, so a random window or route split leaks a route's local segments across the
boundary and makes the validation curve optimistic.  B2D is 44 types x exactly 5 routes with
one row of Y per route, so whole types are the only grouping that splits neither a route nor a
type: A = A_train + A_val by whole scenario type.
Determinism: the draw of inner-val types is `default_rng(IV_RNG_BASE + draw).choice(...)` over
`sorted(set(types[tr]))`.  Its only inputs are the draw index and the outer-training type set,
and the outer split itself is `unified_split(draw, utypes, J)` = `RandomState(1000 + draw)`,
identical in every arm.  Nothing in the chain reads the model seed, the arm, the architecture
or any corruption seed, so all arms and all seeds are selected on the identical inner-val set.
`fingerprint()` writes a hash of the chosen type names into every output npz so cross-arm
identity is checkable after the fact rather than asserted in prose.


*** THE HELD-OUT BLOCK IS NEVER TOUCHED DURING STAGES 1 AND 2. ***  Not for selection, not for
stopping, not for reporting during training.  The contract here is STRICTER than NAVSIM's.
NAVSIM calls `pred_on(te)` on every improving epoch — labels never enter selection so it is not
statistical leakage, but the held-out inputs are forwarded many times during training and
"held-out untouched" is not literally true there.  For B2D the held-out block is forwarded
EXACTLY ONCE, after both stages are over, and the check is not the weak "does corrupting the
held-out block move the selection" but the strong **"how many held-out forwards happened during
stages 1 and 2?  It must be 0."**  Four runtime mechanisms, all reachable asserts, not comments:
  1. HeldOutGuard      every route index and window row that enters a forward pass, the
                       standardisation accumulator, graph_stats or a likelihood block is
                       checked against the held-out sets while training is open.
  2. HeldOutGuard.heldout()   the ONLY sanctioned way to name the held-out block.  It refuses
                       to run while training is open, and it COUNTS.  `n_heldout_open` is the
                       instrumented number the report quotes; it must be 0.
  3. erase_heldout     the response block that reaches the GPU has its held-out columns set to
                       NaN, so the observation mask is identically zero there and no held-out
                       cell can reach any loss even through an indexing mistake.
  4. selftest          each draw feeds one KNOWN held-out route into the guard and requires it
                       to raise.  Without this, a guard that silently never fires would look
                       exactly like a guard that passes.
`end_training()` is the single point where the held-out block becomes readable, and it is
called after stage 2's last epoch.

NAVSIM IS NOT RE-RUN, AND THAT IS A DECISION, NOT AN OMISSION.  run_navsim already selects the
best-inner-val checkpoint inside every fold, so the frozen NAVSIM numbers already carry this
treatment; re-running them under a second implementation of the same rule would produce a
second set of numbers to reconcile and no new information.  What NAVSIM does NOT carry is the
stage-2 full-A refit — its selected checkpoint is the one trained on the fold's inner-train
block — so NAVSIM's frozen numbers are single-stage-selected, and that difference is stated
wherever the two domains are put in the same table.  `--early-stop` asserts on `--domain
navsim` rather than silently doing nothing.
"""
import hashlib
import os

import numpy as np

IV_TYPES = 6            # inner-val scenario types carved out of the 36 outer-training types
IV_RNG_BASE = 2000      # depends on the DRAW only, never on the model seed or the arm


# ── the split ────────────────────────────────────────────────────────────────
def inner_split(draw, types, tr):
    """Carve IV_TYPES whole scenario types out of the OUTER-TRAINING types.

    -> (itr, iv) boolean route masks, both subsets of tr.  Grouped by type, so no scenario type
    and no route is split across the two sides."""
    tr_types = np.array(sorted(set(types[tr])))
    rng = np.random.default_rng(IV_RNG_BASE + draw)
    iv_types = set(tr_types[rng.choice(len(tr_types), IV_TYPES, replace=False)].tolist())
    iv = np.isin(types, sorted(iv_types)) & tr
    itr = tr & ~iv
    assert set(types[itr]).isdisjoint(set(types[iv])), 'a scenario type straddles the inner split'
    assert not (iv & ~tr).any(), 'the inner-val split reached outside the training block'
    assert itr.sum() and iv.sum()
    return itr, iv


def no_split(tr):
    """The frozen path: inner-train IS the whole training block, inner-val is empty."""
    return tr, np.zeros_like(tr)


def fingerprint(types, iv):
    """Short hash of the inner-val TYPE NAMES.  Equal across arms <=> identical selection set."""
    s = '|'.join(sorted(set(str(t) for t in types[iv])))
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _vhash(v):
    """Short hash of a float vector, rounded so it is stable under float32/64 round-trips."""
    return hashlib.sha256(np.round(np.asarray(v, np.float64), 6).tobytes()).hexdigest()[:12]


# ── the leakage boundary ─────────────────────────────────────────────────────
class HeldOutGuard:
    """Holds every route index and window row the outer split holds out for this draw and
    refuses any forward pass, standardisation pass or likelihood block that names one, until
    end_training() is called.  A check on what the code ACTUALLY passed, not on a promise.

    It also COUNTS, because the contract this analysis makes is a number, not an adjective:
        n_open        screened calls made while stages 1 and 2 were running
        n_heldout_open   of those, how many named a held-out route or row  -> MUST BE 0
        n_heldout     completed held-out passes after end_training()       -> MUST BE 1 per draw
    The guard survives across BOTH stages: one guard per draw, opened before stage 1 and closed
    after stage 2, so the counters cover the whole of training and not just one stage."""

    def __init__(self, te_routes, te_rows, hp, keepJ):
        self.te_routes = np.asarray(sorted(te_routes), np.int64)
        self.te_rows = None if te_rows is None else np.asarray(sorted(te_rows), np.int64)
        self.hp = set(int(x) for x in hp)
        assert not (set(int(x) for x in keepJ) & self.hp), 'a held-out planner entered keepJ'
        self.open = True
        self.n_checks = 0
        self.n_open = 0             # screened calls made while training was open
        self.n_heldout_open = 0     # of those, calls that named held-out data.  MUST stay 0.
        self.n_heldout = 0          # sanctioned held-out passes, after end_training()
        self._in_selftest = False

    def _count_open(self, hit):
        if self.open and not self._in_selftest:
            self.n_open += 1
            self.n_heldout_open += int(hit)

    def routes(self, sel, what):
        sel = np.asarray(sel, np.int64)
        if self.open:
            bad = np.intersect1d(sel, self.te_routes)
            self._count_open(len(bad) > 0)
            assert len(bad) == 0, \
                f'HELD-OUT LEAK: {what} named held-out routes {bad[:8].tolist()}'
            self.n_checks += 1
        return sel

    def rows(self, rows, what):
        rows = np.asarray(rows, np.int64)
        if self.open and self.te_rows is not None:
            bad = np.intersect1d(rows, self.te_rows)
            self._count_open(len(bad) > 0)
            assert len(bad) == 0, \
                f'HELD-OUT LEAK: {what} named held-out window rows {bad[:8].tolist()}'
            self.n_checks += 1
        return rows

    def heldout(self, sel, what='held-out prediction'):
        """The ONE sanctioned read of the held-out block.  Legal only after end_training(), and
        counted.  Call it once per draw with the WHOLE held-out index set even when the forward
        is then chunked, so `n_heldout` counts passes over C and not minibatches of one pass."""
        assert not self.open, f'{what} tried to read the held-out block while training is open'
        sel = np.asarray(sel, np.int64)
        assert np.isin(sel, self.te_routes).all(), f'{what} named a route that is not held out'
        self.n_heldout += 1
        return sel

    def selftest(self):
        """Prove the assert is REACHABLE.  A guard that can never fire is not a guard.
        Probes are excluded from the leak counters (self._in_selftest) so the reported
        `n_heldout_open` counts real training calls only."""
        assert self.open, 'selftest must run while the guard is open'
        self._in_selftest = True
        fired = 0
        try:
            self.routes(self.te_routes[:1], 'selftest')
        except AssertionError:
            fired += 1
        if self.te_rows is not None:
            try:
                self.rows(self.te_rows[:1], 'selftest')
            except AssertionError:
                fired += 1
        try:
            self.heldout(self.te_routes[:1], 'selftest')
        except AssertionError:
            fired += 1
        self._in_selftest = False
        want = 2 if self.te_rows is None else 3
        assert fired == want, \
            f'HeldOutGuard is INERT: {fired}/{want} probes raised. Every "no leak" line is void.'
        assert self.n_heldout == 0 and self.n_heldout_open == 0, 'selftest polluted the counters'
        return fired

    def end_training(self, what='held-out prediction'):
        """The single point at which the held-out block becomes readable.  Called after STAGE 2's
        last epoch — never between the stages, and never before."""
        assert self.open, 'end_training called twice'
        self.open = False
        return what

    def leak_line(self, draw):
        return (f'  [b2d draw {draw}] LEAK LEDGER  screened calls during stages 1+2 '
                f'{self.n_open}, of which named held-out data {self.n_heldout_open} '
                f'(must be 0) | sanctioned held-out passes after end_training '
                f'{self.n_heldout} (must be 1)')


def erase_heldout(Yk, te):
    """Y[keepJ] with the held-out columns set to NaN, so the observation mask is zero there and
    no held-out cell can contribute to any loss even by an indexing mistake.  A numeric no-op
    for a correct loop — that is the point."""
    Ytr = np.asarray(Yk, np.float64).copy()
    Ytr[:, te] = np.nan
    return Ytr


def assert_masked(torch, Yd, Md, te, dev):
    te_t = torch.tensor(np.where(te)[0], device=dev)
    assert float(Md[:, te_t].sum()) == 0.0, 'held-out cells are observable in the loss mask'
    assert float(Yd[:, te_t].abs().sum()) == 0.0, 'held-out responses reached the GPU'


# ── the selection metric ─────────────────────────────────────────────────────
def cell_nll(torch, bv, cols, Yd, Md, THE, gx, lgw, sg):
    """The frozen B2D cell likelihood on one column block, marginalised over eps by
    Gauss-Hermite, per observed cell.  Identical to the training loss MINUS the
    0.05*(log sigma)^2 term, which is a prior on sigma and not a data term."""
    s = torch.tensor(np.asarray(cols, np.int64), device=Yd.device)
    z = (bv[None, :, None] + sg * gx[None, None, :]) - THE[:, None, None]
    p = torch.sigmoid(z)
    yy = Yd[:, s]; mm = Md[:, s]
    llc = (yy[:, :, None] * torch.log(p + 1e-7)
           + (1 - yy[:, :, None]) * torch.log(1 - p + 1e-7)) * mm[:, :, None]
    return float(-torch.logsumexp(llc.sum(0) + lgw[None, :], 1).sum() / mm.sum())


class Selector:
    """Stage-1 checkpoint selection: records the inner-val curve and returns ONE integer, e*.

    It does not snapshot weights.  Under the two-stage rule stage 1's weights are discarded and
    stage 2 re-initialises from the same seed, so a state_dict snapshot would be dead machinery
    that looks load-bearing.  What stage 1 hands to stage 2 is an epoch count, nothing else."""

    def __init__(self, epochs):
        self.curve = np.full((epochs, 3), np.nan)     # train_loss, inner-val nll, sigma
        self.best = (np.inf, -1)
        self.sigma = np.nan

    def update(self, ep, train_loss, iv_nll, sigma):
        self.curve[ep] = [train_loss, iv_nll, sigma]
        if iv_nll < self.best[0]:
            self.best = (iv_nll, ep)
            self.sigma = sigma

    @property
    def best_ep(self):
        return self.best[1]


# ── the rule: stage 1 selects e*, stage 2 refits on the full outer-train block ─
class Stage:
    """One pass of the per-draw loop.  `train` is the route mask this stage FITS ON — theta, the
    feature standardisation, seg_norm/graph_stats and the minibatches all follow it."""

    __slots__ = ('no', 'name', 'train', 'iv', 'epochs', 'select', 'final')

    def __init__(self, no, name, train, iv, epochs, select, final):
        self.no, self.name = no, name
        self.train, self.iv = train, iv
        self.epochs, self.select, self.final = epochs, select, final


class TwoStage:
    """Drives the uniform two-stage rule.  Every B2D arm writes

        plan = es.TwoStage(draw, types, tr, a.epochs, a.early_stop)
        for stg in plan.stages():
            <re-seed; fit theta on stg.train; rebuild the model; train stg.epochs epochs;
             if stg.select: score inner-val each epoch and call plan.update(...)>
            plan.end_stage(stg, sigma)
        guard.end_training()
        <forward the held-out block once>

    so the schedule, the masks and the epoch counts are decided HERE and cannot drift apart
    between arms.  Without --early-stop `stages()` yields a single stage identical to the
    frozen recipe: train on all of the outer-train block for the full 30 epochs, select
    nothing.  That is why the flag-off path reproduces the frozen predictions exactly."""

    def __init__(self, draw, types, tr, epochs, enabled):
        self.draw, self.types, self.tr, self.epochs = draw, types, tr, epochs
        self.enabled = bool(enabled)
        self.itr, self.iv = inner_split(draw, types, tr) if self.enabled else no_split(tr)
        self.picker = Selector(epochs)
        self.theta = {}          # stage no -> theta vector fitted for that stage
        self.sigma = {}          # stage no -> learned sigma at the end of that stage
        self.n_train = {}        # stage no -> routes that stage fitted on
        self._done = set()

    # -- the schedule -----------------------------------------------------
    def stages(self):
        if not self.enabled:
            yield Stage(1, 'frozen', self.tr, self.iv, self.epochs, False, True)
            assert 1 in self._done, 'end_stage was not called'
            return
        yield Stage(1, 'stage1-select', self.itr, self.iv, self.epochs, True, False)
        assert 1 in self._done, 'end_stage was not called for stage 1'
        e = self.picker.best_ep
        assert e >= 0, 'stage 1 finished without ever scoring an inner-val epoch'
        # e* is an epoch INDEX; training epochs 0..e* inclusive is e*+1 epochs.
        yield Stage(2, 'stage2-refit', self.tr, np.zeros_like(self.tr), e + 1, False, True)
        assert 2 in self._done, 'end_stage was not called for stage 2'
        assert self.n_train[2] > self.n_train[1], \
            (f'stage 2 fitted on {self.n_train[2]} routes but stage 1 on {self.n_train[1]}: '
             'stage 2 must see the FULL outer-training block, which is strictly larger')
        assert self.theta_delta > 0, \
            'theta is identical in both stages, so stage 2 did not refit it on the full block'

    # -- what the arm reports back ----------------------------------------
    def note_theta(self, stg, theta):
        """Record the theta this stage was fitted under.  Stage 1 must get theta_inner
        (A_train responses), stage 2 theta_outer (full A responses); they are compared at the
        end of stages() so a copy-paste that reused the wrong mask cannot pass silently."""
        self.theta[stg.no] = np.asarray(theta, np.float64).copy()
        self.n_train[stg.no] = int(np.asarray(stg.train).sum())
        return theta

    def update(self, stg, ep, train_loss, iv_nll, sigma):
        assert stg.select, 'only the selecting stage may score the inner-val block'
        self.picker.update(ep, train_loss, iv_nll, sigma)

    def end_stage(self, stg, sigma):
        assert stg.no in self.theta, f'{stg.name} never called note_theta'
        self.sigma[stg.no] = float(sigma)
        self._done.add(stg.no)

    # -- derived quantities -----------------------------------------------
    @property
    def best_ep(self):
        return self.picker.best_ep if self.enabled else -1

    @property
    def stage2_epochs(self):
        return self.picker.best_ep + 1 if self.enabled else self.epochs

    @property
    def theta_delta(self):
        """max |theta_outer - theta_inner|.  The size of the objective shift that e* is being
        transferred across.  0.0 would mean stage 2 never refitted theta."""
        if not self.enabled or 2 not in self.theta:
            return 0.0
        return float(np.abs(self.theta[2] - self.theta[1]).max())

    @property
    def sigma_final(self):
        return self.sigma[max(self.sigma)] if self.sigma else float('nan')

    # -- logging -----------------------------------------------------------
    def head(self, stg):
        if not self.enabled:
            return (f'  [b2d draw {self.draw}] frozen recipe: train {int(stg.train.sum())} '
                    f'routes / {len(set(self.types[stg.train]))} types, {stg.epochs} epochs, '
                    f'no selection')
        if stg.no == 1:
            return (f'  [b2d draw {self.draw}] STAGE 1 (select e*)  A_train '
                    f'{int(self.itr.sum())} routes / {len(set(self.types[self.itr]))} types  '
                    f'A_val {int(self.iv.sum())} routes / {len(set(self.types[self.iv]))} types  '
                    f'fp {fingerprint(self.types, self.iv)}  theta_inner on A_train responses '
                    f'(hash {_vhash(self.theta[1])})')
        return (f'  [b2d draw {self.draw}] STAGE 2 (refit)  FULL A {int(stg.train.sum())} '
                f'routes / {len(set(self.types[stg.train]))} types  '
                f'(= A_train {self.n_train[1]} + A_val {int(self.iv.sum())})  '
                f'epochs 0..{self.best_ep} = {stg.epochs}  '
                f'theta_outer on full-A responses (hash {_vhash(self.theta[2])}, '
                f'max|theta_outer-theta_inner| {self.theta_delta:.4f})')

    def line(self, n_te):
        if not self.enabled:
            return f'  [b2d draw {self.draw}] frozen recipe, nothing selected'
        c = self.picker.curve
        return (f'  [b2d draw {self.draw}] CHECKPOINT SELECTION  e* = {self.best_ep} of '
                f'{self.epochs} (inner-val nll {self.picker.best[0]:.4f}); epoch '
                f'{self.epochs - 1} nll {c[-1, 1]:.4f} (+{c[-1, 1] - self.picker.best[0]:.4f}) '
                f'| sigma at e* {self.picker.sigma:.3f} -> stage-2 final '
                f'{self.sigma_final:.3f} '
                f'| stage 2 trained {self.n_train[2]} routes x {self.stage2_epochs} epochs '
                f'| held out and never read during either stage: {n_te} routes')


# ── the per-draw record every arm writes ─────────────────────────────────────
class Ledger:
    """The per-draw bookkeeping, in one place so all four arms save the IDENTICAL keys.

    Splat `led.fields()` into np.savez.  The keys are:
        early_stop          bool, whether the two-stage rule ran at all
        es_curve            (draws, epochs, 3) train loss / inner-val nll / sigma, stage 1
        es_best_ep          (draws,) e*, an epoch INDEX; -1 on the frozen path
        es_stage2_epochs    (draws,) e*+1, the epochs stage 2 actually ran
        sigma               (draws,) learned sigma of the FINAL fitted model (stage 2)
        es_sigma_stage1     (draws,) sigma at the selected stage-1 epoch e* (NaN when frozen)
        es_train_routes     (draws, 2) routes each stage fitted on; col 1 must exceed col 0
        es_theta_delta      (draws,) max|theta_outer - theta_inner|
        es_theta_hash       (draws, 2) hash of each stage's theta; the two must differ
        es_heldout_open     (draws,) held-out reads during stages 1+2.  MUST be all 0.
        es_heldout_passes   (draws,) held-out passes after end_training.  MUST be all 1.
        iv_fingerprint      (draws,) hash of the inner-val TYPE NAMES
    """

    def __init__(self, draws, epochs, enabled):
        self.enabled = bool(enabled)
        self.curve = np.full((draws, epochs, 3), np.nan)
        self.best_ep = np.full(draws, -1, np.int64)
        self.s2_epochs = np.full(draws, -1, np.int64)
        self.sigma = np.full(draws, np.nan)
        self.sigma_s1 = np.full(draws, np.nan)
        self.n_train = np.zeros((draws, 2), np.int64)
        self.theta_delta = np.full(draws, np.nan)
        self.theta_hash = [['', ''] for _ in range(draws)]
        self.ho_open = np.full(draws, -1, np.int64)
        self.ho_pass = np.full(draws, -1, np.int64)
        self.fp = [''] * draws

    def record(self, draw, plan, guard):
        """Called once per draw, after the held-out forward.  Asserts the contract rather than
        merely logging it: a violated count aborts the run instead of reaching the npz."""
        self.curve[draw] = plan.picker.curve
        self.best_ep[draw] = plan.best_ep
        self.s2_epochs[draw] = plan.stage2_epochs
        self.sigma[draw] = plan.sigma_final
        self.sigma_s1[draw] = plan.picker.sigma      # sigma at the SELECTED stage-1 epoch e*
        self.n_train[draw] = [plan.n_train.get(1, 0), plan.n_train.get(2, plan.n_train.get(1, 0))]
        self.theta_delta[draw] = plan.theta_delta
        self.theta_hash[draw] = [_vhash(plan.theta[k]) if k in plan.theta else ''
                                 for k in (1, 2)]
        self.ho_open[draw] = guard.n_heldout_open
        self.ho_pass[draw] = guard.n_heldout
        self.fp[draw] = fingerprint(plan.types, plan.iv) if plan.enabled else ''
        assert guard.n_heldout_open == 0, \
            f'draw {draw}: {guard.n_heldout_open} held-out reads during stages 1 and 2'
        assert guard.n_heldout == 1, \
            f'draw {draw}: {guard.n_heldout} held-out passes, the contract is exactly 1'
        if plan.enabled:
            assert self.theta_hash[draw][0] != self.theta_hash[draw][1], \
                f'draw {draw}: both stages used the same theta'

    def fields(self):
        return dict(early_stop=bool(self.enabled), es_curve=self.curve,
                    es_best_ep=self.best_ep, es_stage2_epochs=self.s2_epochs,
                    sigma=self.sigma, es_sigma_stage1=self.sigma_s1,
                    es_train_routes=self.n_train, es_theta_delta=self.theta_delta,
                    es_theta_hash=np.array(self.theta_hash), es_heldout_open=self.ho_open,
                    es_heldout_passes=self.ho_pass, iv_fingerprint=np.array(self.fp))


# ── initialisation audit ──────────────────────────────────────────────
def init_hash(m):
    """SHA256 over the freshly built parameters, before the first optimiser step.  Used to
    show (a) that the seeds now give DIFFERENT initial weights under --proper-init and
    (b) that stage 1 and stage 2 of one run still start from the SAME weights."""
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# ── output location ──────────────────────────────────────────────────────────
def out_path(rg, name, early_stop, proper_init=False):
    """Two-stage runs go to rg/es/.  The scripts hardcode their output names and would
    otherwise overwrite the frozen primary results, which has already happened once.
    --proper-init runs are a SEPARATE stochastic design (init varies across seeds, not only
    minibatch order) and go to rg/es_pinit/.  They must never be pooled with rg/es/."""
    if not early_stop:
        assert not proper_init, '--proper-init requires the two-stage --early-stop protocol'
        return f'{rg}/{name}'
    d = f'{rg}/es_pinit' if proper_init else f'{rg}/es'
    os.makedirs(d, exist_ok=True)
    p = f'{d}/{name}'
    assert '/frozen30/' not in p and '/unblind_snapshot/' not in p, \
        'a checkpoint-selection run must never write into the frozen results'
    return p
