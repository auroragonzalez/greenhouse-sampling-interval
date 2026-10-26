"""
sampling_interval.py
====================

Why a one-step model learned at ``dt >> tau`` loses the effect of its own
inputs — and how to put step tests, block experiments and learned simulators on
one comparable scale.

What is established, and what is not
-----------------------------------
The empirical finding is solid and survives every control run so far: on the
August campaign, a one-step model fitted at **dt = 1 h** returns the wrong sign
for the ventilation inputs (VENT_ROOF +0.21 C, VENT_SIDE +0.30 C) while the same
data at **dt = 15 min** returns the physical one (-1.61, -2.12). Controls:

* **Not sample size.** Subsampling the 15-min frame to the 1 h frame's n = 621
  keeps the signs correct (-0.98, -1.49). At equal n and equal learner, the
  interval is what changes the answer.
* **Not overfitting.** Heavy regularisation of the 1 h model does not repair the
  sign (+0.21 -> +0.49 with 4 leaves and min_samples_leaf 40).
* **Not the seed.** Identical to two decimals across five seeds.
* **Not feature redundancy.** Dropping ``dT_ext`` changes nothing.
* **A flexible learner fails first, but OLS is not immune.** Plain OLS on the
  same 1 h data keeps every sign correct (-0.30, -1.34, -0.48), and it holds at
  45 min too. Pushed to 90 min (``dt/tau`` = 5.6) it inverts as well
  (VENT_ROOF +0.09, FOG +0.40). So the learner sets *where* the threshold sits,
  not *whether* there is one: the tree ensemble breaks at ``dt/tau`` ~ 3.7 and
  OLS at ~ 5.6. An earlier draft of this module claimed OLS never inverted; that
  was true only of the range that had been tried.

**The mechanism is now identified**, after two hypotheses were falsified here
rather than asserted. Both of these are wrong:

1. *"An hour steps over the response."* False on the arithmetic. For a
   first-order plant with the input held over the step, the true one-step
   sensitivity ``K (1 - e^{-D/tau})`` **grows** with the step and saturates at
   ``K``. A longer step makes the effect easier to see, not harder.
2. *"``T_t`` absorbs the input, so the two go collinear."* Not reproduced.
   ``corr(u, T)`` stays flat across every sampling interval and does not track
   the error at all; it is useless as a diagnostic.

What does reproduce it is the **experiment's own block structure aliasing with
the sampling grid**. Every block ends with a scheduled treatment -> off
transition, and during that washout the plant relaxes *back up*. Sampled at a
``dt`` comparable to the treatment length, those transitions systematically pair
a high ``u_t`` with a positive ``dT``. At ``dt`` = 1 h against a 2 h treatment,
one sample in three is exactly that pairing; at 15 min it is one in twelve.

The inversion needs three conditions together, and removing any one of them
prevents it (see :func:`factorial_mechanism`):

===================  ========  ====  ====================
washout              inputs    dt    ratio to truth
===================  ========  ====  ====================
none                 7         60    0.43  (attenuated)
60 min               1         60    0.84  (fine)
60 min               7         15    0.70  (fine)
**60 min**           **7**     **60**  **-0.07  inverted**
===================  ========  ====  ====================

That last cell returns +0.21 C against a true -2.93; the real campaign at 1 h
returns +0.21 C for VENT_ROOF. Same sign, same magnitude, from a plant whose
gain is known by construction.

How bad it gets is governed by ``dt/tau``, even though *whether* it happens at
all is governed by the washout. :func:`sweep_taus` runs five time constants
across nine sampling intervals under the campaign's geometry and tests which
quantity the curves collapse onto. Grouping the ratios by ``dt/tau`` gives a
spread of 0.24; grouping them by ``dt`` at a fixed treatment length gives 0.57.
They collapse on ``dt/tau``, 2.4x more tightly, and the sign turns over around

    **dt / tau ~ 4**

which is one of the two usable design thresholds. The other comes from
:func:`sweep_treatment`, which holds ``dt/tau`` fixed at 3.75 and varies the
treatment length over five seeds: no seed inverts at ``dt/treat <= 0.5``, and
two to five of five invert at ``dt/treat >= 1``. The two are **not** redundant —
each sweep holds the other quantity fixed and still finds an effect — so the
rule of thumb needs both:

    **safe when dt/tau < 4 AND dt/treatment < 0.5**

The August campaign sits exactly on both edges at 1 h (3.75 and 0.50) and
comfortably inside them at 15 min (0.94 and 0.125), which is the whole story in
two numbers.

Tested out of sample on the real data at eight sampling intervals, the rule is
right **8 times out of 8**, and six of those eight points (20, 30, 40, 45, 90 and
120 min) played no part in deriving it. The zero crossing lands exactly on the
predicted boundary: VENT_ROOF runs -1.61, -1.41, -0.72, -0.45, -0.08 and then
+0.21 at 60 min.

The physics behind the ``dt/tau`` half is now consistent:
during the washout the plant relaxes back up with time constant ``tau``, so when
``dt >> tau`` that relaxation *completes inside a bin that still carries a high
input*, and the contamination is maximal. When ``dt << tau`` only a fraction of
it lands in the bin.

Worth noting for honesty's sake: the "~4 tau" figure quoted in earlier drafts
was approximately the right threshold for entirely the wrong reason.

The irony worth stating plainly: **the washout that makes the block design clean
for block-level analysis is exactly what poisons the one-step model.** The two
analyses want opposite things from the same schedule.

Practical consequences:

* **Do not read causal gradients off a non-parametric one-step model fitted at a
  ``dt`` comparable to the treatment duration.** A linear fit is a useful
  cross-check because it degrades later — but it degrades, so agreement between
  the two is evidence and disagreement is a warning, not a proof either way.
  OLS also overestimates by 1.3-2.6x at coarse ``dt``.
* The pre-fit diagnostic is :func:`mixed_bin_fraction` plus the ratio of ``dt``
  to the *treatment* length, not ``corr(u, T)`` and not ``dt/tau`` alone.
* When a schedule is designed to serve both a block analysis and a one-step
  model, the washout has to be either short relative to ``dt`` or explicitly
  modelled — or the one-step fit has to drop the end-of-treatment rows.

Contents
--------
``simulate_first_order``  synthetic plant with block-held random inputs
``sweep_step``            fit the same one-step model at many ``dt``, recover
                          the gradient, compare against the analytic truth
``equivalent_gain``       convert an effect measured at some horizon into the
                          static gain ``K``, so estimates from different
                          experiments can be compared
``reconcile``             the campaign's three independent routes to the same
                          number, on one scale
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------- #
# Synthetic plant
# --------------------------------------------------------------------------- #

def simulate_first_order(tau_min: float = 16.0, K: float = -3.0,
                         days: int = 31, base_dt_s: int = 10,
                         block_min: float = 180.0, levels=(0.0, 0.25, 0.5, 0.75, 1.0),
                         n_inputs: int = 1, input_corr: float = 0.0, K_others=None,
                         washout_min: float = 0.0, washout_share: float = 1.0,
                         ext_mean: float = 30.0, ext_amp: float = 8.0,
                         noise_c: float = 0.05, interlock_c: float | None = None,
                         seed: int = 0) -> pd.DataFrame:
    """Integrate ``dT/dt = (T_ext(t) + K u - T)/tau`` on a fine grid.

    The input is drawn once per block and held, exactly like the campaign's
    schedule — that hold is what creates the collinearity the sweep measures,
    so a study with white-noise inputs would miss the whole effect.

    ``interlock_c`` adds the piece the first version of this study was missing:
    a safety rule that forces the input to full whenever ``T`` exceeds it. That
    is a **feedback path from the state to the input**, and it is present in the
    real greenhouse (the 45 C rule opens both window banks, touching 32 % of
    campaign hours). Without it the sweep shows no failure at all, which is how
    we learned that the coarse-step failure is not about missing the transient.

    ``tau_min`` and ``K`` are known by construction, which is the point: every
    estimate downstream can be scored against the truth instead of against
    another estimate.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 3600 / base_dt_s)
    t_s = np.arange(n) * base_dt_s
    t_h = t_s / 3600.0

    # Diurnal exterior forcing plus a slow day-to-day drift, so the regression
    # has something to control for and the setting is not artificially clean.
    ext = (ext_mean
           + ext_amp * np.sin(2 * np.pi * (t_h - 9) / 24)
           + 3.0 * np.sin(2 * np.pi * t_h / (24 * 7))
           + rng.normal(0, 0.3, n).cumsum() * 0.002)

    per_block = int(block_min * 60 / base_dt_s)
    n_blocks = int(np.ceil(n / per_block))

    # Several block-held inputs, optionally co-activated. ``input_corr`` is the
    # probability that a secondary input copies the primary one's level in a
    # block, which is how a screening design makes actuators travel together.
    lv = np.asarray(levels, dtype=float)
    U_blocks = rng.choice(lv, size=(n_blocks, n_inputs))
    if n_inputs > 1 and input_corr > 0:
        copy = rng.random((n_blocks, n_inputs - 1)) < input_corr
        U_blocks[:, 1:] = np.where(copy, U_blocks[:, :1], U_blocks[:, 1:])
    U = np.repeat(U_blocks, per_block, axis=0)[:n]

    # The protocol's washout: the last ``washout_min`` of every block is forced
    # to zero. This is the ingredient the first synthetic study lacked, and it
    # is not cosmetic — it means every block ends with a scheduled
    # treatment -> off transition, during which the plant relaxes *back up*.
    # Sampled at a dt comparable to the treatment length, those transitions
    # systematically pair a high ``u_t`` with a positive ``dT``, which is the
    # textbook recipe for reading the sign backwards.
    if washout_min > 0:
        per_wash = int(washout_min * 60 / base_dt_s)
        phase = np.arange(n) % per_block
        tail = phase >= (per_block - per_wash)
        if washout_share >= 1.0:
            U[tail] = 0.0
        else:
            # Only a share of the blocks end in a washout; in the rest the tail
            # gets a fresh random level instead, which is how the September
            # campaign's paired half-blocks switch (no return to zero).
            blk = np.arange(n) // per_block
            has_wash = rng.random(n_blocks) < washout_share
            fresh = np.repeat(rng.choice(lv, size=(n_blocks, n_inputs)), per_block, axis=0)[:n]
            w = tail & has_wash[blk]
            f = tail & ~has_wash[blk]
            U[w] = 0.0
            U[f] = fresh[f]
    u = U[:, 0].copy()
    gains = np.asarray([K] + list(K_others if K_others is not None
                                  else [K * 0.6] * (n_inputs - 1)), dtype=float)
    drive = U @ gains

    a = np.exp(-base_dt_s / (tau_min * 60.0))
    T = np.empty(n)
    T[0] = ext[0]
    eps = rng.normal(0, noise_c, n)
    if interlock_c is None:
        eq = ext + drive
        for i in range(1, n):
            T[i] = a * T[i - 1] + (1 - a) * eq[i - 1] + eps[i]
    else:
        # The scheduled level still holds per block, but the safety rule can
        # override it — so the realized input is partly a function of the state.
        for i in range(1, n):
            if T[i - 1] > interlock_c:
                # Realized primary input goes to full; the drive is corrected by
                # the difference between realized and scheduled for that input.
                drive[i - 1] += gains[0] * (1.0 - U[i - 1, 0])
                u[i - 1] = 1.0
            T[i] = a * T[i - 1] + (1 - a) * (ext[i - 1] + drive[i - 1]) + eps[i]

    idx = pd.to_datetime("2026-08-03") + pd.to_timedelta(t_s, unit="s")
    df = pd.DataFrame({"T": T, "ext": ext, "u": u}, index=idx)
    for j in range(1, n_inputs):
        df[f"u{j}"] = U[:, j]
    df.attrs.update(tau_min=tau_min, K=K, block_min=block_min,
                    base_dt_s=base_dt_s, interlock_c=interlock_c,
                    n_inputs=n_inputs, input_corr=input_corr,
                    washout_min=washout_min, washout_share=washout_share)
    return df


def mixed_bin_fraction(df: pd.DataFrame, dt_min: float) -> float:
    """Share of resampling bins that straddle a change in the realized input.

    A bin that contains a level change averages two regimes: its ``u`` is a
    blend that was never commanded and its ``T`` covers a transient. In the real
    campaign these are common at 1 h — motors ramp, the interlock fires
    mid-hour — while a synthetic study with block lengths that divide the bin
    has none at all, which is exactly the difference this quantity exposes.
    """
    u = df.filter(regex=r"^u").copy()
    chg = (u.diff().abs() > 1e-9).any(axis=1)
    return float(chg.resample(f"{int(dt_min)}min").max().mean())


def true_gradient(K: float, dt_min: float, tau_min: float) -> float:
    """Analytic one-step sensitivity ``dT_{t+dt}/du`` for the held-input plant.

    Note the sign of the trend: this **increases** in ``dt``. Any estimator that
    returns less as ``dt`` grows is losing the effect for statistical reasons,
    not physical ones.
    """
    return K * (1.0 - np.exp(-dt_min / tau_min))


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #

def _fit_and_probe(d: pd.DataFrame, target: str, learner: str, seed: int):
    """Fit one-step model on (T, ext, u) and read back its do-gradient."""
    feats = ["T", "ext"] + [c for c in d.columns if c.startswith("u")]
    y = d["T"].shift(-1) - (d["T"] if target == "delta" else 0.0)
    m = pd.concat([d[feats], y.rename("y")], axis=1).dropna()
    if len(m) < 50:
        return np.nan, np.nan

    X, yy = m[feats].values, m["y"].values
    if learner == "ols":
        mod = LinearRegression().fit(X, yy)
    else:
        mod = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                            random_state=seed).fit(X, yy)

    # do(u = 1) - do(u = 0) holding the rest of the row fixed
    ui = [feats.index(c) for c in feats if c.startswith("u")]
    lo, hi = X.copy(), X.copy()
    lo[:, ui], hi[:, ui] = 0.0, 0.0
    hi[:, feats.index("u")] = 1.0
    g = float(np.mean(mod.predict(hi) - mod.predict(lo)))
    return g, float(np.corrcoef(m["u"], m["T"])[0, 1])


def sweep_step(df: pd.DataFrame, dts_min=(1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120),
               targets=("delta", "level"), learners=("ols", "hgb"),
               seed: int = 0) -> pd.DataFrame:
    """Recover the input gradient at a range of sampling intervals.

    Returns one row per (dt, target, learner) with the estimated gradient, the
    analytic truth at that same dt, their ratio, and ``corr_uT`` — the
    collinearity that explains the ratio.
    """
    tau, K = df.attrs["tau_min"], df.attrs["K"]
    rows = []
    for dt in dts_min:
        d = df.resample(f"{int(dt)}min").mean()
        truth = true_gradient(K, dt, tau)
        for tgt in targets:
            for lrn in learners:
                g, c = _fit_and_probe(d, tgt, lrn, seed)
                rows.append(dict(dt_min=dt, dt_over_tau=dt / tau, target=tgt,
                                 learner=lrn, estimado=g, verdadero=truth,
                                 ratio=g / truth if truth else np.nan,
                                 corr_uT=c, n=len(d)))
    out = pd.DataFrame(rows)
    out.attrs.update(df.attrs)
    return out


def sweep_taus(taus=(4, 8, 16, 32, 64), K: float = -3.0, days: int = 31,
               block_min: float = 180.0, washout_min: float = 60.0,
               n_inputs: int = 7, input_corr: float = 0.3,
               learner: str = "hgb", target: str = "delta",
               dts_min=(5, 10, 15, 20, 30, 45, 60, 90, 120),
               seed: int = 0, **kw) -> pd.DataFrame:
    """The sweep at several time constants, under the campaign's geometry.

    The point is a **collapse test**, and it is the one that decides which
    quantity belongs in a design rule. If the failure were governed by
    ``dt/tau``, the curves would collapse when plotted against that. The
    mechanism identified here says it is governed by ``dt`` relative to the
    **treatment** length instead, so the collapse should happen against
    ``dt_over_treat`` and *not* against ``dt_over_tau``.

    Both columns are returned so the reader can see which one lines up.
    """
    treat = block_min - washout_min
    frames = []
    for tau in taus:
        df = simulate_first_order(tau_min=tau, K=K, days=days,
                                  block_min=block_min, washout_min=washout_min,
                                  n_inputs=n_inputs, input_corr=input_corr,
                                  seed=seed, **kw)
        s = sweep_step(df, dts_min=dts_min, targets=(target,),
                       learners=(learner,), seed=seed)
        s["tau_min"] = tau
        s["dt_over_treat"] = s.dt_min / treat
        s["bins_mezcla"] = [mixed_bin_fraction(df, d) for d in s.dt_min]
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def sweep_treatment(treats_min=(15, 30, 45, 60, 90, 120, 180, 240),
                    washout_frac: float = 1 / 3, tau_min: float = 16.0,
                    dt_min: float = 60.0, K: float = -3.0, days: int = 31,
                    n_inputs: int = 7, input_corr: float = 0.3,
                    learner: str = "hgb", target: str = "delta",
                    seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """At a fixed sampling interval, vary the **treatment** length.

    This is the knob a practitioner actually owns before the campaign runs.
    Sampling faster afterwards is an analysis choice and cannot undo a schedule
    whose blocks are short relative to the grid; lengthening the treatment is a
    design choice that can. Washout is kept at a constant fraction of the block
    so the geometry stays proportional, as the protocol's does.

    A single run per cell is **not** enough: with one seed the curve comes out
    non-monotone and invites the wrong conclusion. Averaged over five, it is
    clean, and the spread between treatment lengths (0.52) clearly exceeds the
    spread between seeds (0.32), so the treatment length matters on its own
    account and not merely through ``dt/tau``, which is held fixed at 3.75 here
    by construction. Zero of five seeds invert at ``dt/treat <= 0.5``; two to
    five of five invert at ``dt/treat >= 1``.
    """
    truth = true_gradient(K, dt_min, tau_min)
    rows = []
    for treat in treats_min:
        block = treat / (1 - washout_frac)
        wash = block - treat
        gs, cs, mix = [], [], np.nan
        for sd in seeds:
            df = simulate_first_order(tau_min=tau_min, K=K, days=days,
                                      block_min=block, washout_min=wash,
                                      n_inputs=n_inputs, input_corr=input_corr,
                                      seed=sd)
            g, c = _fit_and_probe(df.resample(f"{int(dt_min)}min").mean(),
                                  target, learner, sd)
            gs.append(g); cs.append(c)
            if np.isnan(mix):
                mix = mixed_bin_fraction(df, dt_min)
        r = np.array(gs) / truth
        rows.append(dict(tratamiento_min=treat, lavado_min=round(wash, 1),
                         dt_min=dt_min, dt_over_treat=dt_min / treat,
                         dt_over_tau=dt_min / tau_min,
                         estimado=float(np.mean(gs)), verdadero=truth,
                         ratio=float(r.mean()), ratio_sd=float(r.std()),
                         n_invertidas=int((r < 0).sum()), n_semillas=len(seeds),
                         bins_mezcla=mix, corr_uT=float(np.mean(cs))))
    return pd.DataFrame(rows)


def factorial_mechanism(tau_min: float = 16.0, K: float = -3.0, days: int = 31,
                        block_min: float = 180.0, washouts=(0.0, 60.0),
                        n_inputs=(1, 7), dts=(15, 60), input_corr: float = 0.3,
                        seed: int = 0) -> pd.DataFrame:
    """The three-condition factorial that isolates the inversion.

    Reproduces the campaign's own geometry by default (180 min blocks, 120 of
    treatment and 60 of washout, seven inputs, sampled at 1 h) and returns the
    recovered-to-true ratio in every cell. The inverted cell is the one that
    matches the real data.
    """
    rows = []
    for w in washouts:
        for k in n_inputs:
            df = simulate_first_order(tau_min=tau_min, K=K, days=days,
                                      block_min=block_min, washout_min=w,
                                      n_inputs=k, input_corr=input_corr, seed=seed)
            for dt in dts:
                s = sweep_step(df, dts_min=(dt,), targets=("delta",),
                               learners=("ols", "hgb"), seed=seed)
                for _, r in s.iterrows():
                    rows.append(dict(lavado_min=w, n_entradas=k, dt_min=dt,
                                     aprendiz=r.learner, estimado=r.estimado,
                                     verdadero=r.verdadero, ratio=r.ratio,
                                     invertido=bool(r.ratio < 0),
                                     bins_mezcla=mixed_bin_fraction(df, dt)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Putting real estimates on one scale
# --------------------------------------------------------------------------- #

def equivalent_gain(effect: float, tau_min: float, kind: str,
                    horizon_min: float) -> float:
    """Convert an effect measured one way into the static gain ``K``.

    Three experiments in this project measure "what does this actuator do", at
    three different horizons, and they are **not** comparable as printed:

    * ``"step"``     - an asymptotic step-test gain. Already ``K``.
    * ``"onestep"``  - a simulator ``do()`` at horizon ``h``; the plant has only
      travelled ``1 - e^{-h/tau}`` of the way, so divide by that.
    * ``"blockmean"``- the mean over a treatment window of length ``h`` starting
      from equilibrium; the time-average of the step response is
      ``K (1 - (tau/h)(1 - e^{-h/tau}))``, so divide by that factor.

    Using each group's **own** tau matters, and it has to come from the *same
    signal* as the outcome: taking FOG's tau as the median over all its fits
    pulls in the humidity fit at 115 min and lands on 77 min instead of the
    39 min the temperature fits give.

    Results on the campaign (see ``07.sampling interval.ipynb``). VENT_ROOF
    reconciles well - blockmean -2.28, onestep -1.91, step -1.81, a spread of
    24 % across three experiments at three different horizons, which is the
    validation the learned simulator otherwise lacks. The failures are
    informative rather than noisy:

    * **FOG** gives -10.9 and -9.1 from the two campaign routes but -2.5 from
      the step test. The step trial ran at 06:00 with the interior at **91 % RH**
      and zero radiation; evaporative cooling scales with vapour pressure
      deficit, so that trial measured fog where it has almost no capacity. It is
      a different operating point, not a contradiction.
    * **LIGHTS** gives -3.80 from a step trial that started at 18:03 with PAR at
      709 and the interior at 46 C. That "cooling" is the evening solar decline;
      the trial is unusable for a gain whatever the trend term does.
    * **SHADE** does not reconcile because it was never identified in the first
      place - see ``campaign_effects.py``.

    So the correction is validated where the inputs are clean, and where it
    fails the data says why. A single step test is one operating point, and
    comparing it against a campaign average requires the operating points to
    match.
    """
    h, tau = float(horizon_min), float(tau_min)
    if kind == "step":
        return effect
    if kind == "onestep":
        f = 1.0 - np.exp(-h / tau)
    elif kind == "blockmean":
        f = 1.0 - (tau / h) * (1.0 - np.exp(-h / tau))
    else:
        raise ValueError(kind)
    return effect / f if f else np.nan


def reconcile(estimates: pd.DataFrame) -> pd.DataFrame:
    """Common-scale table from a frame of (grupo, ruta, efecto, tau, horizonte).

    ``ruta`` takes the ``kind`` values of :func:`equivalent_gain`. The output
    adds ``K_equivalente`` and, per group, the spread across routes — which is
    the honest measure of whether the three experiments agree.
    """
    e = estimates.copy()
    e["K_equivalente"] = [equivalent_gain(r.efecto, r.tau_min, r.ruta, r.horizonte_min)
                          for r in e.itertuples()]
    piv = e.pivot_table(index="grupo", columns="ruta", values="K_equivalente")
    piv["rango"] = piv.max(axis=1) - piv.min(axis=1)
    piv["dispersion_rel"] = piv["rango"] / piv.drop(columns="rango").abs().mean(axis=1)
    return piv.sort_values("dispersion_rel")
