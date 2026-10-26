"""
campaign_effects.py
===================

Causal read-out of the August-2026 campaign: dose-response curves, the effects
table, and the old-vs-new sign-flip validation that the protocol names as the
headline result (``EXPERIMENT_PROTOCOL.md`` section 7).

The one methodological choice worth stating up front
----------------------------------------------------
Every estimate here is **intention-to-treat**: the treatment variable is the
level the seeded schedule *commanded* for a block, never the level the
actuators were *observed* at. That distinction is the whole point of the
campaign. The commanded level is exogenous by construction — it was drawn from
a pre-generated schedule keyed on the timestamp — whereas the observed level is
contaminated by the 45 C interlock, which opens both window banks precisely when
the greenhouse is hottest. The interlock touches **32 % of campaign hours**,
concentrated between 11:00 and 19:00, i.e. exactly the band where ventilation
matters, so conditioning on the realized level would quietly rebuild the
confounding the campaign was run to destroy.

Using the commanded level makes the interlock a *mediator* sitting on the
causal path rather than a confounder, which is the correct treatment: the
estimate answers "what happens if I schedule the vents open", which is also the
question a controller has to answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import protocol_check as pc

GROUPS = ["LIGHTS", "FOG", "VENT_ROOF", "VENT_SIDE", "SHADE", "RECIRC", "RECIRC2"]

#: Exterior covariates. Randomisation means these are not needed for
#: unbiasedness; they are in the model only to soak up variance and tighten the
#: intervals.
CONTROLS = ["ext_temp", "ext_par", "ext_hum", "ext_wind"]

#: Daylight band used for the "does ventilation cool?" question. At night an
#: empty greenhouse in August sits close to ambient, so opening a vent moves
#: almost nothing and pooling the two bands only dilutes the daytime effect.
DAY_HOURS = (9, 19)


def itt_blocks(rep, wx: pd.DataFrame | None = None,
               phases: tuple[str, ...] = ("B", "C", "D")) -> pd.DataFrame:
    """One row per protocol block: commanded levels, outcome, weather.

    The outcome is the mean interior state over the block's **treatment
    window** (the first ``pc.TREATMENT_HOURS`` of the 3 h block); the washout
    hour is excluded so a block is not scored on its own recovery. Blocks from
    Phase A are dropped by default: they are long single-factor step trials on
    a different timing grid, and :func:`protocol_check.phase_a_dynamics`
    already reads them properly.
    """
    if wx is None:
        wx = pc.load_weather(freq="1min")
    bt = rep.blocks[rep.blocks.phase.isin(phases)]
    sens = rep.sens
    rows = []
    for t0, r in bt.iterrows():
        t1 = t0 + pd.Timedelta(hours=pc.TREATMENT_HOURS)
        seg, wseg = sens.loc[t0:t1], wx.loc[t0:t1]
        if seg.empty or wseg.empty:
            continue
        pre = sens.loc[t0 - pd.Timedelta(minutes=30):t0]
        d = dict(block=t0, day=int(r.day), phase=r.phase, hour=t0.hour,
                 temp=seg.temp_centro.mean(), hum=seg.hum_centro.mean(),
                 temp_pre=pre.temp_centro.mean() if len(pre) else np.nan,
                 temp_max=seg.temp_centro.max())
        for c in CONTROLS:
            d[c] = wseg[c].mean()
        # An unlisted group is off, not unknown: the schedule commands every
        # group in every block.
        for g in GROUPS:
            d[g] = float(r[g]) if pd.notna(r[g]) else 0.0
        rows.append(d)
    return pd.DataFrame(rows).set_index("block")


def _fit(B: pd.DataFrame, outcome: str, regressors: list[str],
         standardise: bool = True):
    d = B[[outcome] + regressors].dropna()
    X = d[regressors]
    if standardise:
        sd = X.std().replace(0, np.nan)
        X = ((X - X.mean()) / sd).dropna(axis=1)
    return sm.OLS(d[outcome], sm.add_constant(X)).fit(cov_type="HC1"), d


def effects(B: pd.DataFrame, outcome: str = "temp",
            groups: list[str] | None = None,
            controls: list[str] | None = None,
            standardise: bool = True) -> pd.DataFrame:
    """ITT effect of each group on the block-mean outcome, with robust CIs.

    Coefficients are per standard deviation of the commanded level, so the
    seven groups — which live on completely different scales (0-10 intensity,
    seconds of spray, aperture %) — can be read on one axis. Heteroskedasticity-
    robust (HC1) standard errors, because block variance is much larger at
    midday than at night.
    """
    groups = groups or GROUPS
    controls = controls if controls is not None else CONTROLS
    fit, d = _fit(B, outcome, groups + controls, standardise)
    ci = fit.conf_int()
    out = pd.DataFrame({
        "effect": fit.params[groups],
        "se": fit.bse[groups],
        "t": fit.tvalues[groups],
        "p": fit.pvalues[groups],
        "ci_lo": ci.loc[groups, 0],
        "ci_hi": ci.loc[groups, 1],
    })
    out.attrs["n"] = len(d)
    out.attrs["r2"] = fit.rsquared
    return out.sort_values("effect")


def effects_by_daypart(B: pd.DataFrame, outcome: str = "temp") -> pd.DataFrame:
    """The same table split into day and night, side by side.

    Worth doing every time: LIGHTS is invisible by day (solar gain swamps a few
    kW of lamps) and clearly warming at night, while the vents are the mirror
    image. A pooled estimate hides both.
    """
    day = B[(B.hour >= DAY_HOURS[0]) & (B.hour < DAY_HOURS[1])]
    night = B[(B.hour >= 21) | (B.hour < 6)]
    a, b = effects(day, outcome), effects(night, outcome)
    out = pd.concat([a[["effect", "t"]].rename(columns=lambda c: c + "_day"),
                     b[["effect", "t"]].rename(columns=lambda c: c + "_night")],
                    axis=1)
    out.attrs["n_day"], out.attrs["n_night"] = a.attrs["n"], b.attrs["n"]
    return out.sort_values("effect_day")


def dose_response(B: pd.DataFrame, group: str, outcome: str = "temp",
                  daytime: bool = True, adjust: bool = True) -> pd.DataFrame:
    """Mean outcome per commanded level, optionally weather-adjusted.

    Randomisation makes the raw per-level means unbiased already, but August
    weather still varies a lot between the days a level happened to land on, so
    the adjusted column (residual after regressing out the exterior covariates,
    re-centred) is usually the one to plot. Both are returned so the reader can
    see they agree.
    """
    d = B
    if daytime:
        d = d[(d.hour >= DAY_HOURS[0]) & (d.hour < DAY_HOURS[1])]
    d = d[[group, outcome] + CONTROLS].dropna()
    if adjust and len(d) > len(CONTROLS) + 2:
        fit = sm.OLS(d[outcome], sm.add_constant(d[CONTROLS])).fit()
        d = d.assign(adj=fit.resid + d[outcome].mean())
    else:
        d = d.assign(adj=d[outcome])
    g = d.groupby(d[group].round(0))
    out = g.agg(n=(outcome, "size"), raw=(outcome, "mean"),
                adjusted=("adj", "mean"), sd=("adj", "std"))
    out["se"] = out["sd"] / np.sqrt(out["n"])
    return out.drop(columns="sd")


# --------------------------------------------------------------------------- #
# The headline validation
# --------------------------------------------------------------------------- #

#: Observational actuator columns (on-minutes per hour) mapped onto the same
#: seven groups, so the old and new estimates can be put on one chart.
OLD_MAP = {
    "VENT_ROOF": ["act_vent_cen_este", "act_vent_cen_oeste"],
    "VENT_SIDE": ["act_vent_lat_norte", "act_vent_lat_sur", "act_vent_lat_oeste"],
    "SHADE":     ["act_pantalla_cenital"],
    "FOG":       ["act_fog"],
    "RECIRC":    ["act_recirculador"],
    "RECIRC2":   ["act_recirculador2"],
}


def old_effects(standardise: bool = True) -> pd.DataFrame:
    """The same regression shape on the year-long *observational* data.

    This is the control condition for the whole campaign. Nothing about the
    model changes — same outcome, same exterior covariates, same standardisation
    — so any difference in the coefficients is a difference in the data, which
    is precisely the claim being tested.
    """
    import greenhouse_dataset as gd

    old = gd.build_hourly_dataset().loc[gd.ACTIVE_START:].copy()
    old = old.rename(columns={"ext_rad_par": "ext_par", "ext_viento": "ext_wind"})
    for g, cols in OLD_MAP.items():
        have = [c for c in cols if c in old]
        old[g] = old[have].mean(axis=1) if have else np.nan
    old["temp"] = old["temp_centro"]
    groups = list(OLD_MAP)
    fit, d = _fit(old, "temp", groups + CONTROLS, standardise)
    out = pd.DataFrame({"effect": fit.params[groups], "t": fit.tvalues[groups]})
    out["raw_corr"] = [old[g].corr(old["temp_centro"]) for g in groups]
    out.attrs["n"] = len(d)
    return out


def sign_flip(B: pd.DataFrame) -> pd.DataFrame:
    """Old (observational) versus new (interventional) effect, group by group.

    ``flipped`` marks the groups where the campaign changed the *sign* of the
    estimated effect — the single comparison section 7 of the protocol nominates
    as proof that the exercise worked.
    """
    new = effects(B[(B.hour >= DAY_HOURS[0]) & (B.hour < DAY_HOURS[1])])
    old = old_effects()
    out = pd.concat([
        old[["raw_corr", "effect", "t"]].rename(
            columns={"raw_corr": "old_raw_corr", "effect": "old_effect", "t": "old_t"}),
        new[["effect", "t"]].rename(columns={"effect": "new_effect", "t": "new_t"}),
    ], axis=1).dropna(subset=["new_effect"])
    # LIGHTS has no observational counterpart (the lamps were not logged as an
    # actuator before the campaign), so it gets no verdict rather than a
    # spurious one from comparing against NaN.
    out["flipped"] = pd.array(
        np.where(out.old_effect.notna(),
                 np.sign(out.old_effect) != np.sign(out.new_effect), None),
        dtype="boolean")
    out["new_significant"] = out.new_t.abs() > 2
    out["old_significant"] = out.old_t.abs() > 2
    return out.sort_values("new_effect")
