"""Paper B: the digital twin asked what-if questions the randomised blocks answer.

A twin is useful for the questions it answers, not for its one-step error. The
August schedule randomised all seven actuator groups over 2-h blocks, so the
blocks give a measured answer to each of the questions a grower would put to a
twin, short of designing a controller:

1. **Per actuator.** What if this group had been fully on, rather than off, for
   the 2-h block? Temperature and humidity, for each of the seven groups.
2. **Dose.** How much does each extra 25 % of roof-vent opening cool?
3. **Choice.** Which single measure cools the greenhouse most at midday?

Twin answers come from rolling the models of ``consequence.py`` forward over
each daytime block, from the observed state, under the logged weather and the
logged levels of every group not being changed; blocks are always simulated by
a model that did not see their day. Measured answers come from the
regression of the block means on the commanded levels and the weather; the
dose--response uses one indicator per roof-vent level.

    python3 scripts/whatif.py      # -> results/whatif_*.csv
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm

import consequence as cq
from consequence import crl, FEATS, FOLDS, TREAT_MIN

DATA = cq.DATA
DTS = (10, 15, 30, 60)
WEATHER = ["ext_temp", "ext_par", "ext_hum", "ext_wind"]
DOSES = (0.0, 25.0, 50.0, 75.0, 100.0)


def scenario_set() -> dict[str, dict]:
    """Named forcings applied during the treatment bins."""
    s = {}
    for g in crl.GROUPS:
        s[f"{g}:on"] = {g: crl.FULL[g]}
        s[f"{g}:off"] = {g: 0.0}
    for d in DOSES:
        s[f"ROOF:{d:g}"] = {"VENT_ROOF": d}
    return s


def rollouts(mt, mh, df: pd.DataFrame, t0: pd.Timestamp, dt: int,
             scen: dict[str, dict]) -> dict[str, tuple[float, float]]:
    """Mean simulated (T, RH) over ``cq.window`` for every scenario at once,
    stepping the same way as ``cq.rollout``."""
    step = pd.Timedelta(minutes=dt)
    t = t0 - step
    if t not in df.index:
        return {}
    names = list(scen)
    n = len(names)
    T = np.full(n, float(df.at[t, "temp_centro"]))
    H = np.full(n, float(df.at[t, "hum_centro"]))
    col = {c: i for i, c in enumerate(FEATS)}
    sT, sH = [], []
    for k in range(TREAT_MIN // dt + 1):
        if t not in df.index:
            return {}
        x = np.tile(df.loc[t, FEATS].astype(float).values, (n, 1))
        x[:, col["temp_centro"]], x[:, col["hum_centro"]] = T, H
        x[:, col["dT_ext"]] = T - x[:, col["ext_temp"]]
        if t >= t0:
            for i, nm in enumerate(names):
                for g, v in scen[nm].items():
                    x[i, col[g]] = v
        if np.isnan(x).any():
            return {}
        T = np.clip(T + mt.predict(x), *crl.T_CLIP)
        H = np.clip(H + mh.predict(x), *crl.H_CLIP)
        t = t + step
        if k >= 1:
            sT.append(T.copy()); sH.append(H.copy())
    mT, mH = np.mean(sT, axis=0), np.mean(sH, axis=0)
    return {nm: (mT[i], mH[i]) for i, nm in enumerate(names)}


def simulate(dt: int, blocks: pd.DataFrame) -> pd.DataFrame:
    df = cq.design(dt)
    m = df.dropna(subset=FEATS + ["dT", "dH"])
    fold = cq.fold_of_day(df["day"])
    scen = scenario_set()
    rows = []
    for f in range(FOLDS):
        tr = m[m["day"].map(fold) != f]
        mt = cq.hgb().fit(tr[FEATS].values, tr["dT"].values)
        mh = cq.hgb().fit(tr[FEATS].values, tr["dH"].values)
        for t0, r in blocks.iterrows():
            if fold.get(t0.normalize()) != f:
                continue
            out = rollouts(mt, mh, df, t0, dt, scen)
            if not out:
                continue
            w = df.loc[cq.window(t0, dt)]
            row = dict(block=t0, dt_min=dt, meas_T=w.temp_centro.mean(),
                       meas_H=w.hum_centro.mean(),
                       **{g: r[f"cmd_{g}"] for g in crl.GROUPS},
                       **{c: w[c].mean() for c in WEATHER})
            for nm, (a, b) in out.items():
                row[f"T|{nm}"], row[f"H|{nm}"] = a, b
            rows.append(row)
    return pd.DataFrame(rows)


def measured(d: pd.DataFrame, y: str) -> pd.DataFrame:
    """Full-on minus off effect of every group on block-mean ``y``."""
    xs = list(crl.GROUPS) + WEATHER
    d = d.dropna(subset=xs + [y])
    fit = sm.OLS(d[y], sm.add_constant(d[xs])).fit(cov_type="HC1")
    return pd.DataFrame({"measured": fit.params[crl.GROUPS] * pd.Series(crl.FULL),
                         "se": fit.bse[crl.GROUPS] * pd.Series(crl.FULL)})


def measured_dose(d: pd.DataFrame) -> pd.DataFrame:
    """Roof-vent level indicators (0 % as reference), other groups linear."""
    others = [g for g in crl.GROUPS if g != "VENT_ROOF"] + WEATHER
    lv = pd.get_dummies(d.VENT_ROOF.round(), prefix="R", dtype=float)
    lv = lv.drop(columns="R_0.0")
    X = sm.add_constant(pd.concat([d[others], lv], axis=1))
    ok = X.notna().all(axis=1) & d.meas_T.notna()
    fit = sm.OLS(d.meas_T[ok], X[ok]).fit(cov_type="HC1")
    out = [dict(level=0.0, measured=0.0, se=0.0)]
    for c in lv.columns:
        out.append(dict(level=float(c[2:]), measured=fit.params[c], se=fit.bse[c]))
    return pd.DataFrame(out)


def main() -> None:
    blocks = cq.day_blocks()
    effects, doses, choice = [], [], []
    for dt in DTS:
        d = simulate(dt, blocks)
        d.to_csv(DATA / f"whatif_blocks_{dt}.csv", index=False)
        for y, v in (("meas_T", "T"), ("meas_H", "H")):
            me = measured(d, y)
            for g in crl.GROUPS:
                c = d[f"{v}|{g}:on"] - d[f"{v}|{g}:off"]
                effects.append(dict(dt_min=dt, var=v, group=g, blocks=len(d),
                                    measured=me.at[g, "measured"], se=me.at[g, "se"],
                                    simulated=c.mean(), share_same_sign=
                                    (np.sign(c) == np.sign(me.at[g, "measured"])).mean()))
        md = measured_dose(d)
        for _, r in md.iterrows():
            s = (d[f"T|ROOF:{r.level:g}"] - d["T|ROOF:0"]).mean()
            doses.append(dict(dt_min=dt, level=r.level, measured=r.measured,
                              se=r.se, simulated=s))
        # Choice at midday (block start 11:00-15:00): which single measure,
        # each at full, gives the lowest temperature? Measured ranking from
        # the regression above; twin ranking per block.
        mid = d[(d.block.dt.hour >= 11) & (d.block.dt.hour <= 15)]
        cand = ["VENT_ROOF", "VENT_SIDE", "SHADE", "FOG"]
        me = measured(d, "meas_T")
        best_meas = me.loc[cand, "measured"].idxmin()
        sims = pd.DataFrame({g: mid[f"T|{g}:on"] - mid[f"T|{g}:off"] for g in cand})
        pick = sims.idxmin(axis=1).value_counts(normalize=True)
        choice.append(dict(dt_min=dt, blocks=len(mid), measured_best=best_meas,
                           **{f"twin_picks_{g}": pick.get(g, 0.0) for g in cand},
                           **{f"twin_{g}": sims[g].mean() for g in cand}))
        print(f"dt={dt}: done ({len(d)} blocks)", flush=True)

    E, D, C = pd.DataFrame(effects), pd.DataFrame(doses), pd.DataFrame(choice)
    E.to_csv(DATA / "whatif_effects.csv", index=False)
    D.to_csv(DATA / "whatif_dose.csv", index=False)
    C.to_csv(DATA / "whatif_choice.csv", index=False)
    pd.set_option("display.width", 200)
    print(E.round(2).to_string(index=False)); print()
    print(D.round(2).to_string(index=False)); print()
    print(C.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
