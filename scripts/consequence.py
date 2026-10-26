"""Paper B: what the sign inversion costs a practitioner, short of control.

Two uses of a learned one-step model that come before any controller:

1. **Choosing the interval by validation score.** Every interval is scored by
   day-blocked cross-validation, the usual way to pick a model. If the score
   favours the interval that inverts, standard practice selects the wrong model.

2. **Scenario simulation (digital twin).** Each model is rolled forward over the
   2-h treatment of every daytime August block (phases B-D), from the observed
   state just before the block, under the logged actuator levels and exterior
   weather. Each block is simulated twice more with both vent groups forced
   fully open and fully closed. The randomised blocks give a measured answer to
   the same what-if question: the regression of the measured block-mean
   temperature on the commanded vent levels. Models are always evaluated on
   days they were not trained on.

    python3 scripts/consequence.py      # -> results/consequence_*.csv
"""
from __future__ import annotations

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import GroupKFold

import campaign_rl as crl
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import two_campaigns as tc

DATA = ROOT / "results"
WINDOW = ("2026-08-03", "2026-09-02 23:59")
BLOCKS = ROOT / "data" / "blocks" / "august_blocks.csv"
FEATS = list(crl.FEATURES)
VENTS = ("VENT_ROOF", "VENT_SIDE")
TREAT_MIN = 120
DTS = (10, 15, 20, 30, 40, 60, 120)        # intervals that divide the treatment
FOLDS = 5


def hgb(seed: int = 0) -> HGB:
    return HGB(max_iter=400, learning_rate=0.06, random_state=seed)


def design(dt: int) -> pd.DataFrame:
    """Binned frame with next-bin increments of temperature and humidity."""
    df = tc.with_exterior(crl.build_frame(freq=f"{dt}min", window=WINDOW), f"{dt}min")
    nxt = df.shift(-1)
    ok = df.index.to_series().diff().shift(-1) == pd.Timedelta(minutes=dt)
    df["dT"] = (nxt.temp_centro - df.temp_centro).where(ok)
    df["dH"] = (nxt.hum_centro - df.hum_centro).where(ok)
    df["day"] = df.index.normalize()
    return df


def fold_of_day(days: pd.Index) -> dict:
    u = np.array(sorted(set(days)))
    return {d: i % FOLDS for i, d in enumerate(u)}


# --------------------------------------------------------------------------- #
# 1. Model selection by validation score
# --------------------------------------------------------------------------- #

def selection(df: pd.DataFrame, dt: int) -> dict:
    m = df.dropna(subset=FEATS + ["dT"])
    X, y, g = m[FEATS].values, m["dT"].values, m["day"].values
    pred = np.empty_like(y)
    for tr, te in GroupKFold(FOLDS).split(X, y, g):
        pred[te] = hgb().fit(X[tr], y[tr]).predict(X[te])
    T0 = m.temp_centro.values
    sse = np.sum((y - pred) ** 2)
    # R2 of the increment, and of the next level (what most papers report)
    r2_inc = 1 - sse / np.sum((y - y.mean()) ** 2)
    lvl = T0 + y
    r2_lvl = 1 - sse / np.sum((lvl - lvl.mean()) ** 2)
    rmse = np.sqrt(sse / len(y))
    return dict(dt_min=dt, n=len(y), rmse_c=rmse, r2_increment=r2_inc,
                r2_level=r2_lvl, rmse_per_h=rmse * np.sqrt(60 / dt))


# --------------------------------------------------------------------------- #
# 2. Scenario simulation over the August blocks
# --------------------------------------------------------------------------- #

def day_blocks() -> pd.DataFrame:
    b = pd.read_csv(BLOCKS, parse_dates=["start_local"])
    b = b[b.phase.isin(["B", "C", "D"])]
    b = b[(b.start_local.dt.hour >= 9) & (b.start_local.dt.hour < 19)]
    for gr in crl.GROUPS:
        b[f"cmd_{gr}"] = b[f"cmd_{gr}"].fillna(0.0)
    return b.set_index("start_local")


def window(t0: pd.Timestamp, dt: int) -> slice:
    """Bins a one-step model can see the treatment in: the input in bin t only
    reaches bin t + dt, so the window is shifted one bin past the block start
    (it ends in the first washout bin)."""
    return slice(t0 + pd.Timedelta(minutes=dt), t0 + pd.Timedelta(minutes=TREAT_MIN))


def rollout(mt, mh, df: pd.DataFrame, t0: pd.Timestamp, dt: int,
            force: dict | None = None) -> float:
    """Mean simulated temperature over ``window(t0, dt)``.

    Starts from the observed state one bin before the block, steps once with
    the logged inputs of that bin, then once per treatment bin with the logged
    (or forced) inputs; only the states that follow a treatment input count.
    """
    step = pd.Timedelta(minutes=dt)
    t = t0 - step
    if t not in df.index:
        return np.nan
    T, H = float(df.at[t, "temp_centro"]), float(df.at[t, "hum_centro"])
    temps = []
    for k in range(TREAT_MIN // dt + 1):
        if t not in df.index:
            return np.nan
        x = df.loc[t, FEATS].astype(float).copy()   # weather and inputs at t
        x["temp_centro"], x["hum_centro"] = T, H
        x["dT_ext"] = T - x.ext_temp
        if force and t >= t0:
            for gname, v in force.items():
                x[gname] = v
        if x.isna().any():
            return np.nan
        v = x.values.reshape(1, -1)
        T = float(np.clip(T + mt.predict(v)[0], *crl.T_CLIP))
        H = float(np.clip(H + mh.predict(v)[0], *crl.H_CLIP))
        t = t + step
        if k >= 1:
            temps.append(T)
    return float(np.mean(temps))


def scenarios(df: pd.DataFrame, dt: int, blocks: pd.DataFrame) -> pd.DataFrame:
    m = df.dropna(subset=FEATS + ["dT", "dH"])
    fold = fold_of_day(df["day"])
    open_ = {g: crl.FULL[g] for g in VENTS}
    shut = {g: 0.0 for g in VENTS}
    rows = []
    for f in range(FOLDS):
        tr = m[m["day"].map(fold) != f]
        mt = hgb().fit(tr[FEATS].values, tr["dT"].values)
        mh = hgb().fit(tr[FEATS].values, tr["dH"].values)
        for t0, r in blocks.iterrows():
            if fold.get(t0.normalize()) != f:
                continue
            meas = df.temp_centro.loc[window(t0, dt)].mean()
            rows.append(dict(
                block=t0, dt_min=dt, measured=meas,
                simulated=rollout(mt, mh, df, t0, dt),
                vents_open=rollout(mt, mh, df, t0, dt, open_),
                vents_shut=rollout(mt, mh, df, t0, dt, shut),
                **{g: r[f"cmd_{g}"] for g in crl.GROUPS},
                **{c: df.loc[window(t0, dt), c].mean()
                   for c in ("ext_temp", "ext_par", "ext_hum", "ext_wind")}))
    return pd.DataFrame(rows)


def measured_vent_effect(sc: pd.DataFrame) -> float:
    """Ground truth from the randomised blocks: measured block-mean temperature
    regressed on commanded levels and weather; both vent groups fully open
    versus closed, in degrees C."""
    d = sc.drop_duplicates("block")
    xs = list(crl.GROUPS) + ["ext_temp", "ext_par", "ext_hum", "ext_wind"]
    d = d.dropna(subset=xs + ["measured"])
    fit = sm.OLS(d.measured, sm.add_constant(d[xs])).fit(cov_type="HC1")
    eff = sum(fit.params[g] * crl.FULL[g] for g in VENTS)
    cov = fit.cov_params().loc[list(VENTS), list(VENTS)].values
    w = np.array([crl.FULL[g] for g in VENTS])
    se = float(np.sqrt(w @ cov @ w))
    return eff, se, len(d)


def main() -> None:
    sel, sc = [], []
    blocks = day_blocks()
    for dt in DTS:
        df = design(dt)
        sel.append(selection(df, dt))
        s = scenarios(df, dt, blocks)
        sc.append(s)
        print(f"dt={dt:3d}  " + "  ".join(f"{k}={v:.3f}" for k, v in sel[-1].items()
                                           if k not in ("dt_min", "n")), flush=True)
    sel = pd.DataFrame(sel)
    sc = pd.concat(sc, ignore_index=True)
    sel.to_csv(DATA / "consequence_selection.csv", index=False)
    sc.to_csv(DATA / "consequence_scenarios.csv", index=False)

    print()
    rows = []
    for dt, g in sc.groupby("dt_min"):
        truth, se, n = measured_vent_effect(g)
        g = g.dropna(subset=["simulated", "vents_open", "vents_shut"])
        err = g.simulated - g.measured
        scen = g.vents_open - g.vents_shut
        rows.append(dict(dt_min=dt, blocks=len(g), measured_effect=truth,
                         measured_se=se, simulated_effect=scen.mean(),
                         share_warming=(scen > 0).mean(),
                         rmse_block_mean=np.sqrt((err ** 2).mean()),
                         bias_block_mean=err.mean()))
        print(f"dt={dt:3d}  blocks={len(g):3d}  measured open-shut={truth:+.2f} (se {se:.2f})  "
              f"simulated={scen.mean():+.2f} C (warming in {100 * (scen > 0).mean():.0f} % "
              f"of blocks)  block-mean RMSE={np.sqrt((err**2).mean()):.2f} "
              f"bias={err.mean():+.2f}")
    pd.DataFrame(rows).to_csv(DATA / "consequence_summary.csv", index=False)


if __name__ == "__main__":
    main()
