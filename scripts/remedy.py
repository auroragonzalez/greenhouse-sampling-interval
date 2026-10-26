"""Paper B: does giving the one-step model the input trajectory fix the sign?

The mechanism (Section 5) says that with bin means the change from bin t to
bin t+1 is set by the input in bin t+1, not by the one in bin t. Two framings
put the input the model is credited with back where the plant sees it:

``media``      the framing in the draft: bin means of state and inputs at t,
               target mean T(t+1) - mean T(t). The baseline that inverts.
``siguiente``  the same bin means plus the inputs of the next bin. The
               ``do()`` query raises the *next* bin's input, which is the
               action a controller would choose for the coming interval.
``zoh``        zero-order-hold alignment: the state is read at the instant t
               (the bin's first sample), the inputs and the exterior are means
               over the interval that follows, [t, t+dt), and the target is
               T(t+dt) - T(t), again instantaneous. This is the framing of
               Eq. (1) in the paper, so on the synthetic plant its truth is
               exactly K (1 - exp(-dt/tau)) whenever the input is held.

Nothing here needs data a controller would not have: the inputs over the coming
interval are the action it is choosing.

    python3 scripts/remedy.py             # synthetic + real, point estimates
    python3 scripts/remedy.py synth       # one part
    python3 scripts/remedy.py real 40     # real, with a 40-replicate day bootstrap
"""
from __future__ import annotations

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import LinearRegression

import sampling_interval as si
import campaign_rl as crl
import protocol_check as pc
import two_campaigns as tc

DATA = ROOT / "results"
FRAMINGS = ("media", "siguiente", "zoh")


# --------------------------------------------------------------------------- #
# Synthetic plant
# --------------------------------------------------------------------------- #

def _synth_frame(df: pd.DataFrame, dt: int, framing: str):
    """Design matrix for one framing on the simulated plant."""
    ucols = [c for c in df.columns if c.startswith("u")]
    f = f"{dt}min"
    if framing == "zoh":
        st = df[["T"]].resample(f).first()
        fwd = df[["ext"] + ucols].resample(f).mean()
        d = st.join(fwd)
    else:
        d = df.resample(f).mean()
    y = d["T"].shift(-1) - d["T"]
    feats = ["T", "ext"] + ucols
    if framing == "siguiente":
        nxt = d[ucols].shift(-1).add_suffix("_next")
        d = d.join(nxt)
        feats = feats + list(nxt.columns)
    m = pd.concat([d[feats], y.rename("y")], axis=1).dropna()
    return m, feats, ucols


def _synth_probe(m, feats, ucols, framing, learner, seed):
    X, y = m[feats].values, m["y"].values
    mod = (LinearRegression() if learner == "ols"
           else HGB(max_iter=300, learning_rate=0.08, random_state=seed)).fit(X, y)
    lo = X.copy()
    for c in feats:
        if c.startswith("u"):
            lo[:, feats.index(c)] = 0.0
    hi = lo.copy()
    hi[:, feats.index("u_next" if framing == "siguiente" else "u")] = 1.0
    return float(np.mean(mod.predict(hi) - mod.predict(lo)))


def synth(seeds=range(5), dts=(15, 30, 45, 60, 90, 120)) -> pd.DataFrame:
    """August geometry (3 h blocks, 60 min washout, 7 inputs) and September's
    (1 h blocks, 30 min holds, 39 % of blocks ending in a washout)."""
    geoms = {
        "agosto": dict(block_min=180.0, washout_min=60.0, washout_share=1.0),
        "septiembre": dict(block_min=60.0, washout_min=30.0,
                           washout_share=59 / 153),
    }
    rows = []
    for gname, g in geoms.items():
        for sd in seeds:
            df = si.simulate_first_order(tau_min=16.0, K=-3.0, days=31,
                                         n_inputs=7, input_corr=0.3, seed=sd, **g)
            for dt in dts:
                truth = si.true_gradient(-3.0, dt, 16.0)
                for fr in FRAMINGS:
                    m, feats, ucols = _synth_frame(df, dt, fr)
                    for lrn in ("hgb", "ols"):
                        e = _synth_probe(m, feats, ucols, fr, lrn, sd)
                        rows.append(dict(geometria=gname, seed=sd, dt_min=dt,
                                         framing=fr, learner=lrn, estimado=e,
                                         verdadero=truth, ratio=e / truth))
            print("synth", gname, "seed", sd, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "remedy_synth.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# Real campaigns (station exterior, as in Table 2)
# --------------------------------------------------------------------------- #

def real_frame(campaign: str, dt: int, framing: str):
    if framing == "media":
        return tc.frame(campaign, f"{dt}min")
    w = tc.WINDOWS[campaign]
    f = f"{dt}min"
    sens1 = pc.load_sensors(pc.DATA_DIR)
    if framing == "zoh":
        # State at the instant t: the first minute of the bin.
        sens = sens1.resample("1min").mean().resample(f).first()
    else:
        sens = sens1.resample(f).mean()
    lev = crl.group_levels(pc.DATA_DIR, f)        # mean over [t, t+dt)
    df = sens.join(lev).join(tc.exterior(f)).loc[w[0]:w[1]].copy()
    hour = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dT_ext"] = df["temp_centro"] - df["ext_temp"]
    feats = list(crl.FEATURES)
    if framing == "siguiente":
        nxt = df[crl.GROUPS].shift(-1).add_suffix("_next")
        df = df.join(nxt)
        feats = feats + list(nxt.columns)
    d = df.dropna(subset=feats).copy()
    y = d["temp_centro"].shift(-1) - d["temp_centro"]
    ok = d.index.to_series().shift(-1) - d.index.to_series() == pd.Timedelta(f)
    m = pd.concat([d[feats], y.rename("y")], axis=1)[ok].dropna()
    return m, feats


def real_effects(m, feats, framing, learner="hgb", seed=0) -> dict[str, float]:
    X, y = m[feats].values, m["y"].values
    mod = (LinearRegression() if learner == "ols"
           else HGB(max_iter=400, learning_rate=0.06, random_state=seed)).fit(X, y)
    day = m[(m.index.hour >= 10) & (m.index.hour < 19)]
    lo = day[feats].values.copy()
    for c in feats:
        if c.split("_next")[0] in crl.GROUPS:
            lo[:, feats.index(c)] = 0.0
    out = {}
    for g in tc.PROBE:
        hi = lo.copy()
        col = f"{g}_next" if framing == "siguiente" else g
        hi[:, feats.index(col)] = crl.FULL[g]
        out[g] = float(np.mean(mod.predict(hi) - mod.predict(lo)))
    return out


def real(B: int = 0, dts=(15, 30, 45, 60, 90, 120)) -> pd.DataFrame:
    rows = []
    for camp in tc.WINDOWS:
        for dt in dts:
            for fr in FRAMINGS:
                m, feats = real_frame(camp, dt, fr)
                for lrn in ("hgb", "ols"):
                    point = real_effects(m, feats, fr, lrn)
                    boot = None
                    if B and lrn == "hgb":
                        rng = np.random.default_rng(0)
                        days = np.array(sorted(set(m.index.normalize())))
                        by_day = {d: m[m.index.normalize() == d] for d in days}
                        boot = pd.DataFrame([
                            real_effects(pd.concat([by_day[d] for d in
                                                    rng.choice(days, len(days))]),
                                         feats, fr, lrn, seed=b)
                            for b in range(B)])
                    for g in tc.PROBE:
                        r = dict(campaign=camp, dt_min=dt, framing=fr, learner=lrn,
                                 group=g, n=len(m), effect=point[g],
                                 criterion=tc.criterion(camp, dt))
                        if boot is not None:
                            r.update(lo=float(boot[g].quantile(.05)),
                                     hi=float(boot[g].quantile(.95)),
                                     p_positive=float((boot[g] > 0).mean()))
                        rows.append(r)
            print("real", camp, dt, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / ("remedy_real.csv" if B else "remedy_real_point.csv"),
               index=False)
    return out


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if part in ("all", "synth"):
        s = synth()
        print(s[s.learner == "hgb"].groupby(["geometria", "dt_min", "framing"])
              .ratio.agg(["mean", "std", lambda r: int((r < 0).sum())])
              .round(2).unstack("framing").to_string())
    if part in ("all", "real"):
        r = real(B)
        print(r[r.learner == "hgb"].pivot_table(
            index=["campaign", "dt_min"], columns=["group", "framing"],
            values="effect").round(2).to_string())
