"""Paper B: the sampling-interval sweep on both campaigns (August and September).

Exterior features come from the on-site station for both campaigns (the
export of 2026-10-19 covers September; the earlier one stopped on 2026-09-07,
which is why a previous version used the Open-Meteo archive). Temperature,
humidity and PAR are the station's. Wind is the station's too, except where the
anemometer has no reading or is clearly failing: on every September morning it
logs nothing from about 06:00 and then saturates at up to 25 m/s until about
11:00, while Open-Meteo gives under 2 m/s. Those bins are filled with Open-Meteo
wind rescaled to the station (see ``exterior``).

Outputs go to ``results/`` as CSV, which is what the R figure script reads.
"""
from __future__ import annotations

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import LinearRegression

import campaign_rl as crl
import protocol_check as pc

DATA = ROOT / "results"
OM_FILE = ROOT / "data" / "openmeteo" / "openmeteo_archive_2026-08-01_2026-09-23.csv"

#: Campaign windows, local time. September ends at the last block (07:00 on
#: the 23rd, an hour before the transplant).
WINDOWS = {"august": ("2026-08-03", "2026-09-02 23:59"),
           "september": ("2026-09-16 14:00", "2026-09-23 06:59")}

#: Geometry used by the criterion: time constant (Phase A, temperature fits)
#: and the length for which a level is held before it changes.
TAU_MIN = 16.1
HOLD_MIN = {"august": 120.0, "september": 30.0}

PROBE = ("VENT_ROOF", "VENT_SIDE")

#: Open-Meteo 10-m wind -> station wind, least squares on the 10-min bins of
#: 3 Aug - 23 Sep where the station reading is valid (r = 0.75).
WIND_FILL = (0.302, 0.642)
#: A station wind bin is a failure, not wind, when it exceeds the rescaled
#: Open-Meteo value by more than this (flags 10 bins in August and 49 in
#: September, all of the latter between 08:00 and 11:00).
WIND_BAD_MS = 4.0


def openmeteo(freq: str) -> pd.DataFrame:
    """Open-Meteo archive on a local-time grid.

    Instantaneous channels are labelled at their hour; shortwave radiation is a
    preceding-hour mean, so it is moved to the middle of that hour before
    interpolating (``protocol_check`` fact 5).
    """
    raw = pd.read_csv(OM_FILE, parse_dates=["time"]).set_index("time")
    inst = raw[["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]].copy()
    inst.columns = ["om_temp", "om_hum", "om_wind"]
    inst["om_wind"] /= 3.6                                   # km/h -> m/s
    rad = raw["shortwave_radiation"].rename("om_rad")
    rad.index = rad.index - pd.Timedelta("30min")
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="1min")
    out = (pd.concat([inst, rad], axis=1).reindex(grid.union(rad.index).union(inst.index))
           .interpolate(method="time", limit_area="inside").reindex(grid))
    return out.resample(freq).mean()


def exterior(freq: str) -> pd.DataFrame:
    """Station exterior on a local-time grid, wind gaps and failures filled.

    Same grid and interpolation as ``protocol_check.load_outside`` (and hence
    ``campaign_rl.build_frame``); only the wind bins that are missing or fail
    the check against Open-Meteo are replaced.
    """
    ext = pc.load_outside(pc.DATA_DIR, freq=freq)
    om = openmeteo(freq)["om_wind"].reindex(ext.index)
    fill = WIND_FILL[0] + WIND_FILL[1] * om
    bad = ext["ext_wind"].isna() | (ext["ext_wind"] - fill > WIND_BAD_MS)
    ext["ext_wind"] = ext["ext_wind"].where(~bad, fill)
    ext["wind_filled"] = bad.astype(float)
    return ext


def with_exterior(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Swap the exterior of a ``campaign_rl.build_frame`` frame for ``exterior``."""
    ext = exterior(freq).reindex(df.index)
    df = df.copy()
    for c in ("ext_temp", "ext_hum", "ext_par", "ext_wind"):
        df[c] = ext[c]
    df["dT_ext"] = df["temp_centro"] - df["ext_temp"]
    return df


def frame(campaign: str, freq: str, exterior_: str = "station") -> tuple[pd.DataFrame, list[str]]:
    """One-step design matrix: features at t, target T(t+dt) - T(t)."""
    w = WINDOWS[campaign]
    sens = pc.load_sensors(pc.DATA_DIR).resample(freq).mean()
    lev = crl.group_levels(pc.DATA_DIR, freq)
    df = sens.join(lev).join(exterior(freq)).loc[w[0]:w[1]].copy()
    hour = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dT_ext"] = df["temp_centro"] - df["ext_temp"]
    feats = list(crl.FEATURES)
    d = df.dropna(subset=feats).copy()
    y = d["temp_centro"].shift(-1) - d["temp_centro"]
    ok = d.index.to_series().shift(-1) - d.index.to_series() == pd.Timedelta(freq)
    m = pd.concat([d[feats], y.rename("y")], axis=1)[ok].dropna()
    return m, feats


def do_effects(m: pd.DataFrame, feats: list[str], learner: str = "hgb",
               seed: int = 0) -> dict[str, float]:
    """``do()`` effect of opening each vent group fully, daytime rows, all
    other actuators set to zero — the query of Section 3 of the paper."""
    X, y = m[feats].values, m["y"].values
    mod = (LinearRegression() if learner == "ols"
           else HGB(max_iter=400, learning_rate=0.06, random_state=seed)).fit(X, y)
    day = m[(m.index.hour >= 10) & (m.index.hour < 19)]
    lo = day[feats].values.copy()
    for g in crl.GROUPS:
        lo[:, feats.index(g)] = 0.0
    out = {}
    for g in PROBE:
        hi = lo.copy(); hi[:, feats.index(g)] = crl.FULL[g]
        out[g] = float(np.mean(mod.predict(hi) - mod.predict(lo)))
    return out


def day_bootstrap(m: pd.DataFrame, feats: list[str], learner: str, B: int,
                  seed: int = 0) -> pd.DataFrame:
    """Resample whole days with replacement and refit — the unit of the
    schedule is the day, so rows within a day are not independent."""
    rng = np.random.default_rng(seed)
    days = np.array(sorted(set(m.index.normalize())))
    by_day = {d: m[m.index.normalize() == d] for d in days}
    rows = []
    for b in range(B):
        pick = rng.choice(days, size=len(days), replace=True)
        mb = pd.concat([by_day[d] for d in pick])
        rows.append(do_effects(mb, feats, learner, seed=b))
    return pd.DataFrame(rows)


def mixed_bins(campaign: str, dt_min: int) -> float:
    """Share of dt-bins in which some commanded group changes level."""
    w = WINDOWS[campaign]
    lev = crl.group_levels(pc.DATA_DIR, "1min").loc[w[0]:w[1]].round(3)
    chg = (lev.diff().abs() > 1e-9).any(axis=1)
    return float(chg.resample(f"{dt_min}min").max().mean())


def criterion(campaign: str, dt_min: float) -> str:
    safe = dt_min / TAU_MIN < 4 and dt_min / HOLD_MIN[campaign] < 0.5
    return "safe" if safe else "unsafe"


RUNS = [("august", "station"), ("september", "station")]


def sweep(dts=(10, 15, 20, 30, 40, 45, 60, 90, 120), B: int = 0,
          runs=RUNS) -> pd.DataFrame:
    rows = []
    for camp, ext in runs:
        for dt in dts:
            m, feats = frame(camp, f"{dt}min", ext)
            for lrn in ("hgb", "ols"):
                point = do_effects(m, feats, lrn)
                boot = day_bootstrap(m, feats, lrn, B) if B else None
                for g in PROBE:
                    r = dict(campaign=camp, exterior=ext, dt_min=dt, learner=lrn,
                             group=g, n=len(m), effect=point[g],
                             dt_over_tau=dt / TAU_MIN,
                             dt_over_hold=dt / HOLD_MIN[camp],
                             criterion=criterion(camp, dt))
                    if boot is not None:
                        r.update(lo=float(boot[g].quantile(.05)),
                                 hi=float(boot[g].quantile(.95)),
                                 p_positive=float((boot[g] > 0).mean()))
                    rows.append(r)
            print(camp, ext, dt, len(m), flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # ``python3 two_campaigns.py``      point estimates, both campaigns
    # ``python3 two_campaigns.py 40``   + day-bootstrap (B=40)
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    s = sweep(B=B, runs=RUNS)
    s.to_csv(DATA / ("real_sweep.csv" if B else "real_sweep_point.csv"), index=False)
    mb = pd.DataFrame([dict(campaign=c, dt_min=dt, mixed=mixed_bins(c, dt))
                       for c in WINDOWS for dt in (10, 15, 20, 30, 40, 45, 60, 90, 120)])
    mb.to_csv(DATA / "mixed_bins.csv", index=False)
    print(mb.pivot(index="dt_min", columns="campaign", values="mixed").round(3))
