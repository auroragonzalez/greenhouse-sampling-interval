"""Paper B: the what-if of ``whatif.py`` with forecast weather instead of the station.

Used for decisions, a twin must simulate the next hours with weather that has
not happened yet. This repeats the August what-if (15 min and 1 h) with the
exterior that would have been available at the block start, from two Open-Meteo
archives of forecasts actually issued (not reanalysis):

* ``short``: Historical Forecast API, the latest run at each hour (lead of a
  few hours);
* ``dayahead``: Previous Runs API, the run of the day before (24-48 h lead).

Each is used raw and corrected with the station (``mos``: per-variable linear
regression of the station reading on the forecast and the time of day, fitted
on the days of the other folds). ``anchor`` adds to the corrected short-range
forecast the station-minus-forecast offset of temperature and humidity seen in
the bin before the block, i.e. what an operator knows when deciding.

Models are those of ``whatif.py`` (trained on the station exterior, the block's
day held out); only the weather fed to the rollout changes. The bin before the
block keeps the station reading, as it would in real use.

    python3 scripts/forecast_check.py   # -> results/forecast_check*.csv
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import consequence as cq
import two_campaigns as tc
import whatif as wi
from consequence import crl, FEATS, FOLDS, TREAT_MIN

DATA = cq.DATA
DTS = (15, 60)
FILES = {"short": cq.ROOT / "data" / "openmeteo" / "openmeteo_histforecast_2026-08-01_2026-09-23.csv",
         "dayahead": cq.ROOT / "data" / "openmeteo" / "openmeteo_dayahead_2026-08-01_2026-09-23.csv"}
EXT = ["ext_temp", "ext_hum", "ext_par", "ext_wind"]
#: Shortwave -> PAR, both in W m-2, for the raw forecasts. The station reports
#: PAR as energy: its noon readings are about half the forecast shortwave.
SW_TO_PAR = 0.48
ANCHORED = ["ext_temp", "ext_hum"]
CAND = ["VENT_ROOF", "VENT_SIDE", "SHADE", "FOG"]


def forecast(src: str, freq: str) -> pd.DataFrame:
    """Forecast on a local-time grid, interpolated as ``two_campaigns.openmeteo``."""
    raw = pd.read_csv(FILES[src], skiprows=3, parse_dates=["time"]).set_index("time")
    raw.columns = ["temp", "hum", "wind", "rad"]
    inst = raw[["temp", "hum", "wind"]].copy()
    inst["wind"] /= 3.6
    rad = raw["rad"].copy()
    rad.index = rad.index - pd.Timedelta("30min")
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="1min")
    out = (pd.concat([inst, rad], axis=1).reindex(grid.union(rad.index).union(inst.index))
           .interpolate(method="time", limit_area="inside").reindex(grid))
    out = out.resample(freq).mean()
    return pd.DataFrame({"ext_temp": out.temp, "ext_hum": out.hum,
                         "ext_par": out.rad * SW_TO_PAR, "ext_wind": out.wind})


def mos(fc: pd.DataFrame, st: pd.DataFrame, fold: dict) -> pd.DataFrame:
    """Station ~ forecast + time of day, each day predicted from the other folds."""
    h = 2 * np.pi * (fc.index.hour + fc.index.minute / 60) / 24
    out = fc.copy()
    f_of = fc.index.normalize().map(lambda d: fold.get(d, -1))
    for c in EXT:
        X = np.column_stack([np.ones(len(fc)), fc[c], np.sin(h), np.cos(h)])
        y = st[c].reindex(fc.index).values
        ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        for f in range(FOLDS):
            tr = ok & (f_of != f) & (f_of >= 0)
            beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            te = (f_of == f) | (f_of < 0)
            out.loc[te, c] = X[te] @ beta
    out["ext_hum"] = out.ext_hum.clip(0, 100)
    out["ext_par"] = out.ext_par.clip(lower=0).where(fc.ext_par > 0, 0.0)
    out["ext_wind"] = out.ext_wind.clip(lower=0)
    return out


def variants(dt: int, fold: dict) -> dict[str, pd.DataFrame]:
    freq = f"{dt}min"
    st = tc.exterior(freq)
    v = {}
    for src in FILES:
        fc = forecast(src, freq)
        v[f"{src}_raw"] = fc
        v[f"{src}_mos"] = mos(fc, st, fold)
    v["short_mos_anchor"] = v["short_mos"]          # offset added per block
    return v


def block_frame(df: pd.DataFrame, t0: pd.Timestamp, dt: int, ext: pd.DataFrame,
                anchor: bool) -> pd.DataFrame | None:
    step = pd.Timedelta(minutes=dt)
    w = df.loc[t0 - step: t0 + pd.Timedelta(minutes=TREAT_MIN)].copy()
    fut = w.index >= t0
    e = ext.reindex(w.index)
    if e.loc[fut, EXT].isna().any().any():
        return None
    for c in EXT:
        val = e[c].values.copy()
        if anchor and c in ANCHORED:
            val = val + (w[c].iloc[0] - e[c].iloc[0])
        w.loc[fut, c] = val[fut]
    w["dT_ext"] = w["temp_centro"] - w["ext_temp"]
    return w


def run(dt: int, blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = cq.design(dt)
    m = df.dropna(subset=FEATS + ["dT", "dH"])
    fold = cq.fold_of_day(df["day"])
    V = variants(dt, fold)
    scen = wi.scenario_set() | {"logged": {}}
    rows, err = [], []
    for f in range(FOLDS):
        tr = m[m["day"].map(fold) != f]
        mt = cq.hgb().fit(tr[FEATS].values, tr["dT"].values)
        mh = cq.hgb().fit(tr[FEATS].values, tr["dH"].values)
        for t0, r in blocks.iterrows():
            if fold.get(t0.normalize()) != f:
                continue
            w = df.loc[cq.window(t0, dt)]
            base = dict(block=t0, dt_min=dt, meas_T=w.temp_centro.mean(),
                        meas_H=w.hum_centro.mean(),
                        **{g: r[f"cmd_{g}"] for g in crl.GROUPS},
                        **{c: w[c].mean() for c in wi.WEATHER})
            sims = {"station": df}
            for name, ext in V.items():
                bf = block_frame(df, t0, dt, ext, anchor=name.endswith("anchor"))
                if bf is None:
                    break
                sims[name] = bf
                tw = slice(t0, t0 + pd.Timedelta(minutes=TREAT_MIN - dt))
                err.append(dict(dt_min=dt, variant=name, block=t0,
                                **{c: (bf.loc[tw, c] - df.loc[tw, c]).abs().mean() for c in EXT}))
            else:
                outs = {k: wi.rollouts(mt, mh, s, t0, dt, scen) for k, s in sims.items()}
                if all(outs.values()):
                    for k, out in outs.items():
                        row = dict(base, variant=k)
                        for nm, (a, b) in out.items():
                            row[f"T|{nm}"], row[f"H|{nm}"] = a, b
                        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(err)


def summarise(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eff, choice = [], []
    for (dt, var), g in d.groupby(["dt_min", "variant"], sort=False):
        ref = d[(d.dt_min == dt) & (d.variant == "station")].set_index("block")
        g = g.set_index("block")
        me = wi.measured(ref.reset_index(), "meas_T")
        mh = wi.measured(ref.reset_index(), "meas_H")
        row = dict(dt_min=dt, variant=var, blocks=len(g),
                   level_mae_T=(g["T|logged"] - g.meas_T).abs().mean(),
                   level_mae_H=(g["H|logged"] - g.meas_H).abs().mean())
        for v, M in (("T", me), ("H", mh)):
            ce = pd.DataFrame({gr: g[f"{v}|{gr}:on"] - g[f"{v}|{gr}:off"] for gr in crl.GROUPS})
            cr = pd.DataFrame({gr: ref[f"{v}|{gr}:on"] - ref[f"{v}|{gr}:off"] for gr in crl.GROUPS})
            row[f"effect_shift_{v}"] = (ce - cr).abs().mean().mean()   # per block, vs station
            for gr in crl.GROUPS:
                eff.append(dict(dt_min=dt, variant=var, var=v, group=gr,
                                measured=M.at[gr, "measured"], simulated=ce[gr].mean(),
                                station=cr[gr].mean()))
        mid = g[(g.index.hour >= 11) & (g.index.hour <= 15)]
        mref = ref.loc[mid.index]
        pick = pd.DataFrame({c: mid[f"T|{c}:on"] - mid[f"T|{c}:off"] for c in CAND}).idxmin(axis=1)
        pref = pd.DataFrame({c: mref[f"T|{c}:on"] - mref[f"T|{c}:off"] for c in CAND}).idxmin(axis=1)
        row["midday_same_pick"] = (pick == pref).mean()
        row |= {f"picks_{c}": (pick == c).mean() for c in CAND}
        choice.append(row)
    return pd.DataFrame(eff), pd.DataFrame(choice)


def main() -> None:
    blocks = cq.day_blocks()
    D, E = zip(*(run(dt, blocks) for dt in DTS))
    d, e = pd.concat(D), pd.concat(E)
    eff, summ = summarise(d)
    err = e.groupby(["dt_min", "variant"], sort=False)[EXT].mean().reset_index()
    summ = summ.merge(err, on=["dt_min", "variant"], how="left")
    eff.to_csv(DATA / "forecast_check_effects.csv", index=False)
    summ.to_csv(DATA / "forecast_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(2).to_string(index=False)); print()
    print(eff[eff["var"] == "T"].pivot_table(index=["dt_min", "group"], columns="variant",
                                             values="simulated", sort=False).round(2))


if __name__ == "__main__":
    main()
