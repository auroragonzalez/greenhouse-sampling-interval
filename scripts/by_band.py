"""Paper B: vent effect by interior-exterior temperature difference.

Answers a question from the greenhouse side: a vent does not act on the
greenhouse directly, it couples the interior to the outside air, so its effect
should depend on whether outside is colder or warmer. Here the one-step model's
``do()`` effect is computed row by row (daytime, 10-19 h) and averaged within
bands of ``T_int - T_ext``. Exterior is the on-site station for both
campaigns, as in ``two_campaigns.py``.

    python3 scripts/by_band.py      # -> results/vent_by_band.csv
"""
from __future__ import annotations

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

import campaign_rl as crl
from two_campaigns import DATA, frame

BANDS = [-20, 0, 3, 6, 10, 15, 40]


def per_row(campaign: str, dt_min: int, seed: int = 0) -> pd.DataFrame:
    m, feats = frame(campaign, f"{dt_min}min")
    mod = HGB(max_iter=400, learning_rate=0.06, random_state=seed).fit(
        m[feats].values, m["y"].values)
    day = m[(m.index.hour >= 10) & (m.index.hour < 19)]
    lo = day[feats].values.copy()
    for g in crl.GROUPS:
        lo[:, feats.index(g)] = 0.0
    out = pd.DataFrame({"dT": day["dT_ext"].values}, index=day.index)
    for g in ("VENT_ROOF", "VENT_SIDE"):
        hi = lo.copy(); hi[:, feats.index(g)] = crl.FULL[g]
        out[g] = mod.predict(hi) - mod.predict(lo)
    return out


if __name__ == "__main__":
    rows = []
    for camp in ("august", "september"):
        for dt in (15, 60):
            r = per_row(camp, dt)
            r["band"] = pd.cut(r.dT, BANDS)
            share_cold = float((r.dT < 0).mean())
            for band, sub in r.groupby("band", observed=True):
                for g in ("VENT_ROOF", "VENT_SIDE"):
                    rows.append(dict(campaign=camp, dt_min=dt, band=str(band),
                                     lo=band.left, hi=band.right, group=g,
                                     n=len(sub), effect=float(sub[g].mean()),
                                     share_interior_colder=share_cold))
            print(camp, dt, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "vent_by_band.csv", index=False)
    print(out.pivot_table(index=["campaign", "band"], columns=["dt_min", "group"],
                          values="effect").round(2).to_string())
    print(out.groupby(["campaign", "dt_min"]).share_interior_colder.first().round(3))
