"""Paper B: the checks of Table 3 (Section 4.1) on the August campaign.

Each row refits the gradient-boosting one-step model under one change and
reports the ``do()`` effect of opening the roof vents fully, at 15 min and at
1 h, with the same exterior as everything else (``two_campaigns.frame``):

* reference model;
* the 15-min data subsampled (evenly spaced rows) to the 1-h row count;
* shallow trees at 1 h (4 leaves, >= 40 samples per leaf, 200 iterations);
* five random seeds;
* seven feature ablations, one at a time: the interior-exterior difference,
  interior humidity, the two time-of-day terms (together), and each of the four
  exterior channels;
* the other six actuator groups removed (only the roof vents left);
* a minimal three-feature model: interior temperature, interior-exterior
  difference and roof-vent level.

    python3 scripts/robust.py      # -> results/robust.csv
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

import two_campaigns as tc
from two_campaigns import crl

DATA = tc.DATA
G = "VENT_ROOF"
ABLATIONS = {
    "dT_ext": ["dT_ext"], "hum_centro": ["hum_centro"],
    "hour": ["hour_sin", "hour_cos"], "ext_temp": ["ext_temp"],
    "ext_hum": ["ext_hum"], "ext_par": ["ext_par"], "ext_wind": ["ext_wind"],
}
MINIMAL = ["temp_centro", "dT_ext", G]


def effect(m: pd.DataFrame, feats: list[str], seed: int = 0, n_sub: int | None = None,
           **kw) -> float:
    if n_sub is not None and n_sub < len(m):
        m = m.iloc[np.linspace(0, len(m) - 1, n_sub).astype(int)]
    par = {"max_iter": 400, "learning_rate": 0.06} | kw
    mod = HGB(random_state=seed, **par).fit(m[feats].values, m["y"].values)
    day = m[(m.index.hour >= 10) & (m.index.hour < 19)]
    lo = day[feats].values.copy()
    for g in crl.GROUPS:
        if g in feats:
            lo[:, feats.index(g)] = 0.0
    hi = lo.copy()
    hi[:, feats.index(G)] = crl.FULL[G]
    return float(np.mean(mod.predict(hi) - mod.predict(lo)))


def main() -> None:
    m15, f = tc.frame("august", "15min")
    m60, _ = tc.frame("august", "60min")
    rows = []

    def add(check, dt, value, detail=""):
        rows.append(dict(check=check, dt_min=dt, effect=value, detail=detail))

    for m, dt in ((m15, 15), (m60, 60)):
        add("reference", dt, effect(m, f), f"n={len(m)}")
    add("subsampled", 15, effect(m15, f, n_sub=len(m60)), f"n={len(m60)}")
    add("shallow", 60, effect(m60, f, max_leaf_nodes=4, min_samples_leaf=40,
                              max_iter=200, learning_rate=0.05))
    for m, dt in ((m15, 15), (m60, 60)):
        s = [effect(m, f, seed=k) for k in range(5)]
        add("seeds_range", dt, max(s) - min(s), "max - min over 5 seeds")
        for name, drop in ABLATIONS.items():
            add("ablation", dt, effect(m, [c for c in f if c not in drop]), name)
    others = [c for c in f if c not in crl.GROUPS or c == G]
    add("others_removed", 60, effect(m60, others))
    add("minimal", 60, effect(m60, MINIMAL))
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "robust.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
