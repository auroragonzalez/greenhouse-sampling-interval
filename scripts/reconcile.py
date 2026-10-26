"""Paper B: the three routes to an actuator's gain (August), written to CSV.

Phase-A step tests, block-mean ITT contrasts over the 2-h treatment and the 15-min one-step
simulator, each converted to an equivalent static gain with
``sampling_interval.equivalent_gain`` and that group's own temperature tau.
"""
from __future__ import annotations

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import campaign_effects as ce
import campaign_rl as crl
import protocol_check as pc
import sampling_interval as si
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import two_campaigns as tc

DATA = ROOT / "results"

if __name__ == "__main__":
    rep = pc.run_all(verbose=False)
    tt = rep.dynamics[(rep.dynamics.usable) & (rep.dynamics.signal == "temp_centro")]
    tau_g = tt.groupby("group").tau_min.median().to_dict()
    TAU = float(tt.tau_min.median())

    df = tc.with_exterior(crl.build_frame(), crl.STEP)
    sim = crl.fit_simulator(df)
    B = ce.itt_blocks(rep)
    day = B[(B.hour >= 9) & (B.hour < 19)]
    ef = ce.effects(day, standardise=False)
    d = df.dropna(subset=crl.FEATURES)
    dd = d[(d.index.hour >= 10) & (d.index.hour < 19)][crl.FEATURES].values

    rows = [dict(grupo=r.group, ruta="step", efecto=r.gain,
                 tau_min=tau_g.get(r.group, TAU), horizonte_min=np.nan)
            for _, r in tt.iterrows()]
    rows += [dict(grupo=g, ruta="blockmean", efecto=float(ef.loc[g, "effect"] * day[g].max()),
                  tau_min=tau_g.get(g, TAU), horizonte_min=120.0) for g in ce.GROUPS]
    rows += [dict(grupo=g, ruta="onestep", efecto=float(sim.do_effect(dd, {g: crl.FULL[g]})[0]),
                  tau_min=tau_g.get(g, TAU), horizonte_min=15.0) for g in crl.GROUPS]
    E = (pd.DataFrame(rows).groupby(["grupo", "ruta"], as_index=False)
         .agg(efecto=("efecto", "mean"), tau_min=("tau_min", "first"),
              horizonte_min=("horizonte_min", "first")))
    E["K_equivalente"] = [si.equivalent_gain(r.efecto, r.tau_min, r.ruta, r.horizonte_min)
                          for r in E.itertuples()]
    E.to_csv(DATA / "reconcile.csv", index=False)
    pd.Series(tau_g).rename("tau_min").to_csv(DATA / "tau_by_group.csv")
    print(E.pivot_table(index="grupo", columns="ruta", values="K_equivalente").round(2))
    print("median tau", round(TAU, 1), {k: round(v, 1) for k, v in tau_g.items()})
