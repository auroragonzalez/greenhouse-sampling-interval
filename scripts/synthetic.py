"""Paper B: every synthetic experiment behind the figures, written to CSV.

The factorial, the time-constant collapse, the treatment-length sweep and
the washout-share sweep that places the September campaign's geometry on the
same axis as August's.

    python3 scripts/synthetic.py            # all
    python3 scripts/synthetic.py share      # one part
"""
from __future__ import annotations

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import sampling_interval as si

DATA = ROOT / "results"
DATA.mkdir(exist_ok=True)

#: September: 153 one-hour blocks, of which 59 (S2 + S3) end in a 30-min
#: washout to zero and 94 are paired halves that switch to a new level.
SEP_SHARE = 59 / 153


def factorial() -> None:
    si.factorial_mechanism().to_csv(DATA / "synth_factorial.csv", index=False)


def taus() -> None:
    si.sweep_taus().to_csv(DATA / "synth_taus.csv", index=False)


def treatment() -> None:
    si.sweep_treatment().to_csv(DATA / "synth_treatment.csv", index=False)


def share(seeds=range(5), dts=(15, 30, 45, 60, 90, 120),
          shares=(0.0, 0.25, SEP_SHARE, 0.6, 1.0)) -> None:
    """September geometry (60-min blocks, 30 of treatment) with a varying share
    of blocks that end in a washout. At share 1 every switch is a
    treatment -> off transition, as in August; at 0 every switch is to a fresh
    random level."""
    rows = []
    for sh in shares:
        for sd in seeds:
            df = si.simulate_first_order(tau_min=16.0, K=-3.0, days=31,
                                         block_min=60.0, washout_min=30.0,
                                         washout_share=sh, n_inputs=7,
                                         input_corr=0.3, seed=sd)
            s = si.sweep_step(df, dts_min=dts, targets=("delta",),
                              learners=("hgb",), seed=sd)
            for _, r in s.iterrows():
                rows.append(dict(share=sh, seed=sd, dt_min=r.dt_min,
                                 dt_over_tau=r.dt_over_tau, ratio=r.ratio))
        print("share", round(sh, 3), "done", flush=True)
    pd.DataFrame(rows).to_csv(DATA / "synth_share.csv", index=False)


if __name__ == "__main__":
    parts = {"factorial": factorial, "taus": taus, "treatment": treatment,
             "share": share}
    for name in (sys.argv[1:] or parts):
        print("->", name, flush=True)
        parts[name]()
