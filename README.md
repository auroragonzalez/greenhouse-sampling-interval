# Sampling interval and sign inversion in learned greenhouse models

Code for the paper

> **Sampling interval and sign inversion of actuator effects in one-step models
> learned from randomised greenhouse experiments**
> A. González-Vidal, Y. Ye, P. J. Fernández-Campillo, M. F. García-Cruz,
> M. A. Zamora-Izquierdo, A. F. Skarmeta. University of Murcia. *Manuscript
> submitted, 2026.*

The paper fits gradient-boosting and linear one-step models to two randomised
campaigns run in an empty 76.8 m² research greenhouse in Murcia, Spain (August
2026: 2 h treatments + 1 h washouts; September 2026: 30 min holds). It finds
that the roof vents seem to cool the greenhouse when the model is learned at
15 min and seem to warm it when the model is learned at 1 h. The paper
explains this as aliasing between the schedule and the sampling grid, shows the
same effect on a synthetic first-order plant, and proposes a rule you can check
before fitting: Δt/τ < 4 and Δt/hold < 0.5.

> [!IMPORTANT]
> **The campaign data are not in this repository yet.** We are preparing the
> dataset for publication on Zenodo, and it is not ready. Until it is out,
> `make data` writes a **synthetic, randomly generated** stand-in with the same
> file names, columns and time conventions, so that every script runs. The
> stand-in comes from a toy plant. **Results computed from it are not the
> paper's results and should not be quoted.** This README will link the Zenodo
> record (DOI) once it is published.

## Quick start

```bash
pip install -r requirements.txt     # Python >= 3.10
make data                           # synthetic placeholder data -> data/  (~15 s)
make results                        # every analysis -> results/*.csv     (~1 h)
make figs                           # figures -> figs/*.pdf               (R)
```

`make figs` needs R ≥ 4.3 with `ggplot2`, `dplyr`, `tidyr`, `readr`,
`patchwork` and `scales`. The figures use the *Latin Modern Roman* font
through `cairo_pdf`.

## Repository layout

```
src/                      shared library
  protocol_check.py         reads the raw exports; decodes the protocol, per-block levels, Phase-A step fits
  campaign_rl.py            campaign design matrix, realised group levels, one-step simulator
  campaign_effects.py       intention-to-treat block contrasts
  sampling_interval.py      synthetic first-order plant and the sampling-interval mechanism
scripts/                  one script per analysis of the paper (each writes CSV to results/)
  two_campaigns.py          sweep of the sampling interval on both campaigns (Fig. 1, Appendix table)
  synthetic.py              synthetic-plant experiments (Figs. 2-6)
  reconcile.py              step tests vs block means vs one-step model (Fig. 7)
  by_band.py                vent effect by interior-exterior temperature difference (Fig. 8)
  robust.py                 robustness checks (Table "robust")
  remedy.py                 whether giving the model the input trajectory restores the sign
  consequence.py            interval choice by validation score; scenario simulation
  whatif.py                 what-if questions answered by the randomised blocks (twin tables)
  forecast_check.py         the same what-if questions with forecast weather
  figs.R                    all figures, from results/
tools/
  make_synthetic_data.py    writes the synthetic placeholder data
data/                     input data (see data/README.md); generated, not tracked
results/                  CSV outputs (not tracked)
figs/                     PDF figures (not tracked)
```

`synthetic.py` uses only `sampling_interval.py`. Its output, and Figs. 2-6,
do not depend on the campaign data, so those results are the paper's results
even before the data are released.

## Using the real data

Once the Zenodo dataset is out, download it and point the code at the raw
exports instead of running `make data`:

```bash
export OSIRIS_DATA_DIR=/path/to/raw/exports   # default: data/raw/
make results figs
```

The scripts expect the layout described in `data/README.md`. If the published
layout is different, we will provide a script to map it.

## Notes

* All timestamps in the raw exports are UTC. The loaders convert them to
  Murcia local time (CEST, UTC+2), and every window in the code is in local
  time.
* The modules in `src/` come from the project's working repository. Some
  docstrings mention notebooks that are not part of this repository.
  `campaign_effects.old_effects` and `sign_flip` need the year-long operation dataset
  and is not used by the paper.
* An interactive explorer of the 15 min and 1 h twins, DTWIG, is available at
  <https://dtwig.d.openscienco.org/>.

## Citation

If you use this code, please cite the paper (the reference will be added when
it is published).

## License

Code: MIT (see `LICENSE`). The dataset will carry its own license on Zenodo.
