# Code for "Sampling interval and sign inversion of actuator effects in one-step
# models learned from randomised greenhouse experiments" (Paper B).
#
#   make data      write the SYNTHETIC placeholder data to data/ (~15 s)
#   make results   run every analysis, CSV to results/ (~1 h)
#   make figs      draw the figures from results/ into figs/ (R)
#   make all       the three, in order
#
# With the real data (Zenodo, forthcoming) skip `make data` and point
# OSIRIS_DATA_DIR at the raw exports; see README.md.
PY ?= python3 -W ignore

all: results figs

data:
	$(PY) tools/make_synthetic_data.py

results:
	mkdir -p results
	$(PY) scripts/two_campaigns.py
	$(PY) scripts/two_campaigns.py 40
	$(PY) scripts/synthetic.py
	$(PY) scripts/reconcile.py
	$(PY) scripts/by_band.py
	$(PY) scripts/remedy.py
	$(PY) scripts/robust.py
	$(PY) scripts/consequence.py
	$(PY) scripts/whatif.py
	$(PY) scripts/forecast_check.py

figs:
	Rscript scripts/figs.R

clean:
	rm -rf results/*.csv figs/*.pdf

.PHONY: all data results figs clean
