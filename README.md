# Exploration cost in onboard RL schedulers: simulation code

Code and results for the preprint *Exploration Cost: The Exploration Coefficient in Onboard
Reinforcement-Learning Schedulers Is an Independent Driver of Battery Aging* (J. Goodman, 2026; engrXiv, doi:[10.31224/8438](https://doi.org/10.31224/8438)).

Question: at **fixed delivered mission value**, does an epsilon-greedy scheduler's exploration
rate still cost physical wear? Each arm runs until it reaches the same cumulative reward, then
total wear is compared between epsilon = 1 (full exploration) and epsilon = 0 (greedy), across
30 independent task-value draws.

| Script | Model | Wear law |
|---|---|---|
| `run_assay.py` | 400 km orbit, toy 100 Wh battery | Wohler power law, DoD^1.87 |
| `run_assay_starlink_scale.py` | 550 km, 1200 Wh, parameters from arXiv:2603.04372 | log-linear cycle life, sigma = 0.8 |
| `run_assay_actuator_wear_generalization.py` | Open Ant actuator (arXiv:2607.18488) | Archard, torque x displacement |

```bash
pip install -r requirements.txt
python run_assay.py                      # writes leo_battery_result.json
python run_assay_starlink_scale.py       # writes leo_battery_starlink_scale_result.json
python run_assay_actuator_wear_generalization.py   # writes actuator_wear_result.json
python run_assay.py --no-warmstart       # sensitivity run: writes *_no_warmstart.json
```

Runs are seeded and deterministic on a given software stack. With numpy 2.4.6, scipy 1.17.1 and Python 3.11 a fresh run of `run_assay.py` reproduces its committed file byte for byte (checked for `run_assay.py` only). On a different stack (a clean-room re-run on Kaggle's current Python and library versions) the three top-level scripts ran but wrote files that were not byte-identical to the committed ones; the headline ratios from `fresh_stream_retest` matched exactly there. Compare numerical values rather than file bytes if your library versions differ.

## Headline results (30 independent draws per model)

| Model | Median wear ratio (eps=1 / eps=0) | Draws reversed | Wilcoxon p |
|---|---|---|---|
| 1, Wohler | 1.47x | 0/30 | 9.3e-10 |
| 2, log-linear | 1.58x | 0/30 | 9.3e-10 |
| 3, Archard | 1.71x (range 0.83-3.30x) | 2/30 | 9.3e-9 |


## Fresh-stream re-test (October 2026)

A pre-registered re-run of the two battery models on 30 new task streams with independent scheduler seeds agrees with the table above (medians 1.52x and 1.58x; 1 and 0 of 30 streams reversed). Code, stored output and the Zenodo pre-registration are in [`fresh_stream_retest/`](fresh_stream_retest/).

## Initialisation sensitivity: read this before reusing the comparison

A greedy agent whose value estimates all start at zero picks task 0 every run (`argmax` over
ties), so a naive epsilon = 0 baseline can stay stuck on a poor arm. All results above use a warm
start (every task tried once, in random order, for every epsilon). `--no-warmstart` reproduces
the naive initialisation. Over the same 30 draws it reverses the effect in 14/30 (Model 1), 13/30
(Model 2) and 19/30 (Model 3) draws, with median ratios 1.04x, 1.04x and 0.73x, none significant.
The effect is conditional on a fairly initialised greedy baseline.

## Changes and superseded results

An earlier single-draw version of this study, also shown on solver.press, reported larger ratios
(1.76-2.33x). It used the naive initialisation on one task draw and is superseded by the
30-draw results above. `notes/` keeps the original write-ups with their corrections. Counts in
those notes were re-measured on 4 Oct 2026 with `--no-warmstart`, replacing earlier figures of
15/30 and 21/30.

## Limitations

Simulation only: synthetic task streams, no flight or hardware data. Model 2's charge and
task-energy scales are inherited from Model 1, and its battery-specific aging constant is
arbitrary (every comparison is a ratio, so it cancels). sigma = 0.8 was extracted from the source
paper by an AI-assisted read. See the preprint for the full list.

## Licence

MIT.
