# Fresh-stream re-test of the two battery models (added October 2026)

A pre-registered re-run of `run_assay.py` (Model 1) and `run_assay_starlink_scale.py` (Model 2) on 30 NEW task streams (generator seeds 1000-1029) with scheduler seeds that are independent across streams (50000 + 100 x stream + s, s = 0..11; the original runs reuse the same 12 scheduler seeds in every stream). The warm start is on in every arm. Model 3 (actuator wear) is not part of it.

The protocol and decision rule were frozen before the run and are deposited on Zenodo (**doi:10.5281/zenodo.23461605**, which holds `PREREG.md` as frozen, `decide.py`, `harness.py` and the kernel file). `harness.py` and `decide.py` here are the same files.

```bash
pip install -r ../requirements.txt
python harness.py --out s3_result.json     # about 10 seconds on a CPU; deterministic
```

`s3_result_kaggle_run.json` is the stored output of the run reported in Section 5.5 of the preprint (version 2); a fresh local run reproduces its fresh-stream ratios exactly (largest difference 0.0).

| | Model 1 (400 km, Wohler) | Model 2 (550 km, log cycle life) |
|---|---|---|
| Median aging ratio, eps=1 / eps=0 | 1.52x | 1.58x |
| Streams where eps=1 is cheaper (of 30) | 1 (smallest ratio 0.97x) | 0 (smallest 1.13x) |
| Wilcoxon, ratio > 1 | p = 1.9e-9 | p = 9.3e-10 |

Controls run in the same session (all in the JSON): the original 30 ratios are reproduced exactly; the same fresh streams without the warm start are judged not robust by the same rule (21 of 30 streams backwards in both models); an eps=0 against eps=0 null gives median ratios of 1.001x. Same author and same simulators, so this checks stability and implementation, not independence.
