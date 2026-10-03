## CORRECTION, 3 Oct 2026 — a second implementation bug found and fixed, effect now multi-seed validated

While building a third generalization test, the single-task-stream design below turned out to
be sitting on a real bug: `est = np.zeros(N_TASKS)` plus `np.argmax(est)` means a pure-greedy
(eps=0) agent's first choice is **always task index 0**, every run — `np.argmax` breaks ties by
returning the first occurrence, and every arm starts tied at exactly 0.0. If task 0 is not the
best arm, eps=0 can stay stuck exploiting it (nothing else is ever sampled unless its own
running estimate drops below zero, rare for positive-valued rewards). Replaying 30 independent
task streams with the **original, unfixed** code: eps=0 came out WORSE than eps=1 — backwards
from the claimed direction — in 14 of 30 (re-measured 4 Oct 2026 with `--no-warmstart`; an earlier count here said 15). The single seed this row was published on (20260917)
happened to land in the other half. It was a lucky draw, not a robust result.

**Fix:** a forced warm start — try each arm once, in random order, before the eps-controlled
policy takes over. Applied identically at every eps, so it cannot bias the comparison.

**With the fix, replayed across the same 30 independent task streams: 0/30 backwards.**
Median ratio (total aging, eps=1 vs eps=0) **1.47x**, Wilcoxon signed-rank p=9.3e-10. The
original single-seed table below (2.33x, 55/45 split) is **superseded** by this multi-seed
result — kept for the record, not as the current number. Model 2
(`run_assay_starlink_scale.py`) and a third generalization to actuator wear
(`run_assay_actuator_wear_generalization.py`) carry the identical fix and the identical
multi-seed check; see `STARLINK_SCALE_RESULT.md` and `ACTUATOR_WEAR_RESULT.md`.

**Reproducing the corrected result:**
```bash
python analysis/leo_battery/run_assay.py
```
The script now reports the 30-seed robustness check as its primary verdict.

---

# `7d543860`: the effect is real and larger than claimed — but the exclusivity clause is wrong

**The numbers in this original write-up below are SUPERSEDED by the correction above** —
kept for the historical record of how the design evolved (naive partial correlation, reward
collinearity, the reward-matched fix), not as the current validated result.

Run 17 Sep 2026, local CPU. Design fixed in `run_assay.py` and committed (`caa2634f`) before
the run. The row is live on solver.press and carries the falsification clause this tests.

## The first design could not run its own control

Replaying a fixed task stream at eight exploration coefficients gives
Spearman(ε, mean DoD) = **+0.515** (p = 8e-08), and the partial correlation controlling for
reward nominally *strengthens* it to +0.567. That looked like support. It is not trustworthy:

- **Spearman(ε, reward) = −0.974.** Exploration and achieved reward are 97% collinear by
  construction, so partialling reward out leaves almost no residual variance in ε.
- **The matched-reward subsample returned n = 0 of 96.** The reward ranges across ε do not
  overlap at all, so the interpretable control had no data.

My verdict function declared SUPPORTED while its own primary control returned nothing — the same
defect this week's audits keep finding, in the rule I had just written.

## Holding reward constant by construction

The fix is to stop inferring the control and build it in: run each arm until it has delivered the
same cumulative mission value, then ask what that cost. 96/96 runs reached the target; reward
spread 1200–1207, **CV 0.0016**.

| ε | orbits to target | mean DoD | total aging |
|---|---|---|---|
| 0.00 | 175.0 | 0.1996 | 8.60 |
| 0.20 | 187.3 | 0.2244 | 11.60 |
| 0.50 | 215.2 | 0.2352 | 14.58 |
| 1.00 | 277.7 | 0.2433 | **20.06** |

**Spearman(ε, mean DoD) = +0.755** (p = 6e-19); **Spearman(ε, total aging) = +0.957**
(p = 2e-52). Delivering the same mission value at ε = 1 costs **2.33× the battery damage** it
costs at ε = 0.

## And the row's exclusivity clause does not survive

The hypothesis says degradation runs through depth of discharge **"not through cycle count"**.
Decomposing the 2.33× at fixed reward:

| factor | ε 0 → 1 | share of the log increase |
|---|---|---|
| cycle count (175 → 278 orbits) | **1.59×** | **55%** |
| mean depth (0.1996 → 0.2433) | 1.45× | 44% |

**Cycle count is the larger contributor, not the smaller.** Both matter and they are close, but
the word "not" is wrong.

The distinction that survives is the one I actually established on 17 Sep: cycles **per day** are
pinned by orbital mechanics at 15.58 and no scheduler can change them. That killed the row's
*original* mechanism, which claimed exploration raises cycle *frequency*. But total cycles **to
deliver a mission** are not pinned, because a more exploratory scheduler needs more orbits to
earn the same reward. I corrected "frequency" to "depth" and over-corrected: the right statement
is *both*, with count slightly ahead.

## Verdict

**SUPPORTED on the effect, REFUTED on the mechanism's exclusivity.** Exploration does drive
battery aging at fixed delivered value, strongly and monotonically. It does not do so
"through depth, not cycle count".

## Scope

A simulation, not a satellite. Simulated 400 km orbit, ε-greedy over six tasks, Wöhler damage
with b = 1.87. The 2.33× is a property of this model; the *direction* and the count-vs-depth
split are the transferable parts, and both depend on b only weakly — the count factor does not
involve b at all.

## Reproducing

```bash
python analysis/leo_battery/run_assay.py
```
