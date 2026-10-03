# `7d543860` generalization: actuator wear — real, but weaker than the battery result

Run 3 Oct 2026. `analysis/leo_battery/run_assay_actuator_wear_generalization.py`. A third
transfer test of the published row's mechanism, moving entirely off battery cycling onto
mechanical actuator wear (Archard's law, a real robot's documented torque/displacement
ranges — see the script's own docstring for the arXiv:2607.18488 sourcing).

## The bug this test exposed before anything else

The first run came back non-monotonic and with eps=0 the WORST arm, not the best — immediately
suspicious. Tracing it found the same bug later confirmed in Models 1 and 2:
`np.argmax` on all-zero-tied value estimates always starts a pure-greedy agent on task index 0,
regardless of which task is actually best. Replaying 30 independent task streams with the
**original, unfixed** code: backwards (eps=0 worse than eps=1) in 19 of 30 (re-measured 4 Oct 2026 with `--no-warmstart`; an earlier count here said 21) — the majority
outcome, not an edge case. This is what led to re-checking Models 1 and 2, which turned out to
share the same bug (14/30 and 13/30, measured with `--no-warmstart` — see `RESULT.md` and
`STARLINK_SCALE_RESULT.md`).

**Fix:** the same forced warm start used in both battery models — try each task once, in random
order, before the eps-controlled policy takes over. Applied identically at every eps.

## Result, with the fix

30 independent task streams (torque/displacement redrawn from the Open Ant's documented ranges
each time), 12 scheduler seeds per stream per eps, same reward-matched design as the battery
models.

**Backwards (eps=0 wear > eps=1 wear) in 2 of 30 streams** — not zero, unlike both battery
models (0/30 each). Median ratio (total wear, eps=1 vs eps=0) **1.71x** where the effect holds,
but the full range is **[0.83x, 3.30x]** — the low end is a genuine reversal, not noise. Wilcoxon
signed-rank across all 30 (ratio > 1): p = 9.3e-9, still significant overall, but visibly less
uniform than either battery model.

## Reading this honestly

The mechanism generalizes beyond battery cycling to mechanical wear — the direction is right in
28 of 30 independent environments, and the median effect size (1.71x) is comparable to the
battery models' (1.47x, 1.58x). But it is **not as robust**: roughly 1 in 15 random task
configurations reverses it, against 0 in 30 for both battery models. Plausible reason: the
Archard wear law is LINEAR in torque x displacement with no saturating/depth regime, so the
balance between "more steps" and "wear per step" is more sensitive to exactly which tasks get
explored than the battery models' aging laws were (both of which are superlinear/exponential in
depth, giving the "more cycles" channel more room to dominate consistently).

## Verdict

**SUPPORTED, with a real and disclosed exception rate.** This is weaker evidence than the
battery result and should not be quoted with the same confidence. If this generalization is
ever written up alongside the battery finding, the exception rate (2/30, with reversals as
large as 0.83x) belongs in the same sentence as the 1.71x median, not in a footnote.

## Reproducing

```bash
python analysis/leo_battery/run_assay_actuator_wear_generalization.py
```
