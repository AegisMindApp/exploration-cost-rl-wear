## CORRECTION, 3 Oct 2026 — same argmax-tie bug as Model 1, same fix, now multi-seed validated

This model shared Model 1's bug: `np.argmax` on all-zero-tied estimates always starts a
pure-greedy (eps=0) agent on task 0, regardless of which task is actually best. Fixed with the
identical forced warm-start (try each task once before the eps-controlled policy takes over).

**Replayed across 30 independent task streams with the fix: 0/30 backwards.** Median ratio
(total aging, eps=1 vs eps=0) **1.58x**, range [1.15x, 2.02x], Wilcoxon p=9.3e-10. The
single-seed numbers below (1.76x, 82/18 split) are superseded by this multi-seed result.

```bash
python analysis/leo_battery/run_assay_starlink_scale.py
```

---

# `7d543860`: the effect survives transfer to a realistic constellation profile

**Numbers below are SUPERSEDED by the correction above.**

Run 2 Oct 2026, local CPU. `analysis/leo_battery/run_assay_starlink_scale.py`, ahead of any
outreach to a real LEO constellation operator — see the row's EVP for why a pre-outreach
validation step was wanted.

## Why this run, not just the original

`run_assay.py` established SUPPORTED at fixed reward (Spearman(eps, aging) = +0.957,
p = 2e-52, 2.33x damage at eps=1 vs eps=0) on a 400 km orbit, 100 Wh battery, b=1.87 Wohler
damage law — its own RESULT.md calls this "a simulation, not a satellite." Before trusting the
finding enough to pitch it to anyone, the honest next step is checking whether the EFFECT
survives when every physical parameter is replaced with a published one from an independent
source, under a structurally different aging law — not just re-running the same toy model at a
different altitude.

## Where the new numbers come from

arXiv:2603.04372, "Unseen Cost of Space Computing: Quantifying LEO Battery Aging via
Physics-Driven Modeling" (2026) — flagged as adjacent literature on this row's EVP on 2 Oct
2026, now actually used rather than just cited:

- **550 km altitude, 53° inclination** — Starlink Shell 1's actual inclination; period and
  eclipse fraction computed here from the altitude via standard orbital mechanics (95.50 min,
  15.08 orbits/day, 37.2% eclipse at the beta=0 maximum-eclipse bound), not copied from the
  paper's rounded figure.
- **1200 Wh battery, 20% minimum state of charge** — a hard floor enforced in the sim: a task
  that would drive the battery below it is skipped that orbit.
- **75 W housekeeping** (midpoint of the paper's stated 50–100 W band).
- **A different FUNCTIONAL FORM of aging law**: log10(L) + σ·d = ε (log-linear cycle life in
  depth, σ = 0.8), against the original's power-law Wöhler damage (∝ d^1.87). Not a relabeled
  version of the same curve — this is the actual point of the run.

Charge-per-sun and task energy are scaled by the same ratio-to-capacity the original model
used (documented in the script, since the source paper's panel-area/efficiency ranges are wide
enough that picking one point would be no better grounded).

## Result

| eps | orbits | mean DoD | aging | reward |
|---|---|---|---|---|
| 0.00 | 175.0 | 0.1989 | 252.4 | 1203.6 |
| 0.20 | 187.3 | 0.2298 | 286.6 | 1203.4 |
| 0.50 | 215.2 | 0.2434 | 337.7 | 1204.1 |
| 1.00 | 277.7 | 0.2534 | 444.2 | 1201.7 |

96/96 runs reached the fixed reward target (1200, CV 0.0016 across arms — constant by
construction, same design as the original).

**Spearman(ε, mean DoD) = +0.755, p = 6.5e-19. Spearman(ε, total aging) = +0.985, p = 6.2e-74.**
Both *stronger* than the original run's p-values. **1.76x total aging at ε=1 vs ε=0** (down from
2.33x — the magnitude is model-specific, as expected; RESULT.md said so already).

## What changed, and why it matters more than the headline ratio

**The cycle-count/depth split moved from 55/45 to 82/18.** The published log-cycle-life law is
less steeply sensitive to depth over the observed DoD range (0.20–0.25) than the Wöhler power
law was, so under a realistic aging model, "more orbits needed to deliver the same mission
value" dominates the aging cost even more than depth does. The original row's hedge — "both
matter, count slightly ahead" — undersells it on a realistic battery; depth is a minor
contributor here, not a comparable one.

## Verdict

**SUPPORTED, and the core claim is robust to a different orbit, a different battery scale, and
a structurally different aging law.** What is NOT robust, and should not be quoted as a fixed
number: the 2.33x (now 1.76x) ratio and the depth/count split (55/45, now 82/18). The
transferable claim is: exploration independently drives battery aging at fixed delivered
mission value, through both more cycles and deeper cycles, with cycle count usually the larger
term — not a specific multiplier.

## Scope, unchanged from the original

Still two simulations, not a satellite. Nobody outside the operator has Starlink's actual
battery chemistry, task mix, or scheduler. This is the strongest public substitute available —
real orbit geometry at a named operator's altitude/inclination, a real battery capacity and SOC
floor, and an aging law from an independent paper — not a claim about any deployed system.

## Reproducing

```bash
python analysis/leo_battery/run_assay_starlink_scale.py
```
