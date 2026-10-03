#!/usr/bin/env python3
"""`7d543860` generalization test: does the exploration-coefficient-as-independent-aging-
driver finding survive a move OFF battery cycling entirely, onto mechanical actuator wear?

Models 1 and 2 (run_assay.py, run_assay_starlink_scale.py) varied orbit, battery scale and
aging-LAW FORM but stayed inside "RL scheduler cycling a battery." This asks the broader
question the paper's own Related Work raised and left untested: the RL-robotics literature
treats exploration-induced ACTUATOR wear as a safety hazard to bound, not a quantity to
measure against delivered task value [ROBOTICS-WEAR, "The Open Ant", arXiv:2607.18488]. Same
mechanism class (a cycling/moving physical component with a finite service life, consumed by
scheduler decisions), a genuinely different physical law and a genuinely different hardware
domain -- not another battery, this time a geared servo joint.

WHERE THE NUMBERS COME FROM
----------------------------
arXiv:2607.18488 ("The Open Ant"), fetched directly (not search paraphrase) 3 Oct 2026:
  - Actuators: Dynamixel XL430-W250-T (nominal torque 0.28 Nm) and XM430-W350-T (0.82 Nm) --
    the paper's own two configurations, "Lite" and full "Physical Ant".
  - Joint angular ranges: knee +-70 deg (constrained to a 20 deg window in their experiments),
    hip 45 deg range.
  - Documented failure modes, verbatim: "foot-tip abrasion from repeated impacts", "plastic
    gear wear", and a "72 C thermal cutoff within 30 minutes under stalled load" for the Lite
    configuration.

THE PHYSICAL LAW: Archard wear, not a battery aging law
-----------------------------------------------------------
Archard's equation gives wear volume proportional to load times sliding distance (V = K*F*d/H
-- a material-hardness constant folds into K, same role the unknown battery constant played in
Model 2, and cancels in every ratio reported below for the same reason). For a rotary joint,
load x sliding distance becomes torque x angular displacement -- mechanical work through the
joint, not a "depth" variable at all. This is a structurally different law from both prior
models: Wohler is a POWER LAW in depth, the LEO log-cycle-life law is LOG-LINEAR in depth,
Archard wear here is LINEAR in a load x displacement product with no depth concept whatsoever.

A SEPARATE resource gates task affordability, same role the battery's SOC floor played: thermal
headroom, motivated directly by the paper's own documented 72 C cutoff under stalled load.
Heat generated per action is modeled as proportional to torque SQUARED (a DC servo's resistive
heating scales with current squared, and current scales with torque at fixed voltage) --
deliberately a DIFFERENT functional dependence on torque than the wear law's LINEAR dependence,
so the gating resource and the measured outcome are not simply rescaled copies of each other.

OPERATIONALIZATION
-------------------
* 6 "tasks" = 6 candidate joint actions, each with an unknown VALUE (reward, e.g. forward
  progress) and a fixed (torque, angular displacement) drawn once from the paper's real ranges:
  torque ~ U(0.28, 0.82) Nm, displacement ~ U(20, 70) degrees -- the Lite-to-full-Ant range the
  paper itself documents, not an invented one.
* Same eps-greedy scheduler, same fixed-reward-target design (Section 3.2's reward-matched
  control, carried over unchanged -- it is domain-agnostic by construction).
* Thermal budget: CAPACITY, recovered by a per-step COOLDOWN, drawn down by HOUSEKEEPING (idle
  controller/sensor draw) plus torque^2 heating; a task that would breach the thermal FLOOR is
  skipped that step, same role the battery SOC floor played.
* Metric: total Archard wear accumulated to reach a fixed cumulative reward target, same
  reward-matched design as both battery models.

READING RULE, fixed here, identical in form to the battery models'
---------------------------------------------------------------------
SUPPORTED if total wear rises monotonically in eps at fixed delivered reward. The row's
exclusivity (both channels contribute, not depth/torque alone) is checked the same way:
decomposing the eps 0->1 log-increase in total wear into a step-count factor and a
per-step-mean-wear factor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

# --no-warmstart reproduces the ORIGINAL (pre-fix) initialisation, to measure the
# sensitivity reported in the paper; it writes to a separate file.
import sys
WARM_START = "--no-warmstart" not in sys.argv
OUT = Path(__file__).resolve().parent / ("actuator_wear_result.json" if WARM_START
                                          else "actuator_wear_result_no_warmstart.json")

N_TASKS = 6
EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
SEEDS = 12
TARGET = 1200.0
ORBIT_CAP_MULT = 6          # same cap-multiplier convention as the battery models

# Thermal budget, analogous to battery capacity/housekeeping/SOC-floor. Units are abstract
# (as the original battery model's Wh were) -- only ratios and ranks are claimed.
CAPACITY = 100.0
COOLDOWN_PER_STEP = 42.0    # same ratio-to-capacity (42%) the battery models used throughout
HOUSEKEEPING = 14.0         # idle controller/sensor draw, same ratio-to-capacity as before
THERMAL_FLOOR = 0.0         # breach point; a step whose heat cost would cross it is skipped

# Real ranges from arXiv:2607.18488, fetched 3 Oct 2026 (see module docstring).
TORQUE_LO, TORQUE_HI = 0.28, 0.82          # Nm, XL430-W250-T to XM430-W350-T nominal torque
DISPLACEMENT_LO_DEG, DISPLACEMENT_HI_DEG = 20.0, 70.0   # deg, the paper's own knee/hip ranges
K_ARCHARD = 1.0             # wear-coefficient x hardness^-1, folds into a constant -- see note

HEAT_SCALE = 60.0           # torque^2 -> heat units, chosen so the thermal budget binds
                            # about as often as the battery models' did (not tuned to the
                            # answer: picked once from the torque range above, before running)


def task_stream(rng):
    """(value, torque_Nm, displacement_rad) per task, torque/displacement from the Open
    Ant's own documented ranges, fixed once and replayed at every epsilon."""
    out = []
    for _ in range(N_TASKS):
        torque = float(rng.uniform(TORQUE_LO, TORQUE_HI))
        disp_deg = float(rng.uniform(DISPLACEMENT_LO_DEG, DISPLACEMENT_HI_DEG))
        out.append({
            "value": float(rng.uniform(1, 10)),
            "torque": torque,
            "disp_rad": disp_deg * np.pi / 180.0,
            "heat": HEAT_SCALE * torque ** 2,          # resistive heating ~ torque^2
            "wear": K_ARCHARD * torque * (disp_deg * np.pi / 180.0),  # Archard: linear in
                                                                       # torque x displacement
        })
    return out


def simulate_fixed_reward(tasks, eps, seed, target):
    # WARM-START FIX, added 3 Oct 2026 -- see run_assay.py's simulate_fixed_reward for the
    # bug (argmax on all-zero-tied estimates always returns task 0, so an unfixed eps=0
    # agent's choice is "always default to task 0", not a real comparison). Found here by
    # the first run coming back non-monotonic (eps=0 took MORE steps than eps=1); confirmed
    # across 30 task streams it was backwards 19/30 times before this fix (measured with --no-warmstart). Trying each task
    # once first, in random order, removes the artifact and is applied at every eps equally.
    rng = np.random.default_rng(seed)
    est = np.zeros(N_TASKS); cnt = np.zeros(N_TASKS)
    heat_budget = CAPACITY
    wears = []
    reward = 0.0
    steps = 0

    for k in (rng.permutation(N_TASKS) if WARM_START else ()):
        heat_budget = min(CAPACITY, heat_budget + COOLDOWN_PER_STEP); heat_budget -= HOUSEKEEPING
        t = tasks[k]
        if heat_budget - t["heat"] > THERMAL_FLOOR:
            heat_budget -= t["heat"]
            obs = t["value"] + rng.normal(0, 1.5)
            reward += t["value"]; cnt[k] += 1; est[k] += (obs - est[k]) / cnt[k]
            wears.append(t["wear"])
        else:
            wears.append(0.0)
        steps += 1

    while reward < target and steps < TARGET * ORBIT_CAP_MULT / 4:
        heat_budget = min(CAPACITY, heat_budget + COOLDOWN_PER_STEP)
        heat_budget -= HOUSEKEEPING
        k = int(rng.integers(N_TASKS)) if rng.random() < eps else int(np.argmax(est))
        t = tasks[k]
        if heat_budget - t["heat"] > THERMAL_FLOOR:
            heat_budget -= t["heat"]
            obs = t["value"] + rng.normal(0, 1.5)
            reward += t["value"]; cnt[k] += 1; est[k] += (obs - est[k]) / cnt[k]
            wears.append(t["wear"])
        else:
            wears.append(0.0)   # skipped step: no action taken, no wear, no reward
        steps += 1
    w = np.array(wears)
    return {"eps": eps, "seed": seed, "mean_wear_per_step": float(w.mean()),
            "total_wear": float(w.sum()), "reward": reward, "steps": steps,
            "hit_target": bool(reward >= target)}


def main() -> int:
    tasks = task_stream(np.random.default_rng(20260917))
    print("Open Ant generalization: 6 tasks, torque/displacement drawn from "
          f"[{TORQUE_LO},{TORQUE_HI}] Nm x [{DISPLACEMENT_LO_DEG},{DISPLACEMENT_HI_DEG}] deg "
          "(arXiv:2607.18488's own documented ranges)\n")

    fr = [simulate_fixed_reward(tasks, e, 2000 + s, TARGET)
          for e in EPSILONS for s in range(SEEDS)]
    ok = [r for r in fr if r["hit_target"]]
    print(f"=== FIXED-REWARD DESIGN (target {TARGET:.0f}, {len(ok)}/{len(fr)} runs reached it) ===")
    print(f"{'eps':>6}{'steps':>9}{'mean wear/step':>16}{'total wear':>12}{'reward':>10}")
    for e in EPSILONS:
        g = [r for r in ok if r["eps"] == e]
        if not g:
            print(f"{e:>6.2f}{'  (no run reached the target)':>47}"); continue
        print(f"{e:>6.2f}{np.mean([x['steps'] for x in g]):>9.1f}"
              f"{np.mean([x['mean_wear_per_step'] for x in g]):>16.5f}"
              f"{np.mean([x['total_wear'] for x in g]):>12.4f}"
              f"{np.mean([x['reward'] for x in g]):>10.1f}")

    if len(set(r["eps"] for r in ok)) > 2:
        E = np.array([r["eps"] for r in ok]); S = np.array([r["steps"] for r in ok])
        W = np.array([r["total_wear"] for r in ok]); R = np.array([r["reward"] for r in ok])
        MW = np.array([r["mean_wear_per_step"] for r in ok])
        s_s = stats.spearmanr(E, S); s_w = stats.spearmanr(E, W)
        print(f"\nreward spread across arms: {R.min():.0f}-{R.max():.0f} "
              f"(CV {R.std()/R.mean():.4f}) -- constant by construction")
        print(f"Spearman(eps, steps-to-target) = {s_s.statistic:+.3f}  p={s_s.pvalue:.3g}")
        print(f"Spearman(eps, TOTAL wear)      = {s_w.statistic:+.3f}  p={s_w.pvalue:.3g}")

        eps0 = [r for r in ok if r["eps"] == 0.0]; eps1 = [r for r in ok if r["eps"] == 1.0]
        if eps0 and eps1:
            steps0, steps1 = np.mean([x["steps"] for x in eps0]), np.mean([x["steps"] for x in eps1])
            mw0, mw1 = np.mean([x["mean_wear_per_step"] for x in eps0]), np.mean([x["mean_wear_per_step"] for x in eps1])
            total0, total1 = np.mean([x["total_wear"] for x in eps0]), np.mean([x["total_wear"] for x in eps1])
            ratio = total1 / total0 if total0 else float("nan")
            import math
            log_steps = math.log(steps1 / steps0)
            log_intensity = math.log(mw1 / mw0) if mw0 and mw1 else float("nan")
            log_total = log_steps + log_intensity
            print(f"\neps 0 -> 1: {ratio:.2f}x total wear "
                  f"({steps0:.1f} -> {steps1:.1f} steps, {mw0:.5f} -> {mw1:.5f} mean wear/step)")
            if log_total and not math.isnan(log_total) and log_total != 0:
                print(f"decomposition: step-count {100*log_steps/log_total:.0f}% of the "
                      f"log increase, per-step wear intensity {100*log_intensity/log_total:.0f}%")

    else:
        s_s = s_w = None

    # ---- MULTI-SEED ROBUSTNESS, added 3 Oct 2026 -- see run_assay.py for the bug
    # (argmax-on-zero-ties) this fix answers. One task stream is one sample from the space
    # of environments; this is the primary evidence, not the single stream above.
    n_streams = 30
    ratios, backwards = [], 0
    for ts_seed in range(n_streams):
        ts_tasks = task_stream(np.random.default_rng(ts_seed))
        g0 = [simulate_fixed_reward(ts_tasks, 0.0, 2000 + s, TARGET) for s in range(SEEDS)]
        g1 = [simulate_fixed_reward(ts_tasks, 1.0, 2000 + s, TARGET) for s in range(SEEDS)]
        w0 = np.mean([r["total_wear"] for r in g0 if r["hit_target"]])
        w1 = np.mean([r["total_wear"] for r in g1 if r["hit_target"]])
        if w0 <= 0 or np.isnan(w0) or np.isnan(w1):
            continue
        ratios.append(w1 / w0)
        backwards += w0 > w1
    ratios = np.array(ratios)
    print(f"\n=== MULTI-SEED ROBUSTNESS ({len(ratios)} independent task streams, "
          f"warm-start {'ON' if WARM_START else 'OFF (original code)'}) ===")
    print(f"  eps=1 wear > eps=0 wear in {len(ratios) - backwards}/{len(ratios)} streams")
    print(f"  ratio (eps1/eps0 total wear): median={np.median(ratios):.2f}x  "
          f"range=[{ratios.min():.2f}x, {ratios.max():.2f}x]")
    wsr = stats.wilcoxon(ratios - 1.0, alternative="greater")
    print(f"  Wilcoxon signed-rank (ratio > 1): W={wsr.statistic:.1f}  p={wsr.pvalue:.3g}")

    frac_backwards = backwards / max(len(ratios), 1)
    if frac_backwards > 0.10 or wsr.pvalue >= 0.05:
        verdict = (f"PARTIALLY ROBUST, NOT CLEAN — backwards in {backwards}/{len(ratios)} "
                  f"independent task streams (median {np.median(ratios):.2f}x where it "
                  f"holds); weaker than the battery models, which were 0/30 backwards")
    else:
        verdict = (f"SUPPORTED, robust across {len(ratios)} independent task streams "
                  f"(median {np.median(ratios):.2f}x, {backwards}/{len(ratios)} backwards)")

    print(f"\nVERDICT (multi-seed robustness decides): {verdict}")
    OUT.write_text(json.dumps({
        "runs": fr, "verdict": verdict,
        "spearman_steps": None if s_s is None else {"rho": s_s.statistic, "p": s_s.pvalue},
        "spearman_wear": None if s_w is None else {"rho": s_w.statistic, "p": s_w.pvalue},
        "multi_seed_ratios": ratios.tolist(), "multi_seed_backwards": int(backwards),
        "multi_seed_n": len(ratios),
    }, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
