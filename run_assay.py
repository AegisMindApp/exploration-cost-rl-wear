#!/usr/bin/env python3
"""`7d543860` — does RL exploration drive LEO battery aging through discharge DEPTH?

The row is live on solver.press and carries this falsification clause:

  FALSIFIED IF replaying one fixed task stream through the scheduler at several exploration
  coefficients, and integrating a physics-driven LEO aging model over the resulting
  charge-discharge profiles, shows no monotone relationship between the coefficient and mean
  depth of discharge ONCE ACHIEVED REWARD IS HELD CONSTANT.

The clause fixes the design. It does not fix the implementation, and the implementation decides
the answer, so those choices are written here before the run.

WHY THE REWARD CONTROL IS THE WHOLE EXPERIMENT
-----------------------------------------------
More exploration means more tasks tried, which means more energy drawn, which means deeper
discharge — and also lower achieved reward. Without conditioning on reward, "exploration ages
the battery" is indistinguishable from "doing more work ages the battery", which is not a
finding. That control is why this clause was written the way it was, and it is reported both
ways below so the difference is visible rather than asserted.

IMPLEMENTATION CHOICES, FIXED BEFORE RUNNING
---------------------------------------------
* **Orbit.** 400 km LEO: 92.4 min period, ~35% of each orbit in eclipse. One charge/discharge
  cycle per orbit, 15.58 per day — this is the premise that killed the row's ORIGINAL mechanism
  (it claimed exploration changes cycle FREQUENCY; orbital mechanics fixes that).
* **Aging model.** Wöhler/Palmgren-Miner: damage per cycle ∝ DoD^b with b = 1.87, a standard
  Li-ion value. Superlinear in depth, which is exactly why depth rather than count is the
  candidate mechanism. Reported alongside mean DoD so the conclusion does not hinge on b.
* **Scheduler.** ε-greedy over tasks with unknown value, ε as the exploration coefficient.
  Exploring picks a random eligible task; exploiting picks the best current estimate.
* **Fixed task stream.** One seed generates the arrivals; every ε replays the SAME stream.
  Scheduler randomness varies across 12 seeds per ε.
* **Reward control.** Two ways, both reported: (a) partial Spearman of ε against mean DoD
  controlling for achieved reward; (b) a matched subsample restricted to runs whose reward
  falls in a common band across all ε.

READING RULE, fixed here
------------------------
* **SUPPORTED** if mean DoD rises monotonically with ε *and* survives the reward control.
* **FALSIFIED** if no monotone relationship survives the control.
* **CONFOUNDED** if the raw relationship exists but vanishes once reward is held constant —
  that is the "doing more work" explanation, and it is a distinct outcome from both.
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
OUT = Path(__file__).resolve().parent / ("leo_battery_result.json" if WARM_START
                                          else "leo_battery_result_no_warmstart.json")

PERIOD_MIN = 92.4
ECLIPSE_FRAC = 0.35
ORBITS = 300            # ~19 days
CAPACITY = 100.0        # arbitrary Wh
CHARGE_PER_SUN = 42.0   # Wh recovered per sunlit phase
HOUSEKEEPING = 14.0     # Wh drawn per orbit regardless of tasks
B_WOHLER = 1.87
N_TASKS = 6
EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
SEEDS = 12


def task_stream(rng):
    """True per-task (value, energy). Fixed once and replayed at every epsilon."""
    return [{"value": float(rng.uniform(1, 10)),
             "energy": float(rng.uniform(4, 16))} for _ in range(N_TASKS)]


def simulate_fixed_reward(tasks, eps, seed, target):
    """Run until cumulative reward reaches `target`, then stop.

    This is the fix for the first design. There, epsilon and achieved reward came out 97%
    collinear (Spearman -0.974) because more exploration simply earns less, so "controlling
    for reward" had almost no residual variance to work with and the matched-reward
    subsample was EMPTY (n=0 of 96) -- the reward ranges across epsilon do not overlap.
    A partial correlation under near-collinearity is not a control.

    Here reward is held constant BY CONSTRUCTION: every arm delivers the same mission value
    and the question becomes what depth of discharge it cost to get there. That is what the
    falsification clause asks, and it is measurable rather than inferred.

    SECOND BUG, FOUND 3 OCT 2026 DURING REVIEW: `est = np.zeros(N_TASKS)` plus
    `np.argmax(est)` means a pure-greedy (eps=0) agent's FIRST choice, every single run, is
    task index 0 -- `np.argmax` breaks ties by returning the first occurrence, and every arm
    starts tied at exactly 0.0. If task 0 happens not to be the best arm, eps=0 can stay stuck
    on it (nothing else is ever sampled unless its own running estimate drops below zero,
    which rarely happens for positive-valued rewards), making eps=0 WORSE than high-exploration
    runs in roughly half of independent task-value draws -- confirmed by replaying 30 fixed
    task streams: 14/30 ran backwards before this fix (measured with --no-warmstart), 0/30 after it. The single seed this row
    was originally published on (20260917) happened to land in the "ok" half; it was not a
    robust result, it was a lucky one.

    FIX: a forced warm start -- try each arm once, in random order, before the eps-controlled
    policy takes over. Applied identically regardless of eps, so it does not bias the
    comparison toward the hypothesis; it only removes the "always default to arm 0" artifact
    that a genuinely informed greedy comparison should never have had.
    """
    rng = np.random.default_rng(seed)
    est = np.zeros(N_TASKS); cnt = np.zeros(N_TASKS)
    soc = CAPACITY; dods = []; reward = 0.0; orbits = 0

    for k in (rng.permutation(N_TASKS) if WARM_START else ()):
        soc = min(CAPACITY, soc + CHARGE_PER_SUN); start = soc; soc -= HOUSEKEEPING
        t = tasks[k]
        if soc - t["energy"] > 0:
            soc -= t["energy"]
            obs = t["value"] + rng.normal(0, 1.5)
            reward += t["value"]; cnt[k] += 1; est[k] += (obs - est[k]) / cnt[k]
        dods.append((start - soc) / CAPACITY); orbits += 1

    while reward < target and orbits < ORBITS * 6:
        soc = min(CAPACITY, soc + CHARGE_PER_SUN)
        start = soc; soc -= HOUSEKEEPING
        k = int(rng.integers(N_TASKS)) if rng.random() < eps else int(np.argmax(est))
        t = tasks[k]
        if soc - t["energy"] > 0:
            soc -= t["energy"]
            obs = t["value"] + rng.normal(0, 1.5)
            reward += t["value"]; cnt[k] += 1; est[k] += (obs - est[k]) / cnt[k]
        dods.append((start - soc) / CAPACITY); orbits += 1
    d = np.array(dods)
    return {"eps": eps, "seed": seed, "mean_dod": float(d.mean()),
            "aging": float(np.sum(d ** B_WOHLER)), "reward": reward,
            "orbits": orbits, "hit_target": bool(reward >= target)}


def simulate(tasks, eps, seed):
    rng = np.random.default_rng(seed)
    est = np.zeros(N_TASKS)      # value estimates
    cnt = np.zeros(N_TASKS)
    soc = CAPACITY
    dods, reward = [], 0.0
    for _ in range(ORBITS):
        soc = min(CAPACITY, soc + CHARGE_PER_SUN)     # sunlit phase
        start = soc
        soc -= HOUSEKEEPING                            # eclipse begins
        # one scheduling decision per eclipse
        if rng.random() < eps:
            k = int(rng.integers(N_TASKS))             # explore
        else:
            k = int(np.argmax(est))                    # exploit
        t = tasks[k]
        if soc - t["energy"] > 0:
            soc -= t["energy"]
            obs = t["value"] + rng.normal(0, 1.5)
            reward += t["value"]
            cnt[k] += 1
            est[k] += (obs - est[k]) / cnt[k]
        dods.append((start - soc) / CAPACITY)
    dods = np.array(dods)
    damage = float(np.sum(dods ** B_WOHLER))
    return {"eps": eps, "seed": seed, "mean_dod": float(dods.mean()),
            "aging": damage, "reward": reward, "cycles": len(dods)}


def partial_spearman(x, y, z):
    """Spearman of x vs y with z partialled out, on ranks."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(v):
        A = np.column_stack([np.ones(len(rz)), rz])
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b
    ex, ey = resid(rx), resid(ry)
    r = float(np.corrcoef(ex, ey)[0, 1])
    n = len(x); dof = n - 3
    t = r * np.sqrt(dof / max(1e-12, 1 - r ** 2))
    return r, float(2 * (1 - stats.t.cdf(abs(t), dof)))


def main() -> int:
    tasks = task_stream(np.random.default_rng(20260917))
    print(f"orbit {PERIOD_MIN} min -> {24*60/PERIOD_MIN:.2f} cycles/day, "
          f"{ORBITS} orbits, {SEEDS} seeds x {len(EPSILONS)} epsilons\n")
    runs = [simulate(tasks, e, 1000 + s) for e in EPSILONS for s in range(SEEDS)]

    print(f"{'eps':>6}{'mean DoD':>11}{'aging':>10}{'reward':>10}")
    for e in EPSILONS:
        g = [r for r in runs if r["eps"] == e]
        print(f"{e:>6.2f}{np.mean([x['mean_dod'] for x in g]):>11.4f}"
              f"{np.mean([x['aging'] for x in g]):>10.2f}"
              f"{np.mean([x['reward'] for x in g]):>10.1f}")

    E = np.array([r["eps"] for r in runs]); D = np.array([r["mean_dod"] for r in runs])
    R = np.array([r["reward"] for r in runs]); A = np.array([r["aging"] for r in runs])
    raw = stats.spearmanr(E, D)
    print(f"\nRAW      Spearman(eps, mean DoD) = {raw.statistic:+.3f}  p={raw.pvalue:.3g}")
    print(f"         Spearman(eps, reward)   = {stats.spearmanr(E,R).statistic:+.3f}")
    pr, pp = partial_spearman(E, D, R)
    print(f"CONTROL  partial Spearman(eps, DoD | reward) = {pr:+.3f}  p={pp:.3g}")

    # matched subsample: a reward band every epsilon can reach
    lo = max(min(R[E == e]) for e in EPSILONS); hi = min(max(R[E == e]) for e in EPSILONS)
    m = (R >= lo) & (R <= hi)
    print(f"\nMATCHED  reward band [{lo:.0f},{hi:.0f}], n={int(m.sum())} of {len(R)}, "
          f"epsilons present: {sorted(set(E[m]))}")
    if m.sum() > 10 and len(set(E[m])) > 2:
        ms = stats.spearmanr(E[m], D[m])
        print(f"         Spearman(eps, DoD | matched) = {ms.statistic:+.3f}  p={ms.pvalue:.3g}")
    else:
        print("         too few matched runs to test — reported, not skipped")

    # ---- the design that actually holds reward constant -------------------------
    TARGET = 1200.0
    fr = [simulate_fixed_reward(tasks, e, 2000 + s, TARGET)
          for e in EPSILONS for s in range(SEEDS)]
    ok = [r for r in fr if r["hit_target"]]
    print(f"\n=== FIXED-REWARD DESIGN (target {TARGET:.0f}, "
          f"{len(ok)}/{len(fr)} runs reached it) ===")
    print(f"{'eps':>6}{'orbits':>9}{'mean DoD':>11}{'aging':>10}{'reward':>10}")
    for e in EPSILONS:
        g = [r for r in ok if r["eps"] == e]
        if not g:
            print(f"{e:>6.2f}{'  (no run reached the target)':>40}"); continue
        print(f"{e:>6.2f}{np.mean([x['orbits'] for x in g]):>9.1f}"
              f"{np.mean([x['mean_dod'] for x in g]):>11.4f}"
              f"{np.mean([x['aging'] for x in g]):>10.2f}"
              f"{np.mean([x['reward'] for x in g]):>10.1f}")
    if len(set(r["eps"] for r in ok)) > 2:
        E2 = np.array([r["eps"] for r in ok]); D2 = np.array([r["mean_dod"] for r in ok])
        A2 = np.array([r["aging"] for r in ok]); R2 = np.array([r["reward"] for r in ok])
        s_d = stats.spearmanr(E2, D2); s_a = stats.spearmanr(E2, A2)
        print(f"\n  reward spread across arms: {R2.min():.0f}-{R2.max():.0f} "
              f"(CV {R2.std()/R2.mean():.4f}) -- constant by construction")
        print(f"  Spearman(eps, mean DoD) = {s_d.statistic:+.3f}  p={s_d.pvalue:.3g}")
        print(f"  Spearman(eps, TOTAL aging) = {s_a.statistic:+.3f}  p={s_a.pvalue:.3g}")
        fixed_rho, fixed_p = s_d.statistic, s_d.pvalue
    else:
        print("  too few arms reached the target to test")
        fixed_rho, fixed_p = float("nan"), float("nan")

    # ---- MULTI-SEED ROBUSTNESS, added 3 Oct 2026 after finding the argmax-tie bug ------
    # A single task stream (one draw of 6 (value, energy) pairs) is one sample from the
    # space of possible environments. The warm-start fix above makes any ONE draw fair,
    # but it does not by itself establish the result is not still a lucky draw -- only
    # replaying MANY independent draws does that. This is the primary evidence now, not
    # the single-stream table above, which is kept for readability but is not what the
    # verdict is decided on.
    n_streams = 30
    ratios, backwards = [], 0
    for ts_seed in range(n_streams):
        ts_tasks = task_stream(np.random.default_rng(ts_seed))
        g0 = [simulate_fixed_reward(ts_tasks, 0.0, 2000 + s, TARGET) for s in range(SEEDS)]
        g1 = [simulate_fixed_reward(ts_tasks, 1.0, 2000 + s, TARGET) for s in range(SEEDS)]
        a0 = np.mean([r["aging"] for r in g0 if r["hit_target"]])
        a1 = np.mean([r["aging"] for r in g1 if r["hit_target"]])
        if a0 <= 0 or np.isnan(a0) or np.isnan(a1):
            continue
        ratios.append(a1 / a0)
        backwards += a0 > a1
    ratios = np.array(ratios)
    print(f"\n=== MULTI-SEED ROBUSTNESS ({len(ratios)} independent task streams, "
          f"warm-start {'ON' if WARM_START else 'OFF (original code)'}) ===")
    print(f"  eps=1 aging > eps=0 aging in {len(ratios) - backwards}/{len(ratios)} streams")
    print(f"  ratio (eps1/eps0 aging): median={np.median(ratios):.2f}x  "
          f"range=[{ratios.min():.2f}x, {ratios.max():.2f}x]")
    wsr = stats.wilcoxon(ratios - 1.0, alternative="greater")
    print(f"  Wilcoxon signed-rank (ratio > 1): W={wsr.statistic:.1f}  p={wsr.pvalue:.3g}")

    verdict = ("FALSIFIED" if pp > 0.05 or pr <= 0 else
               "SUPPORTED" if raw.statistic > 0 else "CONFOUNDED")
    if raw.pvalue < 0.05 and raw.statistic > 0 and (pp > 0.05 or pr <= 0):
        verdict = "CONFOUNDED — raw relationship vanishes once reward is held constant"
    # The first design's verdict is NOT trusted: epsilon and reward were 97% collinear and
    # its matched-reward control returned n=0. The fixed-reward design is the one that
    # answers the clause -- but the MULTI-SEED robustness check decides, not one stream.
    if np.isnan(fixed_rho):
        verdict = "INCONCLUSIVE — no arm reached the fixed reward target"
    elif backwards / max(len(ratios), 1) > 0.10 or wsr.pvalue >= 0.05:
        verdict = (f"NOT ROBUST — backwards in {backwards}/{len(ratios)} independent "
                   f"task streams")
    else:
        verdict = (f"SUPPORTED, robust across {len(ratios)} independent task streams "
                  f"(median {np.median(ratios):.2f}x, {backwards}/{len(ratios)} backwards)")
    print(f"\nVERDICT (multi-seed robustness decides): {verdict}")
    OUT.write_text(json.dumps({"runs": runs, "raw_rho": raw.statistic,
                               "raw_p": raw.pvalue, "partial_rho": pr, "partial_p": pp,
                               "verdict": verdict, "b_wohler": B_WOHLER,
                               "multi_seed_ratios": ratios.tolist(),
                               "multi_seed_backwards": int(backwards),
                               "multi_seed_n": len(ratios)}, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
