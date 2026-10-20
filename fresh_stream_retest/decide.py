"""Decision code for S3 (self-replication of discovery 7d543860, LEO battery RL exploration cost). DRAFT.
The ORIGINAL rule, copied unchanged from analysis/leo_battery/run_assay.py and run_assay_starlink_scale.py (commit 151514b0), applied per model:
  for each of N independent task streams, ratio_i = mean aging(eps=1) / mean aging(eps=0) at a fixed delivered reward (12 scheduler seeds per arm, runs that hit the target only);
  backwards_i = (mean aging eps=0 > mean aging eps=1);  Wilcoxon signed-rank on (ratio - 1), alternative 'greater'.
  Original verdict: NOT ROBUST iff backwards/n > 0.10 or Wilcoxon p >= 0.05; otherwise SUPPORTED.
Here: part verdict Supported = original SUPPORTED, Refuted = original NOT ROBUST.  Fewer than MIN_STREAMS usable streams -> Inconclusive (plumbing guard; new, cannot change a 30/30 outcome).
Claim verdict (Models 1 and 2): Supported iff both parts Supported; Refuted if either is Refuted; else Inconclusive.
python decide.py --plant : planted-effect table (stream-level lognormal ratios; executed, see plant_table.json)."""
import sys, json
import numpy as np
from scipy import stats
MAX_BACK_FRAC, ALPHA, MIN_STREAMS = 0.10, 0.05, 25
def part(ratios):
    r = np.asarray([x for x in ratios if np.isfinite(x)], float); n = len(r)
    if n < MIN_STREAMS: return {"n": n, "verdict": "Inconclusive", "why": f"only {n} usable streams"}
    back = int((r < 1.0).sum())      # a0 > a1  <=>  ratio a1/a0 < 1 (ties are not backwards, as in the original)
    d = r - 1.0
    p = 1.0 if np.all(d == 0) else float(stats.wilcoxon(d, alternative="greater").pvalue)
    robust = not (back / n > MAX_BACK_FRAC or p >= ALPHA)
    return {"n": n, "backwards": back, "median_ratio": float(np.median(r)), "min": float(r.min()), "max": float(r.max()), "wilcoxon_p": p,
            "original_label": "SUPPORTED" if robust else "NOT ROBUST", "verdict": "Supported" if robust else "Refuted"}
def decide(m1, m2):
    p1, p2 = part(m1), part(m2); vs = (p1["verdict"], p2["verdict"])
    v = "Supported" if vs == ("Supported", "Supported") else "Refuted" if "Refuted" in vs else "Inconclusive"
    agree = {"Supported": "agrees with the original verdict (SUPPORTED, robust across 30 task streams, 0/30 backwards both models)",
             "Refuted": "disagrees with the original verdict (SUPPORTED, robust across 30 task streams, 0/30 backwards both models)",
             "Inconclusive": "neither agrees nor disagrees with the original verdict (SUPPORTED, robust across 30 task streams, 0/30 backwards both models)"}[v]
    return {"verdict": v + " under protocol X", "agreement": agree, "model1_400km_wohler": p1, "model2_550km_log_cycle_life": p2}
def plant(reps=4000, seed=0, n=30):
    rng = np.random.default_rng(seed); rows = []
    # Stream-level ratio ~ lognormal(log m, s); both models drawn with the same parameters.  Row label carries the TRUE backwards probability q = P(ratio < 1).
    def row(tag, med, s):
        c = {}; q = float(stats.norm.cdf(-np.log(med) / s))
        for _ in range(reps):
            a = np.exp(rng.normal(np.log(med), s, n)); b = np.exp(rng.normal(np.log(med), s, n))
            v = decide(a, b)["verdict"]; c[v] = c.get(v, 0) + 1
        rows.append({"scenario": tag, "true_backwards_prob": round(q, 4), **{k: round(v / reps, 3) for k, v in sorted(c.items())}}); print(json.dumps(rows[-1]))
    row("as original: median 1.5x, log-sd 0.17 (original streams: 0.181 and 0.160)", 1.5, 0.17)
    row("original-like median 1.5x but heavier tail (log-sd 0.25)", 1.5, 0.25)
    row("actuator-like: median 1.71x, log-sd 0.38 (original actuator: 2/30 backwards)", 1.71, 0.38)
    row("weaker: median 1.35x, log-sd 0.17", 1.35, 0.17)
    row("weaker: median 1.25x, log-sd 0.17", 1.25, 0.17)
    row("weaker: median 1.15x, log-sd 0.17", 1.15, 0.17)
    row("null: median 1.0x, log-sd 0.17", 1.0, 0.17)
    row("pre-fix bug-like: about 45% backwards (median 1.0x-ish, log-sd 0.35 with slight positive shift)", 1.05, 0.35)
    return rows
if __name__ == "__main__" and "--plant" in sys.argv: json.dump(plant(), open("plant_table.json", "w"), indent=1)
