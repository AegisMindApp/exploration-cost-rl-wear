"""S3 harness: self-replication of 7d543860 (LEO battery RL exploration cost). DRAFT. python harness.py --out OUT.json [--smoke]
Re-implements the two battery simulators of analysis/leo_battery/run_assay.py (Model 1: 400 km, Wohler b=1.87) and run_assay_starlink_scale.py (Model 2: 550 km, 1200 Wh,
20% SOC floor, log-cycle-life sigma=0.8), parameters and code paths unchanged, warm start ON.  Pure numpy/scipy, no pandas (Kaggle default image).
Controls, all executed in every run and all gating:
  C1 reproduction: original streams 0-29 and scheduler seeds 2000-2011 must reproduce the stored multi_seed_ratios of both original result files to 1e-9 (else NO VERDICT: our port differs).
  C2 bug detector (positive control): the PRE-FIX code (warm start OFF) on the FRESH streams must be judged NOT ROBUST by the same rule in both models (else NO VERDICT: the rule could not fire).
  C3 null (negative control): eps=0 against eps=0 with disjoint scheduler seeds on the FRESH streams must be judged NOT ROBUST in both models (else NO VERDICT).
  C4 fixed reward holds: every counted run hit the target; reward CV across arms reported.
Fresh: task streams default_rng(1000..1029) (original: 0..29 and 20260917); scheduler seeds 50000 + 100*stream_index + s, s=0..11 (original: 2000+s, shared across streams); null arm 60000 + 100*i + s."""
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from decide import decide, part
N_TASKS, SEEDS, TARGET = 6, 12, 1200.0
STREAMS_FRESH = list(range(1000, 1030)); STREAMS_ORIG = list(range(0, 30))
def _m1():
    return dict(name="model1_400km_wohler", ORBITS=300, CAP=100.0, CHG=42.0, HK=14.0, FLOOR=0.0, E_LO=4.0, E_HI=16.0, DOD_DEN=100.0, law="wohler")
def _m2():
    mu, re, h = 398600.4418, 6371.0, 550.0; a = re + h; per = 2 * math.pi * math.sqrt(a ** 3 / mu)
    cap = 1200.0; sc = cap / 100.0
    return dict(name="model2_550km_log_cycle_life", ORBITS=300, CAP=cap, CHG=42.0 * sc, HK=75.0 * (per / 3600.0), FLOOR=0.2 * cap, E_LO=4.0 * sc, E_HI=16.0 * sc, DOD_DEN=cap - 0.2 * cap, law="loglife")
def task_stream(M, rng):
    return [{"value": float(rng.uniform(1, 10)), "energy": float(rng.uniform(M["E_LO"], M["E_HI"]))} for _ in range(N_TASKS)]
def damage(M, d):
    return np.sum(d ** 1.87) if M["law"] == "wohler" else np.sum(10.0 ** (0.8 * d))
def sim(M, tasks, eps, seed, warm=True):
    rng = np.random.default_rng(seed); est = np.zeros(N_TASKS); cnt = np.zeros(N_TASKS)
    soc = M["CAP"]; dods = []; reward = 0.0; orbits = 0
    for k in (rng.permutation(N_TASKS) if warm else ()):
        soc = min(M["CAP"], soc + M["CHG"]); start = soc; soc -= M["HK"]; t = tasks[k]
        if soc - t["energy"] > M["FLOOR"]:
            soc -= t["energy"]; obs = t["value"] + rng.normal(0, 1.5); reward += t["value"]; cnt[k] += 1; est[k] += (obs - est[k]) / cnt[k]
        dods.append((start - soc) / M["DOD_DEN"]); orbits += 1
    while reward < TARGET and orbits < M["ORBITS"] * 6:
        soc = min(M["CAP"], soc + M["CHG"]); start = soc; soc -= M["HK"]
        k = int(rng.integers(N_TASKS)) if rng.random() < eps else int(np.argmax(est)); t = tasks[k]
        if soc - t["energy"] > M["FLOOR"]:
            soc -= t["energy"]; obs = t["value"] + rng.normal(0, 1.5); reward += t["value"]; cnt[k] += 1; est[k] += (obs - est[k]) / cnt[k]
        dods.append((start - soc) / M["DOD_DEN"]); orbits += 1
    d = np.array(dods)
    return {"aging": float(damage(M, d)), "reward": reward, "orbits": orbits, "hit": bool(reward >= TARGET)}
def arm(M, tasks, eps, seeds, warm=True):
    runs = [sim(M, tasks, eps, s, warm) for s in seeds]; ok = [r for r in runs if r["hit"]]
    return (float(np.mean([r["aging"] for r in ok])) if ok else float("nan")), runs
def ratios(M, streams, seed_fn, warm=True, eps_a=0.0, eps_b=1.0, seed_fn_b=None, nseeds=SEEDS):
    out, nonhit, rew = [], 0, []
    for i, ts in enumerate(streams):
        tasks = task_stream(M, np.random.default_rng(ts))
        a0, r0 = arm(M, tasks, eps_a, [seed_fn(i, s) for s in range(nseeds)], warm)
        a1, r1 = arm(M, tasks, eps_b, [(seed_fn_b or seed_fn)(i, s) for s in range(nseeds)], warm)
        nonhit += sum(not r["hit"] for r in r0 + r1); rew += [r["reward"] for r in r0 + r1 if r["hit"]]
        out.append(a1 / a0 if (a0 > 0 and np.isfinite(a0) and np.isfinite(a1)) else float("nan"))
    return out, {"runs_not_hitting_target": nonhit, "reward_cv": float(np.std(rew) / np.mean(rew)) if rew else None}
def boot_median(r, B=10000, seed=0):
    r = np.asarray([x for x in r if np.isfinite(x)]); g = np.random.default_rng(seed)
    m = np.median(g.choice(r, (B, len(r))), axis=1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

# generated from the two original result files (multi_seed_ratios), 30 floats each
REF = {"m1": [1.3758383667642327, 1.5791220374436292, 2.1325530713094616, 1.3781254532434928, 1.2543901201373604, 1.3311762635302595, 1.3692945519695352, 1.8268674206465783, 2.2439722571506304, 1.39990445746972, 1.6428383543879634, 1.8284903579964191, 1.4158609086476501, 1.2417718338435353, 1.3971291301997906, 1.035031758013163, 1.4674515170141285, 1.8987973703151513, 1.558038458546044, 1.8968887130172427, 1.7648284287066371, 1.2883203397126182, 1.4801902376488159, 1.1774142903169573, 1.6546103009905646, 1.6223613517577853, 1.2523794501973218, 1.5398774435307703, 1.2119490256048147, 1.609737455818735], "m2": [1.347563037921761, 1.4716756122497416, 1.7456769999543753, 1.5564103141705345, 1.1464605593348611, 1.5980793875416086, 1.5311060810255372, 1.6004461385870248, 2.0209856494556138, 1.379743489659098, 1.3701820705914685, 1.9960174189283317, 1.7294002463228062, 1.327110829318982, 1.2152043345693473, 1.1595851067371978, 1.727321283535479, 1.602015996705262, 1.6424225146323832, 1.7056427951367905, 1.6346404767239153, 1.3353578801067876, 1.6985275866909726, 1.3134303969823988, 1.359853168430244, 1.983429656734264, 1.372681148717693, 1.8261819847672867, 1.336012806428226, 1.9148325570911702]}

def main(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else "s3_result.json"; smoke = "--smoke" in argv; t0 = time.time()
    ns_fresh = 3 if smoke else 30; nseeds = 3 if smoke else SEEDS; nrepro = 2 if smoke else 30
    R = {"smoke": smoke, "flags": [], "controls": {}}; Ms = {"m1": _m1(), "m2": _m2()}; fresh = STREAMS_FRESH[:ns_fresh]
    # C1 reproduction (always the full 12 scheduler seeds: the stored ratios used 12)
    c1 = {}
    for k, M in Ms.items():
        rr, _ = ratios(M, STREAMS_ORIG[:nrepro], lambda i, s: 2000 + s); ref = REF[k][:nrepro]
        c1[k] = {"max_abs_diff": float(np.max(np.abs(np.array(rr) - np.array(ref)))), "n": nrepro}
    R["controls"]["C1_reproduction"] = c1
    if not all(v["max_abs_diff"] < 1e-9 for v in c1.values()): R["flags"].append("NO VERDICT: C1 port does not reproduce the original ratios")
    # fresh main + C2 + C3
    main_r, c2r, c3r, info = {}, {}, {}, {}
    for k, M in Ms.items():
        main_r[k], info[k] = ratios(M, fresh, lambda i, s: 50000 + 100 * i + s, nseeds=nseeds)
        c2r[k], _ = ratios(M, fresh, lambda i, s: 50000 + 100 * i + s, warm=False, nseeds=nseeds)
        c3r[k], _ = ratios(M, fresh, lambda i, s: 50000 + 100 * i + s, eps_a=0.0, eps_b=0.0, seed_fn_b=lambda i, s: 60000 + 100 * i + s, nseeds=nseeds)
    R["fresh_ratios"] = main_r; R["fresh_info"] = info
    R["controls"]["C2_prefix_bug_detector"] = {k: part(v) for k, v in c2r.items()}
    R["controls"]["C3_null_eps0_vs_eps0"] = {k: part(v) for k, v in c3r.items()}
    # in full mode the controls must read NOT ROBUST (Refuted) in both models
    if not smoke:
        for nm in ("C2_prefix_bug_detector", "C3_null_eps0_vs_eps0"):
            if not all(v["verdict"] == "Refuted" for v in R["controls"][nm].values()): R["flags"].append(f"NO VERDICT: {nm} did not fire (rule not shown able to return NOT ROBUST)")
    for k in Ms:
        if info[k]["runs_not_hitting_target"]: R["flags"].append(f"{k}: {info[k]['runs_not_hitting_target']} runs did not reach the reward target (excluded as in the original; reported)")
    dec = decide(main_r["m1"], main_r["m2"])
    R["magnitude_descriptive_not_gating"] = {k: {"median": float(np.nanmedian(v)), "boot95_median": boot_median(v)} for k, v in main_r.items()}
    if smoke: dec = {"verdict": "SMOKE: plumbing only, no verdict", "detail": dec}
    elif any(f.startswith("NO VERDICT") for f in R["flags"]): dec = {"verdict": "Inconclusive under protocol X", "why": [f for f in R["flags"] if f.startswith("NO VERDICT")], "detail": dec}
    R["decision"] = dec; R["wall_s"] = round(time.time() - t0, 1)
    Path(out).write_text(json.dumps(R, indent=1, default=float)); print(json.dumps({"decision": dec["verdict"], "flags": R["flags"], "wall_s": R["wall_s"]}, default=float))
if __name__ == "__main__": main(sys.argv)
