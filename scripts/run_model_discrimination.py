"""Can the crack tilt theta tell us which K model to use (mode I, projection, K_I + K_II mixed mode)?

Two parts, bar and plate at a 1 mm detected crack:
  1. Model gap: spread of predicted log10 life across the four tilt models at
     nominal inputs, against tilt (from mixed_mode_step1.json).
  2. Identification: the true crack follows one of the four models (equal
     prior odds), its tilt is UT-sized to +-5 deg about theta, every other
     input is at industry-typical uncertainty. Repeat inspections (as in
     run_observations.py, depth error 0.17 mm) update the model odds:
     P(M | y) proportional to the ensemble-mean likelihood of y under M.
     Reported: mean posterior probability of the true model after k inspections
     (0.25 = no information).

Writes results/2026-10-23_nasgro_only/model_discrimination.json.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import cases, mixed_mode as mm, runner  # noqa: E402
from nasgro_scenarios import observations as ob  # noqa: E402

THETAS = (0.0, 15.0, 30.0, 45.0, 60.0)
MODELS = (1, 0, 2, 3)                     # full mode I, projection, energy, Tanaka (indices into mm.MODELS)
N_ENS, N_TRUTH, SIGMA_M = 200_000, 100, 0.17e-3
INTERVAL = {"bar": 35_000.0, "plate": 2_500.0}
K, SHOW_K = 10, (1, 3, 6)


def sample(case, theta, model, n, seed):
    runner.set_standard()
    s = cases.sample(case, 1e-3, n, seed=seed, fixed=runner.fx(("tilt",)))
    s["mm_theta"] = np.deg2rad(np.clip(np.abs(theta + 5.0 * cases._z("mm_theta", n, seed)), 0.0, 80.0))
    s["mm_model"] = np.full(n, model)
    s["tilt"] = np.zeros(n)
    return s


def hist(args, chunk=20_000):
    case, theta, model, n, seed = args
    s = sample(case, theta, model, n, seed)
    times = INTERVAL[case] * np.arange(1, K + 1)
    parts = [ob.histories(case, {k: v[i:i + chunk] for k, v in s.items()}, times, profile=mm.mm_profile)
             for i in range(0, n, chunk)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def run(case, theta, pool):
    times = INTERVAL[case] * np.arange(1, K + 1)
    ens = pool.map(hist, [(case, theta, m, N_ENS, 2000 + m) for m in MODELS])
    tru = pool.map(hist, [(case, theta, m, 3 * N_TRUTH, 3000 + m) for m in MODELS])
    eps = np.random.default_rng(11).standard_normal((3 * N_TRUTH, K))
    p_true = {k: [] for k in range(K + 1)}
    for mi, (Lt, Ht) in enumerate(tru):
        for i in np.flatnonzero(np.isfinite(Lt))[:N_TRUTH]:
            y = Ht[i] + SIGMA_M * eps[i]
            logw = [np.zeros(len(L)) for L, _ in ens]
            for k in range(K + 1):
                if k > 0:
                    if Lt[i] <= times[k - 1]:
                        break
                    for j, (L, H) in enumerate(ens):
                        logw[j] += ob.update(L, H[:, k - 1:k], times[k - 1:k], y[k - 1:k], SIGMA_M, 1)
                ml = np.array([logsumexp(lw) - np.log(len(lw)) for lw in logw])
                p_true[k].append(float(np.exp(ml[mi] - logsumexp(ml))))
    return {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in p_true.items() if len(v) >= 20}


def main(cases=("bar", "plate")):
    runner.set_standard()
    gap = {}
    step1 = runner.load("mixed_mode_step1.json")["results"]
    for case in ("bar", "plate"):
        rows = [r for r in step1 if r["case"] == case and r["depth_mm"] == 1.0]
        th = rows[0]["theta_deg"]
        lives = np.array([[np.nan if x is None else x for x in r["log10_life"]] for r in rows])
        gap[case] = dict(theta=th, models=[r["model"] for r in rows], log10_life=lives.tolist(),
                         spread=np.nanmax(lives, 0).tolist(), spread_min=np.nanmin(lives, 0).tolist(),
                         arrest=[bool(np.isnan(lives[:, j]).any()) for j in range(len(th))])
    path = os.path.join(runner.RESULTS, "model_discrimination.json")
    out = runner.load("model_discrimination.json") if os.path.exists(path) else {"identification": {}}
    out.update(gap=gap, sigma_m_mm=SIGMA_M * 1e3, models=[mm.MODELS[m] for m in MODELS])
    with Pool(4) as pool:
        for case in cases:
            ident = {}
            for theta in THETAS:
                ident[str(theta)] = run(case, theta, pool)
                print(case, theta, {k: round(v["mean"], 2) for k, v in ident[str(theta)].items()}, flush=True)
            out["identification"][case] = ident
            print(runner.save("model_discrimination.json", out), flush=True)   # saved after each case


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("bar", "plate"))
