"""Spread of RUL after each inspection, and periodic inspection against continuous monitoring.

Same prior ensemble (2 million members, the seeds of run_observations.py) and
the same 300 true cracks per case, but the depth histories are stored on a grid
ten times finer than the inspection interval, so that a permanently installed
sensor can read the depth ten times per interval. Regimes:

  periodic 0.5 / 0.17 / 0.05   one reading per interval, error sd in mm (as in
                               run_observations.py, same noise draws)
  continuous 0.5               ten readings per interval, independent error sd 0.5 mm
  continuous 0.5 + bias 0.25   as above plus a fixed sensor offset per crack,
                               sd 0.25 mm, unknown to the analyst (correlated
                               error; integrated out exactly in the likelihood)

Per regime and number of elapsed intervals k: median U90 over the true cracks,
coverage, and the 90 % RUL band relative to the true RUL. For one example crack
(the true crack nearest the median life, as in Figure 13) it stores 2,000
weighted resamples of the predicted life at every k.

Writes results/2026-10-23_nasgro_only/inspection_regimes.json.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import runner  # noqa: E402
from nasgro_scenarios import observations as ob  # noqa: E402
from run_observations import (CHUNK, ENS_SEED, INTERVAL, K, N_ENS, N_TRUTH, NOISE_SEED,  # noqa: E402
                              TRUTH_SEED)

FINE = 10
REGIMES = {
    "periodic 0.5": dict(sigma=0.5e-3, every=FINE, bias=0.0),
    "periodic 0.17": dict(sigma=0.17e-3, every=FINE, bias=0.0),
    "periodic 0.05": dict(sigma=0.05e-3, every=FINE, bias=0.0),
    "continuous 0.5": dict(sigma=0.5e-3, every=1, bias=0.0),
    "continuous 0.5 + bias 0.25": dict(sigma=0.5e-3, every=1, bias=0.25e-3),
}
N_SAMPLES = 2000


def fine_times(case):
    return INTERVAL[case] / FINE * np.arange(1, K * FINE + 1)


def chunk(args):
    case, c = args
    runner.set_standard()
    s, L, H = ob.ensemble(case, CHUNK, ENS_SEED + c, fine_times(case))
    return s["a_i"], L, H.astype(np.float32)


def noise(case, n_truth):
    """Reading errors per truth: periodic ones reuse run_observations.py's draws."""
    per = np.random.default_rng(NOISE_SEED).standard_normal((n_truth, K))
    cont = np.random.default_rng(6).standard_normal((n_truth, K * FINE))
    bias = np.random.default_rng(7).standard_normal(n_truth)
    return per, cont, bias


def readings(reg, i, Ht, per, cont, bias):
    """(fine-grid indices, values) of the readings of truth i under a regime."""
    if reg["every"] == FINE:
        idx = np.arange(FINE - 1, K * FINE, FINE)
        e = per[i]
    else:
        idx = np.arange(K * FINE)
        e = cont[i]
    return idx, Ht[i, idx] + reg["sigma"] * e + reg["bias"] * bias[i]


def track(L, H, Lt_i, idx, y, reg, times_f, interval, keep_samples=None, rng=None, a_i=None):
    """Posterior after each elapsed interval k = 0..K for one truth."""
    n = len(L)
    s1, s2 = np.zeros(n), np.zeros(n)
    alive = np.ones(n, bool)
    sg2, sb2 = reg["sigma"] ** 2, reg["bias"] ** 2
    m, j0, out, samples = 0, 0, [], []
    for k in range(K + 1):
        tk = k * interval
        if Lt_i <= tk:
            break
        while j0 < len(idx) and times_f[idx[j0]] <= tk + 1e-6:
            a = H[:, idx[j0]]
            alive &= np.isfinite(a) & (L > times_f[idx[j0]])
            r = np.where(alive, y[j0] - np.nan_to_num(a), 0.0)
            s1 += r
            s2 += r * r
            m += 1
            j0 += 1
        c = sb2 / (sg2 * (sg2 + m * sb2)) if m else 0.0
        logw = np.where(alive, -0.5 * (s2 / sg2 - c * s1 ** 2), -np.inf)
        r, w = ob.rul_posterior(L, logw, tk)
        out.append((k, r))
        if keep_samples is not None:
            fin = np.flatnonzero(np.isfinite(L) & (w > 0))
            pick = rng.choice(fin, keep_samples, p=w[fin] / w[fin].sum())
            samples.append(dict(k=k, life=L[pick].tolist(), a_i=a_i[pick].tolist(), p_arrest=r["p_arrest"],
                                n_readings=m))
    return out, samples


def run_case(case, pool):
    parts = pool.map(chunk, [(case, c) for c in range(N_ENS // CHUNK)])
    A = np.concatenate([p[0] for p in parts])
    L = np.concatenate([p[1] for p in parts])
    H = np.concatenate([p[2] for p in parts])
    del parts
    times_f = fine_times(case)
    ts, Lt, Ht = ob.ensemble(case, 2 * N_TRUTH, TRUTH_SEED, times_f)
    truths = np.flatnonzero(np.isfinite(Lt))[:N_TRUTH]
    per, cont, bias = noise(case, len(Lt))
    ex = truths[np.argmin(np.abs(np.log10(Lt[truths]) - np.median(np.log10(Lt[truths]))))]
    out = {"interval": INTERVAL[case], "readings_per_interval_continuous": FINE, "n_truth": len(truths),
           "example": {"truth_life": float(Lt[ex]), "truth_a_i": float(ts["a_i"][ex]), "regimes": {}}, "regimes": {}}
    for name, reg in REGIMES.items():
        def one(i):
            idx, y = readings(reg, i, Ht, per, cont, bias)
            res, _ = track(L, H, Lt[i], idx, y, reg, times_f, INTERVAL[case])
            rows = []
            for k, r in res:
                true_rul = Lt[i] - k * INTERVAL[case]
                lt = np.log10(true_rul)
                rows.append((k, r["u90"], float(r["q05"] <= lt <= r["q95"]), 10 ** (r["q05"] - lt),
                             10 ** (r["q95"] - lt), r["ess"]))
            return rows
        with ThreadPoolExecutor(8) as ex_pool:
            allrows = [row for rows in ex_pool.map(one, truths) for row in rows]
        a = np.array(allrows)
        per_k = []
        for k in range(K + 1):
            m = a[:, 0] == k
            if m.sum() < 10:
                break
            per_k.append(dict(k=k, n=int(m.sum()), u90_median=float(np.median(a[m, 1])),
                              u90_q25=float(np.percentile(a[m, 1], 25)), u90_q75=float(np.percentile(a[m, 1], 75)),
                              coverage=float(a[m, 2].mean()),
                              lo_ratio_median=float(np.median(a[m, 3])), hi_ratio_median=float(np.median(a[m, 4])),
                              ess_median=float(np.median(a[m, 5])), ess_min=float(a[m, 5].min())))
        out["regimes"][name] = per_k
        print(case, name, [(p["k"], round(p["u90_median"], 3), round(p["coverage"], 2), int(p["ess_median"]))
                           for p in per_k], flush=True)
        idx, y = readings(reg, ex, Ht, per, cont, bias)
        res, samples = track(L, H, Lt[ex], idx, y, reg, times_f, INTERVAL[case], keep_samples=N_SAMPLES,
                             rng=np.random.default_rng(1), a_i=A)
        out["example"]["regimes"][name] = dict(
            samples=samples, bands=[dict(k=k, **r) for k, r in res],
            readings_t=times_f[idx].tolist(), readings_y=y.tolist())
    return out


def main(which=("bar", "plate")):
    runner.set_standard()
    path = os.path.join(runner.RESULTS, "inspection_regimes.json")
    res = runner.load("inspection_regimes.json") if os.path.exists(path) else {}
    with Pool(min(16, os.cpu_count())) as pool:
        for case in which:
            res[case] = run_case(case, pool)
            print(runner.save("inspection_regimes.json", res), flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("bar", "plate"))
