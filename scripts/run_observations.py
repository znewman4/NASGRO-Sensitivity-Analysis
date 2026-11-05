"""RUL certainty as repeat crack-depth measurements are added (question: value of extra observations).

For each case at a 1 mm detected crack: a prior ensemble of 2 million members
(NASGRO-equation inputs at industry-typical uncertainty) and 300 "true"
cracks drawn from the same prior on another seed (cracks that fail only). Each
true crack is inspected every `INTERVAL` cycles (about a tenth of the median
life) with depth error sd sigma_m in {0.5, 0.17, 0.05} mm, until it fails.
After each inspection the ensemble is reweighted (observations.update) and the
90 % band of log10 RUL is read off.

Writes results/2026-10-23_nasgro_only/observations.json.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import cases, life, runner  # noqa: E402
from nasgro_scenarios import observations as ob  # noqa: E402

N_ENS, CHUNK, N_TRUTH = 2_000_000, 100_000, 300
INTERVAL = {"bar": 35_000.0, "plate": 2_500.0}       # cycles, ~0.1 x prior median life at 1 mm
K = 10
PRECISIONS_MM = (0.5, 0.17, 0.05)
SD_AT = (1, 3, 6)
ENS_SEED, TRUTH_SEED, NOISE_SEED = 1000, 99, 5


def chunk(args):
    case, c = args
    runner.set_standard()
    times = INTERVAL[case] * np.arange(1, K + 1)
    s, L, H = ob.ensemble(case, CHUNK, ENS_SEED + c, times)
    keep = {v[0] for v in ob.LEARNED[case].values()}
    return {k: s[k] for k in keep}, L, H


def ensemble(case, pool):
    parts = pool.map(chunk, [(case, c) for c in range(N_ENS // CHUNK)])
    s = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    return s, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])


def regenerate(case, idx):
    """Full inputs of ensemble members by global index (chunks are re-drawn from their seeds)."""
    runner.set_standard()
    out = []
    for c in np.unique(idx // CHUNK):
        s = cases.sample(case, 1e-3, CHUNK, seed=ENS_SEED + c, fixed=runner.fx(()))
        loc = idx[idx // CHUNK == c] % CHUNK
        out.append({k: v[loc] for k, v in s.items()})
    order = np.concatenate([np.flatnonzero(idx // CHUNK == c) for c in np.unique(idx // CHUNK)])
    full = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return {k: v[inv] for k, v in full.items()}


def depth_band(case, s, w, t_grid, qs=(0.05, 0.5, 0.95)):
    """Weighted depth quantiles on a time grid (members still intact at each time)."""
    total, h = ob.histories(case, s, t_grid)
    band = []
    for j, t in enumerate(t_grid):
        ok = np.isfinite(h[:, j])
        band.append(ob.weighted_quantiles(h[ok, j], w[ok], qs).tolist() if ok.sum() > 10 else [np.nan] * 3)
    return np.array(band)


def run_case(case, pool):
    times = INTERVAL[case] * np.arange(1, K + 1)
    s, L, H = ensemble(case, pool)
    ts, Lt, Ht = ob.ensemble(case, 2 * N_TRUTH, TRUTH_SEED, times)
    truths = np.flatnonzero(np.isfinite(Lt))[:N_TRUTH]
    eps = np.random.default_rng(NOISE_SEED).standard_normal((len(Lt), K))
    sd0 = ob.prior_sd(s, case)
    life_log = np.where(np.isfinite(L), np.log10(np.where(np.isfinite(L), L, 1.0)), np.nan)
    out = {"interval": INTERVAL[case], "n_ensemble": N_ENS, "n_truth": len(truths),
           "prior_median_log10_life": float(np.nanmedian(life_log[np.isfinite(L)])), "precisions": {}}
    for sm_mm in PRECISIONS_MM:
        sm = sm_mm * 1e-3
        rows, sd_rows = [], {k: [] for k in SD_AT}
        for i in truths:
            y = Ht[i] + sm * eps[i]
            logw = np.zeros(len(L))
            for k in range(K + 1):
                tk = 0.0 if k == 0 else times[k - 1]
                if Lt[i] <= tk:
                    break
                if k > 0:
                    logw += ob.update(L, H[:, k - 1:k], times[k - 1:k], y[k - 1:k], sm, 1)
                r, w = ob.rul_posterior(L, logw, tk)
                truth = np.log10(Lt[i] - tk)
                rows.append([k, r["u90"], r["ess"], float(r["q05"] <= truth <= r["q95"]), r["q50"] - truth])
                if k in SD_AT:
                    nz = w > 0
                    ratios = ob.posterior_sd_ratio({kk: v[nz] for kk, v in s.items()}, w[nz], case, sd0)
                    fin = nz & np.isfinite(L)
                    x = life_log[fin]
                    m = np.sum(w[fin] * x) / w[fin].sum()
                    ratios["predicted life (all inputs combined)"] = float(
                        np.sqrt(np.sum(w[fin] * (x - m) ** 2) / w[fin].sum()) / np.nanstd(life_log))
                    sd_rows[k].append(ratios)
        rows = np.array(rows)
        per_k = []
        for k in range(K + 1):
            m = rows[:, 0] == k
            if m.sum() < 10:
                break
            u = rows[m, 1]
            per_k.append(dict(k=k, n=int(m.sum()), u90_median=float(np.median(u)),
                              u90_q25=float(np.percentile(u, 25)), u90_q75=float(np.percentile(u, 75)),
                              coverage=float(rows[m, 3].mean()), ess_median=float(np.median(rows[m, 2])),
                              ess_min=float(rows[m, 2].min()), bias_median=float(np.median(rows[m, 4]))))
        sd = {k: {name: float(np.median([r[name] for r in v])) for name in v[0]} for k, v in sd_rows.items() if v}
        out["precisions"][str(sm_mm)] = dict(per_k=per_k, sd_ratio=sd)
        print(case, sm_mm, [(p["k"], round(p["u90_median"], 3), round(p["coverage"], 2), int(p["ess_median"]))
                            for p in per_k], flush=True)

    # Example crack for the figure: the truth closest to the median life
    i = truths[np.argmin(np.abs(np.log10(Lt[truths]) - np.median(np.log10(Lt[truths]))))]
    t_grid = np.linspace(0.0, Lt[i] * 1.05, 120)
    ex = {"truth_life": float(Lt[i]), "t_grid": t_grid.tolist(), "times": times.tolist()}
    true_s = {k: v[i:i + 1] for k, v in ts.items()}
    ex["truth_depth"] = ob.histories(case, true_s, t_grid)[1][0].tolist()
    rng = np.random.default_rng(0)
    prior_idx = rng.choice(N_ENS, 20_000, replace=False)
    ex["prior_band"] = depth_band(case, regenerate(case, prior_idx), np.ones(len(prior_idx)), t_grid).tolist()
    ex["panels"] = {}
    for sm_mm in (0.5, 0.05):
        sm = sm_mm * 1e-3
        y = Ht[i] + sm * eps[i]
        n_obs = int(np.sum(times < Lt[i]))
        k_show = min(3, n_obs)
        logw = ob.update(L, H, times, y, sm, k_show)
        r, w = ob.rul_posterior(L, logw, times[k_show - 1])
        idx = np.flatnonzero(w > 0)
        if len(idx) > 20_000:
            idx = rng.choice(idx, 20_000, replace=False, p=w[idx] / w[idx].sum())
            ww = np.ones(len(idx))
        else:
            ww = w[idx]
        ex["panels"][str(sm_mm)] = dict(k=k_show, y=y[:k_show].tolist(),
                                        band=depth_band(case, regenerate(case, idx), ww, t_grid).tolist(),
                                        rul=r)
    out["example"] = ex
    return out


def main():
    runner.set_standard()
    res = {}
    with Pool(min(16, os.cpu_count())) as pool:
        for case in ("bar", "plate"):
            res[case] = run_case(case, pool)
    print(runner.save("observations.json", res))


if __name__ == "__main__":
    main()
