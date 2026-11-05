"""Repeat inspections and permanent monitoring with BS 7910:2019 sizing errors (new script).

As run_observations.py / run_inspection_regimes.py, but:
  * the crack is detected at 2 mm, the smallest depth focussed phased array and
    TOFD can reliably size (Table T.2), not 1 mm;
  * detection AND repeat readings use the same technique's sizing sd (E / 1.3, T.5);
  * a permanently installed sensor with phased-array error reads ten times per interval.
Ensemble 800,000 members, 150 true cracks per case (fewer than the study's 2 million
and 300, for the two-core cloud machine); coverage is reported as the check.

Writes results/2026-11-05_bs7910/observations_bs7910.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import bs7910 as bs  # noqa: E402
from nasgro_scenarios import life, runner  # noqa: E402
from nasgro_scenarios import observations as ob  # noqa: E402

OUT = os.path.join(runner.ROOT, "results", "2026-11-05_bs7910")
A0_MM, N_ENS, CHUNK, N_TRUTH, K, FINE = 2.0, 800_000, 100_000, 150, 10, 10
REGIMES = {   # name: (technique sd mm, readings every `every` fine steps)
    "periodic, phased array (sd 1.15 mm)": (bs.sd_mm("focussed phased array / zonal AUT"), FINE),
    "permanent, phased-array error, 10 per interval": (bs.sd_mm("focussed phased array / zonal AUT"), 1),
    "periodic, TOFD (sd 0.77 mm)": (bs.sd_mm("TOFD (Table T.2)"), FINE),
    "periodic, TOFD trials (sd 0.38 mm)": (bs.sd_mm("TOFD trials, +-0.5 mm"), FINE),
    "periodic, TOFD best trials (sd 0.08 mm)": (bs.sd_mm("TOFD trials, +-0.1 mm"), FINE),
}


def interval(case):
    s = bs.sample(case, A0_MM * 1e-3, 20_000, bs.STUDY_SD_MM, seed=1)
    L = life.chunked(lambda x: life.ca_life(case, x), s)
    return float(0.1 * np.median(L[np.isfinite(L)]))


def ensemble(case, sd, times, seed0, n):
    Ls, Hs = [], []
    for c in range(n // CHUNK):
        s = bs.sample(case, A0_MM * 1e-3, CHUNK, sd, seed=seed0 + c)
        Ls.append(life.ca_life(case, s))
        Hs.append(ob.histories(case, s, times)[1].astype(np.float32))
    return np.concatenate(Ls), np.concatenate(Hs)


def track(L, H, Lt, y, idx, sd, times_f, dt):
    logw = np.zeros(len(L))
    alive = np.ones(len(L), bool)
    j, rows = 0, []
    for k in range(K + 1):
        tk = k * dt
        if Lt <= tk:
            break
        while j < len(idx) and times_f[idx[j]] <= tk + 1e-6:
            a = H[:, idx[j]]
            alive &= np.isfinite(a) & (L > times_f[idx[j]])
            logw += -0.5 * ((y[j] - np.nan_to_num(a)) / sd) ** 2
            j += 1
        r, _ = ob.rul_posterior(L, np.where(alive, logw, -np.inf), tk)
        truth = np.log10(Lt - tk)
        rows.append((k, r["u90"], float(r["q05"] <= truth <= r["q95"]), 10 ** (r["q05"] - truth),
                     10 ** (r["q95"] - truth), r["ess"]))
    return rows


def run_case(case):
    dt = interval(case)
    times_f = dt / FINE * np.arange(1, K * FINE + 1)
    out = {"interval": dt, "a0_mm": A0_MM, "n_ensemble": N_ENS, "regimes": {}}
    cache = {}
    for name, (sd, every) in REGIMES.items():
        if sd not in cache:
            L, H = ensemble(case, sd, times_f, 2000, N_ENS)
            Lt_all, Ht_all = ensemble(case, sd, times_f, 9000, CHUNK)
            truths = np.flatnonzero(np.isfinite(Lt_all))[:N_TRUTH]
            cache = {sd: (L, H, Lt_all[truths], Ht_all[truths])}
        L, H, Lt, Ht = cache[sd]
        rng = np.random.default_rng(5)
        idx = np.arange(every - 1, K * FINE, every)
        rows = []
        for i in range(len(Lt)):
            y = Ht[i, idx] + sd * 1e-3 * rng.standard_normal(len(idx))
            rows += track(L, H, Lt[i], y, idx, sd * 1e-3, times_f, dt)
        a = np.array(rows)
        per_k = []
        for k in range(K + 1):
            m = a[:, 0] == k
            if m.sum() < 10:
                break
            per_k.append(dict(k=k, n=int(m.sum()), u90_median=float(np.median(a[m, 1])),
                              u90_q25=float(np.percentile(a[m, 1], 25)), u90_q75=float(np.percentile(a[m, 1], 75)),
                              coverage=float(a[m, 2].mean()), lo_ratio_median=float(np.median(a[m, 3])),
                              hi_ratio_median=float(np.median(a[m, 4])), ess_median=float(np.median(a[m, 5]))))
        out["regimes"][name] = dict(sd_mm=sd, readings_per_interval=FINE // every, per_k=per_k)
        print(case, name, [(p["k"], round(p["u90_median"], 2), round(p["coverage"], 2), int(p["ess_median"]))
                           for p in per_k], flush=True)
    return out


def main():
    runner.set_standard()
    os.makedirs(OUT, exist_ok=True)
    res = {c: run_case(c) for c in ("bar", "plate")}
    with open(os.path.join(OUT, "observations_bs7910.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
