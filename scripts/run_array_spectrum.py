"""U90 reduction along the array spectrum: one reading vs periodic vs permanent (new script).

Four sensing levels, each given a depth-sizing error (sd, mm) and, for the 2D
array, knowledge of crack shape and tilt. A crack is detected at 2 mm (the
smallest depth phased array and TOFD reliably size, BS 7910 Table T.2). For each
level, three regimes after ten inspection intervals (each 0.1 of the median life):

  one reading   the level's own sizing at detection, no repeat (k = 0)
  periodic      one reading per interval, ten intervals
  permanent     ten readings per interval, ten intervals

Levels (sizing sd, mm) - assumptions, not measured values:
  guided-wave SHM network        2.3   (as conventional UT, BS 7910 T.2: sizes coarsely)
  sparse permanent bulk array    1.15  (phased-array value, T.2), absolute readings;
                                 plus "as change": readings sd 0.17 sharing one unknown
                                 offset of sd 1.15 per crack (baseline subtraction)
  dense linear array             0.38  (about the largest error on real fatigue cracks,
                                 Peng et al. 2018; BS 7910 TOFD trials)
  dense 2D array                 0.38, with crack shape a/c and tilt known exactly

Every U90 is reported with its reduction from the common reference: U90 at
detection sized by conventional UT (sd 2.3 mm). Ensemble 300,000, 80 true cracks
per case and level (two-core machine); coverage reported as the check.

Writes results/2026-11-05_bs7910/array_spectrum_<case>.json. Base modules untouched.
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
A0_MM, N_ENS, CHUNK, N_TRUTH, K, FINE = 2.0, 300_000, 100_000, 80, 10, 10
LEVELS = {  # name: (detection/reading sd mm, fixed inputs, regimes)
    "guided-wave SHM network": (2.3, (), {"periodic": (2.3, 0.0, FINE), "permanent": (2.3, 0.0, 1)}),
    "sparse permanent array": (1.15, (), {"periodic": (1.15, 0.0, FINE), "permanent": (1.15, 0.0, 1),
                                          "permanent, as change": (0.17, 1.15, 1)}),
    "dense linear array": (0.38, (), {"periodic": (0.38, 0.0, FINE), "permanent": (0.38, 0.0, 1)}),
    "dense 2D array": (0.38, ("ac", "tilt"), {"periodic": (0.38, 0.0, FINE), "permanent": (0.38, 0.0, 1)}),
}


def interval(case):
    s = bs.sample(case, A0_MM * 1e-3, 20_000, bs.STUDY_SD_MM, seed=1)
    L = life.chunked(lambda x: life.ca_life(case, x), s)
    return float(0.1 * np.median(L[np.isfinite(L)]))


def ensemble(case, sd, fixed, times, seed0, n):
    Ls, Hs = [], []
    for c in range(max(1, n // CHUNK)):
        s = bs.sample(case, A0_MM * 1e-3, min(n, CHUNK), sd, seed=seed0 + c, fixed=fixed)
        Ls.append(life.ca_life(case, s))
        Hs.append(ob.histories(case, s, times)[1].astype(np.float32))
    return np.concatenate(Ls), np.concatenate(Hs)


def track(L, H, Lt, y, idx, sig, bias, times_f, dt):
    """Posterior after each interval; readings share one Gaussian offset of sd `bias` (integrated out)."""
    n = len(L)
    s1, s2 = np.zeros(n), np.zeros(n)
    alive = np.ones(n, bool)
    sg2, sb2 = sig ** 2, bias ** 2
    m, j, rows = 0, 0, []
    for k in range(K + 1):
        tk = k * dt
        if Lt <= tk:
            break
        while j < len(idx) and times_f[idx[j]] <= tk + 1e-6:
            a = H[:, idx[j]]
            alive &= np.isfinite(a) & (L > times_f[idx[j]])
            r = np.where(alive, y[j] - np.nan_to_num(a), 0.0)
            s1 += r
            s2 += r * r
            m += 1
            j += 1
        c = sb2 / (sg2 * (sg2 + m * sb2)) if m else 0.0
        logw = np.where(alive, -0.5 * (s2 / sg2 - c * s1 ** 2), -np.inf)
        r, _ = ob.rul_posterior(L, logw, tk)
        truth = np.log10(Lt - tk)
        rows.append((k, r["u90"], float(r["q05"] <= truth <= r["q95"]), r["ess"]))
    return rows


def summarise(rows):
    a = np.array(rows)
    out = {}
    for k in (0, K):
        msk = a[:, 0] == k
        if msk.sum() >= 10:
            out[k] = dict(u90=float(np.median(a[msk, 1])), coverage=float(a[msk, 2].mean()),
                          ess=float(np.median(a[msk, 3])), n=int(msk.sum()))
    return out


def run_case(case):
    runner.set_standard()
    dt = interval(case)
    times_f = dt / FINE * np.arange(1, K * FINE + 1)
    ref_s = bs.sample(case, A0_MM * 1e-3, 100_000, 2.3, seed=11)
    ref = float(life.u90(life.chunked(lambda x: life.ca_life(case, x), ref_s)))
    out = {"interval": dt, "a0_mm": A0_MM, "reference_u90_conventional_ut": ref, "levels": {}}
    for li, (name, (sd, fixed, regs)) in enumerate(LEVELS.items()):
        L, H = ensemble(case, sd, fixed, times_f, 3000 + 100 * li, N_ENS)
        Lt_all, Ht_all = ensemble(case, sd, fixed, times_f, 9000 + 100 * li, 2 * N_TRUTH)
        truths = np.flatnonzero(np.isfinite(Lt_all))[:N_TRUTH]
        Lt, Ht = Lt_all[truths], Ht_all[truths]
        one = float(life.u90(L))
        lev = {"sd_mm": sd, "fixed": list(fixed), "one_reading": {"u90": one, "reduction": 1 - one / ref},
               "regimes": {}}
        for rname, (sig, bias, every) in regs.items():
            rng = np.random.default_rng(5)
            idx = np.arange(every - 1, K * FINE, every)
            rows = []
            for i in range(len(Lt)):
                y = Ht[i, idx] + sig * 1e-3 * rng.standard_normal(len(idx)) + bias * 1e-3 * rng.standard_normal()
                rows += track(L, H, Lt[i], y, idx, sig * 1e-3, bias * 1e-3, times_f, dt)
            sm = summarise(rows)
            if K in sm:
                sm[K]["reduction"] = 1 - sm[K]["u90"] / ref
            lev["regimes"][rname] = {"sd_mm": sig, "offset_sd_mm": bias, "readings_per_interval": FINE // every,
                                     "after_ten": sm.get(K), "at_detection": sm.get(0)}
            print(case, name, rname, sm.get(K), flush=True)
        out["levels"][name] = lev
        print(case, name, "one", round(one, 3), "ref", round(ref, 3), flush=True)
        del L, H
    return out


def main():
    runner.set_standard()
    os.makedirs(OUT, exist_ok=True)
    for case in sys.argv[1:] or ("bar", "plate"):
        with open(os.path.join(OUT, f"array_spectrum_{case}.json"), "w") as f:
            json.dump({case: run_case(case)}, f, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
