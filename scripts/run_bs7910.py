"""Rerun of the NASGRO-only study with BS 7910:2019 sizing errors and growth laws (new script).

A  sizing sweep: U90, U90 with depth known exactly, and the 2.3rd percentile of
   log10 life, against detection sizing sd (Annex T techniques and the study's
   0.5 mm), at start depths 1-4 mm, three seeds of 10,000.
B  full U90 ranking (as study Figure 2) with focussed phased array and TOFD sizing.
D  growth-law comparison at a 3 mm reported crack: NASGRO distribution, Annex K
   style probabilistic Paris law, and the deterministic BS 7910 assessment
   (reported depth + E, nominal load, simplified Paris law).

Writes results/2026-11-05_bs7910/bs7910.json. Base modules and earlier results untouched.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import bs7910 as bs  # noqa: E402
from nasgro_scenarios import life, runner  # noqa: E402

OUT = os.path.join(runner.ROOT, "results", "2026-11-05_bs7910")
N, SEEDS = 10_000, (1, 2, 3)
SD_LEVELS = {"study 0.5 mm": 0.5, **{k: bs.sd_mm(k) for k in bs.TECHNIQUES}}
DEPTHS = (1.0, 2.0, 3.0, 4.0)


def lives(case, a_mm, sd, seed, fixed=()):
    s = bs.sample(case, a_mm * 1e-3, N, sd, seed=seed, fixed=fixed)
    return life.chunked(lambda x: life.ca_life(case, x), s)


def sweep():
    out = []
    for case in ("bar", "plate"):
        for a in DEPTHS:
            for name, sd in SD_LEVELS.items():
                u0, u1, pct = [], [], []
                for seed in SEEDS:
                    L0 = lives(case, a, sd, seed)
                    L1 = lives(case, a, sd, seed, fixed=("depth",))
                    u0.append(life.u90(L0))
                    u1.append(life.u90(L1))
                    pct.append(bs.percentiles(L0))
                red = [1 - b / c for b, c in zip(u1, u0)]
                a_min = bs.TECHNIQUES.get(name, {}).get("a_min", 0.0)
                out.append(dict(case=case, depth_mm=a, technique=name, sd_mm=sd, sizable=a >= a_min,
                                u90=float(np.mean(u0)), u90_depth_exact=float(np.mean(u1)),
                                depth_reduction=float(np.mean(red)), depth_reduction_range=[min(red), max(red)],
                                **{k: float(np.mean([p[k] for p in pct])) for k in pct[0]}))
                print(case, a, name, round(out[-1]["u90"], 3), round(out[-1]["depth_reduction"], 3), flush=True)
    return out


def ranking(sd_name):
    sd = bs.sd_mm(sd_name)
    out = []
    for case in ("bar", "plate"):
        inputs = runner.base_inputs(case)
        for a in (1.0, 2.0, 4.0):
            reds = {k: [] for k in inputs}
            u0s = []
            for seed in SEEDS:
                fn = lambda fixed=(): lives(case, a, sd, seed, fixed)  # noqa: E731
                st, red = life.u90_reduction(fn, inputs)
                u0s.append(st["u90"])
                for k, v in red.items():
                    reds[k].append(v)
            out.append(dict(case=case, depth_mm=a, technique=sd_name, sd_mm=sd, u90=float(np.mean(u0s)),
                            reduction={k: float(np.mean(v)) for k, v in reds.items()},
                            reduction_range={k: [min(v), max(v)] for k, v in reds.items()}))
            print("rank", sd_name, case, a, flush=True)
    return out


def growth_laws(a_mm=3.0):
    out = []
    for case in ("bar", "plate"):
        for name in ["study 0.5 mm", *bs.TECHNIQUES]:
            sd = SD_LEVELS[name]
            E = 1.3 * sd
            s = bs.sample(case, a_mm * 1e-3, N, sd, seed=1)
            Ln = life.chunked(lambda x: life.ca_life(case, x), s)
            Lp = bs.prob_life(case, s)
            out.append(dict(case=case, a_reported_mm=a_mm, technique=name, sd_mm=sd, E_mm=E,
                            nasgro={**bs.percentiles(Ln), "u90": float(life.u90(Ln)), "pf": float(np.isfinite(Ln).mean())},
                            annex_k={**bs.percentiles(Lp), "u90": float(life.u90(Lp)), "pf": float(np.isfinite(Lp).mean())},
                            annex_k_positive={**bs.percentiles(bs.prob_life(case, s, positive_only=True))},
                            bs_det_log10=float(np.log10(bs.det_life(case, a_mm * 1e-3, E * 1e-3))),
                            bs_det_positive_log10=float(np.log10(bs.det_life(case, a_mm * 1e-3, E * 1e-3, positive_only=True)))))
            print("laws", case, name, out[-1]["nasgro"]["q50"], out[-1]["annex_k"]["q50"], out[-1]["bs_det_log10"], flush=True)
    return out


def main():
    runner.set_standard()
    os.makedirs(OUT, exist_ok=True)
    res = dict(sd_levels=SD_LEVELS, techniques=bs.TECHNIQUES, n=N, seeds=list(SEEDS))
    res["growth_laws"] = growth_laws()
    res["sweep"] = sweep()
    res["ranking"] = ranking("focussed phased array / zonal AUT") + ranking("TOFD (Table T.2)")
    with open(os.path.join(OUT, "bs7910.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
