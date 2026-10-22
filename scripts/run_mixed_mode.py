"""Mixed-mode (tilted crack) scenario module (separate from the base case).

Step 1, no Monte Carlo: life with every other input at nominal for each tilt
model, tilts 0-60 deg, every depth, both cases. Shows the size of the model
choice next to the parameter spread (baseline U90).

Step 2, Monte Carlo: the tilt has a prior for the component (uniaxial: centred
on 0 deg; inclined cracks: centred on 30 deg), unknown (sd 15 deg) or UT-sized
(sd 5 deg), and the criterion is an uncertain input (projection / energy /
Tanaka). Compares "know the tilt" with "know which model is right".

    python scripts/run_mixed_mode.py [--n 10000]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import cases, life, runner  # noqa: E402
from nasgro_scenarios import mixed_mode as mm  # noqa: E402

THETAS = list(range(0, 61, 5))


def step1():
    out = []
    for case in ("bar", "plate"):
        for d in runner.DEPTHS_MM[case]:
            for k, name in enumerate(mm.MODELS):
                lives = [mm.nominal_life(case, d, th, k) for th in THETAS]
                out.append(dict(case=case, depth_mm=d, model=name, theta_deg=THETAS,
                                log10_life=[float(np.log10(x)) if np.isfinite(x) else None
                                            for x in lives]))
    return out


def step2_task(case, depth_mm, prior, sized, n):
    inputs = {k: v for k, v in runner.base_inputs(case).items() if v != "tilt"}
    inputs["crack tilt (mixed mode)"] = "mm_theta"

    def run(fixed=(), model=None, sized_=sized):
        s = cases.sample(case, depth_mm * 1e-3, n, seed=1, fixed=fixed)
        s = mm.add_inputs(s, prior, sized_, seed=1, fixed=fixed, model=model)
        return life.chunked(lambda x: mm.mm_life(case, x), s)

    st, red = life.u90_reduction(run, inputs)
    base = run()
    u0 = life.u90(base)
    per_model = {mm.MODELS[k]: dict(stats=life.stats(run(model=k)),
                                    reduction=float(1.0 - life.u90(run(model=k)) / u0))
                 for k in mm.CRITERIA}
    red["mixed-mode criterion (mean over the three)"] = float(
        np.mean([v["reduction"] for v in per_model.values()]))
    out = dict(case=case, depth_mm=depth_mm, prior=prior, sized=sized, n=n,
               stats=st, reduction=red, per_model=per_model)
    if not sized:   # value of UT-sizing the tilt: sd 15 -> 5 deg
        out["sizing_reduction"] = float(1.0 - life.u90(run(sized_=True)) / u0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()
    s1 = step1()
    runner.save("mixed_mode_step1.json", dict(results=s1, models=mm.MODELS))
    tasks = [(c, d, p, z, args.n) for p in mm.PRIORS for z in (False, True)
             for c in ("bar", "plate") for d in runner.DEPTHS_MM[c]]
    res = runner.run_pool(step2_task, tasks)
    print(runner.save("mixed_mode.json", dict(results=res, priors=mm.PRIORS)))
    for r in res:
        top = sorted(r["reduction"].items(), key=lambda kv: -kv[1])[:4]
        print(f"{r['prior']:8s} {'sized' if r['sized'] else 'unknown':7s} {r['case']:5s} {r['depth_mm']:.1f} mm"
              f"  U90 {r['stats']['u90']:.2f} | " + ", ".join(f"{k} {v:.0%}" for k, v in top)
              + (f" | UT sizing {r['sizing_reduction']:.0%}" if "sizing_reduction" in r else ""))


if __name__ == "__main__":
    main()
