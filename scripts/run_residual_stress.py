"""Residual-stress scenario module (separate from the base case).

For each scenario in nasgro_scenarios.residual_stress.SCENARIOS and each crack
depth, superposes K_res(a) on the applied K and ranks every input, including the
residual-stress amplitude and decay length, by the U90 reduction when known
exactly. Same random numbers as the baseline, so differences are the scenario's.

    python scripts/run_residual_stress.py [--n 10000] [--only plate_weld,bar_tensile]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import cases, life, runner  # noqa: E402
from nasgro_scenarios import residual_stress as rs  # noqa: E402


def task(name, depth_mm, n):
    sc = rs.SCENARIOS[name]
    case = sc["case"]
    kfn = rs.k_res(case, sc["shape"])
    extra = dict(rs.EXTRA_INPUTS)
    if sc["L"] is None:
        extra.pop("residual stress decay length")

    def fn(fixed):
        s = cases.sample(case, depth_mm * 1e-3, n, seed=1, fixed=runner.fx(fixed))
        s = rs.add_inputs(s, sc, seed=1, fixed=runner.fx(fixed))
        return life.chunked(lambda x: life.ca_life(case, x, kfn), s, size=500)

    out = runner.rank_task(case, depth_mm, fn, extra_inputs=extra, n=n)
    out["scenario"] = name
    out["label"] = sc["label"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--only", default="")
    ap.add_argument("--standard", action="store_true",
                    help="NASGRO-only: kappa and SIF error fixed at 1, results to the dated folder")
    args = ap.parse_args()
    runner.set_standard(args.standard)
    names = args.only.split(",") if args.only else list(rs.SCENARIOS)
    tasks = [(nm, d, args.n) for nm in names for d in runner.DEPTHS_MM[rs.SCENARIOS[nm]["case"]]]
    res = runner.run_pool(task, tasks)
    print(runner.save("residual_stress.json", dict(results=res, scenarios=rs.SCENARIOS)))
    for r in res:
        top = sorted(r["reduction"].items(), key=lambda kv: -kv[1])[:4]
        st, b = r["stats"], r["baseline_stats"]
        print(f"{r['scenario']:16s} {r['depth_mm']:.1f} mm  median {b['median']:.2f}->{st['median']:.2f}"
              f"  Pf {b['pf']:.0%}->{st['pf']:.0%}  U90 {b['u90']:.2f}->{st['u90']:.2f} | "
              + ", ".join(f"{k} {v:.0%}" for k, v in top))


if __name__ == "__main__":
    main()
