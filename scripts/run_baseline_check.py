"""Rebuilt constant-amplitude baseline: U90 ranking by crack depth (cf. Figure 2).

The scenario scripts compare against this. It also checks the rebuild against
the study's published 1 mm statistics.

    python scripts/run_baseline_check.py [--n 10000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import cases, life, runner  # noqa: E402

# Published values in the study document (Figures A1-A3 and Figure 2, 1 mm crack).
PUBLISHED = {
    "bar": dict(mu=5.59, sigma=0.269, pf=0.93,
                top=["crack depth 26%", "material fit 10%", "kappa 10%", "load amplitude 9%"]),
    "plate": dict(mu=4.41, sigma=0.374, pf=0.99,
                  top=["crack depth 29%", "stress amplitude 8%", "kappa 6%", "alpha 3%"]),
}


def task(case, depth_mm, n):
    def fn(fixed):
        s = cases.sample(case, depth_mm * 1e-3, n, seed=1, fixed=runner.fx(fixed))
        return life.chunked(lambda x: life.ca_life(case, x), s)
    return runner.rank_task(case, depth_mm, fn, n=n, with_baseline=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--standard", action="store_true",
                    help="NASGRO-only: kappa and SIF error fixed at 1, results to the dated folder")
    args = ap.parse_args()
    runner.set_standard(args.standard)
    tasks = [(c, d, args.n) for c in ("bar", "plate") for d in runner.DEPTHS_MM[c]]
    res = runner.run_pool(task, tasks)
    print(runner.save("baseline.json", dict(results=res, published=PUBLISHED)))
    for r in res:
        top = sorted(r["reduction"].items(), key=lambda kv: -kv[1])[:4]
        st = r["stats"]
        print(f"{r['case']:5s} {r['depth_mm']:.1f} mm  mu {st['mu']:.2f}  sigma {st['sigma']:.3f}"
              f"  Pf {st['pf']:.0%}  U90 {st['u90']:.2f} | "
              + ", ".join(f"{k} {v:.0%}" for k, v in top))


if __name__ == "__main__":
    main()
