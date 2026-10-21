"""Variable-amplitude scenario module (separate from the base case).

Stage 1: repeating block, no load interaction, every depth, full sample count.
Stage 2: the same block with generalised Willenborg retardation, at a reduced
sample count (cycle-block integration is ~100x slower).
Each input, including the spectrum shape, block length, shut-off ratio and
overload plastic-zone factor, is ranked by the U90 reduction when known exactly.

    python scripts/run_variable_amplitude.py [--n1 10000] [--n2 3000] [--stage 1|2|both]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import cases, life, runner  # noqa: E402
from nasgro_scenarios import variable_amplitude as va  # noqa: E402

STAGE2_DEPTHS_MM = {"bar": [1.0, 4.0], "plate": [0.5, 1.0, 4.0]}


def task(stage, case, depth_mm, n):
    fn_life = va.stage1_life if stage == 1 else va.stage2_life
    extra = va.STAGE1_INPUTS if stage == 1 else va.STAGE2_INPUTS
    size = 500 if stage == 1 else n

    def fn(fixed):
        s = cases.sample(case, depth_mm * 1e-3, n, seed=1, fixed=fixed)
        s = va.add_inputs(s, case, seed=1, fixed=fixed)
        return life.chunked(lambda x: fn_life(case, x), s, size=size)

    out = runner.rank_task(case, depth_mm, fn, extra_inputs=extra, n=n)
    out["stage"] = stage
    if stage == 2:   # stage-1 stats on the same samples, to isolate retardation
        s = va.add_inputs(cases.sample(case, depth_mm * 1e-3, n, seed=1), case, seed=1)
        out["stage1_stats"] = life.stats(life.chunked(lambda x: va.stage1_life(case, x), s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", type=int, default=10000)
    ap.add_argument("--n2", type=int, default=3000)
    ap.add_argument("--stage", default="both")
    args = ap.parse_args()
    tasks = []
    if args.stage in ("1", "both"):
        tasks += [(1, c, d, args.n1) for c in ("bar", "plate") for d in runner.DEPTHS_MM[c]]
    if args.stage in ("2", "both"):
        tasks += [(2, c, d, args.n2) for c in ("bar", "plate") for d in STAGE2_DEPTHS_MM[c]]
    tasks.sort(key=lambda t: -t[0])          # slow stage-2 tasks first
    res = runner.run_pool(task, tasks)
    name = "variable_amplitude.json" if args.stage == "both" else f"variable_amplitude_stage{args.stage}.json"
    print(runner.save(name, dict(results=res, spectrum=va.SPECTRUM)))
    for r in res:
        top = sorted(r["reduction"].items(), key=lambda kv: -kv[1])[:4]
        st, b = r["stats"], r["baseline_stats"]
        print(f"stage {r['stage']} {r['case']:5s} {r['depth_mm']:.1f} mm  median {b['median']:.2f}->{st['median']:.2f}"
              f"  Pf {b['pf']:.0%}->{st['pf']:.0%}  U90 {b['u90']:.2f}->{st['u90']:.2f} | "
              + ", ".join(f"{k} {v:.0%}" for k, v in top))


if __name__ == "__main__":
    main()
