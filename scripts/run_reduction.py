"""Where does reducing input uncertainty stop paying off?

Greedy reduction path: at each stage the group whose spread cut (by
--factor) most raises the RUL lower quantile is chosen. Each row shows the
chosen group, the quantile after the cut, and the gain over the previous
stage; shrinking gains mark diminishing returns.

Usage:
    python scripts/run_reduction.py [--n 4000] [--q 0.05] [--stages 6] [--depths 0.3 1 3] [--seed 4]
"""
from __future__ import annotations

import argparse

import numpy as np

from nasgro_sa import baseline_formulation, baseline_inputs
from nasgro_sa.studies import lower_quantile_objective, reduction_path

MM = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--q", type=float, default=0.05, help="RUL lower quantile to maximise")
    ap.add_argument("--factor", type=float, default=0.5)
    ap.add_argument("--stages", type=int, default=6)
    ap.add_argument("--depths", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    form, base = baseline_formulation(), baseline_inputs()

    for depth in args.depths:
        spec = base.with_nominal(a_init=depth * MM)
        path = reduction_path(form, spec, args.n, lower_quantile_objective(args.q), rng, args.factor, args.stages)
        print(f"\n=== Nominal initial depth {depth:g} mm: {args.q:.0%} RUL quantile, spread cut x{args.factor:g} per stage ===")
        print(f"{'stage':<7}{'cut':<14}{'quantile':>12}{'gain':>10}   runner-up")
        prev = None
        for step in path:
            gain = "" if prev is None else f"{step.value / prev - 1:+.1%}"
            runner = ""
            if step.candidates:
                ranked = sorted(step.candidates.items(), key=lambda kv: -kv[1])
                runner = f"{ranked[1][0]} ({ranked[1][1] / prev - 1:+.1%})"
            print(f"{step.stage:<7}{step.reduced or '-':<14}{step.value:12.4g}{gain:>10}   {runner}")
            prev = step.value


if __name__ == "__main__":
    main()
