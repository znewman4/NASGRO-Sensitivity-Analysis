"""Do the rankings depend on the assumed input distributions?

Repeats the baseline Sobol study under alternative spreads, shapes and C-n
correlations, at fixed decision horizons (the baseline's RUL quantiles),
and reports Kendall tau between each variant's total-effect ranking and the
baseline's.

Usage:
    python scripts/run_robustness.py [--n-sobol 2048] [--depth 1] [--seed 3]
"""
from __future__ import annotations

import argparse

import numpy as np

from nasgro_sa import baseline_formulation, baseline_inputs, rul_quantile
from nasgro_sa.studies import distribution_variants, matrix_table, rank_agreement, sobol_study

MM = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sobol", type=int, default=2048)
    ap.add_argument("--n-mc", type=int, default=20_000)
    ap.add_argument("--depth", type=float, default=1.0, help="nominal initial crack depth (mm)")
    ap.add_argument("--horizon-quantiles", type=float, nargs="+", default=[0.05, 0.5])
    ap.add_argument("--min-events", type=int, default=50,
                    help="blank out variants with fewer expected events (or non-events) in A and B")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    form, spec = baseline_formulation(), baseline_inputs().with_nominal(a_init=args.depth * MM)
    life = form.run(spec.sample(args.n_mc, rng))
    horizons = [float(f"{h:.3g}") for h in rul_quantile(life.rul, args.horizon_quantiles)]
    print(f"Decision horizons (baseline RUL quantiles {args.horizon_quantiles}): {horizons}")

    variants = distribution_variants(spec)
    results = {name: sobol_study(form, v, args.n_sobol, horizons, rng).sobol for name, v in variants.items()}
    groups = results["baseline"].groups
    for j, label in enumerate(results["baseline"].outputs[:-1]):
        mean = np.array([r.mean[j] for r in results.values()])
        estimable = 2 * args.n_sobol * np.minimum(mean, 1 - mean) >= args.min_events
        st = np.column_stack([r.total[:, j] if ok else np.full(len(groups), np.nan)
                              for r, ok in zip(results.values(), estimable)])
        print("\n" + matrix_table(groups, list(results), st, title=f"Total-effect indices, {label}"))
        tau = np.where(estimable, list(rank_agreement(results, output=j).values()), np.nan)
        print(matrix_table(["P(fail)", "Kendall tau"], list(results), np.array([mean, tau])))
        if not estimable.all():
            print(f"(nan: fewer than {args.min_events} expected events at this horizon; indices not estimable)")


if __name__ == "__main__":
    main()
