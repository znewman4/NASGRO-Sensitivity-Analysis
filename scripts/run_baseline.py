"""Baseline study: propagate uncertainty, check numerics, and rank inputs.

Usage:
    python scripts/run_baseline.py [--n-mc 20000] [--n-sobol 2048] [--states 0.3 1 3] [--seed 1]

--states repeats the analysis at several nominal initial crack depths (mm).
Decision horizons default to each state's own RUL quantiles so the failure
indicators stay informative; pass --horizons to fix them in cycles instead.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from nasgro_sa import baseline_formulation, baseline_inputs, rul_quantile
from nasgro_sa.studies import growth_rate_study, integration_convergence, sobol_study, summarise

MM = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mc", type=int, default=20_000)
    ap.add_argument("--n-sobol", type=int, default=2048)
    ap.add_argument("--horizons", type=float, nargs="+", default=None,
                    help="absolute decision horizons (cycles); default: per-state RUL quantiles")
    ap.add_argument("--horizon-quantiles", type=float, nargs="+", default=[0.05, 0.25, 0.5])
    ap.add_argument("--min-events", type=int, default=50,
                    help="flag indicator outputs with fewer expected events (or non-events) in A and B")
    ap.add_argument("--states", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    form, base = baseline_formulation(), baseline_inputs()

    for depth in args.states:
        spec = base.with_nominal(a_init=depth * MM)
        print(f"\n=== Nominal initial depth {depth:g} mm ===")

        t = time.perf_counter()
        x = spec.sample(args.n_mc, rng)
        life = form.run(x)
        horizons = args.horizons or [float(f"{h:.3g}") for h in rul_quantile(life.rul, args.horizon_quantiles)]
        horizons = [h for h in horizons if np.isfinite(h)]
        s = summarise(life, horizons)
        print(f"Propagation, N = {s.n} ({time.perf_counter() - t:.1f} s)")
        print("  outcomes: " + ", ".join(f"{k} {v:.3f}" for k, v in s.fractions.items() if v))
        for h, p, se in zip(horizons, s.p_fail, s.p_fail_se):
            print(f"  P(RUL <= {h:.3g}) = {p:.4f} +/- {se:.4f}")
        print("  RUL quantiles 1/5/10/50%: " + ", ".join(f"{q:.4g}" for q in s.quantiles))

        sub = {k: v[:1000] for k, v in x.items()}
        print("Integration convergence (step-control factor, max rel. life error vs finest, outcome mismatch):")
        for f, err, mis in integration_convergence(form, sub):
            print(f"  {f:<5g} {err:.2e}  {mis:.3f}")

        t = time.perf_counter()
        study = sobol_study(form, spec, args.n_sobol, horizons, rng)
        res = study.sobol
        print(f"Sobol indices, N = {args.n_sobol} x {len(res.groups) + 2} runs ({time.perf_counter() - t:.1f} s)")
        for j in range(len(res.outputs)):
            events = 2 * args.n_sobol * min(res.mean[j], 1 - res.mean[j])
            if events < args.min_events:
                print(f"{res.outputs[j]}: only ~{events:.0f} events (mean {res.mean[j]:.3g}); "
                      f"indices not estimable at this N\n")
            else:
                print(res.table(j), end="\n\n")
        print(study.ks(rng=rng).table(label="RUL (censored)"), end="\n\n")

        c = depth * MM / spec.marginals["aspect_init"].nominal
        rate_res, below = growth_rate_study(form, spec, [(depth * MM, c)], args.n_sobol, rng)
        print(f"Growth rate at the matched nominal state ({below[0]:.1%} of samples below threshold)")
        print(rate_res.table(0), end="\n\n")


if __name__ == "__main__":
    main()
