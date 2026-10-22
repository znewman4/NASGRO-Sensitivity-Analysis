"""Does the sensitivity ranking depend on the growth-model formulation?

NASGRO, Walker and Paris are each calibrated to the same lots of da/dN data
(synthetic until real data are supplied). Each model's parameters form one
"growth" group, so rankings are compared over identical group labels:
the physical inputs plus the growth-rate curve. Paris has no stress-ratio
term and is fitted only to data at the forecast stress ratio. Decision
horizons are the reference (NASGRO) model's RUL quantiles, applied to all.

Usage:
    python scripts/run_model_comparison.py [--n-sobol 2048] [--lots 40] [--depth 1] [--seed 2]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from nasgro_sa import NasgroEquation, baseline_formulation, baseline_inputs, rul_quantile
from nasgro_sa.calibrate import calibrate, standard_fit_specs, synthetic_lots
from nasgro_sa.studies import matrix_table, sobol_study, summarise

MM = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sobol", type=int, default=2048)
    ap.add_argument("--n-mc", type=int, default=20_000)
    ap.add_argument("--lots", type=int, default=40)
    ap.add_argument("--depth", type=float, default=1.0, help="nominal initial crack depth (mm)")
    ap.add_argument("--horizon-quantiles", type=float, nargs="+", default=[0.05, 0.25, 0.5])
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    base = baseline_inputs().with_nominal(a_init=args.depth * MM)
    forecast_r = base.marginals["R"].nominal
    lots = synthetic_lots(NasgroEquation(), base, args.lots, rng)
    print(f"{args.lots} synthetic lots from the baseline NASGRO population; "
          f"{lots[0].dk.size} points each at R = {sorted(set(lots[0].r))}")

    geometry_form = baseline_formulation()
    models = {}
    for name, fit in standard_fit_specs(base, forecast_r).items():
        cal = calibrate(fit, lots)
        models[name] = (replace(geometry_form, rate_model=fit.model), cal.input_spec(base))
        est = np.median(cal.estimates, axis=0)
        print(f"  {name:<7} median residual sd {np.median(cal.residual_sd):.3f} decades; median fit "
              + ", ".join(f"{k}={10**v if k in fit.log10 else v:.3g}" for k, v in zip(fit.free, est)))

    ref_form, ref_spec = models["NASGRO"]
    ref_life = ref_form.run(ref_spec.sample(args.n_mc, rng))
    horizons = [float(f"{h:.3g}") for h in rul_quantile(ref_life.rul, args.horizon_quantiles)]
    print(f"\nDecision horizons (NASGRO RUL quantiles {args.horizon_quantiles}): {horizons}")

    results = {}
    for name, (form, spec) in models.items():
        s = summarise(form.run(spec.sample(args.n_mc, rng)), horizons)
        print(f"  {name:<7} P(RUL <= h) = " + ", ".join(f"{p:.3f}" for p in s.p_fail)
              + "; outcomes " + ", ".join(f"{k} {v:.3f}" for k, v in s.fractions.items() if v))
        results[name] = sobol_study(form, spec, args.n_sobol, horizons, rng).sobol

    groups = results["NASGRO"].groups
    for j, label in enumerate(results["NASGRO"].outputs[:-1]):
        st = np.column_stack([results[m].total[[results[m].groups.index(g) for g in groups], j] for m in results])
        print("\n" + matrix_table(groups, list(results), st, title=f"Total-effect indices, {label}"))


if __name__ == "__main__":
    main()
