"""Sensitivity study with only the NASGRO-equation inputs uncertain.

The crack-opening factor kappa and the SIF-solution error are not terms of the
NASGRO equation; here they are fixed at their nominal value of 1 (not removed
from the code path), so every other input keeps exactly the same random draws.
The run with them ("extended") is repeated on the same seeds for comparison.

Writes to results/2026-10-23_nasgro_only/ and leaves results/ untouched:
  rankings.json   U90 reduction by input and depth, both configurations, 5 seeds
  checks.json     sanity checks: median shift, stress spread, clipped draws
  doubling.json   judgement spreads (alpha, Smax/sigma0, a/c) doubled; bar with
                  alpha and Smax/sigma0 uncertain
  scenarios.json  the study's industrial scenarios (Question 3), NASGRO-only
  phases.json     share of life near threshold / Paris / near critical (Question 1)
  sweeps.json     Giannella-style spread levels of every input at 1 mm (Appendix A)

    python scripts/run_nasgro_only.py [--n 10000] [--seeds 5]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import cases, life, nasgro, runner  # noqa: E402

OUT = runner.STANDARD_DIR
CLOSURE = {"constraint factor alpha": "alpha", "Smax/sigma0": "smax_sigma0"}

# Multipliers on the typical spread at the catalogue's high and low ends (Figure 3).
HIGH = dict(depth=3.6, S=5.0, F=5.0, ac=1.5, alpha=1.5, smax_sigma0=1.5, q=1.67, p=1.67,
            cth_pos=1.67, cth_neg=1.67, Kc=3.0, a0=2.0, dKth0=2.0, Cn=2.0, material=2.0,
            D=2.0, t=2.0, W=2.0, R=3.0, tilt=2.0, kappa=2.0, sif=2.0)
LOW = dict(depth=0.3, S=0.15, F=0.15, ac=0.5, alpha=0.5, smax_sigma0=0.5, q=0.5, p=0.5,
           cth_pos=0.33, cth_neg=0.33, Kc=0.5, a0=0.5, dKth0=0.5, Cn=0.33, material=0.33,
           D=0.33, t=0.33, W=0.33, R=0.4, tilt=0.4, kappa=0.5, sif=0.6)

BAR_AREA = np.pi * cases.BAR["D"] ** 2 / 4.0
SCENARIOS = {   # (case, nominal overrides, spread multipliers, label)
    "bar_low_stress": ("bar", dict(F=200e6 * BAR_AREA), {}, "Low stress (bar 200 MPa)"),
    "bar_high_stress": ("bar", dict(F=400e6 * BAR_AREA), {}, "High stress (bar 400 MPa)"),
    "bar_large": ("bar", dict(D=0.030, F=250e6 * np.pi * 0.030**2 / 4.0), {}, "Larger section (bar D = 30 mm)"),
    "bar_poorly_known": ("bar", {}, HIGH, "Poorly known (all spreads at the high end)"),
    "bar_well_known": ("bar", {}, LOW, "Well known (all spreads at the low end)"),
    "plate_low_stress": ("plate", dict(S=120.0), {}, "Low stress (plate 120 MPa)"),
    "plate_high_R": ("plate", dict(R=0.5), {}, "High stress ratio (plate R = 0.5)"),
    "plate_high_stress": ("plate", dict(S=280.0), {}, "High stress (plate 280 MPa)"),
    "plate_slender": ("plate", dict(ac=0.3), {}, "Slender crack (plate a/c = 0.3)"),
    "plate_thick": ("plate", dict(t=0.020), {}, "Thicker section (plate t = 20 mm)"),
    "plate_poorly_known": ("plate", {}, HIGH, "Poorly known (all spreads at the high end)"),
    "plate_well_known": ("plate", {}, LOW, "Well known (all spreads at the low end)"),
}
DOUBLING = {   # (case, spread, extra ranked inputs, label)
    "plate_alpha_x2": ("plate", dict(alpha=2.0), {}, "alpha spread x2"),
    "plate_smax_x2": ("plate", dict(smax_sigma0=2.0), {}, "Smax/sigma0 spread x2"),
    "plate_ac_x2": ("plate", dict(ac=2.0), {}, "a/c spread x2"),
    "plate_all_x2": ("plate", dict(alpha=2.0, smax_sigma0=2.0, ac=2.0), {}, "alpha, Smax/sigma0 and a/c x2"),
    "bar_closure_x1": ("bar", dict(alpha=1.0, smax_sigma0=1.0), CLOSURE,
                       "bar with alpha (CoV 0.19) and Smax/sigma0 (CoV 0.2) uncertain"),
    "bar_closure_x2": ("bar", dict(alpha=2.0, smax_sigma0=2.0), CLOSURE,
                       "bar with alpha and Smax/sigma0 uncertain, spreads x2"),
}
SWEEP_LEVELS = (("x2", 2.0), ("typical", 1.0), ("x1/3", 1 / 3), ("x1/30", 1 / 30))


def _lives(case, depth_mm, n, seed, fixed=(), spread=None, nominal=None):
    s = cases.sample(case, depth_mm * 1e-3, n, seed=seed, fixed=fixed, spread=spread, nominal=nominal)
    return life.chunked(lambda x: life.ca_life(case, x), s)


def rank(case, depth_mm, n, seed, always=(), spread=None, nominal=None, extra=None):
    inputs = dict(cases.BAR_INPUTS if case == "bar" else cases.PLATE_INPUTS)
    inputs = {k: v for k, v in inputs.items() if v not in always}
    inputs.update(extra or {})
    fn = lambda fixed: _lives(case, depth_mm, n, seed, tuple(fixed) + tuple(always), spread, nominal)  # noqa: E731
    st, red = life.u90_reduction(fn, inputs)
    return dict(case=case, depth_mm=depth_mm, seed=seed, n=n, stats=st, reduction=red)


def ranking_task(config, case, depth_mm, n, seed):
    always = runner.NON_NASGRO if config == "standard" else ()
    return dict(config=config, **rank(case, depth_mm, n, seed, always))


def check_task(case, depth_mm, n):
    out = dict(case=case, depth_mm=depth_mm)
    for config, always in (("extended", ()), ("standard", runner.NON_NASGRO)):
        s = cases.sample(case, depth_mm * 1e-3, n, seed=1, fixed=always)
        L = life.chunked(lambda x: life.ca_life(case, x), s)
        smax = cases.max_stress(case, s)
        out[config] = dict(stats=life.stats(L), stress_mean=float(smax.mean()),
                           stress_cov=float(smax.std() / smax.mean()),
                           kappa_sd=float(s["kappa"].std()), sif_sd=float(s["sif"].std()))
    for label, sp in (("typical", {}), ("x2", dict(alpha=2.0, smax_sigma0=2.0))):
        s = cases.sample("plate", 1e-3, n, seed=1, spread=sp)
        out[f"plate_clipped_{label}"] = dict(
            alpha=float(np.mean((s["alpha"] <= 1.0) | (s["alpha"] >= 3.0))),
            smax_sigma0=float(np.mean((s["smax_sigma0"] <= 0.05) | (s["smax_sigma0"] >= 0.8))),
            ac=float(np.mean((s["ac"] <= 0.2) | (s["ac"] >= 1.0))))
    return out


def doubling_task(name, depth_mm, n):
    case, sp, extra, label = DOUBLING[name]
    return dict(variant=name, label=label,
                **rank(case, depth_mm, n, 1, runner.NON_NASGRO, spread=sp, extra=extra))


def scenario_task(name, depth_mm, n):
    case, nom, sp, label = SCENARIOS[name]
    return dict(scenario=name, label=label,
                **rank(case, depth_mm, n, 1, runner.NON_NASGRO, spread=sp, nominal=nom))


def phase_task(case, depth_mm, n):
    """Share of remaining life spent near threshold, in Paris growth and near Kc.

    Near threshold: the threshold term (1 - dKth/dK)^p at least halves the rate.
    Near critical: the toughness term 1/(1 - Kmax/Kc)^q at least doubles it.
    """
    s = cases.sample(case, depth_mm * 1e-3, n, seed=1, fixed=runner.NON_NASGRO)
    _, a, da = life.depth_grid(s["a_i"], cases.a_limit(case, s), 400)
    kmax = cases.k_unit(case, s, a) * cases.max_stress(case, s)[:, None]
    kmin = kmax * s["R"][:, None]
    p = cases.material(s)
    rate = nasgro.growth_rate(kmax, kmin, a, p)
    R = np.clip(kmin / kmax, nasgro.R_MIN, nasgro.R_MAX)
    f, A0 = nasgro.newman_closure(R, p["alpha"], p["smax_sigma0"])
    dkth = nasgro.threshold(R, f, A0, a, p["dKth0"], p["a0"], p["cth_pos"], p["cth_neg"])
    thr = np.clip(1.0 - dkth / ((1.0 - R) * kmax), 1e-12, None) ** p["p"]
    crit = 1.0 / np.clip(1.0 - kmax / p["Kc"], 1e-6, None) ** p["q"]
    L = life.integrate(rate, kmax, s["Kc"], da)
    fail = np.isfinite(L)
    live = np.cumsum(kmax >= s["Kc"][:, None], axis=1) == 0
    dn = np.where(live & (rate > 0), da / np.where(rate > 0, rate, 1.0), 0.0)
    near_thr = (thr < 0.5) & ~((crit > 2.0) & (np.log(crit) > -np.log(thr)))
    near_crit = (crit > 2.0) & ~near_thr
    tot = dn.sum(1)
    ok = fail & (tot > 0)                  # cracks critical at the start have no life to share
    sh = lambda m: float(np.mean((dn * m).sum(1)[ok] / tot[ok])) if ok.any() else np.nan  # noqa: E731
    return dict(case=case, depth_mm=depth_mm, never_grow=float(1 - fail.mean()),
                critical_at_start=float(np.mean(fail & (tot == 0))),
                near_threshold=sh(near_thr), near_critical=sh(near_crit),
                paris=sh(~near_thr & ~near_crit))


def sweep_task(case, key, n):
    cap = HIGH.get(key, 2.0)
    rows = []
    for label, m in SWEEP_LEVELS:
        mm = min(m, cap) if m > 1 else m
        sp = {"Cn" if key == "Cn" else key: mm}
        s = cases.sample(case, 1e-3, n, seed=1, fixed=runner.NON_NASGRO, spread=sp)
        L = life.chunked(lambda x: life.ca_life(case, x), s)
        rows.append(dict(level=label, multiplier=mm, **life.stats(L)))
    return dict(case=case, key=key, levels=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default="rankings,checks,doubling,scenarios,phases,sweeps")
    args = ap.parse_args()
    parts = args.only.split(",")
    os.makedirs(OUT, exist_ok=True)
    runner.RESULTS = OUT
    depths = runner.DEPTHS_MM
    if "checks" in parts:
        res = runner.run_pool(check_task, [(c, d, args.n) for c in depths for d in depths[c]])
        runner.save("checks.json", dict(results=res))
    if "rankings" in parts:
        tasks = [(cfg, c, d, args.n, sd) for sd in range(1, args.seeds + 1)
                 for cfg in ("extended", "standard") for c in depths for d in depths[c]]
        runner.save("rankings.json", dict(results=runner.run_pool(ranking_task, tasks)))
    if "doubling" in parts:
        tasks = [(nm, d, args.n) for nm, v in DOUBLING.items() for d in depths[v[0]]]
        runner.save("doubling.json", dict(results=runner.run_pool(doubling_task, tasks)))
    if "scenarios" in parts:
        tasks = [(nm, d, args.n) for nm, v in SCENARIOS.items() for d in depths[v[0]]]
        runner.save("scenarios.json", dict(results=runner.run_pool(scenario_task, tasks),
                                           high=HIGH, low=LOW))
    if "phases" in parts:
        ph_depths = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0]
        tasks = [(c, d, args.n) for c in depths for d in ph_depths]
        runner.save("phases.json", dict(results=runner.run_pool(phase_task, tasks)))
    if "sweeps" in parts:
        tasks = [(c, k, args.n) for c in depths
                 for k in runner.base_inputs(c).values() if k not in runner.NON_NASGRO]
        runner.save("sweeps.json", dict(results=runner.run_pool(sweep_task, tasks)))
    print(OUT)


if __name__ == "__main__":
    main()
