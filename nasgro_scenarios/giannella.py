"""Giannella (2022, Int J Fract 235:179-195) round bar in this engine, and the bridge to this study's bar.

Giannella's case: 25CrMo4 bar, D = 18 mm, R = -1, 250 MPa, NASGRO with Newman
closure (Table 2), James-Mills SIF (identical to geometry.bar_F), failure at
a = D/2. Uncertain inputs (Table 4): crack depth N(1, 0.15 mm), diameter and
load amplitude with CoV 1.5 %, and the material: (C, m, p, dKth0) sets drawn
uniformly in the Table 3 ranges and kept when their growth-rate curve fits the
test data with both regime RMS errors below 3 %.

The test data are not published, so a stand-in is generated about the
paper's best-fit curve. `POOLS` keeps two reconstructions:
  * "original", the earlier reproduction: points log-uniform in dK down to
    1.02 x the best-fit threshold, a strict acceptance rule (a curve that
    predicts no growth at a data point is rejected), and q = 1. Its accepted
    dKth0 sits at 5.9-7.2 MPa sqrt(m), below the paper's 6.6-7.6 (Fig. 8), so
    too many cracks grow at 150 MPa (Pf 68 % against 17 %).
  * "corrected": a load-shedding threshold "wall" of points from 1e-11 to
    1e-9 m/cycle, as the paper's Fig. 7 shows; the acceptance arithmetic of a
    direct MATLAB implementation, in which a curve with its threshold above a
    data point gets a complex RMS and is judged on its real part (Fig. 7 shows
    accepted curves to the right of data points); and the printed q = 5,
    which reproduces the paper's stress dependence (Figs 10-11). Three
    stand-in settings were tuned to the paper's stress-level statistics
    (Figs 10-11): data level 0.1 decades below the best fit, Paris-regime
    scatter 0.20 decades, 400 Paris points. Two settings fitted those equally
    well; the Fig. 12 crack-depth sweep chose between them.

`bridge_sample` moves from Giannella's setup to this study's bar one
assumption at a time, on common random numbers.
"""
import os

import numpy as np
from scipy.special import ndtr

from . import cases, nasgro

HERE = os.path.dirname(os.path.abspath(__file__))
R = -1.0
LONG_CRACK = 0.05            # m; crack length given to test data, so the short-crack term vanishes

# Table 2 (q as printed; the "original" pool used q = 1)
FIXED = dict(Kc=131.56, alpha=3.0, q=5.0, a0=3.81e-5, smax_sigma0=0.32, cth_pos=0.115, cth_neg=0.115)
BEST_FIT = dict(C=10 ** -9.814, n=1.796, p=0.718, dKth0=7.192)          # Fig. 8
PRIOR = dict(log10C=(-12.0, -6.9), n=(0.0, 4.0), p=(0.0, 4.4), dKth0=(5.8, 8.0))   # Table 3
NOMINAL = dict(D=0.018, F=63.6e3, a_sd=0.15e-3, D_cov=0.015, F_cov=0.95 / 63.6)  # Table 4

# Published values: (mean, sd) of log10 Nf over propagating cracks, and Pf
PAPER_STRESS = {150: (6.672, 0.646, 0.17), 200: (5.802, 0.194, 0.97), 250: (5.422, 0.118, 1.00),
                300: (5.174, 0.124, 1.00), 400: (4.792, 0.164, 1.00), 500: (4.498, 0.201, 1.00)}
# Figs 12-14: sd of log10 Nf at the Table 4 spread and at the smallest spread swept
PAPER_SWEEP = {"crack depth": (0.118, 0.086), "diameter D": (0.117, 0.109), "load amplitude": (0.118, 0.116)}
PAPER_CORR = {"C-n": -0.99, "C-p": 0.88, "C-dK": -0.47, "n-p": -0.81, "n-dK": 0.40, "p-dK": -0.70}

POOLS = {
    "original": dict(data="log-uniform", data_kw=dict(dk_min_factor=1.02), rule="strict", q=1.0),
    "corrected": dict(data="wall", rule="matlab", q=5.0,
                      data_kw=dict(wall_rates=(-11.0, -9.0), wall_points=40, wall_scatter=0.0,
                                   points=400, scatter_paris=0.20,
                                   generator=dict(C=BEST_FIT["C"] * 10 ** -0.1))),
}


def _closure():
    f, a0 = nasgro.newman_closure(np.array(R), FIXED["alpha"], FIXED["smax_sigma0"])
    return float(f), float(a0)


def long_crack_threshold(dk0):
    f, a0 = _closure()
    return nasgro.threshold(np.array(R), f, a0, LONG_CRACK, dk0, FIXED["a0"], FIXED["cth_pos"], FIXED["cth_neg"])


def log_rate(dk, log_c, n, p, dk0, q=None):
    """log10 da/dN at full-range dK (R = -1, long crack) for parameter arrays; -inf below threshold."""
    f, _ = _closure()
    dkth = long_crack_threshold(np.asarray(dk0))[:, None]
    near_thr = np.clip(1.0 - dkth / dk[None, :], 0.0, None)
    near_crit = 1.0 - dk / (1.0 - R) / FIXED["Kc"]
    with np.errstate(divide="ignore"):
        return (log_c[:, None] + n[:, None] * np.log10((1.0 - f) * dk / (1.0 - R) * 1.0)[None, :]
                + p[:, None] * np.log10(near_thr) - (FIXED["q"] if q is None else q) * np.log10(near_crit)[None, :])


def synthetic_data(rng, dk_min_factor=1.02, points=150, scatter_threshold=0.25, scatter_paris=0.12,
                   dk_split=20.0, dk_max=90.0, generator=None, q=None):
    """Stand-in for the paper's Fig. 3 data: points scattered about a generating curve.

    The generating curve is the paper's best fit unless `generator` overrides
    some of its parameters (C, n, p, dKth0).
    """
    gen = {**BEST_FIT, **(generator or {})}
    dk_th = float(long_crack_threshold(np.array([gen["dKth0"]]))[0])
    dk = np.concatenate([np.exp(rng.uniform(np.log(dk_min_factor * dk_th), np.log(dk_split), points)),
                         np.exp(rng.uniform(np.log(dk_split), np.log(dk_max), points))])
    b = {k: np.array([v]) for k, v in gen.items()}
    y = log_rate(dk, np.log10(b["C"]), b["n"], b["p"], b["dKth0"], q)[0]
    near = dk < dk_split
    return dk, y + rng.normal(0.0, np.where(near, scatter_threshold, scatter_paris)), near


def wall_data(rng, wall_points=100, wall_rates=(-11.0, -9.0), wall_scatter=0.02, points=150,
              dk_split=20.0, dk_max=60.0, scatter_paris=0.12, generator=None, q=None):
    """Stand-in data with a load-shedding threshold wall, as in the paper's Fig. 7.

    Threshold tests step dK down until da/dN falls to about 1e-11 m/cycle, so
    near-threshold points spread evenly in log da/dN at almost constant dK.
    Wall points: log10 da/dN uniform in `wall_rates`, dK from the generating
    curve with lognormal scatter `wall_scatter` (natural log) in dK. Paris
    points: log-uniform dK from the top of the wall to `dk_max`, scatter in
    rate. Both sets are scored as in the paper (near threshold below dk_split).
    """
    gen = {**BEST_FIT, **(generator or {})}
    b = {k: np.array([v]) for k, v in gen.items()}
    curve = lambda dk: log_rate(dk, np.log10(b["C"]), b["n"], b["p"], b["dKth0"], q)[0]   # noqa: E731
    grid = np.exp(np.linspace(np.log(float(long_crack_threshold(b["dKth0"])[0]) * 1.0001), np.log(dk_max), 4000))
    lr = curve(grid)
    target = rng.uniform(*wall_rates, wall_points)
    dk_wall = np.interp(target, lr, grid) * np.exp(rng.normal(0.0, wall_scatter, wall_points))
    y_wall = target
    dk_top = float(np.interp(wall_rates[1], lr, grid))
    dk_p = np.exp(rng.uniform(np.log(dk_top), np.log(dk_max), points))
    y_p = curve(dk_p) + rng.normal(0.0, scatter_paris, points)
    dk = np.concatenate([dk_wall, dk_p])
    return dk, np.concatenate([y_wall, y_p]), dk < dk_split


def log_rate_complex(dk, log_c, n, p, dk0, q=None):
    """log10 da/dN as MATLAB evaluates it: below threshold (1 - dKth/dK)^p is complex, not zero."""
    f, _ = _closure()
    dkth = long_crack_threshold(np.asarray(dk0))[:, None]
    near_thr = np.power((1.0 - dkth / dk[None, :]).astype(complex), p[:, None])
    near_crit = 1.0 - dk / (1.0 - R) / FIXED["Kc"]
    real = (log_c[:, None] + n[:, None] * np.log10((1.0 - f) * dk / (1.0 - R))[None, :]
            - (FIXED["q"] if q is None else q) * np.log10(near_crit)[None, :])
    return real + np.log10(near_thr)


def accept(data, n_accept, rng, tol=0.03, batch=20_000, max_proposals=60_000_000, rule="strict", q=None):
    """Uniform Table 3 proposals kept when both regime RMS errors of log10 da/dN ratio are below tol.

    rule="strict": a curve predicting no growth at a data point has infinite error and is rejected.
    rule="matlab": the arithmetic of a direct MATLAB implementation. Below threshold the rate is
    complex, so the RMS is complex, and MATLAB's `<` compares real parts only; such curves can
    pass. This lets accepted thresholds sit above the lowest data, as the paper's Figs 7-8 show.
    """
    dk, y, near = data
    kept, proposed = [], 0
    while sum(len(k[0]) for k in kept) < n_accept and proposed < max_proposals:
        lc = rng.uniform(*PRIOR["log10C"], batch)
        n, p, dk0 = (rng.uniform(*PRIOR[k], batch) for k in ("n", "p", "dKth0"))
        if rule == "matlab":
            rel = log_rate_complex(dk, lc, n, p, dk0, q) / y[None, :] - 1.0
            rms = lambda m: np.sqrt(np.mean(rel[:, m] ** 2, 1)).real      # noqa: E731
        else:
            rel = log_rate(dk, lc, n, p, dk0, q) / y[None, :] - 1.0
            rms = lambda m: np.sqrt(np.mean(rel[:, m] ** 2, 1))            # noqa: E731
        with np.errstate(invalid="ignore"):
            ok = (rms(near) < tol) & (rms(~near) < tol)
        kept.append((lc[ok], n[ok], p[ok], dk0[ok]))
        proposed += batch
    lc, n, p, dk0 = (np.concatenate(v)[:n_accept] for v in zip(*kept))
    return dict(C=10.0 ** lc, n=n, p=p, dKth0=dk0, rate=len(lc) / proposed)


def pool(name="corrected", n_accept=20_000, seed=7):
    """Accepted material sets, cached in nasgro_scenarios/data."""
    path = os.path.join(HERE, "data", f"giannella_pool_{name}_seed{seed}_n{n_accept}.npz")
    if name == "original" and not os.path.exists(path):
        path = os.path.join(HERE, "data", f"giannella_pool_q1_seed{seed}_n{n_accept}.npz")  # earlier cache
    if os.path.exists(path):
        z = np.load(path)
        return {"C": z["C"], "n": z["n"], "p": z["p"], "dKth0": z["dKth0" if "dKth0" in z else "dK0"],
                "rate": float(z["rate"])}
    cfg = POOLS[name]
    rng = np.random.default_rng(seed)
    make = wall_data if cfg["data"] == "wall" else synthetic_data
    out = accept(make(rng, q=cfg["q"], **cfg["data_kw"]), n_accept, rng, rule=cfg["rule"], q=cfg["q"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    return out


def pool_corr(m):
    x = np.corrcoef([np.log10(m["C"]), m["n"], m["p"], m["dKth0"]])
    return {"C-n": x[0, 1], "C-p": x[0, 2], "C-dK": x[0, 3], "n-p": x[1, 2], "n-dK": x[1, 3], "p-dK": x[2, 3]}


# Bridge steps: each changes one assumption from Giannella's setup towards this study's bar.
STEPS = [
    ("Giannella's setup", dict(material="giannella", a_sd=0.15e-3, F_cov=NOMINAL["F_cov"], extras=False)),
    ("+ crack depth sd 0.15 -> 0.5 mm (UT sizing)", dict(a_sd=0.5e-3)),
    ("+ load CoV 1.5 % -> 10 %", dict(F_cov=0.10)),
    ("+ Kc, a0, R and tilt uncertain", dict(extras=True)),
    ("+ this study's material model", dict(material="study")),
]


def step_configs():
    cfg, out = {}, []
    for label, change in STEPS:
        cfg = {**cfg, **change}
        out.append((label, dict(cfg)))
    return out


def bridge_sample(cfg, n, seed=1, fixed=(), stress=250.0, pool_name="corrected", a_nom=1e-3):
    """Samples for one bridge configuration; inputs named in `fixed` sit at nominal.

    Keys in `fixed`: depth, material, D, F, and (when extras) Kc, a0, R, tilt.
    kappa and the SIF error are always fixed (NASGRO-equation inputs only).
    """
    fx = set(fixed) | {"kappa", "sif"}
    if not cfg["extras"]:
        fx |= {"Kc", "a0", "R", "tilt"}
    if cfg["material"] == "giannella":
        fx_s = fx | {"material"}
    else:
        fx_s = fx
    f_nom = stress * 1e6 * np.pi * NOMINAL["D"] ** 2 / 4.0    # N, from MPa and m^2
    s = cases.sample("bar", a_nom, n, seed=seed, fixed=tuple(fx_s), nominal=dict(F=f_nom))
    # crack depth and load with this configuration's spreads, on the study's streams
    if "depth" not in fx:
        a = a_nom + cfg["a_sd"] * cases._z("depth", n, seed)
        s["a_i"] = np.where(a > 2e-5, a, 2e-5 + np.abs(a - 2e-5))
    s["F"] = np.full(n, f_nom) if "F" in fx else \
        np.clip(f_nom * (1.0 + cfg["F_cov"] * cases._z("F", n, seed)), 0.1 * f_nom, None)
    if cfg["material"] == "giannella":
        for k, v in FIXED.items():
            if k in ("Kc", "a0") and k not in fx:      # extras: keep the study's spread, about Giannella's value
                s[k] = s[k] / cases.BAR[k] * v
            else:
                s[k] = np.full(n, v)
        s["a_lim_ratio"] = np.full(n, 0.5)
        s["q"] = np.full(n, POOLS[pool_name]["q"])
        if "material" in fx:
            for k, v in BEST_FIT.items():
                s[k] = np.full(n, v)
        else:
            m = pool(pool_name)
            idx = np.minimum((ndtr(cases._z("material", n, seed)) * len(m["C"])).astype(int), len(m["C"]) - 1)
            for k in BEST_FIT:
                s[k] = m[k][idx]
    return s
