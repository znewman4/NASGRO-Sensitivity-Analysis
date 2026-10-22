"""Reproduction case: Giannella (2022), "Uncertainty quantification in fatigue crack-growth
predictions", Int J Fract 235:179-195.

Round bar (D = 18 mm) with a straight-fronted crack under tension-compression
(R = -1), 25CrMo4 steel, NASGRO equation with Newman closure (Table 2).
Initial depth, diameter and load amplitude are normal (Table 4).

Material uncertainty follows the paper's procedure (Sect. 2): (C, m, p,
dKth0) are drawn uniformly within the Table 3 ranges, and a set is kept only
if its growth-rate curve fits the test data with RMS errors below 3 % in both
the near-threshold and Paris regimes. The paper's 25CrMo4 test data are not
available, so ``synthetic_test_data`` generates a stand-in set around the
paper's best-fit curve (Fig. 8). Its scatter, and the split between the two
regimes, are the only quantities here not taken from the paper.

Units: metres, cycles, MPa, and kN for the load amplitude.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .calibrate import LONG_CRACK
from .geometry import RoundBarStraightCrack
from .growth import NasgroEquation
from .inputs import EmpiricalGroup, InputSpec, Marginal
from .integrate import FailureCriterion, IntegrationSettings, LifeResult
from .model import Formulation

MM = 1e-3
R = -1.0

# Table 2 (nu in the table is Smax/sigma0; one Cth is given and applies at R = -1).
# Table 2 prints q = 5, but with q = 5 the best-fit curve gives log10 Nf = 5.335
# against the paper's 5.559 (Fig. 9); q = 1 gives 5.564. The reproduction
# therefore uses q = 1; Q_TABLE keeps the printed value for comparison.
Q_TABLE = 5.0
FIXED = dict(Kc=131.56, alpha=3.0, q=1.0, a0_intrinsic=3.81e-5, smax_sigma0=0.32, Cth_pos=0.115, Cth_neg=0.115)
# Fig. 8 best-fit values (C in m/cycle, MPa sqrt(m)); "n" is the paper's m
BEST_FIT = dict(C=10 ** -9.814, n=1.796, p=0.718, dK0=7.192)
# Table 3 sampling ranges
PRIOR = dict(log10C=(-12.0, -6.9), n=(0.0, 4.0), p=(0.0, 4.4), dK0=(5.8, 8.0))

# Table 4: mean and standard deviation of the normal inputs (mm, mm, kN)
NOMINAL = dict(a_init=(1.0, 0.15), D=(18.0, 0.27), F=(63.6, 0.95))
SOURCES = {"a_init": "crack size", "material": "material", "D": "geometry", "F": "loading"}

# Values reported in the paper: (mean, sd) of log10 Nf over propagating cracks, and Pf.
PAPER = {
    "baseline": (5.422, 0.118, 1.00),                                   # Fig. 9
    "stress": {150: (6.672, 0.646, 0.17), 200: (5.802, 0.194, 0.97),    # Fig. 11 (MPa)
               300: (5.174, 0.124, 1.00), 400: (4.792, 0.164, 1.00), 500: (4.498, 0.201, 1.00)},
    "a_init": {0.25: (5.433, 0.184, 0.99), 0.15: (5.422, 0.118, 1.00),  # Fig. 12 (sd in mm)
               0.05: (5.415, 0.081, 1.00), 0.005: (5.416, 0.086, 1.00)},
    "D": {0.6: (5.423, 0.155, 1.00), 0.27: (5.422, 0.117, 1.00),        # Fig. 13 (sd in mm)
          0.1: (5.422, 0.107, 1.00), 0.01: (5.422, 0.109, 1.00)},
    "F": {2.0: (5.423, 0.125, 1.00), 0.95: (5.422, 0.118, 1.00),        # Fig. 14 (sd in kN)
          0.5: (5.422, 0.116, 1.00), 0.05: (5.422, 0.116, 1.00)},
    "contributions": {"crack size": 0.37, "material": 0.31, "geometry": 0.22, "loading": 0.10},   # Fig. 15
}


def nominal_stress(x) -> dict[str, np.ndarray]:
    """sigma_nom = 4 F / (pi D^2), in MPa from F in kN and D in m."""
    return {"S_max": 4.0 * np.asarray(x["F"]) * 1e3 / (np.pi * np.asarray(x["D"]) ** 2) * 1e-6}


def formulation(**settings) -> Formulation:
    """Failure when the crack depth reaches half the diameter (Fig. 9) or Kmax reaches Kc."""
    return Formulation(RoundBarStraightCrack(), NasgroEquation(), FailureCriterion(depth_ratio=0.5),
                       IntegrationSettings(**{"horizon": 1e10, **settings}), derive=nominal_stress)


def _log_rate_terms(dk, n, p, dk0, fixed=FIXED):
    """log10 da/dN minus log10 C at long-crack R = -1 data points, for parameter arrays (n, p, dK0)."""
    model, params = NasgroEquation(), {**fixed, **BEST_FIT}
    kappa = model.threshold(R, LONG_CRACK, params)[0] / params["dK0"]   # dKth = kappa * dK0
    u = model.threshold(R, LONG_CRACK, params)[1]
    near_critical = np.log10(1.0 - dk / (1.0 - R) / fixed["Kc"])
    with np.errstate(divide="ignore", invalid="ignore"):
        near_threshold = np.log10(np.clip(1.0 - kappa * dk0[:, None] / dk[None, :], 0.0, None))
    return n[:, None] * np.log10(u * dk)[None, :] + p[:, None] * near_threshold - fixed["q"] * near_critical


def synthetic_test_data(rng: np.random.Generator, points: int = 150, scatter_threshold: float = 0.25,
                        scatter_paris: float = 0.12, dk_min_factor: float = 1.02, dk_split: float = 20.0,
                        dk_max: float = 90.0, fixed=FIXED):
    """Stand-in for the paper's Fig. 3 data: points scattered about the best-fit curve.

    Returns (dK, da/dN, near_threshold_mask). ``points`` per regime, log-uniform
    in dK from ``dk_min_factor`` times the best-fit threshold; regimes split at
    ``dk_split`` MPa sqrt(m), with lognormal scatter (sd in decades) per regime.
    A set whose threshold exceeds the lowest data point predicts zero growth
    there and is rejected, so the lowest point caps the accepted dKth0; 1.02
    best reproduces the paper's Fig. 8 correlation matrix.
    """
    dk_th = NasgroEquation().threshold(R, LONG_CRACK, {**fixed, **BEST_FIT})[0]
    dk = np.concatenate([np.exp(rng.uniform(np.log(dk_min_factor * dk_th), np.log(dk_split), points)),
                         np.exp(rng.uniform(np.log(dk_split), np.log(dk_max), points))])
    best = {k: np.array([v]) for k, v in BEST_FIT.items()}
    log_rate = np.log10(BEST_FIT["C"]) + _log_rate_terms(dk, best["n"], best["p"], best["dK0"], fixed)[0]
    near = dk < dk_split
    log_rate += rng.normal(0.0, np.where(near, scatter_threshold, scatter_paris))
    return dk, 10.0 ** log_rate, near


def accepted_material(data, n_accept: int, rng: np.random.Generator, tol: float = 0.03, fixed=FIXED,
                      batch: int = 20_000, max_proposals: int = 50_000_000) -> tuple[dict[str, np.ndarray], float]:
    """Uniform proposals within Table 3, kept when both regime RMS errors are below ``tol``.

    RMS error per regime, as in Giannella (2021, Eq. 5):
        sqrt(mean[(log10 da/dN_model / log10 da/dN_data - 1)^2]).
    Returns the accepted sets and the acceptance rate.
    """
    dk, rate, near = data
    y = np.log10(rate)
    kept, proposed = [], 0
    while sum(len(k[0]) for k in kept) < n_accept and proposed < max_proposals:
        log_c = rng.uniform(*PRIOR["log10C"], batch)
        n, p, dk0 = (rng.uniform(*PRIOR[k], batch) for k in ("n", "p", "dK0"))
        rel = (log_c[:, None] + _log_rate_terms(dk, n, p, dk0, fixed)) / y[None, :] - 1.0
        with np.errstate(invalid="ignore"):
            ok = ((np.sqrt(np.mean(rel[:, near] ** 2, axis=1)) < tol)
                  & (np.sqrt(np.mean(rel[:, ~near] ** 2, axis=1)) < tol))
        kept.append((log_c[ok], n[ok], p[ok], dk0[ok]))
        proposed += batch
    log_c, n, p, dk0 = (np.concatenate(v)[:n_accept] for v in zip(*kept))
    return {"C": 10.0 ** log_c, "n": n, "p": p, "dK0": dk0}, len(log_c) / proposed


PARIS_PRIOR = dict(log10C=(-14.0, -6.0), n=(0.0, 6.0))


def accepted_paris(data, n_accept: int, rng: np.random.Generator, tol: float = 0.03, batch: int = 200_000,
                   max_proposals: int = 50_000_000) -> tuple[dict[str, np.ndarray], float]:
    """Paris (C, m) sets accepted against the Paris-regime points of the same data, same RMS rule.

    Paris has no threshold, so only the points above the regime split enter; dK
    is the full range at R = -1, as in the life calculation.
    """
    dk, rate, near = data
    log_dk, y = np.log10(dk[~near]), np.log10(rate[~near])
    kept, proposed = [], 0
    while sum(len(k[0]) for k in kept) < n_accept and proposed < max_proposals:
        log_c, n = rng.uniform(*PARIS_PRIOR["log10C"], batch), rng.uniform(*PARIS_PRIOR["n"], batch)
        rel = (log_c[:, None] + n[:, None] * log_dk[None, :]) / y[None, :] - 1.0
        ok = np.sqrt(np.mean(rel**2, axis=1)) < tol
        kept.append((log_c[ok], n[ok]))
        proposed += batch
    log_c, n = (np.concatenate(v)[:n_accept] for v in zip(*kept))
    return {"C": 10.0 ** log_c, "n": n}, len(log_c) / proposed


def load_or_build_paris_pool(directory, n_accept: int = 20_000, seed: int = 7, fixed=FIXED):
    """Paris sets accepted against the same synthetic data as the NASGRO pool (same seed), cached."""
    path = Path(directory) / f"paris_pool_seed{seed}_n{n_accept}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in ("C", "n")}, float(z["rate"])
    rng = np.random.default_rng(seed)
    pool, rate = accepted_paris(synthetic_test_data(rng, fixed=fixed), n_accept, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, rate=rate, **pool)
    return pool, rate


def load_or_build_pool(directory, n_accept: int = 20_000, seed: int = 7, fixed=FIXED):
    """Accepted material sets and acceptance rate, cached in ``directory`` by q, seed and size."""
    path = Path(directory) / f"material_pool_q{fixed['q']:g}_seed{seed}_n{n_accept}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in ("C", "n", "p", "dK0")}, float(z["rate"])
    rng = np.random.default_rng(seed)
    mat, rate = accepted_material(synthetic_test_data(rng, fixed=fixed), n_accept, rng, fixed=fixed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, rate=rate, **mat)
    return mat, rate


def input_spec(material: dict[str, np.ndarray], fixed=FIXED) -> InputSpec:
    """Table 4 normals (initial depth truncated at 1 um) plus the accepted material sets."""
    (a, sa), (d, sd), (f, sf) = NOMINAL["a_init"], NOMINAL["D"], NOMINAL["F"]
    m = {
        "a_init": Marginal("normal", a * MM, sd=sa * MM, bounds=(1e-6, 0.5 * d * MM)),
        "aspect_init": Marginal("fixed", 1.0),
        "D": Marginal("normal", d * MM, sd=sd * MM),
        "F": Marginal("normal", f, sd=sf),
        "R": Marginal("fixed", R),
        **{k: Marginal("fixed", v) for k, v in {**fixed, **BEST_FIT}.items()},
    }
    return InputSpec(m, empirical={"material": EmpiricalGroup(material, BEST_FIT, log=("C",))})


def log_life_stats(life: LifeResult) -> tuple[float, float, float]:
    """(mean, sd) of log10 Nf over failed samples, and the failed fraction Pf."""
    x = np.log10(life.cycles[life.failed])
    return float(np.mean(x)), float(np.std(x)), float(np.mean(life.failed))
