"""The two cases of the study and their industry-typical input uncertainty.

Nominal values and spreads follow the study document (Figures 3 and A1-A3):
  bar   - 25CrMo4 round bar, D = 18 mm, R = -1, 250 MPa (after Giannella 2022)
  plate - semi-elliptical surface crack, t = 10 mm, W = 100 mm, R = 0.1,
          200 MPa, placeholder aluminium values.

Each uncertain input draws from its own random stream (common random numbers),
so fixing one input at its nominal value leaves every other sample unchanged.
Scenario modules add their own inputs through `extra`.
"""
import zlib
import numpy as np

from . import geometry

# Calibrated so the rebuilt baseline reproduces the study's 1 mm statistics
# (bar: median log10 Nf 5.59, Pf 93 %; plate: 4.41, 99 %); see run_baseline_check.py.
BAR_C = 4.81e-12
PLATE_C = 2.46e-10

BAR = dict(
    name="bar", D=0.018, F=63.6e3, R=-1.0, sigma_y=520.0,
    C=BAR_C, n=3.2, p=0.5, q=0.5, dKth0=5.8, a0=3.81e-5, cth_pos=1.5, cth_neg=0.1,
    alpha=2.5, smax_sigma0=0.3, kappa=1.0, Kc=130.0, sif=1.0, tilt=0.0,
)
PLATE = dict(
    name="plate", t=0.010, W=0.100, ac=0.8, S=200.0, R=0.1, sigma_y=350.0,
    C=PLATE_C, n=3.0, p=0.5, q=1.0, dKth0=2.8, a0=3.81e-5, cth_pos=1.5, cth_neg=0.1,
    alpha=2.0, smax_sigma0=0.3, kappa=1.0, Kc=40.0, sif=1.0, tilt=0.0,
)

# Inputs that the U90 ranking switches off one at a time (display name -> key).
BAR_INPUTS = {
    "crack depth": "depth", "material fit (C, m, p, dKth0)": "material",
    "load amplitude": "F", "crack opening factor kappa": "kappa",
    "SIF solution error": "sif", "diameter D": "D", "crack tilt": "tilt",
    "toughness Kc": "Kc", "intrinsic crack length a0": "a0", "stress ratio R": "R",
}
PLATE_INPUTS = {
    "crack depth": "depth", "stress amplitude": "S", "crack opening factor kappa": "kappa",
    "constraint factor alpha": "alpha", "SIF solution error": "sif",
    "growth curve C-n": "Cn", "crack shape a/c": "ac", "critical-shape exponent q": "q",
    "toughness Kc": "Kc", "intrinsic crack length a0": "a0",
    "threshold-shape exponent p": "p", "Smax/sigma0": "smax_sigma0",
    "threshold dKth0": "dKth0", "stress ratio R": "R", "thickness t": "t",
    "Cth (R >= 0)": "cth_pos", "width W": "W", "Cth (R < 0)": "cth_neg",
    "crack tilt": "tilt",
}


def _z(name, n, seed, k=1):
    """Standard normals for one input, from a stream keyed by its name."""
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    return rng.standard_normal((k, n)) if k > 1 else rng.standard_normal(n)


def _depth(a_nom, n, seed, fixed):
    if fixed:
        return np.full(n, a_nom)
    z = _z("depth", n, seed)
    if a_nom < 1e-3:                       # undetected: lognormal, CoV 0.6
        s = np.sqrt(np.log(1 + 0.6**2))
        return a_nom * np.exp(s * z - 0.5 * s**2)
    a = a_nom + 0.5e-3 * z                 # UT-sized: +-0.5 mm sd
    # truncate at 20 um by reflection (keeps the stream aligned)
    return np.where(a > 2e-5, a, 2e-5 + np.abs(a - 2e-5))


def _normal(name, nom, cov, n, seed, fixed, lo=None):
    if fixed:
        return np.full(n, float(nom))
    x = nom * (1.0 + cov * _z(name, n, seed))
    return np.clip(x, lo, None) if lo is not None else x


def _lognormal(name, nom, cov, n, seed, fixed):
    if fixed:
        return np.full(n, float(nom))
    s = np.sqrt(np.log(1 + cov**2))
    return nom * np.exp(s * _z(name, n, seed))


def sample(case, a_nom, n, seed=1, fixed=(), undetected=None):
    """Draw n samples of every input; inputs named in `fixed` sit at nominal."""
    nom = BAR if case == "bar" else PLATE
    fx = set(fixed)
    s = {k: np.full(n, v, dtype=float) for k, v in nom.items() if k != "name"}
    undetected = a_nom < 1e-3 if undetected is None else undetected
    s["a_i"] = _depth(a_nom, n, seed, "depth" in fx)
    s["tilt"] = np.zeros(n) if "tilt" in fx else \
        np.deg2rad((15.0 if undetected else 5.0) * _z("tilt", n, seed))
    s["sif"] = _normal("sif", 1.0, 0.05, n, seed, "sif" in fx)
    s["kappa"] = _normal("kappa", 1.0, 0.10, n, seed, "kappa" in fx, lo=0.3)
    s["Kc"] = _normal("Kc", nom["Kc"], 0.10, n, seed, "Kc" in fx, lo=0.3 * nom["Kc"])
    s["a0"] = _lognormal("a0", nom["a0"], 0.5, n, seed, "a0" in fx)
    s["R"] = nom["R"] + (np.zeros(n) if "R" in fx else 0.05 * _z("R", n, seed))
    if case == "bar":
        s["F"] = _normal("F", nom["F"], 0.10, n, seed, "F" in fx, lo=0.1 * nom["F"])
        s["D"] = _normal("D", nom["D"], 0.015, n, seed, "D" in fx)
        if "material" not in fx:
            # Accepted pool of fitted (C, m, p, dKth0): level at a pivot, slope,
            # threshold and threshold shape, with the trade-offs a joint fit leaves.
            z = _z("material", n, seed, k=4)
            pivot = 8.0                       # MPa sqrt(m), effective range
            lvl = np.log(1 + 0.15**2) ** 0.5 * z[0]
            s["n"] = nom["n"] * (1 + 0.05 * z[1])
            s["C"] = nom["C"] * pivot ** (nom["n"] - s["n"]) * np.exp(lvl)
            s["dKth0"] = nom["dKth0"] * (1 + 0.10 * z[2])
            zp = -0.5 * z[2] + np.sqrt(0.75) * z[3]
            s["p"] = np.clip(nom["p"] * np.exp(0.7 * zp), 0.05, 3.0)
    else:
        s["S"] = _normal("S", nom["S"], 0.10, n, seed, "S" in fx, lo=0.1 * nom["S"])
        s["t"] = _normal("t", nom["t"], 0.015, n, seed, "t" in fx)
        s["W"] = _normal("W", nom["W"], 0.015, n, seed, "W" in fx)
        s["ac"] = np.clip(_normal("ac", nom["ac"], 0.20, n, seed, "ac" in fx), 0.2, 1.0)
        s["alpha"] = np.clip(_normal("alpha", nom["alpha"], 0.19, n, seed, "alpha" in fx), 1.0, 3.0)
        s["smax_sigma0"] = np.clip(_normal("smax_sigma0", nom["smax_sigma0"], 0.2, n, seed,
                                           "smax_sigma0" in fx), 0.05, 0.8)
        s["q"] = _normal("q", nom["q"], 0.3, n, seed, "q" in fx, lo=0.0)
        s["p"] = _normal("p", nom["p"], 0.3, n, seed, "p" in fx, lo=0.0)
        s["dKth0"] = _normal("dKth0", nom["dKth0"], 0.10, n, seed, "dKth0" in fx)
        s["cth_pos"] = _normal("cth_pos", nom["cth_pos"], 0.3, n, seed, "cth_pos" in fx)
        s["cth_neg"] = _normal("cth_neg", nom["cth_neg"], 0.3, n, seed, "cth_neg" in fx)
        if "Cn" not in fx:
            z = _z("Cn", n, seed, k=2)
            pivot = 10.0
            s["n"] = nom["n"] * (1 + 0.05 * z[1])
            s["C"] = nom["C"] * pivot ** (nom["n"] - s["n"]) * \
                np.exp(np.log(1 + 0.15**2) ** 0.5 * z[0])
    return s


def max_stress(case, s):
    """Applied peak stress (MPa) of the constant-amplitude cycle."""
    if case == "bar":
        return 1e-6 * s["F"] / (np.pi * s["D"] ** 2 / 4.0)   # N/m^2 -> MPa
    return s["S"]


def k_unit(case, s, a):
    """Applied K per MPa of remote stress at depth a (a broadcast against samples).

    Includes the SIF-solution error factor and the mode-I projection of a
    tilted crack (cos^2 of the tilt from the principal plane).
    """
    corr = (s["sif"] * np.cos(s["tilt"]) ** 2)[:, None]
    if case == "bar":
        return corr * geometry.bar_K_uniform(a, s["D"][:, None], 1.0)
    return corr * geometry.plate_K_uniform(a, s["ac"][:, None], s["t"][:, None],
                                           s["W"][:, None], 1.0)


def a_limit(case, s):
    """Geometric end of the solution: 0.7 D (bar) or 0.8 t (plate breakthrough)."""
    return 0.7 * s["D"] if case == "bar" else 0.8 * s["t"]


def material(s):
    keys = ("C", "n", "p", "q", "dKth0", "a0", "cth_pos", "cth_neg",
            "alpha", "smax_sigma0", "kappa", "Kc")
    return {k: s[k][:, None] for k in keys}
