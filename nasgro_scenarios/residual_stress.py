"""Residual-stress scenario: superpose K_res(a) on the applied K.

Profile (through-depth, z from the cracked surface):
    decaying:  sigma_res(z) = A sigma_y (1 - z/L) exp(-z/L)
    uniform:   sigma_res(z) = A sigma_y          (BS 7910 Level-1 style, = check C)
The decaying form has the shape of the BS 7910 Annex Q as-welded profiles:
peak at the surface, a zero crossing at z = L and a compressive balancing zone
beyond it, with zero net force over the depth. Its amplitude A (signed fraction
of yield) and decay length L are the uncertain inputs. The Annex Q polynomial
for a specific joint can be dropped in by replacing `profile`.

K_res comes from the Shen & Glinka weight function (geometry.weight_function_K).
For the bar the crack is treated as a semicircular surface crack (plate
weight function with t = D) and scaled by the ratio of the bar's uniform-stress
solution to the plate's, so the bar's uniform case is reproduced exactly.

Optional first-cycle shakedown caps the surface residual stress so that the
total stress stays within +-sigma_y on the first cycle. No further relaxation or
redistribution as the crack grows is modelled (stated limitation).
"""
import numpy as np
from scipy.special import ndtr

from . import cases, geometry

# Scenario definitions: amplitude A ~ U(lo, hi) of sigma_y; decay length
# L lognormal (median, CoV). Values are engineering judgement anchored on
# BS 7910 practice (as-welded: up to yield; PWHT/fitted: a fraction of it).
SCENARIOS = {
    "plate_weld": dict(case="plate", A=(0.5, 1.0), L=(2.5e-3, 0.3), shape="decaying", shakedown=False,
                       label="Plate, as-welded tensile profile (no relaxation)"),
    "plate_weld_sd": dict(case="plate", A=(0.5, 1.0), L=(2.5e-3, 0.3), shape="decaying", shakedown=True,
                          label="Plate, as-welded tensile profile, first-cycle shakedown"),
    "plate_uniform": dict(case="plate", A=(0.5, 1.0), L=None, shape="uniform", shakedown=False,
                          label="Plate, uniform tensile (check C analogue)"),
    "bar_tensile": dict(case="bar", A=(0.25, 0.75), L=(2.0e-3, 0.3), shape="decaying", shakedown=False,
                        label="Bar, tensile surface profile (fit or weld repair, no relaxation)"),
    "bar_tensile_sd": dict(case="bar", A=(0.25, 0.75), L=(2.0e-3, 0.3), shape="decaying", shakedown=True,
                           label="Bar, tensile surface profile, first-cycle shakedown"),
    "bar_compressive": dict(case="bar", A=(-0.8, -0.4), L=(0.8e-3, 0.3), shape="decaying", shakedown=True,
                            label="Bar, compressive surface (deep rolled or peened), shakedown"),
    "bar_uniform": dict(case="bar", A=(0.25, 0.75), L=None, shape="uniform", shakedown=False,
                        label="Bar, uniform tensile (check C analogue)"),
}

EXTRA_INPUTS = {"residual stress amplitude": "rs_amp", "residual stress decay length": "rs_len"}


def add_inputs(s, sc, seed, fixed=()):
    """Add residual-stress inputs (surface stress sigma_rs in MPa, decay L in m)."""
    n = s["a_i"].size
    lo, hi = sc["A"]
    if "rs_amp" in fixed:
        A = np.full(n, 0.5 * (lo + hi))
    else:
        A = lo + (hi - lo) * ndtr(cases._z("rs_amp", n, seed))   # uniform on (lo, hi)
    sig = A * s["sigma_y"]
    if sc["shakedown"]:
        smax = cases.max_stress(sc["case"], s)
        smin = smax * s["R"]
        sig = np.clip(sig, -s["sigma_y"] - smin, s["sigma_y"] - smax)
    s["rs_sig"] = sig
    if sc["L"] is None:
        s["rs_len"] = np.full(n, np.inf)
    else:
        med, cov = sc["L"]
        z = np.zeros(n) if "rs_len" in fixed else cases._z("rs_len", n, seed)
        s["rs_len"] = med * np.exp(np.sqrt(np.log(1 + cov**2)) * z)
    return s


def profile(z, sig, L):
    """Residual stress (MPa) at depth z for surface value sig and decay length L."""
    x = z / L
    return sig * (1.0 - x) * np.exp(-x)


def k_res(case, shape):
    """Return k_res_fn(s, a) for life.ca_life."""
    def fn(s, a):
        sig = s["rs_sig"][:, None]
        if shape == "uniform":
            return cases.k_unit(case, s, a) * sig
        corr = (s["sif"] * np.cos(s["tilt"]) ** 2)[:, None]
        L = s["rs_len"][:, None]
        if case == "plate":
            ac, t, W = s["ac"][:, None], s["t"][:, None], s["W"][:, None]
            scale = 1.0
        else:
            ac = np.ones_like(a)
            t = s["D"][:, None]
            W = 10.0 * t
            scale = geometry.bar_K_uniform(a, s["D"][:, None], 1.0) / \
                geometry.plate_K_uniform(a, ac, t, W, 1.0)
        Q = geometry.newman_raju_Q(ac)
        Y0, Y1 = geometry.plate_Y0_Y1(a, ac, t, W)
        M1, M2, M3 = geometry.shen_glinka_M(Y0, Y1, Q)
        Kr = geometry.weight_function_K(
            lambda x: profile(x, sig[..., None], L[..., None]), a, M1, M2, M3)
        return corr * scale * Kr
    return fn
