"""Mixed-mode scenario: a crack tilted by theta from the plane normal to the stress.

Four ways to treat it (remote uniaxial stress, K = sigma sqrt(pi a) F):
    projection   re-draw the crack on the principal plane with depth a cos(theta)
                 and grow it in pure mode I (industry flaw-assessment practice)
    full mode I  keep the full depth, ignore the tilt (upper bound on K)
    energy       keep the full depth, K_eq = sqrt(K_I^2 + K_II^2) = K cos(theta),
                 with K_I = K cos^2(theta), K_II = K sin(theta) cos(theta)
    tanaka       keep the full depth, K_eq = (K_I^4 + 8 K_II^4)^(1/4) (Tanaka 1974)
For energy and Tanaka the crack turns towards the principal plane as it grows,
so K_eq relaxes back to mode I over an extension about equal to its start depth:
    K_eq(a) = K(a) [1 - (1 - m(theta)) exp(-(a - a_i) / a_i)],  m = K_eq / K at a_i.
K_eq replaces K in Kmax, Kmin and the Kc check. The base case's own tilt factor
(cos^2 theta on K) is switched off here so the tilt enters through one model only.

In the Monte Carlo scenario the tilt has a prior for the component, and the
choice of criterion is itself an uncertain input taking one of
projection / energy / tanaka with equal probability.
"""
import numpy as np
from scipy.special import ndtr

from . import cases, life, nasgro

MODELS = ("projection", "full mode I", "energy", "tanaka")
CRITERIA = (0, 2, 3)            # the three candidates in the Monte Carlo (indices into MODELS)

# Tilt priors (mean, sd in degrees) before and after ultrasonic sizing.
PRIORS = {
    "uniaxial": dict(mean=0.0, sd_unknown=15.0, sd_sized=5.0,
                     label="Uniaxial loading: tilt centred on 0 deg"),
    "inclined": dict(mean=30.0, sd_unknown=15.0, sd_sized=5.0,
                     label="Inclined cracks (multiaxial or contact loading): tilt centred on 30 deg"),
}
EXTRA_INPUTS = {"crack tilt (mixed mode)": "mm_theta", "mixed-mode criterion": "mm_model"}


def m_factor(model, theta):
    """K_eq / K at the start of growth (before relaxation)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.select([model == 2, model == 3],
                     [c, (c**8 + 8.0 * s**4 * c**4) ** 0.25], 1.0)


def add_inputs(s, prior, sized, seed, fixed=(), model=None):
    """Tilt theta (rad, |N(mean, sd)| capped at 80 deg) and criterion index."""
    n = s["a_i"].size
    p = PRIORS[prior]
    sd = p["sd_sized"] if sized else p["sd_unknown"]
    z = np.zeros(n) if "mm_theta" in fixed else cases._z("mm_theta", n, seed)
    s["mm_theta"] = np.deg2rad(np.clip(np.abs(p["mean"] + sd * z), 0.0, 80.0))
    if model is not None:
        s["mm_model"] = np.full(n, model)
    else:
        u = ndtr(cases._z("mm_model", n, seed))
        s["mm_model"] = np.array(CRITERIA)[np.minimum((3 * u).astype(int), 2)]
    s["tilt"] = np.zeros(n)                     # base tilt factor off
    return s


def mm_life(case, s, n_grid=400):
    """Constant-amplitude life with the tilt treated by each sample's model."""
    model, th = s["mm_model"], s["mm_theta"]
    a_start = np.where(model == 0, s["a_i"] * np.cos(th), s["a_i"])
    sp = dict(s, a_i=a_start)
    _, a, da = life.depth_grid(a_start, cases.a_limit(case, sp), n_grid)
    k1 = cases.k_unit(case, sp, a)
    m = m_factor(model, th)[:, None]
    relax = 1.0 - (1.0 - m) * np.exp(-(a - a_start[:, None]) / a_start[:, None])
    kmax = k1 * relax * cases.max_stress(case, s)[:, None]
    kmin = kmax * s["R"][:, None]
    rate = nasgro.growth_rate(kmax, kmin, a, cases.material(s))
    return life.integrate(rate, kmax, s["Kc"], da)


def nominal_life(case, a_mm, theta_deg, model):
    """Life with every other input at nominal (no Monte Carlo)."""
    allk = tuple(set(cases.BAR_INPUTS.values()) | set(cases.PLATE_INPUTS.values())
                 | {"material", "Cn"})
    s = cases.sample(case, a_mm * 1e-3, 1, seed=1, fixed=allk)
    s["mm_theta"] = np.array([np.deg2rad(theta_deg)])
    s["mm_model"] = np.array([model])
    s["tilt"] = np.zeros(1)
    return float(mm_life(case, s)[0])
