"""Stress-intensity factor solutions.

A geometry exposes:
  * ``inputs``: names of per-sample geometric inputs (e.g. a sampled diameter),
    passed as keyword arrays to the two methods below;
  * ``stress_intensity(a, c, s, **inputs) -> (K_depth, K_surface)`` for crack
    depth ``a`` and surface half-length ``c`` under remote stress ``s``;
  * ``limits(criterion, **inputs) -> (a_limit, c_limit)``, the crack sizes at
    which the failure criterion's geometric limits are reached.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def newman_raju_tension(a, c, t, b, phi):
    """Newman-Raju (1984) boundary-correction factor F and shape factor Q.

    Semi-elliptical surface crack in a plate of thickness t and half-width b
    under remote tension; phi is the parametric angle (pi/2 at the deepest
    point, 0 at the free surface). Valid for 0 < a/c <= 2, a/t < 0.8 and
    c/b < 0.5.
    """
    ac = a / c
    ca = c / a
    at = a / t
    short = ac <= 1.0
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)

    m1 = np.where(short, 1.13 - 0.09 * ac, np.sqrt(ca) * (1.0 + 0.04 * ca))
    m2 = np.where(short, -0.54 + 0.89 / (0.2 + ac), 0.2 * ca**4)
    m3 = np.where(short, 0.5 - 1.0 / (0.65 + ac) + 14.0 * (1.0 - ac) ** 24, -0.11 * ca**4)
    g = 1.0 + (0.1 + 0.35 * np.where(short, 1.0, ca) * at**2) * (1.0 - sin_phi) ** 2
    f_phi = np.where(short, ac**2 * cos_phi**2 + sin_phi**2, ca**2 * sin_phi**2 + cos_phi**2) ** 0.25
    with np.errstate(invalid="ignore"):
        f_w = np.sqrt(1.0 / np.cos(0.5 * np.pi * (c / b) * np.sqrt(at)))
    q = 1.0 + 1.464 * np.where(short, ac, ca) ** 1.65
    return (m1 + m2 * at**2 + m3 * at**4) * g * f_phi * f_w, q


@dataclass(frozen=True)
class SurfaceCrackPlate:
    """Semi-elliptical surface crack, normal to the load, in a plate under remote tension."""

    thickness: float
    width: float
    inputs = ()

    @property
    def half_width(self) -> float:
        return 0.5 * self.width

    def limits(self, criterion):
        return criterion.depth_ratio * self.thickness, criterion.width_ratio * self.half_width

    def stress_intensity(self, a, c, s):
        return _surface_crack_k(a, c, s, self.thickness, self.half_width)


@dataclass(frozen=True)
class SampledSurfaceCrackPlate:
    """SurfaceCrackPlate with thickness and width as per-sample inputs (dimensional tolerances)."""

    inputs = ("thickness", "width")

    def limits(self, criterion, thickness, width):
        return criterion.depth_ratio * np.asarray(thickness), criterion.width_ratio * 0.5 * np.asarray(width)

    def stress_intensity(self, a, c, s, thickness, width):
        return _surface_crack_k(a, c, s, thickness, 0.5 * np.asarray(width))


def _surface_crack_k(a, c, s, t, b):
    """K at the deepest and surface points of a semi-elliptical surface crack (Newman-Raju)."""
    a, c = np.asarray(a, dtype=float), np.asarray(c, dtype=float)
    f_depth, q = newman_raju_tension(a, c, t, b, 0.5 * np.pi)
    f_surface, _ = newman_raju_tension(a, c, t, b, 0.0)
    nominal = s * np.sqrt(np.pi * a / q)
    return nominal * f_depth, nominal * f_surface


@dataclass(frozen=True)
class RoundBarStraightCrack:
    """Straight-fronted crack of depth a in a round bar of diameter D under remote tension.

    James & Mills (1988):
        K = S sqrt(pi a) g [0.752 + 2.02 a/D + 0.37 (1 - sin x)^3],
        g = (1.84/pi) sqrt(tan(x)/x) / cos(x),  x = pi a / (2 D).
    D is a per-sample input. The front is described by its depth alone, so
    both crack-front points report the same K and c simply tracks a.
    """

    inputs = ("D",)

    def limits(self, criterion, D):
        return criterion.depth_ratio * np.asarray(D, dtype=float), np.inf

    def stress_intensity(self, a, c, s, D):
        a = np.asarray(a, dtype=float)
        x = 0.5 * np.pi * a / D
        with np.errstate(invalid="ignore"):
            g = (1.84 / np.pi) * np.sqrt(np.tan(x) / x) / np.cos(x)
        k = s * np.sqrt(np.pi * a) * g * (0.752 + 2.02 * a / D + 0.37 * (1.0 - np.sin(x)) ** 3)
        return k, k


def mixed_mode_factor(theta_deg, criterion: str):
    """Equivalent mode-I factor K_eq / K for a crack tilted theta from the principal plane, uniaxial stress.

    With K_I = K cos^2(theta) and K_II = K sin(theta) cos(theta):
      "energy": K_eq = sqrt(K_I^2 + K_II^2) = K cos(theta)  (energy release rate equivalence);
      "tanaka": K_eq = (K_I^4 + 8 K_II^4)^(1/4)             (Tanaka 1974).
    """
    t = np.radians(np.asarray(theta_deg, dtype=float))
    c, s = np.cos(t), np.sin(t)
    if criterion == "energy":
        return c
    if criterion == "tanaka":
        return (c**8 + 8.0 * s**4 * c**4) ** 0.25
    raise ValueError(f"unknown mixed-mode criterion {criterion!r}")


@dataclass(frozen=True)
class MixedModeOrientation:
    """Tilted initial crack: mixed-mode driving force that relaxes to mode I as the crack grows.

    The initial crack (depth a_start, tilt theta) grows under K_eq = K * m(theta)
    from ``mixed_mode_factor``. Once it kinks towards the principal plane, the
    factor recovers to 1 over a crack extension comparable to the initial
    depth: 1 - (1 - m) exp(-(a - a_start) / a_start). The full depth is kept
    (no projection). This is an engineering approximation; NASGRO itself is
    mode I only.
    """

    base: object
    criterion: str = "energy"

    @property
    def inputs(self) -> tuple[str, ...]:
        return (*self.base.inputs, "theta", "a_start")

    def limits(self, criterion, theta, a_start, **kw):
        return self.base.limits(criterion, **kw)

    def stress_intensity(self, a, c, s, theta, a_start, **kw):
        k_a, k_c = self.base.stress_intensity(a, c, s, **kw)
        grown = np.maximum(np.asarray(a, dtype=float) - a_start, 0.0) / a_start
        factor = 1.0 - (1.0 - mixed_mode_factor(theta, self.criterion)) * np.exp(-grown)
        return k_a * factor, k_c * factor
