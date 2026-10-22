"""Crack-growth rate models.

Units throughout: metres, cycles, MPa and MPa*sqrt(m). Every model is called
as ``model(dk, r, a, params)`` with ``params`` a mapping of parameter name to
scalar or array, so sampled parameters vectorise without copying.
"""
from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np


class RateModel(Protocol):
    parameters: tuple[str, ...]

    def __call__(self, dk, r, a, params: Mapping[str, np.ndarray]) -> np.ndarray: ...


def newman_closure(r, alpha, smax_sigma0):
    """Newman (1984) crack-opening function f = K_op / K_max and its A0 coefficient."""
    r = np.asarray(r, dtype=float)
    a0 = (0.825 - 0.34 * alpha + 0.05 * alpha**2) * np.cos(0.5 * np.pi * smax_sigma0) ** (1.0 / alpha)
    a1 = (0.415 - 0.071 * alpha) * smax_sigma0
    a3 = 2.0 * a0 + a1 - 1.0
    a2 = 1.0 - a0 - a1 - a3
    f_pos = np.maximum(r, a0 + a1 * r + a2 * r**2 + a3 * r**3)
    f_neg = a0 + a1 * np.maximum(r, -2.0)
    return np.where(r >= 0.0, f_pos, f_neg), a0


class NasgroEquation:
    """NASGRO equation (Forman & Mettu 1992) with the NASGRO 4 threshold form.

        da/dN = C [((1 - f)/(1 - R)) dK]^n (1 - dKth/dK)^p / (1 - Kmax/Kc)^q
        dKth  = dK0 sqrt(a/(a + a0)) / [(1 - f)/((1 - A0)(1 - R))]^(1 + Cth R)

    dK0 is the long-crack threshold at R = 0 and a0 the intrinsic crack
    length. Later NASGRO releases reparameterise the threshold term, so check
    this form against the version being emulated before comparing numbers.
    """

    parameters = ("C", "n", "p", "q", "dK0", "Cth_pos", "Cth_neg", "Kc", "alpha", "smax_sigma0", "a0_intrinsic")
    # Optional "closure_factor" kappa scales the open part of the cycle, f' = 1 - kappa (1 - f), in both
    # the growth and threshold terms: kappa > 1 means the crack is open for more of the cycle than
    # Newman's function predicts (less closure), kappa < 1 more closure. Default 1.
    optional = ("closure_factor",)

    def threshold(self, r, a, params):
        f, a0 = newman_closure(r, params["alpha"], params["smax_sigma0"])
        u = params.get("closure_factor", 1.0) * (1.0 - f) / (1.0 - r)
        cth = np.where(r >= 0.0, params["Cth_pos"], params["Cth_neg"])
        small_crack = np.sqrt(a / (a + params["a0_intrinsic"]))
        return params["dK0"] * small_crack / (u / (1.0 - a0)) ** (1.0 + cth * r), u

    def regime_factors(self, dk, r, a, params):
        """Threshold factor (1 - dKth/dK)^p and near-critical factor (1 - Kmax/Kc)^-q.

        Both multiply the Paris term. The first falls from 1 to 0 as dK
        approaches threshold; the second rises from 1 without bound as Kmax
        approaches Kc. Together they define the growth regime at a crack state.
        """
        return self._terms(dk, r, a, params)[:2]

    def _terms(self, dk, r, a, params):
        dk_th, u = self.threshold(r, a, params)
        with np.errstate(divide="ignore", invalid="ignore"):
            near_threshold = np.clip(1.0 - dk_th / dk, 0.0, None) ** params["p"]
            near_critical = np.clip(1.0 - dk / (1.0 - r) / params["Kc"], 0.0, None) ** -params["q"]
        return near_threshold, near_critical, u

    def __call__(self, dk, r, a, params):
        near_threshold, near_critical, u = self._terms(dk, r, a, params)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return params["C"] * (u * dk) ** params["n"] * near_threshold * near_critical


class WalkerEquation:
    """da/dN = C [dK / (1 - R)^(1 - gamma)]^n: a stress-ratio effect, with no threshold or fracture terms."""

    parameters = ("C", "n", "gamma")

    def __call__(self, dk, r, a, params):
        return params["C"] * (np.asarray(dk, dtype=float) / (1.0 - r) ** (1.0 - params["gamma"])) ** params["n"]


class ParisLaw:
    """da/dN = C dK^n, with no stress-ratio, threshold or fracture terms."""

    parameters = ("C", "n")

    def __call__(self, dk, r, a, params):
        return params["C"] * np.asarray(dk, dtype=float) ** params["n"]
