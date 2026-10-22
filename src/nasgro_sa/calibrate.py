"""Calibration of growth-rate models to da/dN test data.

Comparing model forms only means something when every model is calibrated to
the same data. A population of test series ("lots") is fitted lot by lot:
each candidate model is fitted to every lot by least squares on log10(da/dN),
and the lot-to-lot spread of the fitted parameters becomes that model's
parameter distribution. Every model sees the same lots, so differences in
predicted life reflect model form rather than calibration.

Until real test data are available, ``synthetic_lots`` generates lots from a
reference model with parameters drawn from an input specification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .growth import NasgroEquation, ParisLaw, WalkerEquation
from .inputs import InputSpec, Marginal
from .model import STATE_INPUTS

LONG_CRACK = 0.05   # crack length (m) assigned to test data, so small-crack threshold effects vanish
_RATE_FLOOR = 1e-30


@dataclass(frozen=True)
class TestSeries:
    dk: np.ndarray
    r: np.ndarray
    dadn: np.ndarray


def synthetic_series(model, params: Mapping[str, float], r_values: Sequence[float], points_per_r: int,
                     scatter_log10: float, rng: np.random.Generator, upper_fraction: float = 0.9) -> TestSeries:
    """da/dN points log-spaced from just above threshold to ``upper_fraction`` of Kc, per stress ratio."""
    dk_all, r_all, rate_all = [], [], []
    for r in r_values:
        dk_hi = upper_fraction * params["Kc"] * (1.0 - r)
        grid = np.geomspace(0.05, dk_hi, 4000)
        grows = model(grid, r, LONG_CRACK, params) > 0
        dk = np.geomspace(1.05 * grid[np.argmax(grows)], dk_hi, points_per_r)
        rate = model(dk, r, LONG_CRACK, params) * 10.0 ** rng.normal(0.0, scatter_log10, dk.size)
        dk_all.append(dk), r_all.append(np.full(dk.size, r)), rate_all.append(rate)
    return TestSeries(np.concatenate(dk_all), np.concatenate(r_all), np.concatenate(rate_all))


def synthetic_lots(model, spec: InputSpec, n_lots: int, rng: np.random.Generator, r_values=(0.1, 0.4, 0.7),
                   points_per_r: int = 25, scatter_log10: float = 0.05) -> list[TestSeries]:
    """Lots generated from ``model`` with its parameters drawn from ``spec`` (Kc held at nominal)."""
    draws = spec.sample(n_lots, rng)
    lots = []
    for i in range(n_lots):
        params = {k: float(draws[k][i]) for k in model.parameters}
        params["Kc"] = spec.marginals["Kc"].nominal
        lots.append(synthetic_series(model, params, r_values, points_per_r, scatter_log10, rng))
    return lots


@dataclass(frozen=True)
class FitSpec:
    """Which parameters of a model are fitted, how, and to which part of the data."""

    model: object
    initial: Mapping[str, float]                       # free parameters and starting values
    fixed: Mapping[str, float] = field(default_factory=dict)
    log10: tuple[str, ...] = ("C",)                    # fitted (and distributed) on a log scale
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)   # natural units
    r_values: tuple[float, ...] | None = None          # restrict to these stress ratios

    @property
    def free(self) -> tuple[str, ...]:
        return tuple(self.initial)

    def _to_x(self, values: Mapping[str, float]) -> np.ndarray:
        return np.array([np.log10(values[k]) if k in self.log10 else values[k] for k in self.free])

    def _from_x(self, x) -> dict[str, float]:
        return {k: 10.0 ** v if k in self.log10 else v for k, v in zip(self.free, x)}

    def fit(self, series: TestSeries) -> tuple[np.ndarray, float]:
        """Least-squares estimate on the transformed scale and the residual sd in log10(da/dN)."""
        keep = np.ones(series.dk.size, dtype=bool) if self.r_values is None else np.isin(series.r, self.r_values)
        dk, r, log_rate = series.dk[keep], series.r[keep], np.log10(series.dadn[keep])
        default = {k: (1e-300 if k in self.log10 else -np.inf, np.inf) for k in self.free}
        lo = self._to_x({k: self.bounds.get(k, default[k])[0] for k in self.free})
        hi = self._to_x({k: self.bounds.get(k, default[k])[1] for k in self.free})

        def residual(x):
            rate = self.model(dk, r, LONG_CRACK, {**self.fixed, **self._from_x(x)})
            return np.log10(np.maximum(np.nan_to_num(rate, nan=_RATE_FLOOR), _RATE_FLOOR)) - log_rate

        x0 = np.clip(self._to_x(self.initial), lo, hi)
        sol = least_squares(residual, x0, bounds=(lo, hi), x_scale="jac")
        return sol.x, float(np.sqrt(np.mean(sol.fun**2)))


@dataclass
class Calibration:
    fit: FitSpec
    estimates: np.ndarray     # (lots, free) on the transformed scale
    residual_sd: np.ndarray   # (lots,) residual sd in log10(da/dN)

    def input_spec(self, base: InputSpec, label: str = "growth") -> InputSpec:
        """Base physical inputs plus this model's parameters as one Gaussian-copula group.

        Free parameters get the lot-to-lot mean and covariance on the
        transformed scale: lognormal (median held) for log10-fitted ones,
        normal (truncated to the fit bounds) otherwise.
        """
        mean = self.estimates.mean(axis=0)
        cov = np.atleast_2d(np.cov(self.estimates, rowvar=False))
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        m = {k: base.marginals[k] for k in (*STATE_INPUTS, "Kc")}
        for k, v in self.fit.fixed.items():
            m.setdefault(k, Marginal("fixed", v))
        for i, k in enumerate(self.fit.free):
            if k in self.fit.log10:
                sigma = np.log(10.0) * sd[i]
                m[k] = Marginal("lognormal", 10.0 ** mean[i], cov=float(np.sqrt(np.expm1(sigma**2))), hold="median")
            else:
                m[k] = Marginal("normal", float(mean[i]), sd=float(sd[i]), bounds=self.fit.bounds.get(k))
        free = self.fit.free
        correlation = {(free[i], free[j]): float(corr[i, j]) for i in range(len(free)) for j in range(i + 1, len(free))}
        return InputSpec(m, groups={label: free}, correlation=correlation)


def calibrate(fit: FitSpec, lots: Sequence[TestSeries]) -> Calibration:
    results = [fit.fit(lot) for lot in lots]
    return Calibration(fit, np.array([x for x, _ in results]), np.array([s for _, s in results]))


def standard_fit_specs(base: InputSpec, forecast_r: float) -> dict[str, FitSpec]:
    """NASGRO, Walker and Paris fits used for model comparisons.

    NASGRO fits (C, n, p, q, dK0) with the closure constants at their nominal
    values; Walker fits (C, n, gamma) across stress ratios; Paris has no
    stress-ratio term, so it is fitted only to data at the forecast ratio.
    """
    nominal = {k: m.nominal for k, m in base.marginals.items()}
    nasgro_free = ("C", "n", "p", "q", "dK0")
    return {
        "NASGRO": FitSpec(NasgroEquation(), {k: nominal[k] for k in nasgro_free},
                          fixed={k: nominal[k] for k in NasgroEquation.parameters if k not in nasgro_free},
                          bounds={"p": (0.01, 3.0), "q": (0.01, 3.0), "dK0": (0.5, 10.0), "n": (1.0, 8.0)}),
        "Walker": FitSpec(WalkerEquation(), {"C": 1e-10, "n": 3.0, "gamma": 0.5},
                          bounds={"gamma": (0.0, 1.0), "n": (1.0, 8.0)}),
        "Paris": FitSpec(ParisLaw(), {"C": 1e-10, "n": 3.0}, bounds={"n": (1.0, 8.0)}, r_values=(forecast_r,)),
    }
