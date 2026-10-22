"""Vectorised fatigue-life integration for a two-point (depth, surface) crack front."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Mapping

import numpy as np


class Outcome(IntEnum):
    RUNNING = 0
    FRACTURE = 1      # Kmax >= Kc at either crack-front point
    DEPTH_LIMIT = 2   # depth reached the declared fraction of thickness
    WIDTH_LIMIT = 3   # half-length reached the declared fraction of half-width
    ARREST = 4        # neither point grows under the declared loading
    HORIZON = 5       # still growing at the simulation horizon (right-censored)
    INVALID = 6       # left the SIF solution's validity range or the step budget


FAILURE_OUTCOMES = (Outcome.FRACTURE, Outcome.DEPTH_LIMIT, Outcome.WIDTH_LIMIT)
_MARGIN_CODES = np.array([Outcome.FRACTURE, Outcome.DEPTH_LIMIT, Outcome.WIDTH_LIMIT])
REGIMES = ("near-threshold", "Paris", "near-critical")


@dataclass(frozen=True)
class FailureCriterion:
    """Declared end-of-life conditions.

    Failure is the first of: Kmax >= Kc at either point, depth reaching
    ``depth_ratio`` of the geometry's reference depth (plate thickness, bar
    diameter), or half-length reaching ``width_ratio`` of the half-width. The
    plate defaults sit at the edge of the Newman-Raju validity range, so they
    double as a validity limit.
    """

    depth_ratio: float = 0.8
    width_ratio: float = 0.5
    aspect_range: tuple[float, float] = (0.05, 2.0)


@dataclass(frozen=True)
class IntegrationSettings:
    growth_fraction: float = 0.01   # largest relative extension of either point per step
    rel_tol: float = 1e-3           # local error tolerance, relative to crack size, per step
    horizon: float = 1e8            # simulation horizon in cycles; survivors are censored here
    max_steps: int = 20_000
    max_halvings: int = 10
    # Cycle accounting by growth regime (needs a rate model with regime_factors).
    # A step counts as near-threshold when the threshold factor at the deepest
    # point is below the first cutoff (rate at least halved), near-critical when
    # the near-critical factor exceeds the second (rate at least doubled).
    track_regimes: bool = False
    regime_cutoffs: tuple[float, float] = (0.5, 2.0)

    def refined(self, factor: float) -> IntegrationSettings:
        """Step controls multiplied by ``factor`` (< 1 refines), for convergence checks."""
        return replace(self, growth_fraction=self.growth_fraction * factor, rel_tol=self.rel_tol * factor)


@dataclass
class LifeResult:
    cycles: np.ndarray    # cycles at termination
    outcome: np.ndarray   # Outcome codes
    a: np.ndarray         # depth at termination
    c: np.ndarray         # surface half-length at termination
    horizon: float
    regime_cycles: np.ndarray | None = None   # (n, 3) cycles spent per REGIMES entry, if tracked

    @property
    def failed(self) -> np.ndarray:
        return np.isin(self.outcome, FAILURE_OUTCOMES)

    def regime_shares(self) -> np.ndarray:
        """Share of the cycles to failure spent in each regime, for failed samples: (n_failed, 3)."""
        if self.regime_cycles is None:
            raise ValueError("integrate with track_regimes=True to record regimes")
        failed = self.failed
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.regime_cycles[failed] / self.cycles[failed, None]

    @property
    def rul(self) -> np.ndarray:
        """Cycles to failure; inf where the trajectory arrested or was censored."""
        return np.where(self.failed, self.cycles, np.inf)

    def failure_indicator(self, horizons) -> np.ndarray:
        """1.0 where failure occurred within each decision horizon; shape (n, len(horizons))."""
        h = np.atleast_1d(np.asarray(horizons, dtype=float))
        if np.any(h > self.horizon):
            raise ValueError("decision horizons must not exceed the simulation horizon")
        return (self.rul[:, None] <= h[None, :]).astype(float)

    def fractions(self) -> dict[str, float]:
        return {o.name: float(np.mean(self.outcome == o)) for o in Outcome if o is not Outcome.RUNNING}


def rul_quantile(rul, q):
    """Lower quantiles of a right-censored life sample; inf where fewer than q of samples fail."""
    s = np.sort(np.asarray(rul, dtype=float))
    idx = np.ceil(np.atleast_1d(q) * s.size).astype(int) - 1
    return s[np.clip(idx, 0, s.size - 1)]


def integrate_life(geometry, rate_model, params: Mapping[str, np.ndarray], a0, c0, s_max, r,
                   criterion: FailureCriterion = FailureCriterion(),
                   settings: IntegrationSettings = IntegrationSettings(),
                   geometry_params: Mapping[str, np.ndarray] | None = None) -> LifeResult:
    """Integrate constant-amplitude crack growth from (a0, c0) to the first terminating event.

    Array arguments broadcast to a common 1-D sample size; ``params`` maps the
    rate model's parameter names plus "Kc" to scalars or arrays, and
    ``geometry_params`` supplies the geometry's per-sample inputs. Each live
    sample advances by its own cycle increment, sized so the faster-growing
    point extends by at most ``growth_fraction`` of its current size, using the
    explicit midpoint rule. Terminating events inside a step are located by
    linear interpolation of the failure margin.
    """
    geometry_params = geometry_params or {}
    shapes = (np.shape(v) for v in (a0, c0, s_max, r, *params.values(), *geometry_params.values()))
    n = int(np.prod(np.broadcast_shapes(*shapes)))
    full = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (n,))
    a, c = full(a0).copy(), full(c0).copy()
    s_max, r = full(s_max), full(r)
    params = {k: full(v) for k, v in params.items()}
    geometry_params = {k: full(v) for k, v in geometry_params.items()}
    k_crit = params["Kc"]
    a_lim, c_lim = map(full, geometry.limits(criterion, **geometry_params))
    lo_aspect, hi_aspect = criterion.aspect_range

    def evaluate(a, c, idx):
        """Failure margin (>= 0 means terminated), its criterion, and rates at both points."""
        rr = r[idx]
        k_a, k_c = geometry.stress_intensity(a, c, s_max[idx], **{k: v[idx] for k, v in geometry_params.items()})
        g = np.stack([np.maximum(k_a, k_c) / k_crit[idx] - 1.0, a / a_lim[idx] - 1.0, c / c_lim[idx] - 1.0])
        which = np.argmax(g, axis=0)
        p = {k: v[idx] for k, v in params.items()}
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            rate_a = rate_model(k_a * (1.0 - rr), rr, a, p)
            rate_c = rate_model(k_c * (1.0 - rr), rr, c, p)
        return g[which, np.arange(a.size)], _MARGIN_CODES[which], rate_a, rate_c

    regime_cycles = np.zeros((n, len(REGIMES))) if settings.track_regimes else None
    lo_cut, hi_cut = settings.regime_cutoffs

    def regime(a, c, idx):
        """Regime index of each state, judged at the deepest point."""
        rr = r[idx]
        k_a, _ = geometry.stress_intensity(a, c, s_max[idx], **{k: v[idx] for k, v in geometry_params.items()})
        near_threshold, near_critical = rate_model.regime_factors(k_a * (1.0 - rr), rr, a,
                                                                  {k: v[idx] for k, v in params.items()})
        return np.where(near_threshold < lo_cut, 0, np.where(near_critical > hi_cut, 2, 1))

    cycles = np.zeros(n)
    outcome = np.full(n, int(Outcome.RUNNING))
    live = np.arange(n)
    g, code, rate_a, rate_c = evaluate(a, c, live)
    done = g >= 0.0
    outcome[done] = code[done]
    live, g, rate_a, rate_c = live[~done], g[~done], rate_a[~done], rate_c[~done]

    for _ in range(settings.max_steps):
        if live.size == 0:
            break
        ai, ci, ni = a[live], c[live], cycles[live]
        aspect = ai / ci
        invalid = (~np.isfinite(rate_a) | ~np.isfinite(rate_c) | ~np.isfinite(g)
                   | (aspect < lo_aspect) | (aspect > hi_aspect))
        arrested = ~invalid & (rate_a <= 0.0) & (rate_c <= 0.0)
        outcome[live[invalid]] = Outcome.INVALID
        outcome[live[arrested]] = Outcome.ARREST
        go = ~(invalid | arrested)
        idx, ai, ci, ni = live[go], ai[go], ci[go], ni[go]
        g0, ra, rc = g[go], rate_a[go], rate_c[go]

        with np.errstate(divide="ignore"):
            dn = settings.growth_fraction * np.minimum(np.where(ra > 0, ai / ra, np.inf),
                                                       np.where(rc > 0, ci / rc, np.inf))
        dn = np.minimum(dn, settings.horizon - ni)

        # Midpoint rule, halving the step where the Euler-midpoint difference (a
        # local error estimate) exceeds rel_tol of the crack size, or where the
        # midpoint has stopped growing. The latter matters for marginal cracks:
        # stepping straight past an arrest point would turn an arrest into a
        # finite life. Where the midpoint has already terminated (or the halving
        # budget is spent), take an Euler step and let the next step resolve it.
        ram, rcm, euler = np.empty_like(ra), np.empty_like(rc), np.zeros(ra.size, dtype=bool)
        todo = np.arange(ra.size)
        for attempt in range(settings.max_halvings + 1):
            d = dn[todo]
            gm, _, ram_t, rcm_t = evaluate(ai[todo] + 0.5 * d * ra[todo], ci[todo] + 0.5 * d * rc[todo], idx[todo])
            stalled = (ram_t <= 0.0) & (rcm_t <= 0.0)
            eul = (gm >= 0.0) | ~np.isfinite(ram_t) | ~np.isfinite(rcm_t) | stalled
            with np.errstate(invalid="ignore"):
                err = d * np.maximum(np.abs(ram_t - ra[todo]) / ai[todo], np.abs(rcm_t - rc[todo]) / ci[todo])
            ram[todo], rcm[todo], euler[todo] = ram_t, rcm_t, eul
            redo = (gm < 0.0) & (stalled | (~eul & (err > settings.rel_tol)))
            if attempt == settings.max_halvings or not redo.any():
                break
            todo = todo[redo]
            dn[todo] *= 0.5
        ram, rcm = np.where(euler, ra, ram), np.where(euler, rc, rcm)
        a1, c1, n1 = ai + dn * ram, ci + dn * rcm, ni + dn
        g1, code1, ra1, rc1 = evaluate(a1, c1, idx)

        failed = g1 >= 0.0
        s = np.where(failed, g0 / (g0 - np.where(failed, g1, 0.0) - 1e-300), 1.0)
        if regime_cycles is not None:
            regime_cycles[idx, regime(ai, ci, idx)] += s * dn
        a[idx], c[idx], cycles[idx] = ai + s * (a1 - ai), ci + s * (c1 - ci), ni + s * dn
        censored = ~failed & (n1 >= settings.horizon)
        outcome[idx[failed]] = code1[failed]
        outcome[idx[censored]] = Outcome.HORIZON

        keep = ~(failed | censored)
        live, g, rate_a, rate_c = idx[keep], g1[keep], ra1[keep], rc1[keep]
    else:
        if live.size:
            warnings.warn(f"{live.size} trajectories exhausted max_steps; marked INVALID", RuntimeWarning)
            outcome[live] = Outcome.INVALID

    return LifeResult(cycles, outcome, a, c, settings.horizon, regime_cycles)
