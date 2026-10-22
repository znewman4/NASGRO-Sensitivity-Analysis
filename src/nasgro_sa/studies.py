"""Study drivers: propagation, Sobol ranking, robustness, reduction paths and numerical checks."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import stats

from .inputs import InputSpec
from .integrate import LifeResult, Outcome, rul_quantile
from .model import Formulation
from .moment_independent import KSResult, ks_indices
from .sobol import SobolResult, saltelli_blocks, sobol_indices

STATE_GROUPS = ("a_init", "aspect_init", "theta")   # crack-geometry inputs, held fixed at a matched state
RATE_FLOOR = 1e-14                         # m/cycle; below-threshold rates map here in log-rate outputs
RATE_CEILING = 1e-2                        # m/cycle; unstable states (Kmax >= Kc) map here


def decision_outputs(horizons: Sequence[float]) -> Callable[[LifeResult], tuple[np.ndarray, list[str]]]:
    """Failure-within-horizon indicators plus an arrest indicator."""
    def outputs(life: LifeResult):
        y = np.column_stack([life.failure_indicator(horizons), life.outcome == Outcome.ARREST])
        return y, [f"P(RUL<={h:.3g})" for h in horizons] + ["P(arrest)"]
    return outputs


@dataclass
class SobolStudy:
    sobol: SobolResult
    life: LifeResult      # all n (k + 2) trajectories; the first 2n (A and B) are independent draws
    w: np.ndarray         # the standard normals behind them
    groups: list[tuple[str, np.ndarray]]

    def ks(self, n_bins: int = 16, rng: np.random.Generator | None = None) -> KSResult:
        """Moment-independent check on the censored RUL, reusing the independent A and B rows."""
        n2 = 2 * self.sobol.n
        return ks_indices(self.life.rul[:n2], self.w[:n2], self.groups, n_bins=n_bins, rng=rng)


def sobol_study(formulation: Formulation, spec: InputSpec, n: int, horizons: Sequence[float],
                rng: np.random.Generator, n_boot: int = 500) -> SobolStudy:
    """Grouped Sobol indices of the decision outputs; costs n (k + 2) life evaluations."""
    groups = spec.uncertain_groups()
    w = saltelli_blocks(spec.standard_normal(n, rng), spec.standard_normal(n, rng), [c for _, c in groups])
    life = formulation.run(spec.transform(w))
    y, labels = decision_outputs(horizons)(life)
    return SobolStudy(sobol_indices(y, n, [g for g, _ in groups], labels, n_boot=n_boot, rng=rng), life, w, groups)


def growth_rate_study(formulation: Formulation, spec: InputSpec, states: Sequence[tuple[float, float]], n: int,
                      rng: np.random.Generator, n_boot: int = 500) -> tuple[SobolResult, np.ndarray]:
    """Sobol indices of log10 da/dN at the deepest point, at matched crack geometries (a, c).

    Crack size and shape are fixed at each state, so only material and loading
    inputs vary. Rates below threshold are mapped to RATE_FLOOR and unstable
    states (Kmax >= Kc) to RATE_CEILING; the fraction below threshold per state
    is returned alongside.
    """
    groups = [(g, c) for g, c in spec.uncertain_groups() if g not in STATE_GROUPS]
    w = saltelli_blocks(spec.standard_normal(n, rng), spec.standard_normal(n, rng), [c for _, c in groups])
    x = spec.transform(w)
    rates = np.column_stack([formulation.rates_at(x, a, c)[0] for a, c in states])
    labels = [f"log10 da/dN @ a={a * 1e3:g}mm, c={c * 1e3:g}mm" for a, c in states]
    bounded = np.clip(np.nan_to_num(rates, nan=RATE_CEILING, posinf=RATE_CEILING), RATE_FLOOR, RATE_CEILING)
    res = sobol_indices(np.log10(bounded), n, [g for g, _ in groups], labels, n_boot=n_boot, rng=rng)
    return res, np.mean(rates[: 2 * n] <= 0.0, axis=0)


@dataclass
class Summary:
    p_fail: np.ndarray       # P(RUL <= h) per horizon
    p_fail_se: np.ndarray    # Monte Carlo standard error
    quantiles: np.ndarray    # RUL lower quantiles (inf where undefined)
    fractions: dict[str, float]
    n: int


def summarise(life: LifeResult, horizons: Sequence[float], qs: Sequence[float] = (0.01, 0.05, 0.1, 0.5)) -> Summary:
    p = life.failure_indicator(horizons).mean(axis=0)
    n = life.cycles.size
    return Summary(p, np.sqrt(p * (1 - p) / n), rul_quantile(life.rul, qs), life.fractions(), n)


@dataclass
class SweepPoint:
    target: str
    factor: float
    summary: Summary


def variance_sweep(formulation: Formulation, spec: InputSpec, targets: Sequence[str], factors: Sequence[float],
                   n: int, horizons: Sequence[float], rng: np.random.Generator) -> list[SweepPoint]:
    """Rescale one input's (or group's) spread at a time with nominals held fixed.

    Every point reuses the same underlying normals (common random numbers),
    so differences between points reflect the rescaling, not sampling noise.
    """
    w = spec.standard_normal(n, rng)
    return [SweepPoint(t, f, summarise(formulation.run(spec.scaled({t: f}).transform(w)), horizons))
            for t in targets for f in factors]


@dataclass
class ReductionStep:
    stage: int
    reduced: str | None             # group whose spread was cut at this stage
    factors: dict[str, float]       # cumulative spread factors after this stage
    value: float                    # objective after this stage
    candidates: dict[str, float]    # objective for each group considered at this stage


def reduction_path(formulation: Formulation, spec: InputSpec, n: int, objective: Callable[[LifeResult], float],
                   rng: np.random.Generator, factor: float = 0.5, stages: int = 6) -> list[ReductionStep]:
    """Greedy uncertainty-reduction path under a combined budget of one spread cut per stage.

    At each stage every group's spread is trialled at ``factor`` times its
    current value; the cut that most improves ``objective`` (larger is better)
    is kept. The resulting curve shows where further reductions give
    diminishing returns. Common random numbers are used throughout.
    """
    w = spec.standard_normal(n, rng)
    labels = [g for g, _ in spec.uncertain_groups()]
    current = {g: 1.0 for g in labels}
    path = [ReductionStep(0, None, dict(current), objective(formulation.run(spec.transform(w))), {})]
    for stage in range(1, stages + 1):
        candidates = {g: objective(formulation.run(spec.scaled({**current, g: current[g] * factor}).transform(w)))
                      for g in labels}
        best = max(candidates, key=candidates.get)
        current[best] *= factor
        path.append(ReductionStep(stage, best, dict(current), candidates[best], candidates))
    return path


def lower_quantile_objective(q: float = 0.05) -> Callable[[LifeResult], float]:
    """RUL lower quantile: the life that can be cleared at reliability 1 - q."""
    return lambda life: float(rul_quantile(life.rul, q)[0])


def distribution_variants(spec: InputSpec, width_factors=(0.5, 2.0), correlations=(-0.6, -0.95)) -> dict[str, InputSpec]:
    """Alternative input distributions for testing whether rankings depend on the assumptions.

    Only exploratory spreads are rescaled; data-supported ones are left alone.
    The uniform variant keeps each input's mean and sd, except where a uniform
    would cross zero for a positive (lognormal) quantity, which is kept as is.
    Correlation variants replace every declared pairwise correlation.
    """
    exploratory = [g for g, cols in spec.uncertain_groups()
                   if any(spec.marginals[spec.names[c]].basis == "exploratory" for c in cols)]
    out = {"baseline": spec}
    for f in width_factors:
        out[f"spreads x{f:g}"] = spec.scaled({g: f for g in exploratory})
    uniform = dict(spec.marginals)
    for name, m in spec.marginals.items():
        if m.spread == 0.0:
            continue
        parent = m._parent()
        mean, sd = float(parent.mean()), float(parent.std())
        if m.kind == "lognormal" and m.bounds is None and mean - np.sqrt(3.0) * sd <= 0.0:
            continue
        uniform[name] = replace(m, kind="uniform", nominal=mean, sd=sd, cov=None, hold="mean")
    out["uniform shapes"] = replace(spec, marginals=uniform)
    for rho in correlations:
        if spec.correlation:
            out[f"rho = {rho:g}"] = replace(spec, correlation={k: rho for k in spec.correlation})
    return out


def rank_agreement(results: Mapping[str, SobolResult], reference: str = "baseline", output: int = 0) -> dict[str, float]:
    """Kendall tau between each result's total-effect ranking and the reference's."""
    ref = results[reference].total[:, output]
    return {k: float(stats.kendalltau(ref, r.total[:, output]).statistic) for k, r in results.items()}


def matrix_table(row_labels: Sequence[str], col_labels: Sequence[str], values: np.ndarray, title: str = "",
                 fmt: str = "{:9.3f}") -> str:
    width = max(9, *(len(c) + 1 for c in col_labels))
    lines = [title] if title else []
    lines.append(f"{'':<14}" + "".join(f"{c:>{width}}" for c in col_labels))
    for label, row in zip(row_labels, values):
        lines.append(f"{label:<14}" + "".join(f"{fmt.format(v):>{width}}" for v in row))
    return "\n".join(lines)


def integration_convergence(formulation: Formulation, x, refinements: Sequence[float] = (4, 2, 1, 0.5)):
    """Lives with every step control scaled by each refinement, against the finest.

    Returns rows of (refinement, max |relative life error|, fraction of outcomes that differ).
    """
    lives = [replace(formulation, settings=formulation.settings.refined(f)).run(x) for f in refinements]
    ref = lives[-1]
    rows = []
    for f, life in zip(refinements, lives):
        both = life.failed & ref.failed
        rel = np.abs(life.cycles[both] / ref.cycles[both] - 1.0) if both.any() else np.array([np.nan])
        rows.append((f, float(np.max(rel)), float(np.mean(life.outcome != ref.outcome))))
    return rows
