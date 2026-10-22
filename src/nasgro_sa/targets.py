"""Measurement-target analysis: which crack quality is worth measuring, and how precisely.

This is the main analysis. Its measure of RUL uncertainty is U90, the width in
decades of the central 90 % band of log10 RUL among cracks that grow to
failure (``campaign.metrics``). The median and 5 % lower percentile are
reported alongside U90, and the probability of non-propagation separately.

For each crack quality that ultrasound might characterise (size, shape,
orientation, opening/closure), the measurement standard deviation is swept
from coarse to exact while every other input keeps its industry-typical
uncertainty. Material and loading uncertainty therefore stay in the
calculation and set the floor that crack characterisation cannot remove.
Each sweep gives:
  * achievable reduction: U90 with current knowledge minus U90 with the quality known exactly;
  * required precision: the coarsest standard deviation giving REQUIRED_SHARE of that reduction;
  * diminishing-return point: the coarsest standard deviation beyond which all further
    refinement removes less than MARGINAL_SHARE of the baseline U90;
  * whether the reduction is substantial (at least SUBSTANTIAL of the baseline U90).
Reduction curves for every input/group (campaign.reduction_curves) give the
full ranking; Sobol indices are supporting only.

Orientation is tested under several mechanics (``cases.with_orientation``):
projection onto the principal plane, and mixed-mode equivalent driving force
(energy and Tanaka criteria), for aligned and 30-degree tilted cracks.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .campaign import CRACK_INPUTS, ReductionCurves, metrics, plateau_point, reduction_curves
from .cases import Case, with_orientation
from .inputs import InputSpec, Marginal
from .lifestage import StageSobol, stage_sobol
from .uncertainty import UT_CAPABILITY

MM = 1e-3
SUBSTANTIAL = 0.10      # reduction of at least 10 % of the baseline U90
REQUIRED_SHARE = 0.8    # required precision: coarsest sd giving 80 % of the achievable reduction
MARGINAL_SHARE = 0.02   # diminishing returns: all further refinement removes < 2 % of the baseline U90


@dataclass(frozen=True)
class Candidate:
    """A crack quality that a measurement could constrain."""

    key: str                       # "size", "shape", "orientation" or "opening"
    input: str                     # the model input the measurement sets
    unit: str
    mechanics: str = "projection"  # orientation mechanics: "projection", "energy" or "tanaka"
    tilt: float = 0.0              # mean tilt of the crack from the principal plane (deg)

    @property
    def name(self) -> str:
        if self.key == "orientation":
            return f"orientation ({self.mechanics}, {self.tilt:g} deg)"
        return {"size": "size (depth)", "shape": "shape (a/c)", "opening": "opening / closure"}[self.key]

    def grid(self, a: float) -> tuple[float, ...]:
        """Measurement standard deviations, coarse to fine on a halving grid, ending at exact (0)."""
        if self.key == "size":
            return tuple(a / MM * 2.0 ** -k for k in range(6)) + (0.0,)
        start = {"shape": 0.4, "orientation": 32.0, "opening": 0.4}[self.key]
        return tuple(start * 2.0 ** -k for k in range(5)) + (0.0,)

    def measured(self, current: Marginal, sd: float) -> Marginal:
        """The input's distribution after a measurement with standard deviation ``sd`` (in ``unit``)."""
        if self.key == "opening":
            return Marginal("lognormal", current.nominal, cov=sd, hold="median")
        return Marginal("normal", current.nominal, sd=sd * (MM if self.key == "size" else 1.0), bounds=current.bounds)

    def applies(self, spec: InputSpec) -> bool:
        return self.input in spec.marginals and spec.marginals[self.input].spread > 0.0


SIZE = Candidate("size", "a_init", "mm")
SHAPE = Candidate("shape", "aspect_init", "sd of a/c")
OPENING = Candidate("opening", "closure_factor", "cov of kappa")
ORIENTATION = tuple(Candidate("orientation", "theta", "deg", m, t)
                    for m, t in (("projection", 0.0), ("projection", 30.0), ("energy", 30.0), ("tanaka", 30.0)))
CANDIDATES = (SIZE, SHAPE, *ORIENTATION, OPENING)
SCENARIO_CANDIDATES = (SIZE, SHAPE, ORIENTATION[0], ORIENTATION[2], OPENING)
UT_KEY = {"size": "size", "shape": "shape", "orientation": "orientation", "opening": "opening"}


def _with(spec: InputSpec, **marginals: Marginal) -> InputSpec:
    return replace(spec, marginals={**spec.marginals, **marginals})


def _bootstrap_se(life, rng: np.random.Generator, n_boot: int = 200) -> float:
    """Bootstrap standard error of U90, an indication of Monte Carlo noise in the baseline."""
    logs = np.log10(np.maximum(life.cycles[life.failed], 1.0))
    if logs.size < 200:
        return float("nan")
    draws = rng.choice(logs, size=(n_boot, logs.size))
    q = np.percentile(draws, (5, 95), axis=1)
    return float(np.std(q[1] - q[0]))


@dataclass
class Sweep:
    candidate: Candidate
    a: float
    scenario: str
    level: str
    baseline: dict[str, float]      # current (catalogue) knowledge of every input
    floor: dict[str, float]         # every crack quality known exactly; material, loading etc. unchanged
    sds: tuple[float, ...]
    points: list[dict[str, float]]  # metrics at each measurement sd

    def values(self, key: str) -> np.ndarray:
        return np.array([p[key] for p in self.points])

    def reduction(self) -> np.ndarray:
        """U90 reduction (decades) at each measurement sd; negative = worse than current knowledge."""
        return self.baseline["u90"] - self.values("u90")

    @property
    def achievable(self) -> float:
        return float(self.reduction()[-1])

    @property
    def achievable_share(self) -> float:
        return self.achievable / self.baseline["u90"]

    @property
    def substantial(self) -> bool:
        return bool(np.isfinite(self.achievable_share) and self.achievable_share >= SUBSTANTIAL)

    @property
    def required_sd(self) -> float:
        """Coarsest measurement sd giving REQUIRED_SHARE of the achievable reduction (nan if none)."""
        return plateau_point(self.sds, self.reduction(), REQUIRED_SHARE) if self.achievable > 0 else float("nan")

    @property
    def diminishing_sd(self) -> float:
        """Coarsest sd beyond which all further refinement removes < MARGINAL_SHARE of the baseline U90."""
        red, limit = self.reduction(), MARGINAL_SHARE * self.baseline["u90"]
        if not np.all(np.isfinite(red)):
            return float("nan")
        for i in range(len(red)):
            if red[-1] - red[i] < limit:
                return float(self.sds[i])
        return 0.0

    def shift(self, key: str, i: int = -1) -> float:
        """Change in a metric (e.g. log_q50, log_q05) at grid point i relative to current knowledge."""
        return self.points[i][key] - self.baseline[key]

    @property
    def crack_removable_share(self) -> float:
        """Share of the baseline U90 that exact knowledge of every crack quality would remove."""
        return (self.baseline["u90"] - self.floor["u90"]) / self.baseline["u90"]

    @property
    def coarse_achievable(self) -> float:
        """U90 reduction from the coarsest measurement on the grid (an uninformed reference) to exact."""
        u90 = self.values("u90")
        return float(u90[0] - u90[-1])

    @property
    def coarse_required_sd(self) -> float:
        """Coarsest sd giving REQUIRED_SHARE of the reduction measured from the coarsest grid point."""
        u90 = self.values("u90")
        return plateau_point(self.sds, u90[0] - u90, REQUIRED_SHARE) if self.coarse_achievable > 0 else float("nan")


@dataclass
class StateResult:
    case: str
    a: float
    scenario: str
    level: str
    sweeps: dict[str, Sweep] = field(default_factory=dict)       # by candidate name
    curves: ReductionCurves | None = None                        # every input/group (projection, aligned)
    beyond_size: dict[str, float] = field(default_factory=dict)  # U90 reduction from shape once size is measured
    samples: dict[str, np.ndarray] = field(default_factory=dict)  # log10 RUL of failing cracks, for figures
    sobol: StageSobol | None = None                              # supporting: typical-uncertainty Sobol study


def evaluate_state(case: Case, a: float, rng: np.random.Generator, scenario: tuple[str, dict] | None = None,
                   level: str = "typical", candidates=CANDIDATES, n: int = 2000, all_groups: bool = False,
                   factors=(1.0, 0.5, 0.25, 0.1, 0.0), keep_samples: bool = False, sobol_n: int = 0) -> StateResult:
    """Every candidate sweep at one crack state, scenario and catalogue level.

    Candidates sharing orientation mechanics and tilt share a baseline, a floor
    and random numbers, so their sweeps are directly comparable.
    """
    name, overrides = scenario or case.scenarios[0]
    out = StateResult(case.name, a, name, level)
    for mechanics, tilt in dict.fromkeys((c.mechanics, c.tilt) for c in candidates):
        variant = with_orientation(case, mechanics)
        spec = case.industry(a, level).with_nominal(**overrides, **({"theta": tilt} if tilt else {}))
        w = spec.standard_normal(n, rng)

        def run(s: InputSpec, keep: str | None = None, se: bool = False) -> dict[str, float]:
            life = variant.formulation.run(s.transform(w))
            if keep_samples and keep:
                out.samples[keep] = np.log10(np.maximum(life.cycles[life.failed], 1.0))
            m = metrics(life, case.declared_horizons)
            if se:
                m["u90_se"] = _bootstrap_se(life, rng)
            return m

        tag = f"{mechanics}, {tilt:g} deg"
        baseline = run(spec, f"baseline ({tag})", se=True)
        floor = run(_with(spec, **{k: Marginal("fixed", spec.marginals[k].nominal)
                                   for k in CRACK_INPUTS if k in spec.marginals}), f"floor ({tag})")
        for cand in (c for c in candidates if (c.mechanics, c.tilt) == (mechanics, tilt) and c.applies(spec)):
            current = spec.marginals[cand.input]
            sds = cand.grid(a)
            points = [run(_with(spec, **{cand.input: cand.measured(current, sd)}), cand.name if sd == 0 else None)
                      for sd in sds]
            out.sweeps[cand.name] = Sweep(cand, a, name, level, baseline, floor, sds, points)

        if (mechanics, tilt) == ("projection", 0.0):
            if SHAPE.applies(spec) and SIZE.applies(spec):
                size_ut = UT_CAPABILITY["size"][1]
                sized = {"exact": SIZE.measured(spec.marginals["a_init"], 0.0),
                         f"UT-typical ({size_ut:g} mm)": SIZE.measured(spec.marginals["a_init"], size_ut)}
                exact_shape = SHAPE.measured(spec.marginals["aspect_init"], 0.0)
                for label, size_marginal in sized.items():
                    alone = run(_with(spec, a_init=size_marginal))["u90"]
                    both = run(_with(spec, a_init=size_marginal, aspect_init=exact_shape))["u90"]
                    out.beyond_size[f"size {label}"] = alone - both
            if all_groups:
                out.curves = reduction_curves(case, a, n, rng, factors, spec=spec, formulation=variant.formulation)
            if sobol_n:
                out.sobol = stage_sobol(variant.formulation, spec, sobol_n, rng, n_boot=0)
    return out


@dataclass
class TargetRun:
    """All results for one case."""

    case: Case
    base: list[StateResult]                         # base scenario, typical uncertainty, every state
    scenarios: list[StateResult]                    # key states x scenarios (incl. base), typical uncertainty
    levels: list[StateResult]                       # key states, base scenario, catalogue low and high


def run_targets(case: Case, rng: np.random.Generator, states=None, scenario_names=None, n: int = 2000,
                sobol_n: int = 512) -> TargetRun:
    states = states or case.stages
    scenarios = [s for s in case.scenarios if scenario_names is None or s[0] in scenario_names]
    base = [evaluate_state(case, a, rng, n=n, all_groups=True, keep_samples=True, sobol_n=sobol_n) for a in states]
    by_state = {r.a: r for r in base}
    scen = [by_state[a] if s is case.scenarios[0] and a in by_state else
            evaluate_state(case, a, rng, scenario=s, candidates=SCENARIO_CANDIDATES, n=n)
            for a in case.key_stages for s in scenarios]
    levels = [evaluate_state(case, a, rng, level=lv, candidates=SCENARIO_CANDIDATES, n=n)
              for a in case.key_stages for lv in ("low", "high")]
    return TargetRun(case, base, scen, levels)
