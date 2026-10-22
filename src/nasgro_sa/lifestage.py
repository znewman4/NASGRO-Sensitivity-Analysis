"""Life-stage study: questions 1-3 at crack sizes from EICS scale to near the failure limit.

For a case and each nominal initial crack depth:
  Q1  where the remaining life is spent: shares of the cycles to failure in
      the near-threshold, Paris and near-critical regimes, and the
      probability that the crack does not propagate at all;
  Q2  what NASGRO is sensitive to, independently of how uncertain the inputs
      are: the local effect of a 10 % change in each input, and Sobol total
      indices when every input gets the same +/-10 % uniform uncertainty;
  Q3  what matters under industry-typical uncertainty: Sobol total indices
      with the uncertainty catalogue (crack-size knowledge depends on the
      detection stage), at its low, typical and high ends.
Sobol outputs are failure within the population's 5 % and 50 % remaining
life and non-propagation, which stay defined under arrest and censoring.
Population-relative horizons ask what drives the spread of life at a stage;
the case's declared horizons (fixed cycle counts, e.g. inspection intervals)
are added as further outputs and ask what drives the probability of failing
before that many cycles, which is the decision question.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cases import Case
from .inputs import InputSpec, Marginal
from .integrate import LifeResult, Outcome, rul_quantile
from .model import Formulation
from .sobol import SobolResult, saltelli_blocks, sobol_indices
from .studies import growth_rate_study

ABSOLUTE = {"R": 0.1, "theta": 10.0}   # interval-scale inputs: perturbed by +/- these amounts, not +/- 10 %
OUTPUTS = ("fail by 5% life", "fail by median life", "no propagation")


def declared_label(n: float) -> str:
    return f"fail by {n:.3g} cycles"


def screen_spec(case: Case, a: float, rel: float = 0.10) -> InputSpec:
    """Every varied input uniform within +/- rel of nominal (R and theta: +/- the ABSOLUTE amounts)."""
    m = {}
    for k, v in case.nominal_at(a).items():
        if k not in case.varied:
            m[k] = Marginal("fixed", v)
        elif k in ABSOLUTE:
            m[k] = Marginal("uniform", v, sd=ABSOLUTE[k] / np.sqrt(3.0))
        else:
            m[k] = Marginal("uniform", v, cov=rel / np.sqrt(3.0))
    return InputSpec(m, groups={"C-n": ("curve_level", "n")})


@dataclass
class LocalEffects:
    log_life: float                 # log10 nominal life (the horizon if the nominal crack does not fail)
    effect: dict[str, float]        # change in log10 life per +delta: (L(x+d) - L(x-d)) / 2
    bounded: dict[str, bool]        # a run did not fail, so the effect is bounded by the horizon


def local_effects(case: Case, a: float, rel: float = 0.10) -> LocalEffects:
    """Central-difference effect of each varied input at the nominal state.

    Tilt is symmetric about its nominal 0, so its effect is the one-sided
    change L(theta = 10 deg) - L(0).
    """
    nominal = case.nominal_at(a)
    rows = [nominal]
    for k in case.varied:
        d = ABSOLUTE.get(k, rel * abs(nominal[k]))
        rows += [{**nominal, k: nominal[k] + d}, {**nominal, k: nominal[k] - d}]
    life = case.formulation.run({k: np.array([r[k] for r in rows], dtype=float) for k in nominal})
    log_life = np.log10(np.minimum(life.rul, case.formulation.settings.horizon))
    effect, bounded = {}, {}
    for i, k in enumerate(case.varied):
        up, down = 1 + 2 * i, 2 + 2 * i
        effect[k] = log_life[up] - log_life[0] if k == "theta" else 0.5 * (log_life[up] - log_life[down])
        bounded[k] = bool((~life.failed[[0, up, down]]).any())
    return LocalEffects(float(log_life[0]), effect, bounded)


def _head(life: LifeResult, n: int) -> LifeResult:
    regimes = None if life.regime_cycles is None else life.regime_cycles[:n]
    return LifeResult(life.cycles[:n], life.outcome[:n], life.a[:n], life.c[:n], life.horizon, regimes)


@dataclass
class StageSobol:
    sobol: SobolResult
    horizons: tuple[float, float]   # the population's 5 % and 50 % remaining life (horizon if undefined)
    population: LifeResult          # the 2n independent A and B rows
    estimable: np.ndarray           # per output: enough events and non-events for stable indices


def stage_sobol(formulation: Formulation, spec: InputSpec, n: int, rng: np.random.Generator,
                n_boot: int = 200, min_events: int = 50, declared: tuple[float, ...] = ()) -> StageSobol:
    """Grouped Sobol indices of OUTPUTS, plus failure by each declared horizon."""
    groups = spec.uncertain_groups()
    w = saltelli_blocks(spec.standard_normal(n, rng), spec.standard_normal(n, rng), [c for _, c in groups])
    life = formulation.run(spec.transform(w))
    horizon = formulation.settings.horizon
    q05, q50 = (min(float(h), horizon) for h in rul_quantile(life.rul[: 2 * n], (0.05, 0.5)))
    y = np.column_stack([life.failure_indicator([q05, q50]), life.outcome == Outcome.ARREST,
                         life.failure_indicator(declared) if declared else np.empty((len(life.rul), 0))])
    labels = list(OUTPUTS) + [declared_label(h) for h in declared]
    res = sobol_indices(y, n, [lbl for lbl, _ in groups], labels, n_boot=n_boot, rng=rng)
    events = 2 * n * np.minimum(res.mean, 1.0 - res.mean)
    return StageSobol(res, (q05, q50), _head(life, 2 * n), events >= min_events)


@dataclass
class StageResult:
    a: float
    local: LocalEffects
    screen: StageSobol
    industry: dict[str, StageSobol] = field(default_factory=dict)   # by catalogue level
    rate: SobolResult | None = None                                  # log10 da/dN at the matched state
    rate_below_threshold: float = float("nan")

    def summary(self, declared: tuple[float, ...] = (), min_failed: int = 20) -> dict[str, float]:
        """Outcome probabilities, regime shares, life quantiles and P(fail by each declared horizon),
        under typical industry uncertainty.

        Regime shares are left undefined (nan) when fewer than ``min_failed`` cracks fail.
        """
        pop = self.industry["typical"].population
        enough = pop.failed.sum() >= min_failed
        shares = np.nanmean(pop.regime_shares(), axis=0) if enough else np.full(3, np.nan)
        q05, q50 = rul_quantile(pop.rul, (0.05, 0.5))
        return {"p_fail": float(pop.failed.mean()), "p_arrest": float(np.mean(pop.outcome == Outcome.ARREST)),
                "p_censored": float(np.mean(pop.outcome == Outcome.HORIZON)),
                **{f"share_{k}": float(v) for k, v in zip(("threshold", "paris", "critical"), shares)},
                "life_q05": float(q05), "life_q50": float(q50),
                **{f"p_fail_by_{h:.3g}": float(np.mean(pop.rul <= h)) for h in declared}}


def run_stage(case: Case, a: float, rng: np.random.Generator, n_screen: int = 1024, n_industry: int = 1024,
              n_levels: int = 512, n_rate: int = 1024, levels=("typical", "low", "high"),
              n_boot: int = 200) -> StageResult:
    declared = case.declared_horizons
    result = StageResult(a, local_effects(case, a), stage_sobol(case.formulation, screen_spec(case, a), n_screen,
                                                                 rng, n_boot, declared=declared))
    for level in levels:
        n = n_industry if level == "typical" else n_levels
        result.industry[level] = stage_sobol(case.formulation, case.industry(a, level), n, rng, n_boot,
                                             declared=declared)
    c = a / case.base["aspect_init"]
    result.rate, below = growth_rate_study(case.formulation, case.industry(a, "typical"), [(a, c)], n_rate, rng,
                                           n_boot)
    result.rate_below_threshold = float(below[0])
    return result


@dataclass
class RegimeMap:
    a: np.ndarray
    s: np.ndarray
    share: np.ndarray        # (3, len(s), len(a)) regime shares of the nominal crack; nan where it does not fail
    p_propagate: np.ndarray  # (len(s), len(a)) under typical industry uncertainty
    critical_at_start: np.ndarray   # (len(s), len(a)) nominal crack already at a failure limit


def regime_map(case: Case, a_grid, rng: np.random.Generator, n_mc: int = 300, level: str = "typical") -> RegimeMap:
    a_grid, s_grid = np.asarray(a_grid, dtype=float), np.asarray(case.stress_grid, dtype=float)
    cells = [(s, a) for s in s_grid for a in a_grid]
    rows = [{**case.nominal_at(a), **case.stress_inputs(s)} for s, a in cells]
    life = case.formulation.run({k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]})
    share = np.full((3, len(cells)), np.nan)
    share[:, life.failed] = life.regime_shares().T
    draws = [case.industry(a, level).with_nominal(**case.stress_inputs(s)).sample(n_mc, rng) for s, a in cells]
    mc = case.formulation.run({k: np.concatenate([d[k] for d in draws]) for k in draws[0]})
    propagates = (mc.outcome != Outcome.ARREST).reshape(len(cells), n_mc).mean(axis=1)
    shape = (len(s_grid), len(a_grid))
    critical = (life.failed & (life.cycles == 0.0)).reshape(shape)
    return RegimeMap(a_grid, s_grid, share.reshape(3, *shape), propagates.reshape(shape), critical)
