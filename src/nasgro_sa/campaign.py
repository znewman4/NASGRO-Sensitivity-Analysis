"""Campaign runs on the life-stage cases: what better knowledge buys, and how far the rankings carry over.

  reduction_curves       each group's spread reduced progressively, everything else unchanged
  paired_reductions      two groups reduced together versus separately (complementarity)
  physics_vs_knowledge   rankings with equal +/-10 % uncertainty (physics only), with catalogue
                         uncertainty but crack knowledge held constant across stages, and with
                         stage-dependent crack knowledge
  scenario_matrix        rankings across stress, stress-ratio and geometry scenarios
  variant_study          rankings under alternative distribution shapes, widths and dependencies
  model_comparison       consistently calibrated growth models against the benefit of precise inputs

The starting point is the case's typical industry uncertainty, and runs that
are compared share their random numbers. The declared measure of RUL
uncertainty is U90, the width in decades of the central 90 % band of log10
RUL among cracks that grow to failure; the median and 5 % lower percentile
are reported alongside it and the probability of non-propagation separately.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Callable

import numpy as np
from scipy import stats

from .baseline import baseline_inputs
from .calibrate import Calibration, calibrate, standard_fit_specs, synthetic_lots
from .cases import Case
from .growth import NasgroEquation, ParisLaw
from .inputs import EmpiricalGroup, InputSpec, Marginal
from .integrate import LifeResult, Outcome
from .lifestage import OUTPUTS, StageSobol, screen_spec, stage_sobol
from .model import Formulation

MEDIAN = OUTPUTS.index("fail by median life")
CRACK_INPUTS = ("a_init", "aspect_init", "theta", "closure_factor")
OTHER_PHYSICAL_INPUTS = ("S_max", "F", "R", "K_factor", "D", "thickness", "width")


MIN_FAILURES = 200   # cracks that must fail before the RUL band is estimated


def metrics(life: LifeResult, declared=()) -> dict[str, float]:
    """The declared RUL-uncertainty measure and its companions.

    Over cracks that grow to failure: u90 = log10(N95 / N05), the width in
    decades of the central 90 % band (the declared spread measure), and the
    lower percentile log_q05 and median log_q50 (log10 cycles); nan when fewer
    than MIN_FAILURES cracks fail. Also the probability of non-propagation and
    of failure within each declared horizon.
    """
    failed = life.failed
    out = {"p_no_propagation": float(np.mean(life.outcome == Outcome.ARREST)), "p_fail": float(np.mean(failed))}
    if failed.sum() >= MIN_FAILURES:
        q05, q50, q95 = np.percentile(np.log10(np.maximum(life.cycles[failed], 1.0)), (5, 50, 95))
        out.update(log_q05=float(q05), log_q50=float(q50), log_q95=float(q95), u90=float(q95 - q05))
    else:
        out.update({k: float("nan") for k in ("log_q05", "log_q50", "log_q95", "u90")})
    out.update({f"p_fail_by_{h:.3g}": float(np.mean(life.rul <= h)) for h in declared})
    return out


def total_indices(res: StageSobol, output: int = MEDIAN) -> dict[str, float]:
    """Total indices by group for one output; empty when the output is not estimable."""
    if not res.estimable[output]:
        return {}
    return dict(zip(res.sobol.groups, res.sobol.total[:, output]))


def rank_agreement(x: dict[str, float], y: dict[str, float]) -> float:
    """Kendall tau over the groups both rankings share."""
    common = [k for k in x if k in y]
    if len(common) < 3:
        return float("nan")
    return float(stats.kendalltau([x[k] for k in common], [y[k] for k in common]).statistic)


def top_k(x: dict[str, float], k: int = 3) -> list[str]:
    return sorted(x, key=lambda g: -x[g])[:k]


# ---------------------------------------------------------------- value of reducing uncertainty

def plateau_point(xs, reductions, share: float = 0.8) -> float:
    """First x (in the given coarse-to-fine order) at which ``share`` of the final reduction is reached.

    Linear interpolation between grid points; nan when the final reduction is
    not positive.
    """
    xs, red = np.asarray(xs, dtype=float), np.asarray(reductions, dtype=float)
    if not np.isfinite(red[-1]) or red[-1] <= 0.0:
        return float("nan")
    target = share * red[-1]
    for i in range(len(xs)):
        if red[i] >= target:
            if i == 0 or red[i] <= red[i - 1]:
                return float(xs[i])
            return float(xs[i - 1] + (target - red[i - 1]) / (red[i] - red[i - 1]) * (xs[i] - xs[i - 1]))
    return float(xs[-1])


@dataclass
class ReductionCurves:
    a: float
    base: dict[str, float]
    factors: tuple[float, ...]                         # remaining fraction of each group's spread, 1 -> 0
    curves: dict[str, list[dict[str, float]]]          # group -> metrics at each factor

    def shifts(self, group: str, key: str) -> np.ndarray:
        """Change in a metric along the curve, relative to current knowledge."""
        return np.array([m[key] - self.base[key] for m in self.curves[group]])

    def reduction(self, group: str) -> np.ndarray:
        """Reduction in U90 (decades) along the curve; positive = less uncertainty."""
        return -self.shifts(group, "u90")

    def achievable(self, group: str) -> float:
        return float(self.reduction(group)[-1])

    def plateau(self, group: str, share: float = 0.8) -> float:
        """Remaining spread fraction at which ``share`` of the achievable U90 reduction is reached."""
        return plateau_point(self.factors, self.reduction(group), share)

    def ranking(self) -> list[str]:
        """Groups ordered by achievable U90 reduction (largest first)."""
        return sorted(self.curves, key=lambda grp: -np.nan_to_num(self.achievable(grp), nan=-np.inf))


def reduction_curves(case: Case, a: float, n: int, rng: np.random.Generator,
                     factors=(1.0, 0.75, 0.5, 0.25, 0.1, 0.0), spec: InputSpec | None = None,
                     formulation: Formulation | None = None) -> ReductionCurves:
    """Reduce one input/group's spread at a time, for every uncertain group, others unchanged."""
    spec = spec or case.industry(a, "typical")
    formulation = formulation or case.formulation
    w = spec.standard_normal(n, rng)
    run = lambda s: metrics(formulation.run(s.transform(w)), case.declared_horizons)
    base = run(spec)
    curves = {grp: [base if f == 1.0 else run(spec.scaled({grp: f})) for f in factors]
              for grp, _ in spec.uncertain_groups()}
    return ReductionCurves(a, base, tuple(factors), curves)


@dataclass
class PairedReductions:
    a: float
    groups: list[str]
    factor: float
    single: dict[str, float]                 # U90 reduction (decades) with one group reduced
    pair: dict[tuple[str, str], float]       # U90 reduction with both reduced

    def synergy(self) -> np.ndarray:
        """Diagonal: single reductions. Off-diagonal: pair minus the two singles (> 0: complementary)."""
        k = len(self.groups)
        m = np.zeros((k, k))
        for i, gi in enumerate(self.groups):
            m[i, i] = self.single[gi]
            for j, gj in enumerate(self.groups[i + 1:], start=i + 1):
                m[i, j] = m[j, i] = self.pair[(gi, gj)] - self.single[gi] - self.single[gj]
        return m


def paired_reductions(case: Case, a: float, groups: list[str], n: int, rng: np.random.Generator,
                      factor: float = 0.25) -> PairedReductions:
    """Each group's spread cut to ``factor`` alone, and every pair cut together; U90 reductions."""
    spec = case.industry(a, "typical")
    w = spec.standard_normal(n, rng)
    run = lambda s: metrics(case.formulation.run(s.transform(w)))["u90"]
    base = run(spec)
    single = {grp: base - run(spec.scaled({grp: factor})) for grp in groups}
    pair = {(g1, g2): base - run(spec.scaled({g1: factor, g2: factor})) for g1, g2 in combinations(groups, 2)}
    return PairedReductions(a, list(groups), factor, single, pair)


# ---------------------------------------------------------------- what makes the ranking change

@dataclass
class KnowledgeStage:
    a: float
    screen: StageSobol      # every input +/-10 %: physics only
    constant: StageSobol    # catalogue uncertainty, crack knowledge the same at every stage
    staged: StageSobol      # catalogue uncertainty, crack knowledge by detection stage


def physics_vs_knowledge(case: Case, n: int, rng: np.random.Generator, n_boot: int = 0) -> list[KnowledgeStage]:
    return [KnowledgeStage(a,
                           stage_sobol(case.formulation, screen_spec(case, a), n, rng, n_boot),
                           stage_sobol(case.formulation, case.industry(a, "typical", knowledge="constant"), n, rng,
                                       n_boot),
                           stage_sobol(case.formulation, case.industry(a, "typical"), n, rng, n_boot))
            for a in case.stages]


def scenario_matrix(case: Case, n: int, rng: np.random.Generator,
                    n_boot: int = 0) -> dict[float, dict[str, StageSobol]]:
    """Typical-uncertainty Sobol study for every scenario at every key stage."""
    out = {}
    for a in case.key_stages:
        spec = case.industry(a, "typical")
        out[a] = {name: stage_sobol(case.formulation, spec.with_nominal(**over) if over else spec, n, rng, n_boot)
                  for name, over in case.scenarios}
    return out


_Z_GRID = stats.norm.ppf((np.arange(20_001) + 0.5) / 20_001)


def _reshape(spec: InputSpec, kind: str) -> InputSpec:
    """Every uncertain marginal as a uniform (or normal ones as lognormal) with the same realised mean and sd.

    Realised moments include any truncation, so a bounded input keeps the
    spread it actually has, not that of its untruncated parent.
    """
    out = dict(spec.marginals)
    for name, m in spec.marginals.items():
        if m.spread == 0.0:
            continue
        values = m.from_standard_normal(_Z_GRID)
        mean, sd = float(np.mean(values)), float(np.std(values))
        if kind == "uniform":
            if m.kind == "lognormal" and m.bounds is None and mean - np.sqrt(3.0) * sd <= 0.0:
                continue
            out[name] = replace(m, kind="uniform", nominal=mean, sd=sd, cov=None, hold="mean")
        elif kind == "lognormal" and m.kind == "normal" and mean > 0.0:
            out[name] = replace(m, kind="lognormal", nominal=mean, cov=sd / mean, sd=None, hold="mean")
    return replace(spec, marginals=out)


def _scale_judgement(spec: InputSpec, factor: float) -> InputSpec:
    """Rescale groups whose uncertain members all rest on engineering judgement (not literature)."""
    groups = []
    for label, cols in spec.uncertain_groups():
        members = [spec.marginals[spec.names[c]] for c in cols]
        if label not in spec.empirical and all(m.basis == "exploratory" for m in members if m.spread > 0):
            groups.append(label)
    return spec.scaled({grp: factor for grp in groups})


def _independent_empirical(spec: InputSpec, rng: np.random.Generator) -> InputSpec:
    """Each empirical group's columns shuffled separately, keeping marginals but breaking the joint fit."""
    emp = {label: replace(grp, samples={k: np.asarray(v)[rng.permutation(grp.size)] for k, v in grp.samples.items()})
           for label, grp in spec.empirical.items()}
    return replace(spec, empirical=emp)


def distribution_variants(case: Case, a: float, rng: np.random.Generator) -> dict[str, InputSpec]:
    spec = case.industry(a, "typical")
    out = {"typical": spec, "catalogue low": case.industry(a, "low"), "catalogue high": case.industry(a, "high"),
           "judgement x0.5": _scale_judgement(spec, 0.5), "judgement x2": _scale_judgement(spec, 2.0),
           "uniform shapes": _reshape(spec, "uniform"), "lognormal shapes": _reshape(spec, "lognormal")}
    if "C-n" in spec.groups:
        for rho in (-0.6, 0.6):
            out[f"C-n level/slope rho={rho:+g}"] = replace(spec, correlation={("curve_level", "n"): rho})
    if spec.empirical:
        out["growth fit made independent"] = _independent_empirical(spec, rng)
    return out


def variant_study(case: Case, n: int, rng: np.random.Generator, n_boot: int = 0) -> dict[float, dict[str, StageSobol]]:
    return {a: {name: stage_sobol(case.formulation, s, n, rng, n_boot)
                for name, s in distribution_variants(case, a, rng).items()} for a in case.key_stages}


# ---------------------------------------------------------------- model form versus input precision

@dataclass
class ModelEntry:
    formulation: Formulation
    spec: Callable[[float], InputSpec]     # nominal crack depth -> input specification


MATERIAL = ("p", "q", "dK0", "Cth_pos", "Cth_neg", "alpha", "smax_sigma0")


def _calibrated_spec(spec: InputSpec, cal: Calibration) -> InputSpec:
    """Case inputs with the catalogue material replaced by a calibrated parameter group.

    The closure constants are held at the values used in the fit (calibration
    absorbs them); Kc and the intrinsic crack length keep their catalogue
    uncertainty, since long-crack test data do not identify them.
    """
    fitted = cal.input_spec(spec)
    m = {k: v for k, v in spec.marginals.items() if k != "curve_level"}
    m.update({k: Marginal("fixed", spec.marginals[k].nominal) for k in MATERIAL if k in m})
    m.update({k: fitted.marginals[k] for k in cal.fit.free})
    return InputSpec(m, groups={"growth fit": cal.fit.free}, correlation=fitted.correlation)


def plate_models(case: Case, rng: np.random.Generator, n_lots: int = 40) -> dict[str, ModelEntry]:
    """NASGRO, Walker and Paris, each fitted to the same synthetic lots from the plate's material population."""
    base = baseline_inputs()
    lots = synthetic_lots(NasgroEquation(), base, n_lots, rng)
    settings = replace(case.formulation.settings, track_regimes=False)
    out = {}
    for name, fit in standard_fit_specs(base, case.base["R"]).items():
        cal = calibrate(fit, lots)
        out[name] = ModelEntry(replace(case.formulation, rate_model=fit.model, settings=settings),
                               lambda a, cal=cal: _calibrated_spec(case.industry(a, "typical"), cal))
    return out


def bar_models(case: Case, paris_pool: dict[str, np.ndarray]) -> dict[str, ModelEntry]:
    """NASGRO (accepted pool) and Paris accepted against the Paris-regime part of the same data.

    Walker is omitted: with data at a single stress ratio its gamma is not identifiable.
    """
    settings = replace(case.formulation.settings, track_regimes=False)
    ref = {"C": float(np.median(paris_pool["C"])), "n": float(np.median(paris_pool["n"]))}

    def paris_spec(a):
        spec = case.industry(a, "typical")
        return replace(spec, empirical={"growth fit": EmpiricalGroup(paris_pool, ref, log=("C",))})

    return {"NASGRO": ModelEntry(replace(case.formulation, settings=settings), lambda a: case.industry(a, "typical")),
            "Paris": ModelEntry(replace(case.formulation, rate_model=ParisLaw(), settings=settings), paris_spec)}


@dataclass
class ModelComparison:
    a: float
    models: dict[str, dict[str, float]]   # metrics per model
    benefit: dict[str, float]             # reference model: U90 reduction when inputs are known exactly
    disagreement: dict[str, float]        # range across models of log10 q05, log10 q50 and U90 (decades)


def model_comparison(case: Case, models: dict[str, ModelEntry], n: int, rng: np.random.Generator,
                     reference: str = "NASGRO") -> list[ModelComparison]:
    out = []
    for a in case.key_stages:
        mets = {name: metrics(e.formulation.run(e.spec(a).sample(n, rng)), case.declared_horizons)
                for name, e in models.items()}
        ref = models[reference]
        spec = ref.spec(a)
        uncertain = {lbl for lbl, _ in spec.uncertain_groups()}
        w = spec.standard_normal(n, rng)
        run = lambda s: metrics(ref.formulation.run(s.transform(w)))["u90"]
        base = run(spec)
        crack = [k for k in CRACK_INPUTS if k in uncertain]
        inputs = crack + [k for k in OTHER_PHYSICAL_INPUTS if k in uncertain]
        benefit = {"perfect crack characterisation": base - run(spec.scaled({k: 0.0 for k in crack})),
                   "all non-material inputs known": base - run(spec.scaled({k: 0.0 for k in inputs}))}
        with np.errstate(invalid="ignore"):
            disagreement = {key: max(m[key] for m in mets.values()) - min(m[key] for m in mets.values())
                            for key in ("log_q05", "log_q50", "u90")}
        out.append(ModelComparison(a, mets, benefit, disagreement))
    return out
