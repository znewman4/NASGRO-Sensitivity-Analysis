"""Case definitions for the life-stage study.

Every NASGRO input that the formulation uses is exposed, plus crack tilt,
dimensional tolerances and the SIF solution error:
  * the growth-rate curve (C-n) is parameterised by its level at a pivot dK
    ("curve_level", a multiplier) and its slope n about that pivot, and the
    two are always grouped, since C and n are fitted jointly;
  * crack tilt theta is handled by projecting the flaw onto the principal
    plane (as in flaw re-characterisation rules): depth a cos(theta), surface
    length unchanged;
  * "K_factor" is the error of the SIF solution itself (Newman-Raju,
    James-Mills) when its inputs (stress, crack size and shape, dimensions)
    are known exactly, applied as a constant multiplier on K. It is not a
    NASGRO input, and because NASGRO sees K only through dK and Kmax it acts
    exactly like a stress change. It is therefore left out of ``varied`` (the
    model-sensitivity screen) and kept as an uncertainty source in
    ``industry``.

Two cases:
  * round_bar_case: Giannella (2022) round bar, 25CrMo4, R = -1. Material is
    the accepted (C, m, p, dKth0) pool, i.e. calibrated at the service stress
    ratio, so the closure constants and q are absorbed by the fit and held
    fixed in the industry-uncertainty analysis.
  * plate_case: surface crack in a plate, placeholder aluminium values,
    R = 0.1. Material constants come from a database with literature-typical
    scatter; closure and short-crack constants are defaulted, so they carry
    epistemic uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from . import giannella as g
from .geometry import MixedModeOrientation, RoundBarStraightCrack, SampledSurfaceCrackPlate
from .growth import NasgroEquation
from .inputs import EmpiricalGroup, InputSpec, Marginal
from .integrate import FailureCriterion, IntegrationSettings
from .model import Formulation
from .uncertainty import CATALOGUE, crack_marginals

MM = 1e-3

LABELS = {
    "a_init": "crack depth a", "aspect_init": "aspect ratio a/c", "theta": "crack tilt theta",
    "S_max": "stress amplitude", "F": "load amplitude F", "R": "stress ratio R", "K_factor": "SIF solution error",
    "D": "diameter D", "thickness": "thickness t", "width": "width W",
    "C-n": "growth curve C-n", "curve_level": "C-n level", "n": "C-n slope",
    "growth fit": "growth fit (C,m,p,dKth0)",
    "dK0": "threshold dKth0", "p": "threshold shape p", "q": "critical shape q", "Kc": "toughness Kc",
    "alpha": "constraint alpha", "smax_sigma0": "Smax/sigma0", "Cth_pos": "Cth (R>=0)", "Cth_neg": "Cth (R<0)",
    "a0_intrinsic": "intrinsic length a0", "closure_factor": "crack opening factor",
}
CATEGORY = {
    "a_init": "crack", "aspect_init": "crack", "theta": "crack", "closure_factor": "crack",
    "S_max": "loading", "F": "loading", "R": "loading", "K_factor": "analysis",
    "D": "geometry", "thickness": "geometry", "width": "geometry",
    "C-n": "growth curve", "curve_level": "growth curve", "n": "growth curve", "growth fit": "growth curve",
    "dK0": "threshold", "p": "threshold", "a0_intrinsic": "threshold",
    "q": "fracture", "Kc": "fracture", "alpha": "closure", "smax_sigma0": "closure", "Cth_pos": "closure",
    "Cth_neg": "closure",
}
CATEGORY_ORDER = ("crack", "loading", "analysis", "geometry", "growth curve", "threshold", "fracture", "closure")


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    formulation: Formulation
    base: dict[str, float]                          # nominal inputs except crack depth (SI, MPa, kN, deg)
    varied: tuple[str, ...]                         # inputs perturbed in the local and screening analyses
    stages: tuple[float, ...]                       # nominal initial crack depths (m)
    a_limit: float                                  # crack depth at the failure limit (m)
    stress_nominal: float                           # nominal stress amplitude (MPa)
    stress_grid: tuple[float, ...]                  # stresses for the regime map (MPa)
    stress_inputs: Callable[[float], dict[str, float]]
    industry: Callable[..., InputSpec]              # (nominal depth, level, knowledge="staged") -> InputSpec
    key_stages: tuple[float, ...] = ()              # small / first detectable / deep, for the campaign runs (m)
    declared_horizons: tuple[float, ...] = ()       # fixed decision horizons, e.g. inspection intervals (cycles)
    scenarios: tuple[tuple[str, dict], ...] = ()    # (name, nominal overrides); the first is the base case
    notes: tuple[str, ...] = ()

    def nominal_at(self, a: float) -> dict[str, float]:
        return {**self.base, "a_init": a}


@dataclass(frozen=True)
class Derive:
    """C from (curve_level, n) about the pivot, crack tilt, and the K factor on the stress.

    ``orientation="projection"`` re-characterises a tilted crack as its
    projection on the principal plane (depth a cos(theta)); any other value
    keeps the full depth and passes it as ``a_start`` to a MixedModeOrientation
    geometry, which applies the mixed-mode driving force.
    """

    c_ref: float
    n_ref: float
    dk_eff_pivot: float
    stress: Callable | None = None
    orientation: str = "projection"

    def __call__(self, x):
        out = {}
        if "curve_level" in x:
            out["C"] = self.c_ref * np.asarray(x["curve_level"]) * self.dk_eff_pivot ** (self.n_ref - np.asarray(x["n"]))
        if "theta" in x:
            if self.orientation == "projection":
                cos = np.cos(np.radians(np.asarray(x["theta"])))
                out["a_init"] = np.asarray(x["a_init"]) * cos
                out["aspect_init"] = np.asarray(x["aspect_init"]) * cos
            else:
                out["a_start"] = np.asarray(x["a_init"], dtype=float)
        s = self.stress(x) if self.stress else np.asarray(x["S_max"])
        out["S_max"] = s * np.asarray(x.get("K_factor", 1.0))
        return out


def with_orientation(case: Case, mechanics: str) -> Case:
    """The case with its crack-tilt mechanics switched: "projection" (default), "energy" or "tanaka"."""
    if mechanics == "projection":
        return case
    f = case.formulation
    return replace(case, formulation=replace(f, geometry=MixedModeOrientation(f.geometry, mechanics),
                                             derive=replace(f.derive, orientation=mechanics)))


def _effective_pivot(params: dict, r: float, dk_pivot: float) -> float:
    return float(NasgroEquation().threshold(r, 1.0, params)[1]) * dk_pivot


def _catalogue_marginals(names, nominal, level):
    return {k: CATALOGUE[k].marginal(nominal[k], level) for k in names}


def round_bar_case(pool: dict[str, np.ndarray]) -> Case:
    fixed = {**g.FIXED, **g.BEST_FIT}
    pivot = _effective_pivot(fixed, g.R, 33.0)   # geometric centre of the 12-90 MPa sqrt(m) test range
    base = {
        "aspect_init": 1.0, "theta": 0.0, "D": 18.0 * MM, "F": 63.6, "R": g.R, "K_factor": 1.0,
        "curve_level": 1.0, "n": g.BEST_FIT["n"], "p": g.BEST_FIT["p"], "q": g.FIXED["q"], "dK0": g.BEST_FIT["dK0"],
        "Cth_pos": g.FIXED["Cth_pos"], "Cth_neg": g.FIXED["Cth_neg"], "alpha": g.FIXED["alpha"],
        "smax_sigma0": g.FIXED["smax_sigma0"], "a0_intrinsic": g.FIXED["a0_intrinsic"], "Kc": g.FIXED["Kc"],
        "closure_factor": 1.0,
    }
    formulation = Formulation(
        RoundBarStraightCrack(), NasgroEquation(), FailureCriterion(depth_ratio=0.5),
        IntegrationSettings(horizon=1e10, track_regimes=True),
        derive=Derive(g.BEST_FIT["C"], g.BEST_FIT["n"], pivot, lambda x: g.nominal_stress(x)["S_max"]))
    # Reduced material uncertainty shrinks the accepted sets towards the centre of the pool (componentwise
    # median, log scale for C), so the joint fit is kept and the median life does not shift.
    centre = {k: float(10 ** np.median(np.log10(v))) if k == "C" else float(np.median(v)) for k, v in pool.items()}
    d = base["D"]
    load = lambda s, dia=d: {"F": s * np.pi * dia**2 / 4 * 1e6 / 1e3}
    # R is fixed at the calibration value: the material pool was fitted at R = -1 only
    scenarios = (("S=250 (base)", {}), ("S=200", load(200.0)), ("S=300", load(300.0)), ("S=400", load(400.0)),
                 ("D=30mm S=250", {"D": 30.0 * MM, **load(250.0, 30.0 * MM)}))

    def industry(a: float, level: str, knowledge: str = "staged") -> InputSpec:
        crack = crack_marginals(a, 1.0, level, 0.5 * d, knowledge)
        m = {k: Marginal("fixed", v) for k, v in {**base, **g.BEST_FIT}.items() if k != "curve_level"}
        m.update(a_init=crack["a_init"], theta=crack["theta"])
        m.update(_catalogue_marginals(("F", "D", "R", "K_factor", "Kc", "a0_intrinsic", "closure_factor"), base,
                                      level))
        return InputSpec(m, empirical={"growth fit": EmpiricalGroup(pool, centre, log=("C",))})

    return Case(
        "round_bar", "Round bar, 25CrMo4, R = -1 (Giannella 2022)", formulation, base,
        varied=("a_init", "theta", "D", "F", "R", "curve_level", "n", "p", "q", "dK0", "Cth_neg",
                "alpha", "smax_sigma0", "a0_intrinsic", "Kc", "closure_factor"),
        stages=tuple(v * MM for v in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0)), a_limit=0.5 * d,
        stress_nominal=250.0, stress_grid=tuple(np.linspace(120.0, 500.0, 12)),
        stress_inputs=load,
        industry=industry, key_stages=(0.5 * MM, 1.0 * MM, 4.0 * MM), declared_horizons=(3e4, 1e5, 3e5),
        scenarios=scenarios,
        notes=("Material uncertainty = accepted (C, m, p, dKth0) pool, calibrated at the service R = -1, so q and "
               "the closure constants are absorbed by the fit and held fixed under industry uncertainty.",))


def plate_case() -> Case:
    base = {
        "aspect_init": 0.8, "theta": 0.0, "thickness": 10.0 * MM, "width": 100.0 * MM, "S_max": 200.0, "R": 0.1,
        "K_factor": 1.0, "curve_level": 1.0, "n": 3.0, "p": 0.5, "q": 1.0, "dK0": 2.8, "Cth_pos": 1.5,
        "Cth_neg": 0.1, "alpha": 2.0, "smax_sigma0": 0.3, "a0_intrinsic": 0.0381 * MM, "Kc": 40.0,
        "closure_factor": 1.0,
    }
    c_ref = 2.5e-10
    pivot = _effective_pivot({**base, "C": c_ref}, base["R"], 10.0)
    formulation = Formulation(
        SampledSurfaceCrackPlate(), NasgroEquation(), FailureCriterion(depth_ratio=0.8, width_ratio=0.5),
        IntegrationSettings(horizon=1e9, track_regimes=True), derive=Derive(c_ref, base["n"], pivot))
    varied = ("a_init", "aspect_init", "theta", "thickness", "width", "S_max", "R", "curve_level", "n",
              "p", "q", "dK0", "Cth_pos", "alpha", "smax_sigma0", "a0_intrinsic", "Kc", "closure_factor")

    def industry(a: float, level: str, knowledge: str = "staged") -> InputSpec:
        m = {k: Marginal("fixed", v) for k, v in base.items()}
        m.update(crack_marginals(a, base["aspect_init"], level, 0.8 * base["thickness"], knowledge))
        # Cth_neg is inactive at the base R = 0.1 but needed when scenarios move R below zero
        m.update(_catalogue_marginals([k for k in varied if k not in ("a_init", "aspect_init", "theta")]
                                      + ["K_factor", "Cth_neg"], base, level))
        return InputSpec(m, groups={"C-n": ("curve_level", "n")})

    return Case(
        "plate", "Surface crack in a plate, aluminium placeholder values, R = 0.1", formulation, base, varied,
        stages=tuple(v * MM for v in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0)), a_limit=0.8 * base["thickness"],
        stress_nominal=200.0, stress_grid=tuple(np.linspace(60.0, 350.0, 12)),
        stress_inputs=lambda s: {"S_max": s}, industry=industry,
        key_stages=(0.2 * MM, 1.0 * MM, 4.0 * MM), declared_horizons=(5e3, 2e4, 1e5),
        scenarios=(("S=200 R=0.1 (base)", {}),
                   *((f"S={s:g} R={r:g}", {"S_max": s, "R": r}) for s in (120.0, 200.0, 280.0)
                     for r in (-1.0, 0.1, 0.5) if (s, r) != (200.0, 0.1)),
                   ("t=20mm", {"thickness": 20.0 * MM}), ("a/c=0.3", {"aspect_init": 0.3})),
        notes=("Material constants are placeholders with literature-typical scatter; closure and short-crack "
               "constants are database defaults and carry epistemic uncertainty.",))
