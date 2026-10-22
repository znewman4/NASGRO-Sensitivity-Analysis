"""Declared baseline formulation and input distributions.

Scope of the baseline (extensions are added as separate formulations):
  * Semi-elliptical surface crack normal to the load in a finite plate under
    remote tension (Newman-Raju). Inclined / mixed-mode cracks are excluded.
  * Constant-amplitude future loading at a declared stress ratio, with no
    load-interaction (retardation) effects.
  * NASGRO equation with Newman closure.
  * RUL = cycles from the current crack state to the first of: Kmax >= Kc at
    either crack-front point, depth = 0.8 t, or half-length = 0.5 b.

All numerical values are placeholders, chosen to be plausible for an
aluminium-alloy plate. None comes from the NASGRO materials database or from
test data, so every spread is marked exploratory. Replace them, and set
basis="data", as calibrated sources become available.
"""
from __future__ import annotations

from .geometry import SurfaceCrackPlate
from .growth import NasgroEquation
from .inputs import InputSpec, Marginal
from .integrate import FailureCriterion, IntegrationSettings
from .model import Formulation

MM = 1e-3


def baseline_formulation(**settings) -> Formulation:
    return Formulation(geometry=SurfaceCrackPlate(thickness=10 * MM, width=100 * MM),
                       rate_model=NasgroEquation(),
                       criterion=FailureCriterion(depth_ratio=0.8, width_ratio=0.5),
                       settings=IntegrationSettings(**{"horizon": 1e8, **settings}))


def baseline_inputs() -> InputSpec:
    """Placeholder input distributions for the baseline formulation.

    C and n are treated as a single uncertain quantity, the growth-rate curve,
    rather than as two inputs. They are fitted jointly and are highly
    correlated (rho = -0.8 in normal-score space), so their separate effects
    on life are not identifiable. Consequences:
      * the pair receives one grouped Sobol index, "C-n";
      * variance sweeps and reductions rescale both spreads together
        (InputSpec.scaled refuses to rescale C or n alone);
      * ranking C against n is outside the scope of the study.
    """
    m = {
        # Current crack state (as estimated by inspection)
        "a_init": Marginal("lognormal", 1.0 * MM, cov=0.3, bounds=(0.1 * MM, 5.0 * MM)),
        "aspect_init": Marginal("lognormal", 0.8, cov=0.2, bounds=(0.2, 1.5)),
        # Declared future loading
        "S_max": Marginal("normal", 200.0, cov=0.08),
        "R": Marginal("fixed", 0.1),
        # NASGRO equation. C and n come from the same fit and are strongly
        # correlated, so they form one group ("C-n"): a single Sobol index, and
        # spreads rescaled together. See the note on baseline_inputs().
        "C": Marginal("lognormal", 2.5e-10, cov=0.35, hold="median"),
        "n": Marginal("normal", 3.0, sd=0.15),
        "p": Marginal("normal", 0.5, sd=0.1, bounds=(0.1, 1.0)),
        "q": Marginal("normal", 1.0, sd=0.2, bounds=(0.25, 2.0)),
        "dK0": Marginal("lognormal", 2.8, cov=0.15),
        "Cth_pos": Marginal("fixed", 1.5),
        "Cth_neg": Marginal("fixed", 0.1),
        "Kc": Marginal("lognormal", 40.0, cov=0.1),
        "alpha": Marginal("fixed", 2.0),
        "smax_sigma0": Marginal("fixed", 0.3),
        "a0_intrinsic": Marginal("fixed", 0.0381 * MM),
    }
    return InputSpec(m, groups={"C-n": ("C", "n")}, correlation={("C", "n"): -0.8})
