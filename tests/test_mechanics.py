import numpy as np
import pytest

from nasgro_sa import (FailureCriterion, IntegrationSettings, NasgroEquation, Outcome, ParisLaw,
                       SurfaceCrackPlate, integrate_life, newman_closure)


def test_newman_closure_endpoints():
    alpha, ratio = 2.0, 0.3
    f0, a0 = newman_closure(0.0, alpha, ratio)
    assert f0 == pytest.approx(a0)
    assert newman_closure(1.0, alpha, ratio)[0] == pytest.approx(1.0)
    r = np.linspace(-1.0, 0.99, 400)
    assert np.all(np.diff(newman_closure(r, alpha, ratio)[0]) >= -1e-12)


def test_newman_raju_shallow_semicircular():
    # a/c = 1, a/t -> 0, wide plate: F/sqrt(Q) = 1.04/sqrt(2.464) deepest, x1.1 at the surface
    plate = SurfaceCrackPlate(thickness=1.0, width=1e3)
    a = 1e-4
    k_a, k_c = plate.stress_intensity(a, a, 1.0)
    root = np.sqrt(np.pi * a)
    assert k_a / root == pytest.approx(1.04 / np.sqrt(2.464), rel=1e-3)
    assert k_c / root == pytest.approx(1.04 * 1.1 / np.sqrt(2.464), rel=1e-3)


def test_newman_raju_continuous_at_unit_aspect():
    plate = SurfaceCrackPlate(thickness=0.01, width=0.1)
    lo = plate.stress_intensity(0.003, 0.003 / (1 - 1e-9), 100.0)
    hi = plate.stress_intensity(0.003, 0.003 / (1 + 1e-9), 100.0)
    # The published branches differ slightly at a/c = 1 (M2: 0.2017 vs 0.2, M3: -0.106 vs -0.11)
    np.testing.assert_allclose(lo, hi, rtol=1e-3)


class _CentreCrack:
    """K = Y S sqrt(pi a) at both points: closed-form Paris life for checking the integrator."""

    inputs = ()

    def __init__(self, thickness, y=1.12):
        self.thickness, self.y = thickness, y

    def limits(self, criterion):
        return criterion.depth_ratio * self.thickness, np.inf

    def stress_intensity(self, a, c, s):
        return self.y * s * np.sqrt(np.pi * a), self.y * s * np.sqrt(np.pi * c)


def test_integrator_matches_closed_form_paris_life():
    C, m, y, s, a_i, a_f = 1e-11, 3.2, 1.12, 120.0, 1e-3, 8e-3
    geom = _CentreCrack(thickness=a_f / 0.8, y=y)
    life = integrate_life(geom, ParisLaw(), {"C": C, "n": m, "Kc": 1e9}, np.array([a_i]), a_i, s, 0.0,
                          FailureCriterion(aspect_range=(0.0, np.inf)), IntegrationSettings(growth_fraction=0.01))
    exact = (a_i ** (1 - m / 2) - a_f ** (1 - m / 2)) / ((m / 2 - 1) * C * (y * s * np.sqrt(np.pi)) ** m)
    assert life.outcome[0] == Outcome.DEPTH_LIMIT
    assert life.cycles[0] == pytest.approx(exact, rel=1e-4)


NASGRO_PARAMS = dict(C=2.5e-10, n=3.0, p=0.5, q=1.0, dK0=2.8, Cth_pos=1.5, Cth_neg=0.1, Kc=40.0,
                     alpha=2.0, smax_sigma0=0.3, a0_intrinsic=3.81e-5)


def test_nasgro_rate_regimes():
    model = NasgroEquation()
    dk = np.array([1.0, 10.0, 0.9 * 40.0 * 0.9])
    rate = model(dk, 0.1, 0.01, NASGRO_PARAMS)
    assert rate[0] == 0.0                     # below threshold
    assert 1e-8 < rate[1] < 1e-6              # Paris region, plausible for aluminium
    assert rate[2] > 10 * rate[1]             # accelerating towards Kc


def test_outcomes_arrest_fracture_and_censoring():
    plate = SurfaceCrackPlate(thickness=0.01, width=0.1)
    s = np.array([30.0, 600.0, 200.0])
    a0 = np.array([3e-4, 2e-3, 1e-3])
    life = integrate_life(plate, NasgroEquation(), NASGRO_PARAMS, a0, a0 / 0.8, s, 0.1,
                          settings=IntegrationSettings(horizon=1e4))
    assert life.outcome[0] == Outcome.ARREST
    assert life.outcome[1] == Outcome.FRACTURE
    assert life.outcome[2] == Outcome.HORIZON
    assert np.isinf(life.rul[[0, 2]]).all()
    with pytest.raises(ValueError):
        life.failure_indicator([2e4])
