import numpy as np
import pytest

from nasgro_sa import targets as tg
from nasgro_sa.cases import plate_case, with_orientation
from nasgro_sa.geometry import mixed_mode_factor
from nasgro_sa.growth import NasgroEquation
from nasgro_sa.inputs import Marginal

MM = 1e-3
NASGRO = dict(C=2.5e-10, n=3.0, p=0.5, q=1.0, dK0=2.8, Cth_pos=1.5, Cth_neg=0.1, Kc=40.0, alpha=2.0,
              smax_sigma0=0.3, a0_intrinsic=3.81e-5)


@pytest.fixture(scope="module")
def plate():
    return plate_case()


def test_closure_factor_scales_the_open_part_of_the_cycle():
    model = NasgroEquation()
    base = model(np.array([8.0]), 0.1, 0.01, NASGRO)
    assert model(np.array([8.0]), 0.1, 0.01, {**NASGRO, "closure_factor": 1.0}) == pytest.approx(base)
    more_open = model(np.array([8.0]), 0.1, 0.01, {**NASGRO, "closure_factor": 1.1})
    assert more_open > base * 1.1 ** 3   # Paris term ~ kappa^n, and the threshold falls too


def test_mixed_mode_factors():
    assert mixed_mode_factor(0.0, "energy") == pytest.approx(1.0)
    assert mixed_mode_factor(0.0, "tanaka") == pytest.approx(1.0)
    assert mixed_mode_factor(30.0, "energy") == pytest.approx(np.cos(np.radians(30.0)))
    assert mixed_mode_factor(30.0, "tanaka") == pytest.approx((0.75**4 + 8 * 0.4330127**4) ** 0.25, rel=1e-6)


def test_orientation_mechanics(plate):
    x = {k: np.array([v]) for k, v in plate.nominal_at(1 * MM).items()}
    nominal = plate.formulation.run(x).cycles[0]
    for mech in ("energy", "tanaka"):
        variant = with_orientation(plate, mech)
        assert variant.formulation.run(x).cycles[0] == pytest.approx(nominal, rel=1e-9)   # aligned: unchanged
        tilted = variant.formulation.run({**x, "theta": np.array([30.0])}).cycles[0]
        assert tilted > nominal   # lower driving force while the crack is still tilted
    assert with_orientation(plate, "projection") is plate


def test_candidate_grids_and_measurements():
    assert tg.SIZE.grid(1 * MM)[0] == pytest.approx(1.0) and tg.SIZE.grid(1 * MM)[-1] == 0.0
    sds = tg.SIZE.grid(2 * MM)
    assert all(b == pytest.approx(a / 2) for a, b in zip(sds[:-2], sds[1:-1]))   # halving grid
    m = tg.SIZE.measured(Marginal("lognormal", 1e-3, cov=0.6, bounds=(1e-6, 7e-3)), 0.25)
    assert m.kind == "normal" and m.sd == pytest.approx(0.25e-3) and m.bounds == (1e-6, 7e-3)
    assert tg.SIZE.measured(m, 0.0).spread == 0.0
    assert tg.OPENING.measured(Marginal("lognormal", 1.0, cov=0.1), 0.05).cov == pytest.approx(0.05)


def _sweep(u90s, baseline=1.0, floor=0.5):
    sds = (1.0, 0.5, 0.25, 0.125, 0.0)
    pts = [{"u90": u, "log_q50": 4.0, "log_q05": 3.0} for u in u90s]
    return tg.Sweep(tg.SIZE, 1e-3, "base", "typical", {"u90": baseline, "log_q50": 4.0, "log_q05": 3.0},
                    {"u90": floor}, sds, pts)


def test_sweep_summary():
    s = _sweep([1.3, 1.0, 0.8, 0.71, 0.7])
    assert s.achievable == pytest.approx(0.3) and s.achievable_share == pytest.approx(0.3) and s.substantial
    assert s.required_sd == pytest.approx(0.25 + (0.24 - 0.2) / (0.29 - 0.2) * (0.125 - 0.25))
    assert s.diminishing_sd == 0.125   # beyond 0.125 only 0.01 of the 1.0 band remains (< 2 %)
    assert s.crack_removable_share == pytest.approx(0.5)
    assert s.coarse_achievable == pytest.approx(0.6)
    flat = _sweep([1.1, 1.02, 1.0, 0.99, 0.98])
    assert not flat.substantial


def test_evaluate_state_runs_every_candidate(plate):
    rng = np.random.default_rng(5)
    res = tg.evaluate_state(plate, 1 * MM, rng, candidates=(tg.SIZE, tg.SHAPE, tg.ORIENTATION[2], tg.OPENING),
                            n=300, keep_samples=True)
    assert set(res.sweeps) == {"size (depth)", "shape (a/c)", "orientation (energy, 30 deg)", "opening / closure"}
    size = res.sweeps["size (depth)"]
    assert size.floor["u90"] <= size.baseline["u90"]
    assert "size exact" in res.beyond_size
    assert "baseline (energy, 30 deg)" in res.samples and "orientation (energy, 30 deg)" in res.samples
