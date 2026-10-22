import numpy as np
import pytest

from nasgro_sa import giannella as g
from nasgro_sa.cases import plate_case, round_bar_case
from nasgro_sa.lifestage import ABSOLUTE, local_effects, screen_spec
from nasgro_sa.uncertainty import CATALOGUE, CRACK, DETECTION_LIMIT

MM = 1e-3


@pytest.fixture(scope="module")
def plate():
    return plate_case()


@pytest.fixture(scope="module")
def bar():
    pool = {k: np.full(3, v) for k, v in g.BEST_FIT.items()}
    return round_bar_case(pool)


def test_regime_shares_follow_crack_size(plate):
    rows = [plate.nominal_at(a) for a in (0.1 * MM, 6.0 * MM)]
    life = plate.formulation.run({k: np.array([r[k] for r in rows]) for k in rows[0]})
    shares = life.regime_shares()
    np.testing.assert_allclose(shares.sum(axis=1), 1.0, rtol=1e-9)
    assert shares[0, 0] > 0.05          # small crack starts near threshold
    assert shares[1, 2] > 0.5           # deep crack spends most of its life near Kc


def test_regime_shares_need_tracking(plate):
    life = plate.formulation.with_settings(track_regimes=False).run(
        {k: np.array([v]) for k, v in plate.nominal_at(1 * MM).items()})
    with pytest.raises(ValueError):
        life.regime_shares()


def test_screen_spec_bounds(plate):
    x = screen_spec(plate, 1 * MM).sample(4000, np.random.default_rng(0))
    for k in plate.varied:
        nominal = plate.nominal_at(1 * MM)[k]
        half = ABSOLUTE.get(k, 0.10 * abs(nominal))
        assert x[k].min() >= nominal - half - 1e-12 and x[k].max() <= nominal + half + 1e-12
        assert x[k].max() - x[k].min() > 1.8 * half
    assert "C-n" in [lbl for lbl, _ in screen_spec(plate, 1 * MM).uncertain_groups()]


def test_crack_knowledge_depends_on_detection(plate):
    rng = np.random.default_rng(1)
    small = plate.industry(0.1 * MM, "typical").sample(20000, rng)["a_init"]
    large = plate.industry(4.0 * MM, "typical").sample(20000, rng)["a_init"]
    assert np.std(small) / np.median(small) > 0.4                        # undetected: relative scatter
    assert np.std(large) == pytest.approx(CRACK["sized"]["a_init"].typical * MM, rel=0.05)   # sized: absolute
    assert 0.1 * MM < DETECTION_LIMIT <= 4.0 * MM


def test_local_effects_signs(plate):
    eff = local_effects(plate, 1 * MM)
    assert eff.effect["S_max"] < 0 and eff.effect["a_init"] < 0 and eff.effect["Kc"] > 0
    # curve level scales the rate everywhere, so life scales as 1/level exactly
    assert eff.effect["curve_level"] == pytest.approx(0.5 * (np.log10(1 / 1.1) - np.log10(1 / 0.9)), abs=1e-4)
    assert "K_factor" not in eff.effect


def test_sif_solution_error_acts_exactly_like_stress(plate):
    nominal = plate.nominal_at(1 * MM)
    rows = [{**nominal, "K_factor": 1.1}, {**nominal, "S_max": 1.1 * nominal["S_max"]}]
    life = plate.formulation.run({k: np.array([r[k] for r in rows]) for k in nominal})
    assert life.cycles[0] == pytest.approx(life.cycles[1], rel=1e-12)


def test_bar_case_runs_and_matches_reproduction(bar):
    life = bar.formulation.run({k: np.array([v]) for k, v in bar.nominal_at(1 * MM).items()})
    assert np.log10(life.cycles[0]) == pytest.approx(5.564, abs=0.01)   # same best-fit life as giannella


def test_every_varied_input_has_an_industry_uncertainty(plate, bar):
    crack = set(CRACK["sized"])
    for case in (plate, bar):
        spec = case.industry(1 * MM, "typical")
        uncertain = {m for _, cols in spec.uncertain_groups() for m in (spec.names[c] for c in cols)}
        missing = [k for k in case.varied if k not in uncertain]
        # bar: n, p, dK0 are uncertain through the fitted pool (C replaces the curve level); q and the
        # closure constants are absorbed by calibration at the service stress ratio
        expected = {"curve_level", "q", "Cth_neg", "alpha", "smax_sigma0"} if case is bar else set()
        assert set(missing) == expected
        assert all(k in CATALOGUE or k in crack for k in case.varied if k not in expected)
        assert "K_factor" not in case.varied and "K_factor" in uncertain   # Q3 only
