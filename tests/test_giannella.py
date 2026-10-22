import numpy as np
import pytest

from nasgro_sa import giannella as g
from nasgro_sa.geometry import RoundBarStraightCrack
from nasgro_sa.inputs import EmpiricalGroup, InputSpec, Marginal


def test_round_bar_shallow_limit():
    # a/D -> 0: F -> (1.84/pi)(0.752 + 0.37)
    k, _ = RoundBarStraightCrack().stress_intensity(1e-6, 1e-6, 1.0, D=1.0)
    assert k / np.sqrt(np.pi * 1e-6) == pytest.approx(1.84 / np.pi * 1.122, rel=1e-4)


def test_empirical_group_draws_whole_rows_and_shrinks():
    pool = {"x": np.array([1.0, 2.0, 4.0]), "y": np.array([10.0, 20.0, 40.0])}
    spec = InputSpec({"x": Marginal("fixed", 2.0), "y": Marginal("fixed", 20.0), "z": Marginal("normal", 0.0, sd=1.0)},
                     empirical={"m": EmpiricalGroup(pool, {"x": 2.0, "y": 20.0}, log=("y",))})
    s = spec.sample(1000, np.random.default_rng(0))
    np.testing.assert_allclose(s["y"], 10 * s["x"])
    assert set(np.unique(s["x"])) == {1.0, 2.0, 4.0}
    assert [lbl for lbl, _ in spec.uncertain_groups()] == ["m", "z"]
    half = spec.scaled({"m": 0.5}).sample(10, np.random.default_rng(0))
    np.testing.assert_allclose(half["y"] / 20.0, (s["y"][:10] / 20.0) ** 0.5)
    assert [lbl for lbl, _ in spec.scaled({"m": 0.0}).uncertain_groups()] == ["z"]
    with pytest.raises(ValueError):
        spec.scaled({"x": 0.5})


def test_best_fit_life_matches_paper():
    x = {k: np.array([v]) for k, v in {**g.FIXED, **g.BEST_FIT}.items()}
    x.update(a_init=np.array([1e-3]), aspect_init=np.array([1.0]), D=np.array([18e-3]), F=np.array([63.6]),
             R=np.array([-1.0]))
    life = g.formulation().run(x)
    assert np.log10(life.cycles[0]) == pytest.approx(5.559, abs=0.01)   # Fig. 9 best-fit prediction


def test_acceptance_keeps_only_good_fits():
    rng = np.random.default_rng(1)
    data = g.synthetic_test_data(rng, points=60)
    mat, rate = g.accepted_material(data, 50, rng, batch=5000)
    assert len(mat["C"]) == 50 and 0 < rate < 0.05
    assert np.corrcoef(np.log10(mat["C"]), mat["n"])[0, 1] < -0.9
