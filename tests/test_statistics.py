import numpy as np
import pytest
from scipy import stats

from nasgro_sa import InputSpec, Marginal, rul_quantile, saltelli_blocks, sobol_indices


def test_lognormal_holds_declared_statistic_when_rescaled():
    rng = np.random.default_rng(0)
    for hold, stat in (("mean", np.mean), ("median", np.median)):
        base = Marginal("lognormal", 2.0, cov=0.4, hold=hold)
        for factor in (0.5, 1.0, 2.0):
            x = base.scaled(factor).from_standard_normal(rng.standard_normal(400_000))
            assert stat(x) == pytest.approx(2.0, rel=5e-3)
    assert np.all(base.scaled(0.0).from_standard_normal(np.ones(3)) == 2.0)


def test_bounds_and_uniform_spread():
    rng = np.random.default_rng(1)
    x = Marginal("normal", 1.0, sd=1.0, bounds=(0.5, 1.5)).from_standard_normal(rng.standard_normal(10_000))
    assert x.min() >= 0.5 and x.max() <= 1.5
    u = Marginal("uniform", 10.0, cov=0.1).from_standard_normal(rng.standard_normal(400_000))
    assert np.std(u) == pytest.approx(1.0, rel=5e-3)


def test_group_correlation_and_validation():
    spec = InputSpec({"x": Marginal("normal", 0.0, sd=1.0), "y": Marginal("lognormal", 1.0, cov=0.5)},
                     groups={"xy": ("x", "y")}, correlation={("x", "y"): -0.8})
    s = spec.sample(200_000, np.random.default_rng(2))
    assert np.corrcoef(s["x"], np.log(s["y"]))[0, 1] == pytest.approx(-0.8, abs=5e-3)
    with pytest.raises(ValueError):
        InputSpec(spec.marginals, correlation={("x", "y"): 0.5})


def test_sobol_ishigami():
    a, b, n = 7.0, 0.1, 2 ** 15
    rng = np.random.default_rng(3)
    w = saltelli_blocks(rng.standard_normal((n, 3)), rng.standard_normal((n, 3)), [[0], [1], [2]])
    x = -np.pi + 2 * np.pi * stats.norm.cdf(w)
    y = np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])
    res = sobol_indices(y, n, ["x1", "x2", "x3"], n_boot=100, rng=rng)
    np.testing.assert_allclose(res.first[:, 0], [0.3139, 0.4424, 0.0], atol=0.03)
    np.testing.assert_allclose(res.total[:, 0], [0.5576, 0.4424, 0.2437], atol=0.03)
    assert np.all(res.first_ci[0] <= res.first_ci[1])


def test_grouped_sobol_with_dependent_pair():
    # y = x1 + x2 + x3, corr(x1, x2) = 0.5 grouped: Var = 3 + 1, so S(group) = 0.75, S(x3) = 0.25
    spec = InputSpec({k: Marginal("normal", 0.0, sd=1.0) for k in ("x1", "x2", "x3")},
                     groups={"g": ("x1", "x2")}, correlation={("x1", "x2"): 0.5})
    groups = spec.uncertain_groups()
    assert [g for g, _ in groups] == ["g", "x3"]
    n, rng = 2 ** 15, np.random.default_rng(4)
    x = spec.transform(saltelli_blocks(spec.standard_normal(n, rng), spec.standard_normal(n, rng),
                                       [c for _, c in groups]))
    res = sobol_indices(x["x1"] + x["x2"] + x["x3"], n, [g for g, _ in groups])
    np.testing.assert_allclose(res.first[:, 0], [0.75, 0.25], atol=0.02)
    np.testing.assert_allclose(res.total[:, 0], [0.75, 0.25], atol=0.02)


def test_censored_quantiles():
    rul = np.array([1.0, 2.0, 3.0, np.inf, np.inf])
    np.testing.assert_array_equal(rul_quantile(rul, [0.2, 0.6, 0.8]), [1.0, 3.0, np.inf])
