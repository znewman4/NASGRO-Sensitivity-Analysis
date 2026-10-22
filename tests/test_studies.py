import numpy as np
import pytest

from nasgro_sa import Outcome, baseline_formulation, baseline_inputs
from nasgro_sa.studies import (distribution_variants, growth_rate_study, integration_convergence,
                               lower_quantile_objective, rank_agreement, reduction_path, sobol_study,
                               summarise, variance_sweep)


def test_baseline_end_to_end():
    rng = np.random.default_rng(5)
    form, spec = baseline_formulation(), baseline_inputs()
    life = form.run(spec.sample(500, rng))
    assert not np.any(life.outcome == Outcome.INVALID)
    assert life.failed.mean() > 0.5
    s = summarise(life, [3e4, 1e5])
    assert np.all(np.diff(s.p_fail) >= 0)

    study = sobol_study(form, spec, 64, [5e4], rng, n_boot=0)
    assert study.sobol.first.shape == (len(spec.uncertain_groups()), 2)
    ks = study.ks(rng=rng)
    assert ks.index.shape == (len(spec.uncertain_groups()), 1)


def test_grouped_members_cannot_be_rescaled_alone():
    spec = baseline_inputs()
    with pytest.raises(ValueError, match="C-n"):
        spec.scaled({"n": 0.5})
    both = spec.scaled({"C-n": 0.5})
    assert both.marginals["C"].cov == pytest.approx(0.175) and both.marginals["n"].sd == pytest.approx(0.075)


def test_variance_sweep_zero_spread_is_deterministic():
    rng = np.random.default_rng(6)
    form, spec = baseline_formulation(), baseline_inputs()
    zero = spec.scaled({name: 0.0 for name, _ in spec.uncertain_groups()})
    pts = variance_sweep(form, zero, ["S_max"], [0.0], 50, [1e5], rng)
    assert pts[0].summary.p_fail[0] in (0.0, 1.0)


def test_integration_converges():
    rng = np.random.default_rng(7)
    form, spec = baseline_formulation(), baseline_inputs()
    x = spec.with_nominal(a_init=0.3e-3).sample(300, rng)
    rows = integration_convergence(form, x, refinements=(2, 1, 0.5))
    assert rows[1][1] < 1e-3


def test_growth_rate_study_excludes_crack_geometry():
    rng = np.random.default_rng(8)
    res, below = growth_rate_study(baseline_formulation(), baseline_inputs(), [(1e-3, 1.25e-3)], 256, rng, n_boot=0)
    assert "a_init" not in res.groups and "aspect_init" not in res.groups
    assert res.groups[int(np.argmax(res.total[:, 0]))] in ("C-n", "S_max")
    assert below.shape == (1,)


def test_variants_preserve_moments_and_rank_agreement():
    spec = baseline_inputs()
    variants = distribution_variants(spec)
    assert {"baseline", "spreads x0.5", "spreads x2", "uniform shapes", "rho = -0.6"} <= set(variants)
    x = variants["uniform shapes"].sample(200_000, np.random.default_rng(9))["S_max"]
    assert np.mean(x) == pytest.approx(200.0, rel=2e-3) and np.std(x) == pytest.approx(16.0, rel=1e-2)

    rng = np.random.default_rng(10)
    form = baseline_formulation()
    results = {k: sobol_study(form, variants[k], 64, [3e4], rng, n_boot=0).sobol for k in ("baseline", "spreads x2")}
    tau = rank_agreement(results)
    assert tau["baseline"] == pytest.approx(1.0)


def test_reduction_path_improves_lower_quantile():
    rng = np.random.default_rng(11)
    path = reduction_path(baseline_formulation(), baseline_inputs(), 400, lower_quantile_objective(0.05), rng, stages=2)
    assert len(path) == 3 and path[1].reduced is not None
    assert path[2].value >= path[0].value
