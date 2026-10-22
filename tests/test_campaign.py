import numpy as np
import pytest

from nasgro_sa import campaign as cp
from nasgro_sa import giannella as g
from nasgro_sa.cases import plate_case, round_bar_case
from nasgro_sa.integrate import LifeResult, Outcome

MM = 1e-3


@pytest.fixture(scope="module")
def plate():
    return plate_case()


@pytest.fixture(scope="module")
def bar():
    rng = np.random.default_rng(0)
    pool = {"C": g.BEST_FIT["C"] * 10 ** rng.normal(0, 0.1, 50), "n": np.full(50, g.BEST_FIT["n"]),
            "p": np.full(50, g.BEST_FIT["p"]), "dK0": np.full(50, g.BEST_FIT["dK0"])}
    return round_bar_case(pool)


def test_metrics_condition_on_failure():
    logs = np.linspace(3.0, 5.0, 400)
    cycles = np.concatenate([10 ** logs, np.zeros(100)])
    outcome = np.array([Outcome.DEPTH_LIMIT] * 400 + [Outcome.ARREST] * 100)
    m = cp.metrics(LifeResult(cycles, outcome, cycles, cycles, 1e9), declared=(1e4,))
    assert m["p_no_propagation"] == pytest.approx(0.2) and m["p_fail"] == pytest.approx(0.8)
    q05, q50, q95 = np.percentile(logs, (5, 50, 95))
    assert m["u90"] == pytest.approx(q95 - q05) and m["log_q50"] == pytest.approx(q50)
    assert m["p_fail_by_1e+04"] == pytest.approx(np.mean(10 ** logs <= 1e4) * 0.8)
    few = cp.metrics(LifeResult(cycles[:150], outcome[:150], cycles[:150], cycles[:150], 1e9))
    assert np.isnan(few["u90"])   # fewer than MIN_FAILURES failures


def test_plateau_interpolates():
    assert cp.plateau_point((1.0, 0.5, 0.0), (0.0, 0.9, 1.0)) == pytest.approx(1.0 - 0.5 * 0.8 / 0.9)
    assert np.isnan(cp.plateau_point((1.0, 0.5, 0.0), (0.0, -0.1, -0.2)))
    rc = cp.ReductionCurves(1.0, {"u90": 1.0}, (1.0, 0.5, 0.0),
                            {"x": [{"u90": 1.0}, {"u90": 0.6}, {"u90": 0.5}], "y": [{"u90": 1.0}] * 3})
    assert rc.achievable("x") == pytest.approx(0.5) and rc.ranking() == ["x", "y"]


def test_reduction_and_pairs_are_consistent(plate):
    rng = np.random.default_rng(1)
    rc = cp.reduction_curves(plate, 1 * MM, 300, rng, factors=(1.0, 0.5, 0.0))
    spec = plate.industry(1 * MM, "typical")
    assert set(rc.curves) == {lbl for lbl, _ in spec.uncertain_groups()}   # every group, no selection
    for grp, ms in rc.curves.items():
        assert ms[0] is rc.base
    pr = cp.paired_reductions(plate, 1 * MM, rc.ranking()[:2], 300, rng)
    s = pr.synergy()
    np.testing.assert_allclose(s, s.T)


def test_variants_keep_moments_and_break_dependence(plate, bar):
    rng = np.random.default_rng(2)
    variants = cp.distribution_variants(plate, 1 * MM, rng)
    base = variants["typical"].sample(100_000, rng)
    uni = variants["uniform shapes"].sample(100_000, rng)
    for k in ("S_max", "Kc", "alpha"):
        assert np.mean(uni[k]) == pytest.approx(np.mean(base[k]), rel=0.01)
        assert np.std(uni[k]) == pytest.approx(np.std(base[k]), rel=0.03)
    # judgement x2 leaves literature-based spreads alone and doubles judgement-based ones
    j2 = variants["judgement x2"]
    assert j2.marginals["S_max"].spread == variants["typical"].marginals["S_max"].spread
    assert j2.marginals["p"].spread == pytest.approx(2 * variants["typical"].marginals["p"].spread)
    n = np.linspace(1.0, 3.0, 2000)
    joint = round_bar_case({"C": 10 ** (-8.0 - n), "n": n, "p": np.full(2000, 0.7), "dK0": np.full(2000, 7.0)})
    spec = joint.industry(1 * MM, "typical")
    x = spec.sample(30_000, rng)
    assert np.corrcoef(np.log10(x["C"]), x["n"])[0, 1] < -0.99
    x = cp._independent_empirical(spec, np.random.default_rng(3)).sample(30_000, rng)
    assert abs(np.corrcoef(np.log10(x["C"]), x["n"])[0, 1]) < 0.05


def test_calibrated_spec_replaces_material(plate):
    rng = np.random.default_rng(4)
    models = cp.plate_models(plate, rng, n_lots=6)
    spec = models["Paris"].spec(1 * MM)
    assert "curve_level" not in spec.marginals and spec.groups == {"growth fit": ("C", "n")}
    assert spec.marginals["alpha"].spread == 0.0 and spec.marginals["Kc"].spread > 0.0
    life = models["Paris"].formulation.run(spec.sample(50, rng))
    assert life.failed.all()   # Paris has no threshold: every crack grows
