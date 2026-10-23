import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import cases, geometry, life, nasgro  # noqa: E402
from nasgro_scenarios import residual_stress as rs  # noqa: E402
from nasgro_scenarios import variable_amplitude as va  # noqa: E402

ALL = tuple(set(cases.BAR_INPUTS.values()) | set(cases.PLATE_INPUTS.values())
            | {"material", "Cn", "rs_amp", "rs_len", "va_h", "va_nb", "va_solr", "va_beta"})


def nominal(case, a_mm=1.0, n=2):
    return cases.sample(case, a_mm * 1e-3, n, seed=1, fixed=ALL)


@pytest.mark.parametrize("ac", [0.3, 0.8, 1.0])
@pytest.mark.parametrize("a", [2e-4, 1e-3, 5e-3])
def test_weight_function_reproduces_reference_solutions(ac, a):
    t, W = 0.01, 0.1
    A = np.array([a])
    Q = geometry.newman_raju_Q(ac)
    Y0, Y1 = geometry.plate_Y0_Y1(A, ac, t, W)
    M = geometry.shen_glinka_M(Y0, Y1, Q)
    ref = np.sqrt(np.pi * a / Q)
    k_uni = geometry.weight_function_K(lambda x: 1.0 + 0 * x, A, *M)
    k_bend = geometry.weight_function_K(lambda x: 1.0 - 2.0 * x / t, A, *M)
    assert k_uni[0] == pytest.approx(ref * Y0[0], rel=1e-6)
    assert k_bend[0] == pytest.approx(ref * Y0[0] * geometry.newman_raju_H(a, ac, t), rel=1e-6)


def test_growth_law_reduces_to_paris_far_from_threshold_and_kc():
    p = dict(C=1e-11, n=3.0, p=0.5, q=1.0, dKth0=1e-6, a0=1e-9, cth_pos=1.5, cth_neg=0.1,
             alpha=2.0, smax_sigma0=0.3, kappa=1.0, Kc=1e9)
    kmax = np.array([10.0])
    rate = nasgro.growth_rate(kmax, 0.1 * kmax, 1e-3, p)
    f, _ = nasgro.newman_closure(np.array(0.1), 2.0, 0.3)
    assert rate[0] == pytest.approx(1e-11 * ((1 - f) * 10.0) ** 3.0, rel=1e-9)


def test_closed_crack_does_not_grow():
    p = {k: v for k, v in cases.PLATE.items() if k not in ("name",)}
    assert nasgro.growth_rate(np.array([-1.0]), np.array([-3.0]), 1e-3, p)[0] == 0.0


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_zero_residual_stress_gives_baseline(case):
    s = nominal(case)
    sc = dict(rs.SCENARIOS["plate_weld" if case == "plate" else "bar_tensile"])
    s = rs.add_inputs(s, sc, 1, fixed=ALL)
    s["rs_sig"][:] = 0.0
    base = life.ca_life(case, s)
    with_rs = life.ca_life(case, s, rs.k_res(case, "decaying"))
    np.testing.assert_allclose(with_rs, base, rtol=1e-12)


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_uniform_residual_stress_matches_shifted_mean_stress(case):
    """A uniform residual stress equals raising the CA mean stress by the same amount."""
    s = nominal(case)
    s = rs.add_inputs(s, rs.SCENARIOS[f"{case}_uniform"], 1, fixed=ALL)
    l_rs = life.ca_life(case, s, rs.k_res(case, "uniform"))
    smax = cases.max_stress(case, s)
    sig = s["rs_sig"]
    s2 = dict(s)
    s2["R"] = (smax * s["R"] + sig) / (smax + sig)
    if case == "bar":
        s2["F"] = s["F"] * (smax + sig) / smax
    else:
        s2["S"] = smax + sig
    np.testing.assert_allclose(l_rs, life.ca_life(case, s2), rtol=1e-9)


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_tensile_residual_stress_shortens_life(case):
    s = nominal(case)
    s = rs.add_inputs(s, rs.SCENARIOS["plate_weld" if case == "plate" else "bar_tensile"], 1, fixed=ALL)
    assert np.all(life.ca_life(case, s, rs.k_res(case, "decaying")) < life.ca_life(case, s))


def test_spectrum_is_normalised_to_equivalent_amplitude():
    amp, counts = va.spectrum(np.array([1.2, 2.0]), np.array([1e3, 1e5]), 3.0)
    np.testing.assert_allclose(counts.sum(1), [1e3, 1e5], rtol=1e-6)
    s_eq = (np.sum(counts * amp**3, 1) / counts.sum(1)) ** (1 / 3)
    np.testing.assert_allclose(s_eq, 1.0, rtol=1e-12)


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_willenborg_without_retardation_matches_stage1(case):
    s = va.add_inputs(nominal(case), case, 1, fixed=ALL)
    s["va_solr"][:] = 1e9        # phi -> 0: no residual intensity
    l1 = va.stage1_life(case, s)
    l2 = va.stage2_life(case, s, step_frac=0.002)
    np.testing.assert_allclose(np.log10(l2), np.log10(l1), atol=0.02)


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_retardation_extends_life(case):
    s = va.add_inputs(nominal(case), case, 1, fixed=ALL)
    assert np.all(va.stage2_life(case, s) >= va.stage1_life(case, s))


from nasgro_scenarios import mixed_mode as mm  # noqa: E402


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_mixed_mode_models_agree_at_zero_tilt(case):
    lives = [mm.nominal_life(case, 1.0, 0.0, k) for k in range(len(mm.MODELS))]
    s = nominal(case)
    s["tilt"][:] = 0.0
    np.testing.assert_allclose(lives, life.ca_life(case, s)[0], rtol=1e-9)


def test_mixed_mode_equivalent_k_factors():
    th = np.deg2rad(30.0)
    m = mm.m_factor(np.array([0, 1, 2, 3]), th)
    kI, kII = np.cos(th) ** 2, np.sin(th) * np.cos(th)
    np.testing.assert_allclose(m, [1.0, 1.0, np.hypot(kI, kII), (kI**4 + 8 * kII**4) ** 0.25])


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_tilt_lengthens_life_except_full_mode_one(case):
    for k in (0, 2, 3):
        assert mm.nominal_life(case, 1.0, 30.0, k) > mm.nominal_life(case, 1.0, 0.0, k)
    assert mm.nominal_life(case, 1.0, 30.0, 1) == pytest.approx(mm.nominal_life(case, 1.0, 0.0, 1))


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_nasgro_only_fixes_kappa_and_sif_without_touching_other_draws(case):
    ext = cases.sample(case, 1e-3, 500, seed=3)
    std = cases.sample(case, 1e-3, 500, seed=3, fixed=("kappa", "sif"))
    assert np.all(std["kappa"] == 1.0) and np.all(std["sif"] == 1.0)
    for k in ext:
        if k not in ("kappa", "sif"):
            np.testing.assert_array_equal(ext[k], std[k])


@pytest.mark.parametrize("case", ["bar", "plate"])
def test_unit_spread_multipliers_reproduce_default_sampler(case):
    a = cases.sample(case, 0.5e-3, 300, seed=2)
    keys = ("depth", "S", "F", "ac", "alpha", "smax_sigma0", "material", "Cn", "Kc", "a0", "R", "tilt")
    b = cases.sample(case, 0.5e-3, 300, seed=2, spread={k: 1.0 for k in keys if not
                                                         (case == "bar" and k in ("alpha", "smax_sigma0"))})
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
