import numpy as np
import pytest

from nasgro_sa import NasgroEquation, ParisLaw, WalkerEquation, baseline_inputs
from nasgro_sa.calibrate import FitSpec, calibrate, synthetic_lots, synthetic_series
from nasgro_sa.moment_independent import ks_indices

NOMINAL = dict(C=2.5e-10, n=3.0, p=0.5, q=1.0, dK0=2.8, Cth_pos=1.5, Cth_neg=0.1, Kc=40.0,
               alpha=2.0, smax_sigma0=0.3, a0_intrinsic=3.81e-5)
FREE = ("C", "n", "p", "q", "dK0")


def test_nasgro_fit_recovers_noise_free_parameters():
    series = synthetic_series(NasgroEquation(), NOMINAL, (0.1, 0.4, 0.7), 30, 0.0, np.random.default_rng(0))
    start = dict(C=1e-10, n=2.6, p=0.8, q=0.6, dK0=2.0)
    fit = FitSpec(NasgroEquation(), start, {k: v for k, v in NOMINAL.items() if k not in FREE},
                  bounds={"p": (0.01, 3.0), "q": (0.01, 3.0), "dK0": (0.5, 10.0)})
    x, sd = fit.fit(series)
    est = fit._from_x(x)
    assert sd < 1e-6
    for k in FREE:
        assert est[k] == pytest.approx(NOMINAL[k], rel=1e-3)


def test_walker_fit_recovers_gamma():
    true = dict(C=1e-10, n=3.2, gamma=0.6, Kc=40.0)
    series = synthetic_series(WalkerEquation(), true, (0.1, 0.5), 20, 0.0, np.random.default_rng(1))
    x, _ = FitSpec(WalkerEquation(), dict(C=1e-9, n=3.0, gamma=0.5), bounds={"gamma": (0.0, 1.0)}).fit(series)
    assert x[2] == pytest.approx(0.6, rel=1e-6)


def test_calibrated_spec_reflects_lot_spread():
    spec = baseline_inputs()
    rng = np.random.default_rng(2)
    lots = synthetic_lots(NasgroEquation(), spec, 40, rng, scatter_log10=0.02)
    cal = calibrate(FitSpec(ParisLaw(), dict(C=1e-10, n=3.0), r_values=(0.1,)), lots)
    paris = cal.input_spec(spec)
    assert paris.groups == {"growth": ("C", "n")}
    assert paris.correlation[("C", "n")] < -0.5
    assert set(paris.names) >= {"a_init", "aspect_init", "S_max", "R", "Kc", "C", "n"}


def test_ks_index_separates_influential_input_from_dummy():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((8000, 3))
    y = np.where(w[:, 0] > 2.0, np.inf, np.exp(w[:, 0] + 0.2 * w[:, 1]))   # censored upper tail
    res = ks_indices(y, w, [("x0", [0]), ("x1", [1]), ("x2", [2])], rng=rng)
    assert res.index[0, 0] > 0.5
    assert res.index[2, 0] < 2 * res.dummy[0] + 0.02
    assert res.index[1, 0] > res.index[2, 0]
