"""NASGRO crack-growth equation (Forman & Mettu 1992) with Newman (1984) closure.

Every function is vectorised over numpy arrays. Units: crack size in m,
stress intensity in MPa*sqrt(m), growth rate in m/cycle.

The growth law is written in terms of (Kmax, Kmin) rather than (dK, R) so that
anything that shifts both by the same amount (a residual stress intensity, or
the Willenborg residual intensity K_R) enters in exactly the same way:
dK is unchanged, Kmax and R move, and closure and threshold respond through R.
"""
import numpy as np

R_MIN, R_MAX = -2.0, 0.99   # NASGRO: for R < -2 use R = -2


def newman_closure(R, alpha, smax_sigma0):
    """Newman (1984) crack-opening function f = Kop/Kmax and A0."""
    A0 = (0.825 - 0.34 * alpha + 0.05 * alpha**2) * \
        np.cos(np.pi * smax_sigma0 / 2.0) ** (1.0 / alpha)
    A1 = (0.415 - 0.071 * alpha) * smax_sigma0
    A3 = 2.0 * A0 + A1 - 1.0
    A2 = 1.0 - A0 - A1 - A3
    f_pos = np.maximum(R, A0 + A1 * R + A2 * R**2 + A3 * R**3)
    f_neg = A0 + A1 * R
    f = np.where(R >= 0.0, f_pos, f_neg)
    return f, A0


def threshold(R, f, A0, a, dKth0, a0, cth_pos, cth_neg):
    """NASGRO threshold dKth(R, a) with the El Haddad small-crack term."""
    cth = np.where(R >= 0.0, cth_pos, cth_neg)
    ratio = (1.0 - f) / ((1.0 - A0) * (1.0 - R))
    return dKth0 * np.sqrt(a / (a + a0)) / ratio ** (1.0 + cth * R)


def growth_rate(kmax, kmin, a, p):
    """da/dN for one cycle (or a set of cycles) with peak kmax and valley kmin.

    p is a dict of material/closure parameters (arrays broadcastable to kmax):
    C, n, p, q, dKth0, a0, cth_pos, cth_neg, alpha, smax_sigma0, kappa, Kc.
    Returns da/dN >= 0. Cycles with Kmax <= 0 (crack closed) do not grow.
    Kmax >= Kc is failure and is handled by the caller; here the rate is capped.
    """
    kmax = np.asarray(kmax, dtype=float)
    kmin = np.asarray(kmin, dtype=float)
    open_ = kmax > 0.0
    kmax_s = np.where(open_, kmax, 1.0)
    R = np.clip(kmin / kmax_s, R_MIN, R_MAX)
    f, A0 = newman_closure(R, p["alpha"], p["smax_sigma0"])
    dk_c = (1.0 - R) * kmax_s                     # range NASGRO sees at clamped R
    dk_eff = p["kappa"] * (1.0 - f) * kmax_s      # = kappa (1-f)/(1-R) dK
    dk_eff = np.minimum(dk_eff, kmax_s - np.minimum(kmin, kmax_s))
    dkth = threshold(R, f, A0, a, p["dKth0"], p["a0"], p["cth_pos"], p["cth_neg"])
    thr = np.clip(1.0 - dkth / dk_c, 0.0, None)
    crit = np.clip(1.0 - kmax_s / p["Kc"], 1e-6, None)
    rate = p["C"] * dk_eff ** p["n"] * thr ** p["p"] / crit ** p["q"]
    return np.where(open_ & (thr > 0.0), rate, 0.0)
