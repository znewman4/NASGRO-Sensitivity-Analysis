"""Stress-intensity solutions and a weight function for arbitrary stress fields.

Plate: semi-elliptical surface crack, deepest point, Newman & Raju (1981, 1984).
Bar:   surface crack in a solid round bar under tension, Forman & Shivakumar (1986)
       (NASGRO SC07-type solution).
Weight function: Shen & Glinka (1991) three-term form for the deepest point of a
       surface crack, with M1 and M3 solved so that it reproduces the uniform and
       linear reference solutions exactly (M2 = 3).

Lengths in m, stresses in MPa, K in MPa*sqrt(m).
"""
import numpy as np


# --- Plate: Newman & Raju, deepest point (phi = 90 deg), a/c <= 1 -------------

def newman_raju_Q(ac):
    return 1.0 + 1.464 * ac**1.65


def newman_raju_F(a, ac, t, W):
    """Boundary-correction factor F (tension) at the deepest point."""
    c = a / ac
    at = np.clip(a / t, 0.0, 0.95)
    M1 = 1.13 - 0.09 * ac
    M2 = -0.54 + 0.89 / (0.2 + ac)
    M3 = 0.5 - 1.0 / (0.65 + ac) + 14.0 * (1.0 - ac) ** 24
    arg = np.clip(np.pi * c / (2.0 * W) * np.sqrt(at), 0.0, 1.5)
    fw = np.sqrt(1.0 / np.cos(arg))
    return (M1 + M2 * at**2 + M3 * at**4) * fw


def newman_raju_H(a, ac, t):
    """Bending multiplier H at the deepest point (H = H2)."""
    at = a / t
    G1 = -1.22 - 0.12 * ac
    G2 = 0.55 - 1.05 * ac**0.75 + 0.47 * ac**1.5
    return 1.0 + G1 * at + G2 * at**2


def plate_Y0_Y1(a, ac, t, W):
    """Geometry factors for uniform (Y0) and linear sigma0*(1 - x/a) (Y1) stress,
    defined by K = sigma0 * sqrt(pi a / Q) * Y.

    sigma0 (1 - x/a) = sigma0 (1 - t/2a) [tension] + sigma0 t/2a [bending
    sigma_b (1 - 2x/t)], so Y1 = F [1 - (1 - H) t / 2a].
    """
    F = newman_raju_F(a, ac, t, W)
    H = newman_raju_H(a, ac, t)
    return F, F * (1.0 - (1.0 - H) * t / (2.0 * a))


def plate_K_uniform(a, ac, t, W, stress):
    return stress * np.sqrt(np.pi * a / newman_raju_Q(ac)) * newman_raju_F(a, ac, t, W)


# --- Bar: Forman & Shivakumar (1986) -----------------------------------------

def bar_F(a, D):
    xi = np.clip(a / D, 1e-6, 0.75)
    th = np.pi * xi / 2.0
    g = 0.92 * (2.0 / np.pi) / np.cos(th) * np.sqrt(np.tan(th) / th)
    return g * (0.752 + 2.02 * xi + 0.37 * (1.0 - np.sin(th)) ** 3)


def bar_K_uniform(a, D, stress):
    return stress * np.sqrt(np.pi * a) * bar_F(a, D)


# --- Weight function (Shen & Glinka 1991 form) -------------------------------

_GL_U, _GL_W = np.polynomial.legendre.leggauss(32)
_GL_U = 0.5 * (_GL_U + 1.0)        # map to [0, 1]
_GL_W = 0.5 * _GL_W


def shen_glinka_M(Y0, Y1, Q):
    """M1, M2, M3 for the deepest point, solved from the two reference cases."""
    k = np.pi / np.sqrt(2.0 * Q)
    M2 = 3.0
    M1 = 2.0 * k * (2.0 * Y0 - 3.0 * Y1) - 24.0 / 5.0
    M3 = 2.0 * (k * Y0 - 4.0 - M1)
    return M1, M2, M3


def weight_function_K(stress_fn, a, M1, M2, M3):
    """K = int_0^a sigma(x) m(x, a) dx at the deepest point.

    m(x,a) = 2/sqrt(2 pi (a-x)) [1 + M1 s^1/2 + M2 s + M3 s^3/2], s = 1 - x/a.
    With s = u^2 the integrand is smooth: K = 4 sqrt(a/2pi) int_0^1 sigma(a(1-u^2))
    (1 + M1 u + M2 u^2 + M3 u^3) du, evaluated by 32-point Gauss-Legendre.
    stress_fn maps depth x (m, same shape as a[..., None]) to stress (MPa).
    """
    a = np.asarray(a, dtype=float)[..., None]
    u = _GL_U
    x = a * (1.0 - u**2)
    br = 1.0 + M1[..., None] * u + M2 * u**2 + M3[..., None] * u**3
    sig = stress_fn(x)
    return 4.0 * np.sqrt(a[..., 0] / (2.0 * np.pi)) * np.sum(sig * br * _GL_W, axis=-1)
