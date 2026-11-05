"""BS 7910:2019 inputs for the rerun of the NASGRO-only study (new module; base code untouched).

Values are paraphrased from BS 7910:2019 (University of Bristol licence) with the
clause they come from. Nothing here changes cases.py or nasgro.py: sizing error
enters through cases.sample(..., spread={"depth": sd / 0.5 mm}).

Sizing (Annex T):
  * Table T.2 (back-surface / far-side flaws, through-thickness accuracy +-E) and
    T.7.5.2 (TOFD trials +-0.5 to +-0.1 mm). E is a ~80 % two-sided interval, so
    the sizing error sd = E / 1.3 (T.5). Minimum reliably sized depth from Table T.2.
Growth laws (Clause 8, Annex K):
  * Simplified Paris law for steels in air, m = 3, A = 5.21e-13 (da/dN in mm/cycle,
    dK in N/mm^1.5), 8.2.3.5; for another metal A x (E_steel / E)^3 (Equation 8.8).
  * Threshold: unwelded steel, R < 0: dK0 = 170 N/mm^1.5 (Equation 8.7c);
    aluminium in air: 21 N/mm^1.5 (Table 8.5). Below dK0, da/dN = 0.
  * Probabilistic: Table K.2, parent metal in air (Snijder et al.): m = 2.8,
    ln A ~ Normal(-27.76, 0.23) (N, mm); threshold scatter about 10 % (K.6.2.5).
  * Full stress range (Kmax - Kmin) is used, as BS 7910 does for welded joints
    (8.2.1.4); this is conservative for the R = -1 bar.
Assumed, not from BS 7910: E_steel = 207 GPa, E_aluminium = 70 GPa.
"""
import numpy as np

from . import cases, life, runner

N_MM15 = 1000 ** 0.5          # 1 MPa sqrt(m) = 31.62 N/mm^1.5

TECHNIQUES = {                 # Table T.2 / T.7.5.2: +-E (mm), min reliably sized depth (mm)
    "conventional UT": dict(E=3.0, a_min=3.0),
    "focussed phased array / zonal AUT": dict(E=1.5, a_min=2.0),
    "TOFD (Table T.2)": dict(E=1.0, a_min=2.0),
    "TOFD trials, +-0.5 mm": dict(E=0.5, a_min=2.0),
    "TOFD trials, +-0.1 mm": dict(E=0.1, a_min=2.0),
}
STUDY_SD_MM = 0.5             # the study's "typical UT"


def sd_mm(name):
    return TECHNIQUES[name]["E"] / 1.3


E_STEEL, E_AL = 207e3, 70e3   # MPa, assumed
PARIS_DET = {
    "bar": dict(A=5.21e-13, m=3.0, dK0=170.0),                               # 8.2.3.5, 8.7c
    "plate": dict(A=5.21e-13 * (E_STEEL / E_AL) ** 3, m=3.0, dK0=21.0),      # 8.8, Table 8.5
}
PARIS_PROB = dict(m=2.8, lnA_mean=-27.76, lnA_sd=0.23, dK0_cov=0.10)        # Table K.2, K.6.2.5

ALL_KEYS = {
    "bar": ("depth", "tilt", "sif", "kappa", "Kc", "a0", "R", "F", "D", "material"),
    "plate": ("depth", "tilt", "sif", "kappa", "Kc", "a0", "R", "S", "t", "W", "ac", "alpha",
              "smax_sigma0", "q", "p", "dKth0", "cth_pos", "cth_neg", "Cn"),
}


def sample(case, a_nom_m, n, sd, seed=1, fixed=()):
    """NASGRO-only samples with detection sizing error sd (mm)."""
    return cases.sample(case, a_nom_m, n, seed=seed, fixed=runner.fx(tuple(fixed)),
                        spread={"depth": sd / STUDY_SD_MM})


def paris_life(case, s, A, m, dK0, n_grid=400, positive_only=False):
    """Life under da/dN = A dK^m (mm/cycle, N/mm^1.5), zero below dK0; same SIF and failure check as NASGRO.

    positive_only: use Kmax (tensile part) as the range when R < 0 instead of the full range.
    """
    _, a, da = life.depth_grid(s["a_i"], cases.a_limit(case, s), n_grid)
    kmax = cases.k_unit(case, s, a) * cases.max_stress(case, s)[:, None]
    rng_fac = 1.0 - s["R"][:, None]
    if positive_only:
        rng_fac = np.where(s["R"][:, None] < 0, 1.0, rng_fac)
    dk = rng_fac * kmax * N_MM15
    A = np.broadcast_to(np.asarray(A, float), (len(s["a_i"]),))[:, None]
    dK0 = np.broadcast_to(np.asarray(dK0, float), (len(s["a_i"]),))[:, None]
    rate = np.where(dk >= dK0, 1e-3 * A * np.maximum(dk, 0.0) ** m, 0.0)
    return life.integrate(rate, kmax, s["Kc"], da)


def det_life(case, a_reported_m, E_m, positive_only=False):
    """Deterministic BS 7910 life: reported depth + sizing tolerance E, nominal loads, Paris (8.2.3.5)."""
    s = cases.sample(case, a_reported_m + E_m, 1, seed=1, fixed=ALL_KEYS[case])
    p = PARIS_DET[case]
    return float(paris_life(case, s, p["A"], p["m"], p["dK0"], positive_only=positive_only)[0])


def prob_life(case, s, seed=1, positive_only=False):
    """Annex K style life: ln A ~ Normal (Table K.2), dK0 with 10 % scatter, on the NASGRO samples' depth and load."""
    n = len(s["a_i"])
    rng = np.random.default_rng([seed, 7910])
    pp = PARIS_PROB
    lnA = pp["lnA_mean"] + pp["lnA_sd"] * rng.standard_normal(n)
    scale = (E_STEEL / E_AL) ** pp["m"] if case == "plate" else 1.0
    dK0 = PARIS_DET[case]["dK0"] * (1 + pp["dK0_cov"] * rng.standard_normal(n))
    return paris_life(case, s, np.exp(lnA) * scale, pp["m"], dK0, positive_only=positive_only)


def percentiles(L, qs=(2.3, 5, 50, 95)):
    """log10 life percentiles over ALL samples (arrested cracks count as infinite life)."""
    x = np.log10(np.where(np.isfinite(L), L, 1e30))
    return {f"q{q:g}": float(np.percentile(x, q)) for q in qs}
