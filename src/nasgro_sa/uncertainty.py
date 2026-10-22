"""Catalogue of industry-typical input uncertainty (question 3).

Each entry gives a low / typical / high spread, as a coefficient of variation
("cov") or an absolute standard deviation ("sd", in the input's display unit),
with its basis: "literature" when a source reports the value or a range that
brackets it, "judgement" when no source was found and the value is an
engineering estimate to be replaced. Sources are listed with their URLs.

Crack-size uncertainty depends on the stage of life. Below the detection
limit a crack is known only through an assumed or back-calculated
distribution (EICS-type knowledge, large relative scatter); once detected and
sized, the error is roughly absolute, so its relative size shrinks as the
crack grows.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .inputs import Marginal

LEVELS = ("low", "typical", "high")

SOURCES = {
    "TWI-TOFD": "TWI, Structural integrity with TOFD ultrasonic inspection (2005): planar-flaw height error "
                "mean +0.5 mm, sd 1.8 mm (flaws 1.5-30 mm); TOFD struggles below ~2 mm depth. "
                "https://www.twi-global.com/technical-knowledge/published-papers/"
                "structural-integrity-with-time-of-flight-diffraction-tofd-ultrasonic-inspection-july-2005",
    "JSM-PA": "JSM, phased-array IGSCC sizing qualification: depth error mean 0.33 mm, sd 1.92 mm. "
              "https://www.jsm.or.jp/ejam/Vol.4No.4/AA/SP7/47/47.html",
    "Giannella2022": "Giannella (2022) Int J Fract 235:179-195, Table 4: crack depth sd 0.15 mm, diameter cov 1.5 %, "
                     "load cov 1.5 %; Fig. 8: jointly fitted NASGRO parameters.",
    "Giannella2021": "Giannella (2021) Int J Fatigue 144:106044: NDI detects cracks larger than 1-2 mm; "
                     "+/-5 % bounds on K from FEM/DBEM.",
    "eFatigue": "eFatigue, Probabilistic crack growth: load cov ~0.1 when controlled, up to 0.5 for uncontrolled "
                "usage; crack-growth property cov 0.05-0.1 (0.2-0.3 without data). "
                "https://efatigue.com/probabilistic/crackgrowth",
    "ClarkHudak1975": "Clark & Hudak (1975) round robin: inter-laboratory spread ~2:1 typical, ~3:1 overall in "
                      "da/dN at given dK. https://trid.trb.org/View/36287",
    "Threshold-RR": "ASTM STP (1981) threshold round robin (large, procedure-dependent scatter) and "
                    "Materials (2022) on dKth evaluation. https://store.astm.org/stp33450s.html, "
                    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9324856/",
    "Kc-scatter": "Farahmand (2004): Kc cov 5-10 % for aerospace Al; DTIC ADA201435: ~90-100 % in older 4340/7075 "
                  "data. https://store.astm.org/stp11279s.html, https://apps.dtic.mil/sti/tr/pdf/ADA201435.pdf",
    "Newman-alpha": "Constraint factor is a lumped, usually defaulted constant; alpha ~1.15-2.0 used for 2024-T3. "
                    "https://blog.vibrationdata.com/?p=151193; Skorupa (2005) on NASGRO closure treatment.",
    "EIFS": "EIFS distributions are lognormal or Weibull with case-specific parameters (Embraer COBEM 2009; "
            "DSTG 2016). https://www.abcm.org.br/anais/cobem/2009/pdf/COB09-0501.pdf",
}


@dataclass(frozen=True)
class Uncertainty:
    label: str
    kind: str                      # "cov" or "sd"
    low: float
    typical: float
    high: float
    distribution: str = "normal"   # "normal" or "lognormal" (median held)
    basis: str = "judgement"
    sources: tuple[str, ...] = ()
    bounds: tuple[float, float] | None = None
    scale: float = 1.0             # display unit -> SI, for "sd" entries

    def value(self, level: str) -> float:
        return {"low": self.low, "typical": self.typical, "high": self.high}[level]

    def marginal(self, nominal: float, level: str) -> Marginal:
        v = self.value(level)
        spread = {"cov": v} if self.kind == "cov" else {"sd": v * self.scale}
        hold = "median" if self.distribution == "lognormal" else "mean"
        basis = "data" if self.basis == "literature" else "exploratory"
        return Marginal(self.distribution, nominal, **spread, hold=hold, bounds=self.bounds, basis=basis)


MM = 1e-3
DETECTION_LIMIT = 1.0 * MM   # Giannella2021 (1-2 mm); TOFD struggles below ~2 mm (TWI-TOFD)

# Crack descriptors depend on whether the crack has been detected and sized.
CRACK = {
    "sized": {
        "a_init": Uncertainty("crack depth (UT-sized)", "sd", 0.15, 0.5, 1.8, scale=MM, basis="literature",
                              sources=("Giannella2022", "TWI-TOFD", "JSM-PA")),
        "aspect_init": Uncertainty("aspect ratio a/c (sized)", "cov", 0.1, 0.2, 0.3),
        "theta": Uncertainty("tilt from principal plane [deg] (sized)", "sd", 2.0, 5.0, 10.0),
    },
    "undetected": {
        "a_init": Uncertainty("crack depth (undetected / EICS)", "cov", 0.3, 0.6, 1.0, "lognormal",
                              sources=("EIFS",)),
        "aspect_init": Uncertainty("aspect ratio a/c (undetected)", "cov", 0.2, 0.3, 0.5),
        "theta": Uncertainty("tilt from principal plane [deg] (undetected)", "sd", 5.0, 15.0, 30.0),
    },
}

CATALOGUE = {
    # Loading and analysis
    "S_max": Uncertainty("stress amplitude", "cov", 0.015, 0.10, 0.5, "lognormal", "literature",
                         ("Giannella2022", "eFatigue")),
    "F": Uncertainty("load amplitude", "cov", 0.015, 0.10, 0.5, "lognormal", "literature",
                     ("Giannella2022", "eFatigue")),
    "R": Uncertainty("stress ratio (mean / residual stress)", "sd", 0.02, 0.05, 0.15, bounds=(-2.0, 0.95)),
    "K_factor": Uncertainty("SIF solution error (exact inputs)", "cov", 0.03, 0.05, 0.10, "lognormal", "literature",
                            ("Giannella2021",)),
    # Geometry
    "D": Uncertainty("diameter", "cov", 0.005, 0.015, 0.03, basis="literature", sources=("Giannella2022",)),
    "thickness": Uncertainty("thickness", "cov", 0.005, 0.015, 0.03),
    "width": Uncertainty("width", "cov", 0.005, 0.015, 0.03),
    # Growth-rate curve (C-n): level of the curve at the pivot, and slope about the pivot
    "curve_level": Uncertainty("growth-rate level (C at pivot)", "cov", 0.05, 0.15, 0.30, "lognormal",
                               "literature", ("eFatigue", "ClarkHudak1975")),
    "n": Uncertainty("growth-rate slope n (about pivot)", "cov", 0.02, 0.05, 0.10, bounds=(0.5, 8.0)),
    # Near-threshold and near-critical shape
    "dK0": Uncertainty("threshold dKth0", "cov", 0.05, 0.10, 0.20, "lognormal", "literature",
                       ("Threshold-RR", "Giannella2022")),
    "p": Uncertainty("threshold-shape exponent p", "cov", 0.15, 0.30, 0.50, bounds=(0.01, 10.0)),
    "q": Uncertainty("critical-shape exponent q", "cov", 0.15, 0.30, 0.50, bounds=(0.0, 10.0)),
    "Kc": Uncertainty("fracture toughness Kc", "cov", 0.05, 0.10, 0.30, "lognormal", "literature", ("Kc-scatter",)),
    # Closure and short-crack constants (usually defaulted from the database)
    "alpha": Uncertainty("constraint factor alpha", "cov", 0.10, 0.20, 0.30, basis="literature",
                         sources=("Newman-alpha",), bounds=(1.0, 3.0)),
    "smax_sigma0": Uncertainty("Smax/sigma0", "cov", 0.10, 0.20, 0.30, bounds=(0.05, 0.95)),
    "Cth_pos": Uncertainty("Cth (R >= 0)", "cov", 0.10, 0.30, 0.50),
    "Cth_neg": Uncertainty("Cth (R < 0)", "cov", 0.10, 0.30, 0.50),
    "a0_intrinsic": Uncertainty("intrinsic crack length a0", "cov", 0.25, 0.50, 1.0, "lognormal"),
    # In-service opening state relative to Newman's closure function (residual stress, roughness, oxide,
    # load history); scales the open part of the cycle.
    "closure_factor": Uncertainty("crack opening factor kappa", "cov", 0.05, 0.10, 0.20, "lognormal"),
}

# Indicative ultrasonic precision for each measurement target, as standard deviations in the target's
# unit (low / typical / high); None where no established capability was found. Context for the
# measurement-target map only: measurement evidence, not these values, establishes feasibility.
UT_CAPABILITY = {
    "size": (0.15, 0.5, 1.8),        # mm, depth sizing (Giannella2022, TWI-TOFD, JSM-PA)
    "shape": (0.08, 0.16, 0.24),     # sd of a/c, from 10-30 % on a/c = 0.8 (judgement)
    "orientation": (2.0, 5.0, 10.0),  # deg (judgement)
    "opening": None,                 # crack-opening measurement (e.g. nonlinear ultrasonics) is research-stage
}


def crack_stage(a: float) -> str:
    return "sized" if a >= DETECTION_LIMIT else "undetected"


# Crack knowledge held the same at every stage, for separating physics from knowledge:
# depth cov, aspect-ratio cov and tilt sd (deg), at the size-independent end of the catalogue.
CONSTANT_KNOWLEDGE = {"a_init": 0.25, "aspect_init": 0.2, "theta": 5.0}


def crack_marginals(a: float, aspect: float, level: str, a_max: float,
                    knowledge: str = "staged") -> dict[str, Marginal]:
    """Marginals of the crack descriptors at nominal depth ``a`` (m).

    ``knowledge="staged"`` follows the detection stage (undetected below
    DETECTION_LIMIT, UT-sized above); ``"constant"`` uses CONSTANT_KNOWLEDGE at
    every depth, so only the physics changes with crack size.
    """
    if knowledge == "constant":
        k = CONSTANT_KNOWLEDGE
        m = {"a_init": Marginal("lognormal", a, cov=k["a_init"], hold="median"),
             "aspect_init": Marginal("normal", aspect, cov=k["aspect_init"]),
             "theta": Marginal("normal", 0.0, sd=k["theta"])}
    else:
        entries = CRACK[crack_stage(a)]
        m = {"a_init": entries["a_init"].marginal(a, level), "aspect_init": entries["aspect_init"].marginal(aspect, level),
             "theta": entries["theta"].marginal(0.0, level)}
    return {"a_init": replace(m["a_init"], bounds=(1e-6, 0.95 * a_max)),
            "aspect_init": replace(m["aspect_init"], bounds=(0.1, 1.9)),
            "theta": replace(m["theta"], bounds=(-80.0, 80.0))}


def relative_size_uncertainty(a: np.ndarray, level: str = "typical") -> np.ndarray:
    """Coefficient of variation of crack depth knowledge versus nominal depth (m)."""
    a = np.asarray(a, dtype=float)
    sized = CRACK["sized"]["a_init"].value(level) * MM / a
    return np.where(a >= DETECTION_LIMIT, sized, CRACK["undetected"]["a_init"].value(level))
