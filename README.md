# NASGRO RUL sensitivity: residual-stress and variable-amplitude scenario modules

Scenario extensions to the NASGRO remaining-life (RUL) sensitivity study
("Which crack qualities are worth measuring?"). Each scenario is a separate
module with its own script, so the base case is untouched and every result is
tied to its assumptions.

| Script | What it does |
| --- | --- |
| `scripts/run_baseline_check.py` | Rebuilt constant-amplitude baseline: U90 ranking at every depth, checked against the study's published 1 mm statistics |
| `scripts/run_residual_stress.py` | Residual stress: K_res(a) superposed on applied K (7 scenarios) |
| `scripts/run_variable_amplitude.py` | Variable amplitude: stage 1 repeating block without interaction; stage 2 generalised Willenborg retardation |
| `scripts/make_figures.py` | Figures in `results/` |

```bash
pip install numpy scipy matplotlib pytest
python -m pytest -q tests
python scripts/run_baseline_check.py
python scripts/run_residual_stress.py          # ~20 min on 4 cores, n = 10 000
python scripts/run_variable_amplitude.py       # stage 1 n = 10 000, stage 2 n = 3 000
python scripts/make_figures.py
```

## Relation to the original pipeline

The original `nasgro_sa` pipeline was not in this repository, so `nasgro_scenarios`
rebuilds the two cases from the study document (inputs and spreads from its
Figures 3 and A1–A3):

* NASGRO equation (Forman & Mettu 1992) with Newman (1984) closure and the
  NASGRO threshold with the El Haddad intrinsic length a0; κ scales the
  effective range κ(1−f)/(1−R)·ΔK.
* Bar: Forman & Shivakumar (1986) round-bar SIF, D = 18 mm, F = 63.6 kN (250 MPa), R = −1.
  25CrMo4 threshold and slope from the ECF17 data (ΔKth = 13 MPa√m at R = −1, m = 3.2),
  then ΔKth0 and C recalibrated (5.8 MPa√m, 4.81e-12) to the study's 1 mm statistics.
* Plate: Newman & Raju (1981) deepest-point solution, t = 10 mm, W = 100 mm, a/c = 0.8, R = 0.1, 200 MPa.
* Growth is integrated in depth with the shape held fixed (no two-point a–c growth).

Check against the published 1 mm values (`results/baseline.json`):

| | median log10 N (published / rebuilt) | Pf | top inputs, U90 reduction (published → rebuilt) |
| --- | --- | --- | --- |
| Bar | 5.59 / 5.59 | 93 % / 93 % | depth 26 → 31 %, κ 10 → 8 %, load 9 → 8 %, material 10 → 6 % |
| Plate | 4.41 / 4.41 | 99 % / 99 % | depth 29 → 26 %, stress 8 → 9 %, κ 6 → 5 %, α 3 → 2 % |

The rebuilt scatter is wider (σ of log10 N 0.40 against 0.27 for the bar, 0.41 against
0.37 for the plate), and with the shape held fixed a/c is over-weighted for
deep plate cracks (14–21 % against 2–7 %). Scenario results are therefore
reported as changes against this rebuilt baseline, on the same random numbers.

## Residual stress (`nasgro_scenarios/residual_stress.py`)

* K_res(a) is added to both Kmax and Kmin: ΔK is unchanged, R_eff moves, and
  closure, threshold and the Kc term respond (LaRue & Daniewicz 2007; Parker 1982).
* Profile σ_res(z) = A σy (1 − z/L) e^(−z/L): peak at the surface, zero crossing at
  z = L, compressive balance beyond, zero net force. It has the shape of the
  BS 7910 Annex Q as-welded profiles; the Annex Q polynomial for a specific joint can
  replace `profile()`. Uniform A σy (BS 7910 Level-1 style) reproduces check C.
* K_res from the Shen & Glinka (1991) weight function, with M1 and M3 solved so it
  reproduces the Newman–Raju uniform and bending solutions exactly (tested).
* Uncertain inputs: amplitude A ~ U(lo, hi), decay length L lognormal (CoV 0.3).
* Optional first-cycle shakedown: surface σ_res limited so total stress stays
  within ±σy. No relaxation or redistribution as the crack grows (limitation).

## Variable amplitude (`nasgro_scenarios/variable_amplitude.py`)

* Repeating block of n_b cycles, Weibull amplitude exceedance (shape h), clipped at
  10 exceedances per block, peaks capped at 0.9 σy, counted into 17 levels, and
  scaled so the damage-equivalent amplitude equals the CA amplitude.
* Stage 1: growth per block summed over levels, so life is one depth quadrature.
* Stage 2: generalised Willenborg (Willenborg et al. 1971; Gallagher 1974),
  K_R = φ[σy√(βπ(a_ol + r_ol − a)) − Kmax], φ = 1/(SOLR − 1), applied as a shift
  of Kmax and Kmin (like a residual stress). Overload zone carried block to block;
  peaks evenly spaced in 20 segments; steady blocks lumped (≤ 1 % depth per step);
  halving both changes log10 N by < 0.01.
* Uncertain inputs: h, n_b, SOLR, overload plastic-zone factor β (2 plane stress to 6 plane strain).
* Not modelled: underload acceleration, strip-yield closure (Newman 1981).

## Multiple or interacting cracks

Left as a stated scope limit (see the study document).
