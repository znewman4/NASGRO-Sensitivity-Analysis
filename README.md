# NASGRO Sensitivity Analysis

Propagates uncertainty in crack-growth inputs through a NASGRO-equation life
model and ranks which input uncertainties drive predicted remaining useful
life (RUL).

**RUL** is the number of cycles from the current crack state to the declared
failure criterion under the declared future loading (see
[baseline.py](src/nasgro_sa/baseline.py) for the full scope statement).

**Declared measure of RUL uncertainty.** U90 = log10(N95/N05), the width in
decades of the central 90 % band of RUL among cracks that grow to failure.
The median and 5 % lower percentile are reported alongside it, and the
probability that the crack does not propagate separately.

**Main analysis: the measurement-target map** (`scripts/run_targets.py`,
[targets.py](src/nasgro_sa/targets.py)). For each crack quality ultrasound
might characterise (size, shape, orientation, opening/closure), the
measurement standard deviation is swept from coarse to exact while every
other input keeps its industry-typical uncertainty. Material and loading
uncertainty therefore set the floor that crack characterisation cannot
remove. The map reports, per candidate: the achievable U90 reduction, the
precision needed for 80 % of it, the diminishing-return point and the crack
states and scenarios where the reduction is substantial (≥ 10 % of U90).
Reduction curves for every input/group give the full ranking. Orientation is
tested under projection and mixed-mode (energy, Tanaka) mechanics, and
opening/closure through a factor κ on the open part of the cycle. Sobol
indices (life-stage study, campaign) are supporting analyses.

**C and n are one quantity.** C and n are fitted jointly and are highly
correlated, so they are treated as a single uncertain input, the growth-rate
curve:
- they receive one grouped Sobol index ("C-n" in the baseline, "growth" in
  calibrated specs);
- spreads are rescaled together (`InputSpec.scaled` refuses to rescale a
  group member alone);
- ranking C against n is out of scope.

## Layout

| Module | Contents |
| --- | --- |
| `growth.py` | NASGRO equation with Newman closure; Walker; Paris |
| `geometry.py` | Newman–Raju surface crack in a plate (fixed or sampled thickness/width); James–Mills round bar |
| `integrate.py` | Vectorised two-point life integration with step-error control; outcomes (fracture, limits, arrest, censored); cycles per growth regime |
| `uncertainty.py` | Catalogue of industry-typical input uncertainty (low / typical / high, with sources); crack-size knowledge by detection stage |
| `cases.py` | Round-bar (Giannella) and plate cases exposing every NASGRO input, crack tilt, SIF accuracy and tolerances |
| `targets.py` | **Main analysis**: measurement candidates, precision sweeps with the uncertainty floor, measurement-target map |
| `lifestage.py` | Supporting life-stage study: regime shares, local effects, equal-uncertainty screen, Sobol importance |
| `inputs.py` | Marginals with mean/median held fixed under rescaling; Gaussian-copula groups |
| `sobol.py` | Grouped pick-freeze Sobol estimators with bootstrap CIs |
| `moment_independent.py` | Given-data KS sensitivity index with a dummy-input bias floor |
| `calibrate.py` | Least-squares fits of each growth model to the same da/dN lots; lot-to-lot parameter distributions |
| `model.py` | `Formulation`: life from sampled inputs; growth rate at matched states |
| `baseline.py` | Declared baseline formulation and placeholder input distributions |
| `studies.py` | Sobol, growth-rate, variance-sweep, reduction-path and robustness studies; integration convergence |

## Running

```
pip install -e .[dev]
pytest
python scripts/run_targets.py                        # MAIN: measurement-target map -> results/targets/
python scripts/run_life_stages.py                    # supporting: regimes, Sobol by crack size
python scripts/run_campaign.py                       # supporting: pairs, scenarios, variants, models
python scripts/plot_summary.py                       # short-report figures from the results above -> results/summary/
python scripts/run_variable_sweeps.py                # Giannella-style scatter and PDFs for every input -> results/variable_sweeps/
python scripts/run_baseline.py --states 0.3 1 3      # propagation, convergence, Sobol, KS, da/dN
python scripts/run_model_comparison.py               # NASGRO vs Walker vs Paris, same calibration data
python scripts/run_robustness.py                     # rankings under alternative distributions
python scripts/run_reduction.py --depths 0.3 1 3     # greedy uncertainty-reduction path
```

All baseline parameter values, and the calibration lots, are illustrative
placeholders, not NASGRO database values or test data.

## Life-stage study (research questions 1–3)

`python scripts/run_life_stages.py` repeats three analyses at initial crack
depths from EICS scale (0.05 mm) to near the failure limit, for the round bar
and the plate, and writes tables and figures to `results/life_stages/`:

| Question | Analysis | Output |
| --- | --- | --- |
| 1. Which part of life is evaluated? | Share of cycles to failure spent near threshold, in Paris growth and near critical; probability the crack grows at all; a crack-size × stress map | `q1_life_regimes.png`, `stages.csv` |
| 2. What is NASGRO most sensitive to? | Local effect of +10 % in each input, and Sobol total indices with every input ±10 % uniform, so the ranking reflects the model, not assumed uncertainty | `q2_sensitivity.png`, `local_effects.csv`, `screen_ST_*.csv` |
| 3. Which inputs are uncertain in industry? | Sobol total indices with the uncertainty catalogue (crack-size knowledge depends on detection), checked at its low and high ends | `q3_importance.png`, `uncertainty_catalogue.png`, `industry_*_ST_*.csv` |

Catalogue entries marked "judgement" have no source yet and should be
replaced as data are found. "SIF solution error" is the error of the K
solution itself given exact inputs; it is not a NASGRO input and acts exactly
like a stress change, so it appears only in the Q3 analysis.

The same script also reports **fixed decision horizons**: P(failure within a
declared cycle count, e.g. an inspection interval) and its Sobol drivers
(`fixed_horizons.png`). Population-relative horizons ask what drives the
spread of life; fixed horizons ask what drives the decision.

## Campaign (`scripts/run_campaign.py`)

Starting from typical industry uncertainty at three key stages per case (small,
first detectable, deep), [campaign.py](src/nasgro_sa/campaign.py) runs:

| Run | Question | Output |
| --- | --- | --- |
| `reduction` | Where does better knowledge of one input help, and where does it plateau? | `reduction_curves.png/.csv` |
| `pairs` | Is information valuable only when another uncertainty is reduced too? | `paired_reductions.png/.csv` |
| `physics` | Does the ranking change through physics, uncertainty magnitudes, or stage-dependent crack knowledge? | `physics_vs_knowledge.png/.csv` |
| `scenarios` | Which conclusions transfer across stress, stress ratio and geometry? | `scenarios.png/.csv` |
| `distributions` | Are the shortlisted targets robust to shapes, widths and dependencies? | `distributions.png/.csv` |
| `models` | Does disagreement between consistently calibrated models outweigh the benefit of precise inputs? | `model_comparison.png/.csv` |

All runs use the declared U90 measure. Overload and load-history effects are
not yet modelled.

## Reproduction of Giannella (2022)

`python scripts/reproduce_giannella.py` reproduces Figs 9 and 11–15 of
Giannella (2022, Int J Fract 235:179–195): a round bar under R = −1 with the
paper's NASGRO constants, input distributions and accept/reject material
sampling ([giannella.py](src/nasgro_sa/giannella.py)). Tables and figures go
to `results/giannella/`. Known departures from the paper:
- the 25CrMo4 test data are not published, so the material sets are accepted
  against synthetic data around the paper's best-fit curve;
- q = 1 is used instead of the printed q = 5, because q = 1 reproduces the
  paper's best-fit life (log10 N = 5.564 vs 5.559) and q = 5 gives 5.335.

`python scripts/run_variable_sweeps.py` extends Figs 11–14 to every input of
both cases ([sweeps.py](src/nasgro_sa/sweeps.py)). Each input's spread is set
in turn to about twice typical (capped at the catalogue's high end), typical,
a third and a thirtieth, while every other input keeps its industry-typical
uncertainty. The fitted material pool is only shrunk, never widened.
