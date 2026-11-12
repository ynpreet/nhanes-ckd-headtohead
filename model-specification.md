# Model specification — Paper N2 logistic-regression triage model

Full specification of the logistic-regression model, so a third party can compute predictions for a new individual without refitting. Extracted 12 November 2026 by refitting the deposited code (`code/analysis_n2.py`) on the cached CDC NHANES public-use files; the refits reproduce the published metrics exactly (test AUROC 0.7665; held-out AUROC 0.7891).

## How to compute a prediction

1. Take the eight routine predictors below, in this order. 2. Standardise each: z = (x − mean) / sd, using the means and standard deviations in the table. 3. Compute the linear predictor LP = intercept + Σ(coefficient × z). 4. Predicted probability of the CKD proxy = 1 / (1 + exp(−LP)).

Predictors are **not** survey-weighted at prediction time; the survey weights were used only in fitting. Serum creatinine and UACR are outcome variables only and are never predictors. The outcome is the CKD proxy: eGFR <60 mL/min/1.73 m² (CKD-EPI 2021, race-free) **or** UACR ≥30 mg/g, single visit.

## (a) Test-split model — the model behind the Table 2 test-set metrics

Fitted on the 70% training split (n=3,090) of the development complete case (n=4,415), seed 42, stratified on the outcome; survey-weighted. `sklearn` pipeline: `StandardScaler` + `LogisticRegression(max_iter=2000, C=1.0)`.

| Predictor | Mean | SD | Coefficient (standardised scale) |
|---|---:|---:|---:|
| Age (years) | 51.131068 | 17.469140 | +0.488468 |
| Female (0/1) | 0.519741 | 0.499610 | +0.058122 |
| BMI (kg/m^2) | 29.906084 | 7.250529 | +0.062952 |
| Mean systolic BP (mm Hg) | 126.794822 | 19.753728 | +0.415723 |
| Mean diastolic BP (mm Hg) | 72.595793 | 12.764694 | -0.229300 |
| HbA1c (%) | 5.844498 | 1.077999 | +0.353454 |
| Diabetes history (0/1) | 0.159223 | 0.365884 | +0.137228 |
| Hypertension history (0/1) | 0.375081 | 0.484144 | +0.354430 |

Intercept: -2.022085

## (b) Full-development refit — the model behind the held-out (2021–2023) metrics

Same pipeline and settings, fitted on the full development complete case (n=4,415) and applied unchanged to the held-out complete case (n=4,983). This is the model whose held-out performance is reported (AUROC 0.789, Brier 0.116, calibration slope 1.07).

| Predictor | Mean | SD | Coefficient (standardised scale) |
|---|---:|---:|---:|
| Age (years) | 51.056852 | 17.540716 | +0.434158 |
| Female (0/1) | 0.516421 | 0.499730 | +0.087901 |
| BMI (kg/m^2) | 29.814247 | 7.215538 | +0.039885 |
| Mean systolic BP (mm Hg) | 126.621065 | 19.623083 | +0.362840 |
| Mean diastolic BP (mm Hg) | 72.539751 | 12.814263 | -0.167228 |
| HbA1c (%) | 5.855515 | 1.105393 | +0.335964 |
| Diabetes history (0/1) | 0.163080 | 0.369439 | +0.191996 |
| Hypertension history (0/1) | 0.376444 | 0.484493 | +0.337414 |

Intercept: -1.981809

## Gradient boosting and random forest — hyperparameters only

These two models cannot be fully specified in text: gradient boosting `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31, random_state=42)`; random forest `RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=1, random_state=42)`. No fitted model objects are deposited. Both models are reproducible deterministically by rerunning the deposited code (`code/analysis_n2.py`, fixed seeds, single-threaded), which reproduces every published metric exactly; the full run log is `results-n2.md`.
