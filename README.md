# Race-free versus race-based eGFR: a head-to-head reclassification, triage-model and net-benefit comparison in NHANES 2017–2018 with held-out validation in 2021–2023

Sole author: **Preet Mehta** (Independent researcher, Boston, MA, USA)
Contact: preet.mehta.research@gmail.com | ORCID: [0009-0003-8294-1575](https://orcid.org/0009-0003-8294-1575)

Analysis code for the manuscript submitted to the *Journal of the American Medical Informatics Association* (JAMIA), Research and Applications (November 2026). This is a substantially new paper: an earlier descriptive version of this CKD work (a single logistic model with a partial reclassification table) was submitted to the *Journal of Biomedical Informatics* as JBI-26-4252 and desk-rejected on 9 November 2026. The head-to-head equation comparison, the three-model comparison, the decision-curve analysis, and the held-out validation in this repository are all new analyses.

## What the code does

1. **Head-to-head equation comparison** — CKD-EPI 2021 (race-free) vs CKD-EPI 2009 (race-based) as outcome definitions on the same NHANES participants: reclassification counts, G-stage cross-tabulation, net reclassification by survey group, and KDIGO G × A staging.
2. **Triage-model comparison** — logistic regression vs gradient boosting vs random forest on routine-only predictors (age, sex, BMI, mean systolic/diastolic BP, HbA1c, diabetes and hypertension history). Serum creatinine and UACR define the outcome and are never predictors (asserted in code). 70/30 stratified split (seed 42), paired-bootstrap AUROC differences, calibration, and subgroup calibration.
3. **Decision-curve analysis** — net benefit and unnecessary tests avoided per 100 at thresholds 5%–50%, versus "test everyone" and "test no one", on the test set and held-out.
4. **Outcome-definition sensitivity** — each model's predictions scored against both the 2021-defined and the 2009-defined proxy.
5. **Temporal validation** — models refit on the full development case are applied unchanged to NHANES 2021–2023.

## Headline results (from `results-n2.md`)

- CKD proxy (2021 equation): 923/4,827 (19.1% unweighted; 14.0% weighted, 95% CI 12.7–15.4%). Under 2009: 941 (19.5%; 15.0% weighted, 95% CI 13.3–16.6%).
- Reclassification 2021 vs 2009: 88 changed status (35 newly, 53 no longer); G stage changed for 570 (11.8%). Net per 1,000: Non-Hispanic Black +32.3, Non-Hispanic White −22.0. Held-out: 100 reclassified (20 newly, 80 lost).
- Test AUROC (n = 1,325): logistic 0.767, gradient boosting 0.746, random forest 0.769, age-only 0.717. Held-out AUROC (n = 4,983): logistic 0.789, gradient boosting 0.770, random forest 0.794.
- Gradient boosting was significantly worse than logistic (paired-bootstrap AUROC difference 95% CI −0.041 to −0.001) and badly miscalibrated (slope 0.45 test, 0.63 held-out). Random forest held-out: Brier 0.113, slope 0.99, ECE 0.019.
- Decision curves: triage beats "test everyone" above ~6–7% (test) and from 2–5% (held-out); at 15%, the random forest avoids ~29–36 unnecessary tests per 100.

## Data

All data are CDC NHANES public-use files (de-identified; no patient data collected):

- Development, NHANES 2017–2018: `DEMO_J, BPX_J, BMX_J, BIOPRO_J, ALB_CR_J, GHB_J, DIQ_J, BPQ_J`
- Held-out, NHANES August 2021–August 2023: `DEMO_L, BPXO_L, BMX_L, BIOPRO_L, ALB_CR_L, GHB_L, DIQ_L, BPQ_L`

Source: https://wwwn.cdc.gov/nchs/nhanes/Default.aspx. The scripts download any missing XPT files from CDC automatically; place cached files in `--dev-data` / `--val-data` directories to run offline.

**Cystatin C limitation.** No cystatin-C equation was computed: `BIOPRO_J` and `BIOPRO_L` contain no cystatin-C column (verified by column listing, 12 November 2026), and candidate separate files returned 404 from the CDC portal. The head-to-head is creatinine-only plus albuminuria staging. No cystatin value was invented or borrowed.

**Blood-pressure comparability caveat.** Blood-pressure measurement changed between cycles (2017–2018 auscultatory; 2021–2023 oscillometric only), so held-out performance is not a pure transport estimate for blood-pressure effects.

## Reproduction

Requirements: Python 3.10+ with pandas, numpy, scikit-learn, scipy, matplotlib (see `requirements.txt`).

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u code/analysis_n2.py
python3 code/supplement_rf.py
```

Single-threaded execution is recommended: on heavily loaded machines, multithreaded gradient-boosting fitting stalls under CPU contention (observed 12 November 2026). The random forest uses 100 trees at `n_jobs=1`. All results are deterministic (fixed seeds); no fitted-model objects are stored — rerunning the scripts reproduces the fitted models exactly. Full run log: `results-n2.md`.

## Files

- `code/analysis_n2.py` — full run: head-to-head audit, three-model comparison, decision curves, figures.
- `code/supplement_rf.py` — random-forest operating characteristics, subgroup calibration, decision-curve crossover thresholds.
- `results-n2.md` — complete run log with all numbers.
- `requirements.txt` — Python dependencies.

## AI assistance disclosure

The manuscript was drafted with AI assistance (a language model used to structure text and organise the computed results). The author reviewed, corrected, and takes responsibility for all content, numbers, and claims. AI is not listed as an author. All numbers were computed by the code in this repository from the named public dataset, not generated by the language model.

## Licence

Code released for replication of the manuscript's results. Data remain CDC NHANES public-use files under their own terms.
