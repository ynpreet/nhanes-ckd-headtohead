#!/usr/bin/env python3
"""
Paper N2 — Race-free versus race-based eGFR: head-to-head reclassification,
triage-model and net-benefit comparison in NHANES 2017-2018, held-out
validation in NHANES 2021-2023.

Successor to Paper 2 (JBI desk-rejected 9 Nov 2026 as "not of sufficient
importance and interest"). New contributions vs Paper 2:
  (a) HEAD-TO-HEAD equation comparison: CKD-EPI 2021 (race-free) vs CKD-EPI
      2009 (race-based) on the same people — reclassification counts, G-stage
      cross-tabulation, and net reclassification by survey group, plus the
      KDIGO G x A (albuminuria) staging under the 2021 equation.
      Cystatin C: NOT available. BIOPRO_J (2017-2018) and BIOPRO_L
      (2021-2023) contain no cystatin-C column (verified by column listing
      12 Nov 2026); candidate separate files CYS_J / CYST_J / LBXCYS_J
      returned 404 in the Paper 2 run. No cystatin equation is computed and
      no cystatin value is invented or borrowed.
  (b) TRIAGE-MODEL COMPARISON on routine-only predictors (leakage-free:
      creatinine and UACR never predictors): logistic regression vs
      gradient boosting (HistGradientBoostingClassifier) vs random forest,
      same 70/30 split (seed 42), paired-bootstrap AUROC differences,
      calibration curves and Brier scores by subgroup.
  (c) DECISION-CURVE / net-benefit analysis at triage thresholds 5%-50%,
      model vs "test everyone" vs "test no one", on the test set and held-out.
  (d) OUTCOME-DEFINITION SENSITIVITY: each model evaluated against both the
      2021-defined and the 2009-defined proxy, showing the equation choice
      moves model performance, not just prevalence.

Data: cached CDC NHANES public-use XPT files (dev 2017-2018, val 2021-2023).
No downloads are attempted if files are present; download fallback kept.
Run: python3 analysis_n2.py [--dev-data DIR] [--val-data DIR] [--out DIR]
"""
import argparse, os, urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEV_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
VAL_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"
DEV_FILES = ["DEMO_J", "BPX_J", "BMX_J", "BIOPRO_J", "ALB_CR_J", "GHB_J", "DIQ_J", "BPQ_J"]
VAL_FILES = ["DEMO_L", "BPXO_L", "BMX_L", "BIOPRO_L", "ALB_CR_L", "GHB_L", "DIQ_L", "BPQ_L"]
RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other / multi-race"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def egfr2021(scr, age, female):
    k = np.where(female, 0.7, 0.9); alpha = np.where(female, -0.241, -0.302)
    return 142 * np.minimum(scr / k, 1) ** alpha * np.maximum(scr / k, 1) ** -1.200 \
        * 0.9938 ** age * np.where(female, 1.012, 1.0)

def egfr2009(scr, age, female, black):
    k = np.where(female, 0.7, 0.9); alpha = np.where(female, -0.329, -0.411)
    return 141 * np.minimum(scr / k, 1) ** alpha * np.maximum(scr / k, 1) ** -1.209 \
        * 0.993 ** age * np.where(female, 1.018, 1.0) * np.where(black, 1.159, 1.0)

def rd(data_dir, f):
    return pd.read_sas(os.path.join(data_dir, f + ".XPT"), format="xport")

def mean_bp(df, sy_cols, di_cols):
    sy = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in sy_cols}).mean(axis=1)
    di = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in di_cols}).mean(axis=1)
    return sy, di

def g_stage(e):
    return pd.cut(e, [-np.inf, 15, 30, 45, 60, 90, np.inf],
                  labels=["G5 <15", "G4 15-29", "G3b 30-44", "G3a 45-59", "G2 60-89", "G1 >=90"])

def a_stage(u):
    return pd.cut(u, [-np.inf, 30, 300, np.inf], labels=["A1 <30", "A2 30-300", "A3 >300"])

def build(data_dir, files, bp_sy, bp_di, names):
    df = rd(data_dir, names[0])
    for f in names[1:]:
        df = df.merge(rd(data_dir, f), on="SEQN", how="left")
    ad = df[df["RIDAGEYR"] >= 20].copy()
    scr = pd.to_numeric(ad["LBXSCR"], errors="coerce")
    female = ad["RIAGENDR"] == 2
    ad["eGFR2021"] = egfr2021(scr, ad["RIDAGEYR"], female)
    ad["eGFR2009"] = egfr2009(scr, ad["RIDAGEYR"], female, ad["RIDRETH3"] == 4)
    ad["UACR"] = pd.to_numeric(ad["URDACT"], errors="coerce")
    ad["y21"] = ((ad["eGFR2021"] < 60) | (ad["UACR"] >= 30)).astype(float)
    ad["y09"] = ((ad["eGFR2009"] < 60) | (ad["UACR"] >= 30)).astype(float)
    ad["g21"] = g_stage(ad["eGFR2021"]); ad["g09"] = g_stage(ad["eGFR2009"])
    ad["a"] = a_stage(ad["UACR"])
    sy, di = mean_bp(ad, bp_sy, bp_di)
    ad["sbp"], ad["di"] = sy, di
    ad["female"] = female.astype(int)
    ad["bmi"] = pd.to_numeric(ad["BMXBMI"], errors="coerce")
    ad["hba1c"] = pd.to_numeric(ad["LBXGH"], errors="coerce")
    ad["diab_hist"] = (pd.to_numeric(ad["DIQ010"], errors="coerce") == 1).astype(float)
    ad["diab_hist"] = ad["diab_hist"].where(pd.to_numeric(ad["DIQ010"], errors="coerce").isin([1, 2]))
    ad["htn_hist"] = (pd.to_numeric(ad["BPQ020"], errors="coerce") == 1).astype(float)
    ad["htn_hist"] = ad["htn_hist"].where(pd.to_numeric(ad["BPQ020"], errors="coerce").isin([1, 2]))
    ad["agegrp"] = pd.cut(ad["RIDAGEYR"], [19, 44, 64, 200], labels=["20-44", "45-64", "65+"])
    ad["race_label"] = ad["RIDRETH3"].map(RACE)
    ad["w"] = pd.to_numeric(ad["WTMEC2YR"], errors="coerce")
    return ad

def taylor_ci(df, ycol="y21"):
    d = df.dropna(subset=["w", "SDMVSTRA", "SDMVPSU", ycol]).copy()
    p = (d["w"] * d[ycol]).sum() / d["w"].sum()
    d["lin"] = d["w"] * (d[ycol] - p)
    var = 0.0
    for _, s in d.groupby("SDMVSTRA"):
        psu_tot = s.groupby("SDMVPSU")["lin"].sum()
        n = len(psu_tot)
        if n > 1:
            var += n / (n - 1) * ((psu_tot - psu_tot.mean()) ** 2).sum()
    se = np.sqrt(var) / d["w"].sum()
    return p, p - 1.96 * se, p + 1.96 * se

def evaluate(y, p):
    out = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
           "brier": brier_score_loss(y, p)}
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    try:
        cal = LogisticRegression(max_iter=2000, C=1e6)
        cal.fit(lp.reshape(-1, 1), y)
        out["cal_intercept"], out["cal_slope"] = float(cal.intercept_[0]), float(cal.coef_[0][0])
    except Exception:
        out["cal_intercept"], out["cal_slope"] = float("nan"), float("nan")
    order = np.argsort(p); yb, pb = y[order], p[order]
    ece = 0.0
    for chunk_y, chunk_p in zip(np.array_split(yb, 10), np.array_split(pb, 10)):
        ece += abs(chunk_y.mean() - chunk_p.mean()) * len(chunk_y) / len(y)
    out["ece"] = float(ece)
    return out

def subgroup_cal(df, y, p):
    rows = []
    for col, label in [("female", "sex"), ("agegrp", "age group"), ("race_label", "survey group")]:
        for val in sorted(df[col].dropna().unique(), key=str):
            mask = (df[col] == val).values
            disp = "Female" if (col == "female" and val == 1) else "Male" if col == "female" else str(val)
            if mask.sum() >= 30:
                rows.append({"group": label, "value": disp, "n": int(mask.sum()),
                             "observed_pct": round(float(y[mask].mean() * 100), 1),
                             "mean_pred_pct": round(float(p[mask].mean() * 100), 1),
                             "gap_pp": round(float((y[mask].mean() - p[mask].mean()) * 100), 1)})
    return pd.DataFrame(rows)

def net_benefit(y, p, pt):
    """Net benefit at threshold probability pt; also treat-all NB."""
    n = len(y)
    pred_pos = p >= pt
    tp = int(((y == 1) & pred_pos).sum()); fp = int(((y == 0) & pred_pos).sum())
    nb = (tp - fp * pt / (1 - pt)) / n
    prev = y.mean()
    nb_all = prev - (1 - prev) * pt / (1 - pt)
    return nb, nb_all, tp, fp

def dca_table(y, p, thresholds):
    rows = []
    for pt in thresholds:
        nb, nb_all, tp, fp = net_benefit(y, p, pt)
        # reduction in unnecessary tests per 100 vs test-everyone, at same threshold
        red = (nb - nb_all) / (pt / (1 - pt)) * 100 if pt < 1 else np.nan
        rows.append({"threshold": pt, "net_benefit": round(nb, 4), "treat_all_nb": round(nb_all, 4),
                     "treat_none_nb": 0.0, "tp": tp, "fp": fp,
                     "n_flagged": tp + fp, "avoided_tests_per_100": round(red, 1)})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-data", default=os.path.join(ROOT, "..", "nhanes-ckd-risk",
                    "submission-package", "rerun-data", "dev"))
    ap.add_argument("--val-data", default=os.path.join(ROOT, "..", "nhanes-ckd-risk",
                    "submission-package", "rerun-data", "val"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figure"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results-n2.md"))
    a = ap.parse_args()
    for files, base, d in [(DEV_FILES, DEV_BASE, a.dev_data), (VAL_FILES, VAL_BASE, a.val_data)]:
        os.makedirs(d, exist_ok=True)
        for f in files:
            p = os.path.join(d, f + ".XPT")
            if not (os.path.exists(p) and os.path.getsize(p) > 1000):
                print(f"downloading {f} ...")
                urllib.request.urlretrieve(f"{base}/{f}.XPT", p)
    os.makedirs(a.out, exist_ok=True)
    log = []

    def pr(s=""):
        print(s); log.append(s)

    dev = build(a.dev_data, DEV_FILES, ["BPXSY1", "BPXSY2", "BPXSY3"], ["BPXDI1", "BPXDI2", "BPXDI3"],
                ["DEMO_J", "BPX_J", "BMX_J", "BIOPRO_J", "ALB_CR_J", "GHB_J", "DIQ_J", "BPQ_J"])
    outcome = dev[dev[["eGFR2021", "UACR"]].notna().all(axis=1)].copy()
    pr(f"DEV adults 20+: {len(dev)}; outcome complete case: {len(outcome)}")
    pr(f"DEV CKD proxy 2021: {int(outcome['y21'].sum())} ({outcome['y21'].mean()*100:.1f}% unweighted)")
    p21, lo21, hi21 = taylor_ci(outcome, "y21")
    p09, lo09, hi09 = taylor_ci(outcome, "y09")
    pr(f"DEV weighted 2021 proxy: {p21*100:.1f}% (95% CI {lo21*100:.1f}-{hi21*100:.1f}); "
       f"2009 proxy: {int(outcome['y09'].sum())} ({outcome['y09'].mean()*100:.1f}% unw; "
       f"{p09*100:.1f}% weighted (95% CI {lo09*100:.1f}-{hi09*100:.1f})")

    # ---------- (a) HEAD-TO-HEAD ----------
    reclass = outcome[outcome["y21"] != outcome["y09"]]
    newly = int(((outcome["y21"] == 1) & (outcome["y09"] == 0)).sum())
    lost = int(((outcome["y21"] == 0) & (outcome["y09"] == 1)).sum())
    pr(f"\nRECLASS 2021 vs 2009: {len(reclass)} changed status; newly CKD under 2021: {newly}; "
       f"no longer CKD under 2021: {lost}")
    # by group, with direction and net reclassification per 1000
    rows = []
    for grp, sub in outcome.groupby("race_label", observed=True):
        n = len(sub); new = int(((sub["y21"] == 1) & (sub["y09"] == 0)).sum())
        los = int(((sub["y21"] == 0) & (sub["y09"] == 1)).sum())
        rows.append({"survey_group": grp, "n": n, "reclassified": new + los,
                     "newly_2021": new, "lost_2021": los, "net": new - los,
                     "net_per_1000": round((new - los) / n * 1000, 1),
                     "reclass_pct": round((new + los) / n * 100, 2)})
    reclass_by_group = pd.DataFrame(rows).sort_values("reclassified", ascending=False)
    pr("\nReclassification by survey group:\n" + reclass_by_group.to_string(index=False))
    reclass_by_group.to_csv(os.path.join(a.out, "reclassification-by-group.csv"), index=False)
    # eGFR difference by group
    outcome["degfr"] = outcome["eGFR2021"] - outcome["eGFR2009"]
    d = outcome.groupby("race_label", observed=True)["degfr"].agg(["mean", "median", "count"])
    pr("\neGFR2021 - eGFR2009 by group:\n" + d.round(2).to_string())
    d.to_csv(os.path.join(a.out, "egfr-difference-by-group.csv"))
    # G-stage cross-tab (counts) and how many change G stage at all
    gtab = pd.crosstab(outcome["g09"], outcome["g21"])
    pr("\nG-stage cross-tab (rows=2009, cols=2021):\n" + gtab.to_string())
    gtab.to_csv(os.path.join(a.out, "gstage-crosstab.csv"))
    g_change = int((outcome["g09"] != outcome["g21"]).sum())
    pr(f"Changed G stage (any direction): {g_change} ({g_change/len(outcome)*100:.1f}%)")
    # KDIGO G x A heatmap under 2021
    ga = pd.crosstab(outcome["g21"], outcome["a"])
    pr("\nKDIGO G x A under 2021 (counts):\n" + ga.to_string())
    ga.to_csv(os.path.join(a.out, "kdigo-gxa-2021.csv"))
    # albuminuria-only share under each equation
    for col, y in [("2021", "y21"), ("2009", "y09")]:
        alb_only = int(((outcome[y] == 1) & (outcome["eGFR2021" if col == "2021" else "eGFR2009"] >= 60)).sum())
        pr(f"Albuminuria-only proxies under {col}: {alb_only} of {int(outcome[y].sum())} "
           f"({alb_only/int(outcome[y].sum())*100:.1f}%)")
    # weighted reclassification share
    w_reclass = outcome.loc[outcome["y21"] != outcome["y09"], "w"].sum() / outcome["w"].sum() * 100
    pr(f"Weighted share reclassified: {w_reclass:.2f}%")

    # ---------- (b) TRIAGE-MODEL COMPARISON ----------
    feats = ["RIDAGEYR", "female", "bmi", "sbp", "di", "hba1c", "diab_hist", "htn_hist"]
    assert "LBXSCR" not in feats and "URDACT" not in feats
    model_case = outcome.dropna(subset=feats + ["w"]).copy()
    pr(f"\nMODEL complete case: {len(model_case)}; CKD proxy 2021: {int(model_case['y21'].sum())} "
       f"({model_case['y21'].mean()*100:.1f}% unw)")
    X, y21v, y09v, w = model_case[feats].values, model_case["y21"].values, model_case["y09"].values, model_case["w"].values
    Xtr, Xte, ytr, yte, y09tr, y09te, wtr, wte, idx_tr, idx_te = train_test_split(
        X, y21v, y09v, w, model_case.index, test_size=0.30, random_state=42, stratify=y21v)
    te_df = model_case.loc[idx_te]
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                                            max_leaf_nodes=31, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                                n_jobs=1, random_state=42),
    }
    test_p, test_ev = {}, {}
    for name, m in models.items():
        if name == "logistic":
            m.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
        else:
            m.fit(Xtr, ytr, sample_weight=wtr)
        p = m.predict_proba(Xte)[:, 1]
        test_p[name] = p
        ev = evaluate(yte, p); ev["weighted_brier"] = float(np.average((yte - p) ** 2, weights=wte))
        test_ev[name] = ev
        pr(f"\n=== TEST {name} (n={len(yte)}, positives={int(yte.sum())}) ===")
        for k, v in ev.items():
            pr(f"  {k}: {v:.4f}")
    # age-only baseline (logistic)
    age_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    age_pipe.fit(Xtr[:, [0]], ytr, logisticregression__sample_weight=wtr)
    p_age = age_pipe.predict_proba(Xte[:, [0]])[:, 1]
    ev_age = evaluate(yte, p_age)
    pr(f"\n=== TEST age-only ===\n" + "\n".join(f"  {k}: {v:.4f}" for k, v in ev_age.items()))
    # paired bootstrap AUROC differences vs logistic (1000 resamples)
    rng = np.random.default_rng(42); n = len(yte)
    names = list(models.keys())
    boots = {nm: [] for nm in names}; boots["age-only"] = []
    diffs = {"gbm-lr": [], "rf-lr": [], "gbm-rf": [], "lr-age": []}
    for _ in range(1000):
        ii = rng.integers(0, n, n)
        if len(np.unique(yte[ii])) < 2:
            continue
        au = {nm: roc_auc_score(yte[ii], test_p[nm][ii]) for nm in names}
        au["age-only"] = roc_auc_score(yte[ii], p_age[ii])
        for nm in names:
            boots[nm].append(au[nm])
        boots["age-only"].append(au["age-only"])
        diffs["gbm-lr"].append(au["gradient_boosting"] - au["logistic"])
        diffs["rf-lr"].append(au["random_forest"] - au["logistic"])
        diffs["gbm-rf"].append(au["gradient_boosting"] - au["random_forest"])
        diffs["lr-age"].append(au["logistic"] - au["age-only"])
    for nm in names + ["age-only"]:
        pr(f"Bootstrap AUROC 95% CI {nm}: {np.percentile(boots[nm], 2.5):.3f}-{np.percentile(boots[nm], 97.5):.3f}")
    for k, v in diffs.items():
        pr(f"Bootstrap AUROC difference 95% CI {k}: {np.percentile(v, 2.5):.3f} to {np.percentile(v, 97.5):.3f} "
           f"(mean {np.mean(v):.3f})")
    # outcome-definition sensitivity: each test model vs 2009-defined outcome
    pr("\n=== Outcome-definition sensitivity (test set, outcome = 2009 proxy) ===")
    for nm in names:
        ev09 = evaluate(y09te, test_p[nm])
        pr(f"  {nm}: AUROC {ev09['auroc']:.3f} (vs 2021 outcome {test_ev[nm]['auroc']:.3f}), "
           f"AUPRC {ev09['auprc']:.3f}, Brier {ev09['brier']:.3f}")

    # ---------- (c) DECISION CURVES ----------
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    best = "gradient_boosting" if test_ev["gradient_boosting"]["auroc"] >= test_ev["logistic"]["auroc"] else "logistic"
    pr(f"\nBest test AUROC model: {best}")
    dca_rows = []
    for nm in names:
        t = dca_table(yte, test_p[nm], thresholds)
        t["model"] = nm; t["set"] = "test"
        dca_rows.append(t)
        pr(f"\nDCA {nm} (test):\n" + t[["threshold", "net_benefit", "treat_all_nb", "avoided_tests_per_100"]].to_string(index=False))
    dca_all = pd.concat(dca_rows, ignore_index=True)
    dca_all.to_csv(os.path.join(a.out, "decision-curve-test.csv"), index=False)

    # ---------- held-out validation ----------
    val = build(a.val_data, VAL_FILES, ["BPXOSY1", "BPXOSY2", "BPXOSY3"], ["BPXODI1", "BPXODI2", "BPXODI3"],
                ["DEMO_L", "BPXO_L", "BMX_L", "BIOPRO_L", "ALB_CR_L", "GHB_L", "DIQ_L", "BPQ_L"])
    v_out = val[val[["eGFR2021", "UACR"]].notna().all(axis=1)].copy()
    v_case = v_out.dropna(subset=feats + ["w"]).copy()
    pv21, lov, hiv = taylor_ci(v_out, "y21")
    pr(f"\nVAL adults 20+: {len(val)}; outcome complete case: {len(v_out)}; "
       f"CKD proxy 2021: {int(v_out['y21'].sum())} ({v_out['y21'].mean()*100:.1f}% unw; "
       f"{pv21*100:.1f}% weighted, 95% CI {lov*100:.1f}-{hiv*100:.1f})")
    pr(f"VAL model complete case: {len(v_case)}; proxy {int(v_case['y21'].sum())} "
       f"({v_case['y21'].mean()*100:.1f}% unw)")
    # held-out reclassification
    v_new = int(((v_out["y21"] == 1) & (v_out["y09"] == 0)).sum())
    v_los = int(((v_out["y21"] == 0) & (v_out["y09"] == 1)).sum())
    pr(f"VAL reclass 2021 vs 2009: {v_new + v_los} (newly {v_new}, lost {v_los})")
    v_rows = []
    for grp, sub in v_out.groupby("race_label", observed=True):
        nn = len(sub); nw = int(((sub["y21"] == 1) & (sub["y09"] == 0)).sum())
        lw = int(((sub["y21"] == 0) & (sub["y09"] == 1)).sum())
        v_rows.append({"survey_group": grp, "n": nn, "reclassified": nw + lw, "newly_2021": nw,
                       "lost_2021": lw, "net": nw - lw, "net_per_1000": round((nw - lw) / nn * 1000, 1)})
    pd.DataFrame(v_rows).to_csv(os.path.join(a.out, "reclassification-by-group-heldout.csv"), index=False)
    pr("\nVAL reclassification by group:\n" + pd.DataFrame(v_rows).to_string(index=False))
    held_p, held_ev, held_dca = {}, {}, []
    for name, m in models.items():
        if name == "logistic":
            m.fit(X, y21v, logisticregression__sample_weight=w)
        else:
            m.fit(X, y21v, sample_weight=w)
        pv = m.predict_proba(v_case[feats].values)[:, 1]
        held_p[name] = pv
        ev = evaluate(v_case["y21"].values, pv)
        ev["weighted_brier"] = float(np.average((v_case["y21"].values - pv) ** 2, weights=v_case["w"].values))
        held_ev[name] = ev
        pr(f"\n=== HELD-OUT {name} (n={len(v_case)}) ===")
        for k, v in ev.items():
            pr(f"  {k}: {v:.4f}")
        t = dca_table(v_case["y21"].values, pv, thresholds)
        t["model"] = name; t["set"] = "heldout"
        held_dca.append(t)
        pr(f"DCA {name} (held-out):\n" + t[["threshold", "net_benefit", "treat_all_nb", "avoided_tests_per_100"]].to_string(index=False))
    pd.concat(held_dca, ignore_index=True).to_csv(os.path.join(a.out, "decision-curve-heldout.csv"), index=False)
    # subgroup calibration for best model + logistic, test and held-out
    for nm in ["logistic", best]:
        sub = subgroup_cal(te_df, yte, test_p[nm])
        sub.to_csv(os.path.join(a.out, f"subgroup-calibration-test-{nm}.csv"), index=False)
        pr(f"\nSubgroup calibration TEST {nm}:\n" + sub.to_string(index=False))
        subv = subgroup_cal(v_case, v_case["y21"].values, held_p[nm])
        subv.to_csv(os.path.join(a.out, f"subgroup-calibration-heldout-{nm}.csv"), index=False)
        pr(f"\nSubgroup calibration HELD-OUT {nm}:\n" + subv.to_string(index=False))
    # sensitivity/specificity at 10/15/20% thresholds, best model, test + held-out
    for label, yv, pv in [("test", yte, test_p[best]), ("heldout", v_case["y21"].values, held_p[best])]:
        for pt in [0.10, 0.15, 0.20]:
            flag = pv >= pt
            tp = int(((yv == 1) & flag).sum()); fn = int(((yv == 1) & ~flag).sum())
            fp = int(((yv == 0) & flag).sum()); tn = int(((yv == 0) & ~flag).sum())
            pr(f"{label} {best} @ {pt:.0%}: sens {tp/(tp+fn):.3f}, spec {tn/(tn+fp):.3f}, "
               f"PPV {tp/max(tp+fp,1):.3f}, NPV {tn/max(tn+fn,1):.3f}, flagged {int(flag.sum())}/{len(yv)} "
               f"({flag.mean()*100:.1f}%)")

    # ---------- FIGURES ----------
    # Fig 1: reclassification by group (newly vs lost)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    g = reclass_by_group.set_index("survey_group")
    x = np.arange(len(g)); wdt = 0.38
    ax.bar(x - wdt/2, g["newly_2021"], wdt, label="Newly CKD-proxy under 2021", color="#2a6f97")
    ax.bar(x + wdt/2, g["lost_2021"], wdt, label="No longer CKD-proxy under 2021", color="#e07a5f")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("Non-Hispanic ", "NH ") for s in g.index], rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Participants reclassified (n)")
    ax.set_title("Reclassification 2021 vs 2009 CKD-EPI, NHANES 2017-2018 (n=4,827)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig1-reclassification.png"), dpi=200); plt.close(fig)

    # Fig 2: calibration curves, test set (3 models) + held-out best
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
    for ax, (yv, ps, ttl) in zip(axes, [(yte, test_p, "Development test set (n=1,325)"),
                                         (v_case["y21"].values, held_p, "Held-out 2021-2023")]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        for nm, col in [("logistic", "#2a6f97"), ("gradient_boosting", "#e07a5f"), ("random_forest", "#81b29a")]:
            p = ps[nm]
            qs = np.quantile(p, np.linspace(0, 1, 11))
            xs, ys = [], []
            for i in range(10):
                m = (p >= qs[i]) & (p <= qs[i+1] if i == 9 else p < qs[i+1])
                if m.sum() > 0:
                    xs.append(p[m].mean()); ys.append(yv[m].mean())
            ax.plot(xs, ys, "o-", ms=4, color=col, label=nm.replace("_", " "))
        ax.set_xlabel("Mean predicted probability"); ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("Observed proportion")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig2-calibration.png"), dpi=200); plt.close(fig)

    # Fig 3: decision curves, test + held-out (best model + logistic + treat all/none)
    grid = np.arange(0.02, 0.51, 0.01)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
    for ax, (yv, ps, ttl) in zip(axes, [(yte, test_p, "Development test set"),
                                         (v_case["y21"].values, held_p, "Held-out 2021-2023")]):
        prev = yv.mean()
        nb_all = [prev - (1 - prev) * pt / (1 - pt) for pt in grid]
        ax.plot(grid, nb_all, color="gray", ls="--", label="Test everyone")
        ax.plot(grid, np.zeros_like(grid), color="black", ls=":", label="Test no one")
        for nm, col in [("logistic", "#2a6f97"), ("gradient_boosting", "#e07a5f"), ("random_forest", "#81b29a")]:
            nbs = [net_benefit(yv, ps[nm], pt)[0] for pt in grid]
            ax.plot(grid, nbs, color=col, label=nm.replace("_", " "))
        ax.set_xlabel("Threshold probability"); ax.set_xlim(0.02, 0.50)
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("Net benefit")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3-decision-curves.png"), dpi=200); plt.close(fig)

    # summary table CSV
    summ = pd.DataFrame([
        {"model": nm, "set": "test", **{k: round(v, 4) for k, v in test_ev[nm].items()}} for nm in names
    ] + [
        {"model": nm, "set": "heldout", **{k: round(v, 4) for k, v in held_ev[nm].items()}} for nm in names
    ])
    summ.to_csv(os.path.join(a.out, "model-comparison.csv"), index=False)
    pr("\n=== MODEL COMPARISON ===\n" + summ.to_string(index=False))
    with open(a.results, "w") as fh:
        fh.write("# Paper N2 — full run log (real data, NHANES 2017-2018 dev / 2021-2023 held-out)\n\n")
        fh.write("\n".join(log) + "\n")
    pr(f"\nWrote results log to {a.results}; CSVs + figures to {a.out}")

if __name__ == "__main__":
    main()
