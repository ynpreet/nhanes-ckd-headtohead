#!/usr/bin/env python3
"""N2 supplement: random-forest operating characteristics, subgroup calibration,
and exact decision-curve crossover thresholds. Reuses analysis_n2 builders.
Single-threaded (VM is heavily contended; OpenMP thrashes otherwise)."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_n2 import (build, evaluate, subgroup_cal, net_benefit, DEV_FILES, VAL_FILES)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "..", "nhanes-ckd-risk", "submission-package", "rerun-data", "dev")
VAL = os.path.join(ROOT, "..", "nhanes-ckd-risk", "submission-package", "rerun-data", "val")
OUT = os.path.join(ROOT, "figure")
FEATS = ["RIDAGEYR", "female", "bmi", "sbp", "di", "hba1c", "diab_hist", "htn_hist"]

dev = build(DEV, DEV_FILES, ["BPXSY1", "BPXSY2", "BPXSY3"], ["BPXDI1", "BPXDI2", "BPXDI3"],
            ["DEMO_J", "BPX_J", "BMX_J", "BIOPRO_J", "ALB_CR_J", "GHB_J", "DIQ_J", "BPQ_J"])
outcome = dev[dev[["eGFR2021", "UACR"]].notna().all(axis=1)].copy()
mc = outcome.dropna(subset=FEATS + ["w"]).copy()
X, y, w = mc[FEATS].values, mc["y21"].values, mc["w"].values
Xtr, Xte, ytr, yte, wtr, wte, idx_tr, idx_te = train_test_split(
    X, y, w, mc.index, test_size=0.30, random_state=42, stratify=y)
te_df = mc.loc[idx_te]
val = build(VAL, VAL_FILES, ["BPXOSY1", "BPXOSY2", "BPXOSY3"], ["BPXODI1", "BPXODI2", "BPXODI3"],
            ["DEMO_L", "BPXO_L", "BMX_L", "BIOPRO_L", "ALB_CR_L", "GHB_L", "DIQ_L", "BPQ_L"])
v_out = val[val[["eGFR2021", "UACR"]].notna().all(axis=1)].copy()
v_case = v_out.dropna(subset=FEATS + ["w"]).copy()

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=1, random_state=42)
rf.fit(Xtr, ytr, sample_weight=wtr)
p_te = rf.predict_proba(Xte)[:, 1]
rf_full = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=1, random_state=42)
rf_full.fit(X, y, sample_weight=w)
p_ho = rf_full.predict_proba(v_case[FEATS].values)[:, 1]
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
lr.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
p_lr_te = lr.predict_proba(Xte)[:, 1]
lr_full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
lr_full.fit(X, y, logisticregression__sample_weight=w)
p_lr_ho = lr_full.predict_proba(v_case[FEATS].values)[:, 1]

for label, yv, pv in [("test", yte, p_te), ("heldout", v_case["y21"].values, p_ho)]:
    for pt in [0.10, 0.15, 0.20]:
        flag = pv >= pt
        tp = int(((yv == 1) & flag).sum()); fn = int(((yv == 1) & ~flag).sum())
        fp = int(((yv == 0) & flag).sum()); tn = int(((yv == 0) & ~flag).sum())
        print(f"RF {label} @ {pt:.0%}: sens {tp/(tp+fn):.3f}, spec {tn/(tn+fp):.3f}, "
              f"PPV {tp/max(tp+fp,1):.3f}, NPV {tn/max(tn+fn,1):.3f}, flagged {int(flag.sum())}/{len(yv)} ({flag.mean()*100:.1f}%)")

sub = subgroup_cal(te_df, yte, p_te)
sub.to_csv(os.path.join(OUT, "subgroup-calibration-test-random_forest.csv"), index=False)
print("\nRF subgroup calibration TEST:\n" + sub.to_string(index=False))
subv = subgroup_cal(v_case, v_case["y21"].values, p_ho)
subv.to_csv(os.path.join(OUT, "subgroup-calibration-heldout-random_forest.csv"), index=False)
print("\nRF subgroup calibration HELD-OUT:\n" + subv.to_string(index=False))

# exact DCA crossover: lowest threshold where model NB exceeds treat-all NB, fine grid
grid = np.arange(0.01, 0.51, 0.01)
for label, yv, plr, prf in [("test", yte, p_lr_te, p_te),
                            ("heldout", v_case["y21"].values, p_lr_ho, p_ho)]:
    for nm, pv in [("logistic", plr), ("random_forest", prf)]:
        cross = None
        for pt in grid:
            nb, nb_all, _, _ = net_benefit(yv, pv, pt)
            if nb > nb_all and nb > 0:
                cross = pt; break
        # highest threshold where model still beats treat-all (should be 0.50)
        last = max([pt for pt in grid if net_benefit(yv, pv, pt)[0] > net_benefit(yv, pv, pt)[1]], default=None)
        print(f"DCA {label} {nm}: beats treat-all from pt={cross:.2f} through pt={last:.2f}")
print("SUPPLEMENT DONE")
