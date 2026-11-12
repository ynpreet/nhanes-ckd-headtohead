# Paper N2 — full run log (real data, NHANES 2017-2018 dev / 2021-2023 held-out)

DEV adults 20+: 5569; outcome complete case: 4827
DEV CKD proxy 2021: 923 (19.1% unweighted)
DEV weighted 2021 proxy: 14.0% (95% CI 12.7-15.4); 2009 proxy: 941 (19.5% unw; 15.0% weighted (95% CI 13.3-16.6)

RECLASS 2021 vs 2009: 88 changed status; newly CKD under 2021: 35; no longer CKD under 2021: 53

Reclassification by survey group:
      survey_group    n  reclassified  newly_2021  lost_2021  net  net_per_1000  reclass_pct
Non-Hispanic White 1684            37           0         37  -37         -22.0         2.20
Non-Hispanic Black 1084            35          35          0   35          32.3         3.23
Non-Hispanic Asian  679             6           0          6   -6          -8.8         0.88
Other / multi-race  246             5           0          5   -5         -20.3         2.03
    Other Hispanic  464             3           0          3   -3          -6.5         0.65
  Mexican American  670             2           0          2   -2          -3.0         0.30

eGFR2021 - eGFR2009 by group:
                    mean  median  count
race_label                             
Mexican American    2.88    3.14    670
Non-Hispanic Asian  3.28    3.52    679
Non-Hispanic Black -9.76   -9.30   1084
Non-Hispanic White  3.63    3.74   1684
Other / multi-race  3.44    3.50    246
Other Hispanic      3.32    3.50    464

G-stage cross-tab (rows=2009, cols=2021):
g21        G5 <15  G4 15-29  G3b 30-44  G3a 45-59  G2 60-89  G1 >=90
g09                                                                 
G5 <15          9         1          0          0         0        0
G4 15-29        0        25         10          0         0        0
G3b 30-44       0         6         80         22         0        0
G3a 45-59       0         0         12        203        69        0
G2 60-89        0         0          0         43      1266      273
G1 >=90         0         0          0          0       134     2674
Changed G stage (any direction): 570 (11.8%)

KDIGO G x A under 2021 (counts):
a          A1 <30  A2 30-300  A3 >300
g21                                  
G5 <15          0          2        7
G4 15-29        6         12       14
G3b 30-44      51         35       16
G3a 45-59     196         57       15
G2 60-89     1246        189       34
G1 >=90      2659        255       33
Albuminuria-only proxies under 2021: 512 of 923 (55.5%)
Albuminuria-only proxies under 2009: 504 of 941 (53.6%)
Weighted share reclassified: 1.44%

MODEL complete case: 4415; CKD proxy 2021: 829 (18.8% unw)

=== TEST logistic (n=1325, positives=249) ===
  auroc: 0.7665
  auprc: 0.4616
  brier: 0.1280
  cal_intercept: -0.0939
  cal_slope: 0.8232
  ece: 0.0267
  weighted_brier: 0.1048

=== TEST gradient_boosting (n=1325, positives=249) ===
  auroc: 0.7456
  auprc: 0.4093
  brier: 0.1411
  cal_intercept: -0.5032
  cal_slope: 0.4516
  ece: 0.0762
  weighted_brier: 0.1132

=== TEST random_forest (n=1325, positives=249) ===
  auroc: 0.7687
  auprc: 0.4391
  brier: 0.1277
  cal_intercept: -0.1759
  cal_slope: 0.7875
  ece: 0.0248
  weighted_brier: 0.1032

=== TEST age-only ===
  auroc: 0.7169
  auprc: 0.3679
  brier: 0.1381
  cal_intercept: 0.0763
  cal_slope: 0.8716
  ece: 0.0432
Bootstrap AUROC 95% CI logistic: 0.730-0.801
Bootstrap AUROC 95% CI gradient_boosting: 0.708-0.778
Bootstrap AUROC 95% CI random_forest: 0.732-0.802
Bootstrap AUROC 95% CI age-only: 0.679-0.753
Bootstrap AUROC difference 95% CI gbm-lr: -0.041 to -0.001 (mean -0.021)
Bootstrap AUROC difference 95% CI rf-lr: -0.017 to 0.022 (mean 0.002)
Bootstrap AUROC difference 95% CI gbm-rf: -0.041 to -0.005 (mean -0.023)
Bootstrap AUROC difference 95% CI lr-age: 0.027 to 0.072 (mean 0.049)

=== Outcome-definition sensitivity (test set, outcome = 2009 proxy) ===
  logistic: AUROC 0.771 (vs 2021 outcome 0.766), AUPRC 0.464, Brier 0.130
  gradient_boosting: AUROC 0.757 (vs 2021 outcome 0.746), AUPRC 0.431, Brier 0.140
  random_forest: AUROC 0.777 (vs 2021 outcome 0.769), AUPRC 0.458, Brier 0.128

Best test AUROC model: logistic

DCA logistic (test):
 threshold  net_benefit  treat_all_nb  avoided_tests_per_100
      0.05       0.1412        0.1452                   -7.5
      0.10       0.1149        0.0977                   15.5
      0.15       0.0929        0.0446                   27.3
      0.20       0.0758       -0.0151                   36.4
      0.25       0.0576       -0.0828                   42.1
      0.30       0.0470       -0.1601                   48.3
      0.40       0.0226       -0.3535                   56.4
      0.50       0.0098       -0.6242                   63.4

DCA gradient_boosting (test):
 threshold  net_benefit  treat_all_nb  avoided_tests_per_100
      0.05       0.1263        0.1452                  -35.8
      0.10       0.1013        0.0977                    3.2
      0.15       0.0831        0.0446                   21.8
      0.20       0.0694       -0.0151                   33.8
      0.25       0.0556       -0.0828                   41.5
      0.30       0.0427       -0.1601                   47.3
      0.40       0.0186       -0.3535                   55.8
      0.50       0.0030       -0.6242                   62.7

DCA random_forest (test):
 threshold  net_benefit  treat_all_nb  avoided_tests_per_100
      0.05       0.1393        0.1452                  -11.2
      0.10       0.1154        0.0977                   15.9
      0.15       0.0965        0.0446                   29.4
      0.20       0.0811       -0.0151                   38.5
      0.25       0.0652       -0.0828                   44.4
      0.30       0.0522       -0.1601                   49.5
      0.40       0.0231       -0.3535                   56.5
      0.50       0.0038       -0.6242                   62.8

VAL adults 20+: 7809; outcome complete case: 5343; CKD proxy 2021: 925 (17.3% unw; 13.9% weighted, 95% CI 12.5-15.4)
VAL model complete case: 4983; proxy 836 (16.8% unw)
VAL reclass 2021 vs 2009: 100 (newly 20, lost 80)

VAL reclassification by group:
      survey_group    n  reclassified  newly_2021  lost_2021  net  net_per_1000
  Mexican American  357             1           0          1   -1          -2.8
Non-Hispanic Asian  292             2           0          2   -2          -6.8
Non-Hispanic Black  616            20          20          0   20          32.5
Non-Hispanic White 3204            68           0         68  -68         -21.2
Other / multi-race  323             5           0          5   -5         -15.5
    Other Hispanic  551             4           0          4   -4          -7.3

=== HELD-OUT logistic (n=4983) ===
  auroc: 0.7891
  auprc: 0.4514
  brier: 0.1158
  cal_intercept: 0.2348
  cal_slope: 1.0716
  ece: 0.0216
  weighted_brier: 0.0975
DCA logistic (held-out):
 threshold  net_benefit  treat_all_nb  avoided_tests_per_100
      0.05       0.1286        0.1240                    8.7
      0.10       0.1027        0.0753                   24.7
      0.15       0.0808        0.0209                   34.0
      0.20       0.0599       -0.0403                   40.1
      0.25       0.0458       -0.1096                   46.6
      0.30       0.0345       -0.1889                   52.1
      0.40       0.0198       -0.3870                   61.0
      0.50       0.0116       -0.6645                   67.6

=== HELD-OUT gradient_boosting (n=4983) ===
  auroc: 0.7699
  auprc: 0.4234
  brier: 0.1218
  cal_intercept: -0.3468
  cal_slope: 0.6271
  ece: 0.0461
  weighted_brier: 0.1032
DCA gradient_boosting (held-out):
 threshold  net_benefit  treat_all_nb  avoided_tests_per_100
      0.05       0.1206        0.1240                   -6.4
      0.10       0.0945        0.0753                   17.3
      0.15       0.0735        0.0209                   29.8
      0.20       0.0602       -0.0403                   40.2
      0.25       0.0437       -0.1096                   46.0
      0.30       0.0346       -0.1889                   52.1
      0.40       0.0163       -0.3870                   60.5
      0.50      -0.0008       -0.6645                   66.4

=== HELD-OUT random_forest (n=4983) ===
  auroc: 0.7938
  auprc: 0.4760
  brier: 0.1128
  cal_intercept: 0.0319
  cal_slope: 0.9878
  ece: 0.0188
  weighted_brier: 0.0963
DCA random_forest (held-out):
 threshold  net_benefit  treat_all_nb  avoided_tests_per_100
      0.05       0.1249        0.1240                    1.8
      0.10       0.1001        0.0753                   22.3
      0.15       0.0849        0.0209                   36.3
      0.20       0.0689       -0.0403                   43.7
      0.25       0.0575       -0.1096                   50.1
      0.30       0.0447       -0.1889                   54.5
      0.40       0.0231       -0.3870                   61.5
      0.50       0.0132       -0.6645                   67.8

Subgroup calibration TEST logistic:
       group              value   n  observed_pct  mean_pred_pct  gap_pp
         sex               Male 651          18.6           16.8     1.8
         sex             Female 674          19.0           17.5     1.5
   age group              20-44 517           8.5            5.2     3.3
   age group              45-64 484          16.3           17.9    -1.6
   age group                65+ 324          38.9           35.2     3.7
survey group   Mexican American 171          14.6           15.1    -0.4
survey group Non-Hispanic Asian 169          16.6           15.0     1.6
survey group Non-Hispanic Black 305          20.7           17.8     2.9
survey group Non-Hispanic White 487          20.9           18.2     2.8
survey group Other / multi-race  75          18.7           16.5     2.2
survey group     Other Hispanic 118          14.4           17.9    -3.5

Subgroup calibration HELD-OUT logistic:
       group              value    n  observed_pct  mean_pred_pct  gap_pp
         sex               Male 2251          17.7           15.0     2.8
         sex             Female 2732          16.0           15.6     0.4
   age group              20-44 1660           5.3            5.2     0.1
   age group              45-64 1714          13.7           15.5    -1.8
   age group                65+ 1609          31.9           25.5     6.4
survey group   Mexican American  333          12.0           14.4    -2.3
survey group Non-Hispanic Asian  263          15.2           14.2     1.0
survey group Non-Hispanic Black  557          23.9           17.7     6.2
survey group Non-Hispanic White 3022          16.6           15.1     1.5
survey group Other / multi-race  300          18.0           15.8     2.2
survey group     Other Hispanic  508          13.2           14.7    -1.5

Subgroup calibration TEST logistic:
       group              value   n  observed_pct  mean_pred_pct  gap_pp
         sex               Male 651          18.6           16.8     1.8
         sex             Female 674          19.0           17.5     1.5
   age group              20-44 517           8.5            5.2     3.3
   age group              45-64 484          16.3           17.9    -1.6
   age group                65+ 324          38.9           35.2     3.7
survey group   Mexican American 171          14.6           15.1    -0.4
survey group Non-Hispanic Asian 169          16.6           15.0     1.6
survey group Non-Hispanic Black 305          20.7           17.8     2.9
survey group Non-Hispanic White 487          20.9           18.2     2.8
survey group Other / multi-race  75          18.7           16.5     2.2
survey group     Other Hispanic 118          14.4           17.9    -3.5

Subgroup calibration HELD-OUT logistic:
       group              value    n  observed_pct  mean_pred_pct  gap_pp
         sex               Male 2251          17.7           15.0     2.8
         sex             Female 2732          16.0           15.6     0.4
   age group              20-44 1660           5.3            5.2     0.1
   age group              45-64 1714          13.7           15.5    -1.8
   age group                65+ 1609          31.9           25.5     6.4
survey group   Mexican American  333          12.0           14.4    -2.3
survey group Non-Hispanic Asian  263          15.2           14.2     1.0
survey group Non-Hispanic Black  557          23.9           17.7     6.2
survey group Non-Hispanic White 3022          16.6           15.1     1.5
survey group Other / multi-race  300          18.0           15.8     2.2
survey group     Other Hispanic  508          13.2           14.7    -1.5
test logistic @ 10%: sens 0.811, spec 0.584, PPV 0.311, NPV 0.930, flagged 650/1325 (49.1%)
test logistic @ 15%: sens 0.719, spec 0.705, PPV 0.361, NPV 0.916, flagged 496/1325 (37.4%)
test logistic @ 20%: sens 0.639, spec 0.783, PPV 0.405, NPV 0.903, flagged 393/1325 (29.7%)
heldout logistic @ 10%: sens 0.849, spec 0.570, PPV 0.285, NPV 0.949, flagged 2493/4983 (50.0%)
heldout logistic @ 15%: sens 0.733, spec 0.713, PPV 0.340, NPV 0.930, flagged 1804/4983 (36.2%)
heldout logistic @ 20%: sens 0.598, spec 0.805, PPV 0.383, NPV 0.909, flagged 1307/4983 (26.2%)

=== MODEL COMPARISON ===
            model     set  auroc  auprc  brier  cal_intercept  cal_slope    ece  weighted_brier
         logistic    test 0.7665 0.4616 0.1280        -0.0939     0.8232 0.0267          0.1048
gradient_boosting    test 0.7456 0.4093 0.1411        -0.5032     0.4516 0.0762          0.1132
    random_forest    test 0.7687 0.4391 0.1277        -0.1759     0.7875 0.0248          0.1032
         logistic heldout 0.7891 0.4514 0.1158         0.2348     1.0716 0.0216          0.0975
gradient_boosting heldout 0.7699 0.4234 0.1218        -0.3468     0.6271 0.0461          0.1032
    random_forest heldout 0.7938 0.4760 0.1128         0.0319     0.9878 0.0188          0.0963
