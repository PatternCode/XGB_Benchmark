# Illustrative compact decision-tree configurations — Macro-F1

Candidate trees use DT3–DT6 with subsets ranked by Gain, Weight, Cover, or SHAP.
A candidate qualifies when its mean Macro-F1 is no more than 0.010 below the same-depth all-features tree and is at least as high as the matched random-feature baseline.
Among qualifying candidates, the descriptive rule prioritizes fewer selected features, lower maximum depth, fewer mean nodes, higher predictive performance, and finally method name.
These configurations are descriptive examples identified from the same cross-validation results; they are not independently validated optimal models.
Predictive values are reported as mean ± sample SD across the five folds. Random SD is calculated after averaging the 20 repetitions within each fold.

| Dataset | Ranking | k/d | DT max | Actual depth | Features used | Nodes | Macro-F1 | Random Macro-F1 | All-feature Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult Income | Cover | 4/14 | 3 | 3.0 | 3.0 | 15.0 | 0.743 ± 0.006 | 0.595 ± 0.024 | 0.743 ± 0.006 |
| Bank Marketing | Gain | 2/16 | 3 | 3.0 | 2.0 | 15.0 | 0.700 ± 0.010 | 0.502 ± 0.012 | 0.700 ± 0.010 |
| Breast Cancer Wisconsin | SHAP | 2/30 | 3 | 3.0 | 2.0 | 13.8 | 0.916 ± 0.035 | 0.820 ± 0.017 | 0.912 ± 0.024 |
| CIC-IDS2017 | SHAP | 5/65 | 4 | 4.0 | 5.0 | 28.2 | 0.368 ± 0.034 | 0.233 ± 0.010 | 0.349 ± 0.012 |
| Covertype | SHAP | 5/54 | 3 | 3.0 | 3.0 | 15.0 | 0.417 ± 0.002 | 0.162 ± 0.012 | 0.423 ± 0.002 |
| Credit Card Fraud | Gain | 2/30 | 3 | 3.0 | 2.0 | 15.0 | 0.887 ± 0.017 | 0.668 ± 0.032 | 0.888 ± 0.009 |
| Dry Bean | SHAP | 2/16 | 5 | 5.0 | 2.0 | 55.4 | 0.879 ± 0.005 | 0.731 ± 0.022 | 0.888 ± 0.008 |
| Steel Plates Faults | SHAP | 2/27 | 4 | 4.0 | 2.0 | 30.6 | 0.539 ± 0.024 | 0.330 ± 0.024 | 0.489 ± 0.046 |
| UNSW-NB15 | SHAP | 3/39 | 4 | 4.0 | 3.0 | 29.0 | 0.745 ± 0.001 | 0.525 ± 0.026 | 0.734 ± 0.001 |
