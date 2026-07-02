
# Datasets

This document lists the datasets used in the **XGB_Benchmark** project.

The goal of this benchmark is to evaluate whether XGBoost intrinsic feature-importance metrics (Gain, Cover, and Weight) can provide feature subsets with predictive performance comparable to those selected using SHAP across a wide range of tabular datasets.

---

# Dataset Selection Criteria

Datasets included in this benchmark should satisfy the following criteria:

- Tabular data
- Supervised classification task
- Publicly available
- Sufficient number of samples
- Sufficient number of features
- Well-documented source
- Widely used in machine learning research

The benchmark aims to include datasets from multiple application domains, including healthcare, cybersecurity, finance, manufacturing, biology, and general machine learning.

---

# Planned Benchmark Datasets

| Dataset | Domain | Task | Samples | Features | Classes | Size | Priority | Status |
|---------|--------|------|---------:|---------:|--------:|------|----------|--------|
| Adult Income | Census | Binary | 48,842 | 14 | 2 | Small | ★★★★★ | Planned |
| Breast Cancer Wisconsin | Healthcare | Binary | 569 | 30 | 2 | Small | ★★★★★ | Planned |
| Dry Bean | Agriculture | Multiclass | 13,611 | 16 | 7 | Small | ★★★★★ | Planned |
| Bank Marketing | Finance | Binary | 45,211 | 16 | 2 | Medium | ★★★★★ | Planned |
| Steel Plates Faults | Manufacturing | Multiclass | 1,941 | 27 | 7 | Small | ★★★★★ | Planned |
| Covertype | Forestry | Multiclass | 581,012 | 54 | 7 | Large | ★★★★☆ | Planned |
| Credit Card Fraud | Finance | Binary | 284,807 | 30 | 2 | Large | ★★★★☆ | Planned |
| UNSW-NB15 | Cybersecurity | Multiclass | 257,673 | 49 | 10 | Large | ★★★★★ | Planned |
| CIC-IDS2017 | Cybersecurity | Multiclass | >2,800,000 | 78 | 15 | Very Large | ★★★★★ | Planned |
| HIGGS | Particle Physics | Binary | 11,000,000 | 28 | 2 | Very Large | ★★★★☆ | Planned |

---

# Evaluation Strategy

For every dataset, the following pipeline will be applied:

1. Load dataset
2. Data preprocessing
3. Train XGBoost
4. Extract Gain, Cover, and Weight feature importance
5. Compute SHAP values
6. Select the top-k features
7. Train an interpretable downstream model
8. Evaluate predictive performance
9. Compare computational cost
10. Save all results

---

# Downstream Models

The benchmark will initially evaluate:

- Decision Tree
- Logistic Regression
- Linear SVM

Additional interpretable models may be incorporated in future versions.

---

# Evaluation Metrics

The following metrics will be reported where applicable:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- Training Time
- Inference Time
- Feature Selection Time

---

# Feature Selection Methods

Current methods:

- XGBoost Gain
- XGBoost Cover
- XGBoost Weight
- Mean Absolute SHAP

Future methods may include:

- Permutation Importance
- Mutual Information
- ReliefF
- Boruta
- Recursive Feature Elimination (RFE)
- LASSO
- Integrated Gradients
- LIME

---

# Dataset Sources

Datasets will be obtained from one or more of the following repositories:

- UCI Machine Learning Repository
- OpenML
- Kaggle (when licensing permits)
- CIC
- UNSW
- Open government datasets

Only publicly available datasets with appropriate research licenses will be included.