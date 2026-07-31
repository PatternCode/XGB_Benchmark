# XGB Benchmark – Analysis Plan

## Purpose

This document defines the scientific analysis workflow for the XGB Benchmark
project. Its purpose is to ensure that all figures, tables, and conclusions are
generated in a reproducible manner directly from the benchmark outputs.

The analysis is performed exclusively using the combined benchmark results
stored in:

- results/summaries/combined_results_9_datasets.csv
- results/summaries/combined_rankings_9_datasets.csv

No values reported in the paper should be computed manually.

---

# Overall Workflow

Benchmark Experiments
        │
        ▼
Combined Results
        │
        ▼
Result Validation
        │
        ▼
Dataset-level Analysis
        │
        ▼
Cross-dataset Analysis
        │
        ▼
Publication Figures
Publication Tables
        │
        ▼
Paper (Overleaf)

---

# Scientific Objectives

The analysis aims to answer the following research questions.

## RQ1

How do different XGBoost feature-importance methods compare for feature
selection?

Methods:

- Gain
- Weight
- Cover
- SHAP

Baselines:

- Random feature selection
- All features

---

## RQ2

How many features are required to achieve near-optimal predictive performance?

Investigate:

- requested percentage
- actual selected features
- diminishing returns

---

## RQ3

Does the best feature-selection method depend on the downstream classifier?

Models:

- Decision Tree
- Logistic Regression
- XGBoost

---

## RQ4

Are the conclusions consistent across different benchmark datasets?

Datasets:

- Adult Income
- Bank Marketing
- Breast Cancer Wisconsin
- CIC-IDS2017
- Covertype
- Credit Card Fraud
- Dry Bean
- Steel Plates Faults
- UNSW-NB15

(HIGGS will be analysed separately.)

---

## RQ5

Can shallow Decision Trees provide an interpretable model with competitive
performance?

---

# Planned Figures

## Figure 1

Dataset characteristics.

Status:
Not started.

---

## Figure 2

Performance versus feature percentage for each dataset.

Purpose:

Compare

- Gain
- Weight
- Cover
- SHAP
- Random

with

- All-features baseline.

Status:
Not started.

---

## Figure 3

Overall comparison of feature-selection methods.

Status:
Not started.

---

## Figure 4

Performance-complexity trade-off.

Status:
Not started.

---

## Figure 5

Best-performing method for each dataset.

Status:
Not started.

---

## Figure 6

Ranking agreement and stability.

Status:
Not started.

---

# Planned Tables

## Table 1

Benchmark dataset summary.

Contents:

- samples
- features
- classes
- task type

Status:
Not started.

---

## Table 2

Average performance of all feature-selection methods.

Status:
Not started.

---

## Table 3

Best-performing method for every dataset.

Status:
Not started.

---

## Table 4

Performance of shallow Decision Trees.

Status:
Not started.

---

## Supplementary Tables

Additional detailed experimental results.

Status:
Not started.

---

# Analysis Strategy

The analysis follows two levels.

## Level 1

Dataset-level analysis.

Every dataset is analysed independently.

This includes:

- feature-selection behaviour
- performance curves
- best feature percentages
- model comparison

The conclusions are first established separately for every dataset.

---

## Level 2

Cross-dataset analysis.

After completing the dataset-level analysis, summary statistics are computed
across datasets.

Cross-dataset conclusions are drawn only after each dataset contributes equally.

Simple averaging over all experiment rows should be avoided because datasets
contain different numbers of evaluated feature percentages.

---

# Reproducibility

All publication figures and tables must be generated automatically using the
analysis scripts.

No manual calculations should appear in the manuscript.

---

# Current Status

Benchmark framework:
Completed.

Benchmark experiments:
Completed for 9 datasets.

Combined benchmark files:
Completed.

Scientific analysis:
Not started.

Paper writing:
Not started.

HIGGS:
Deferred to a later stage.