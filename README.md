# XGB_Benchmark

A reproducible benchmarking framework for evaluating **XGBoost intrinsic feature importance** against **SHAP-based feature selection** on diverse tabular datasets.

---

## Overview

XGB_Benchmark is an open-source research project that investigates whether the intrinsic feature-importance metrics produced by XGBoost can serve as efficient alternatives to SHAP-based feature selection.

The project focuses on explainable machine learning for tabular data and aims to evaluate the effectiveness of XGBoost's built-in feature-importance metrics—**Gain**, **Cover**, and **Weight**—across a diverse collection of benchmark datasets.

---

## Research Question

Can XGBoost's intrinsic feature-importance metrics provide feature subsets that achieve predictive performance comparable to those selected using SHAP while requiring significantly less computational effort?

---

## Objectives

The objectives of this project are to:

- Compare Gain, Cover, Weight, and SHAP for feature selection.
- Evaluate feature selection across a diverse collection of benchmark datasets.
- Train interpretable downstream models using the selected features.
- Compare predictive performance, interpretability, and computational efficiency.
- Build a reproducible benchmarking framework for future research.

---

## Benchmark Pipeline

For each dataset, the following pipeline will be executed:

```text
Dataset
    │
    ▼
Preprocessing
    │
    ▼
Train XGBoost
    │
    ├──────────────┐
    │              │
    ▼              ▼
Gain         Mean |SHAP|
Cover
Weight
    │
    ▼
Feature Ranking
    │
    ▼
Top-k Feature Selection
    │
    ▼
Decision Tree
    │
    ▼
Evaluation
```

---

## Initial Feature Selection Methods

- Gain
- Cover
- Weight
- Mean Absolute SHAP

Additional feature-selection methods will be incorporated in future releases.

---

# Benchmark Datasets

The benchmark currently supports datasets from multiple public repositories spanning several application domains.

| Dataset | Domain | Task | Source | Download |
|---------|--------|------|--------|----------|
| Adult Income | Census | Binary Classification | OpenML | Automatic |
| Breast Cancer Wisconsin | Healthcare | Binary Classification | scikit-learn | Automatic |
| Dry Bean | Agriculture | Multiclass Classification | UCI ML Repository | Automatic |
| Bank Marketing | Finance | Binary Classification | UCI ML Repository | Automatic |
| Steel Plates Faults | Manufacturing | Multiclass Classification | UCI ML Repository | Automatic |
| Covertype | Forestry | Multiclass Classification | scikit-learn | Automatic |
| Credit Card Fraud | Finance | Binary Classification | Kaggle | Automatic* |
| UNSW-NB15 | Cybersecurity | Multiclass Classification | Kaggle | Automatic* |
| CIC-IDS2017 | Cybersecurity | Multiclass Classification | Canadian Institute for Cybersecurity | Manual |
| HIGGS | Particle Physics | Binary Classification | UCI Machine Learning Repository | Manual |

\* Requires a configured Kaggle account.

---

# Data Pipeline

One of the primary goals of this project is to provide a **reproducible and consistent data pipeline**, regardless of the original dataset source.

Every dataset follows the same lifecycle:

```text
Official Dataset
        │
        ▼
Download
        │
        ▼
Raw Data Standardization
        │
        ▼
data/raw/<dataset>/
    ├── data.csv
    └── metadata.json
        │
        ▼
Data Preparation
        │
        ▼
data/processed/<dataset>/
        │
        ▼
Benchmark Pipeline
```

## Raw Data

The `data/raw/` directory contains **standardized raw datasets**.

Every dataset follows the same structure:

```text
data/raw/
└── <dataset_name>/
    ├── data.csv
    └── metadata.json
```

Datasets are obtained from multiple sources, including:

- OpenML
- scikit-learn
- UCI Machine Learning Repository
- Kaggle
- Manual download

Datasets that require manual acquisition (currently **HIGGS** and **CIC-IDS2017**) additionally include a `README.md` describing:

- where to obtain the original dataset,
- how to standardize the raw files,
- and how to organize them into the expected project structure.

## Processed Data

The `data/processed/` directory contains datasets after preprocessing and transformation.

Regardless of the original source, every processed dataset will share a common format, allowing the benchmarking pipeline to operate independently of the original data source.

---

## Reproducibility

Reproducibility is a primary design goal of this project.

Every experiment is designed to be reproducible using identical:

- datasets
- preprocessing steps
- train/test splits
- model configurations
- random seeds

---

## Repository Structure

```text
XGB_Benchmark/

├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── benchmark/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   └── models/
│
├── experiments/
├── notebooks/
└── results/
```

### Module Responsibilities

| Module | Responsibility |
|---------|----------------|
| **benchmark** | Executes benchmarking experiments and orchestrates the evaluation pipeline. |
| **data** | Dataset registry, downloading, loading, preprocessing, validation, and dataset management. |
| **evaluation** | Performance metrics, explainability analysis, statistical testing, and visualization. |
| **features** | Feature preprocessing, encoding, scaling, and feature selection. |
| **models** | Model training, hyperparameter tuning, cross-validation, and baseline implementations. |

---

## Project Status

🚧 **The repository is currently under active development.**

The first milestone is to build a fully automated benchmarking framework capable of evaluating intrinsic XGBoost feature importance against SHAP across multiple benchmark datasets.

---

## Roadmap

### Phase 1 — Data Infrastructure

- [x] Repository setup
- [x] Dataset registry
- [x] Multi-source dataset downloader
- [x] Raw data standardization
- [ ] Dataset inspection
- [ ] Dataset preparation
- [ ] Dataset validation

### Phase 2 — Benchmark Core

- [ ] XGBoost training pipeline
- [ ] Intrinsic feature importance extraction
- [ ] SHAP implementation
- [ ] Decision Tree benchmark
- [ ] Cross-validation framework

### Phase 3 — Evaluation

- [ ] Benchmark automation
- [ ] Statistical significance analysis
- [ ] Visualization
- [ ] Automatic report generation

### Phase 4 — Research

- [ ] Multi-dataset benchmark
- [ ] Paper submission

---

## Future Work

Future versions of the benchmark may include:

- Additional feature-selection methods
- More explainable machine learning models
- GPU benchmarking
- Distributed benchmarking
- Automatic hyperparameter optimization
- OpenML benchmark integration
- Meta-learning experiments

---

## Installation

Clone the repository:

```bash
git clone https://github.com/PatternCode/XGB_Benchmark.git
cd XGB_Benchmark

conda env create -f environment.yml
conda activate xgb_benchmark

pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome.

If you would like to contribute a new dataset, feature-selection method, benchmark model, or evaluation metric, please open an issue or submit a pull request.

---

## Citation

If you use this repository in your research, please cite the associated publication once it becomes available.

---

## License

This project is released under the MIT License.