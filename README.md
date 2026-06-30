# XGB_Benchmark

A reproducible benchmarking framework for evaluating XGBoost intrinsic feature importance against SHAP-based feature selection on tabular datasets.

---

## Overview

XGB_Benchmark is an open-source research project that investigates whether the intrinsic feature-importance metrics produced by XGBoost can serve as efficient alternatives to SHAP-based feature selection.

The project focuses on explainable machine learning for tabular data and aims to evaluate the effectiveness of XGBoost's built-in feature-importance metrics—Gain, Cover, and Weight—across a diverse collection of benchmark datasets.

---

## Research Question

Can XGBoost's intrinsic feature-importance metrics provide feature subsets that achieve predictive performance comparable to those selected using SHAP while requiring significantly less computational effort?

---

## Objectives

The objectives of this project are to:

- Compare Gain, Cover, Weight, and SHAP for feature selection.
- Evaluate feature selection across a wide range of tabular datasets.
- Train interpretable downstream models using the selected features.
- Compare predictive performance, interpretability, and computational efficiency.
- Build a reproducible benchmarking framework for future research.

---

## Benchmark Pipeline

For each dataset, the following pipeline will be executed:

```
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

## Candidate Datasets

The benchmark will initially focus on publicly available tabular datasets from multiple domains, including:

- Cybersecurity
- Healthcare
- Finance
- Manufacturing
- Physics
- General machine learning benchmarks

---

## Reproducibility

One of the primary goals of this repository is reproducible research.

The benchmark is being developed to ensure that every experiment can be reproduced using identical datasets, preprocessing steps, model configurations, and random seeds.

---

## Repository Structure

```
XGB_Benchmark/

├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
├── experiments/
├── results/
└── notebooks/
```

---

## Project Status

🚧 The repository is currently under active development.

The initial milestone is to implement a fully automated benchmarking pipeline for comparing XGBoost intrinsic feature importance with SHAP on a single dataset. The framework will then be extended to multiple benchmark datasets.

---

## Future Work

Future versions of the benchmark may include:

- Additional feature-selection methods
- More explainable classifiers
- Automated benchmarking
- Statistical significance analysis
- Automatic report generation
- OpenML integration

---

## License

This project is released under the MIT License.

## Installation

Clone the repository:

```bash
git clone https://github.com/PatternCode/XGB_Benchmark.git
cd XGB_Benchmark
conda env create -f environment.yml
conda activate xgb_benchmark
pip install -r requirements.txt

## Contributing

Contributions are welcome.

If you would like to contribute a new dataset, feature-selection method, or evaluation metric, please open an issue or submit a pull request.


## Citation

If you use this repository in your research, please cite the associated publication once it becomes available.
```

## Roadmap

### Phase 1
- [x] Repository setup
- [ ] Dataset loading
- [ ] XGBoost training
- [ ] Feature importance extraction

### Phase 2
- [ ] SHAP implementation
- [ ] Decision Tree benchmark
- [ ] Evaluation pipeline

### Phase 3
- [ ] Multi-dataset benchmark
- [ ] Automatic report generation
- [ ] Statistical significance analysis

### Phase 4
- [ ] Paper submission