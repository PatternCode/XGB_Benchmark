# Canonical Dataset Preparation Policy

## Purpose

This document defines the canonical dataset preparation policy used throughout the **XGB Benchmark** project.

The primary objective of this repository is **not** to maximize predictive performance. Instead, it aims to rigorously evaluate the ability of **XGBoost's internal feature importance measures** to identify informative and discriminative features, and to compare their effectiveness with **SHAP-based feature selection** methods.

To ensure that this comparison is fair, reproducible, and scientifically meaningful, every experiment in this repository must operate on datasets produced by the same deterministic preparation pipeline.

The preparation pipeline performs only **deterministic, representation-level transformations** that preserve the original information content of each dataset. It deliberately avoids transformations that modify the statistical properties of the data or introduce biases that could influence downstream feature-selection methods.

---

# Processing Pipeline

Every dataset follows the same processing workflow:

```text
Raw Dataset
      │
      ▼
inspect_raw.py
      │
      ▼
Inspection Report
      │
      ▼
prepare.py
      │
      ▼
Canonical Processed Dataset
      │
      ▼
Benchmark Experiments
```

The responsibility of each stage is clearly separated:

- **Raw datasets** preserve the original published data.
- **inspect_raw.py** performs quality assessment and generates diagnostic reports.
- **prepare.py** converts raw datasets into a canonical representation.
- **Benchmark experiments** perform all model-specific preprocessing and evaluation.

---

# Design Principles

The preparation pipeline follows the principles below.

## 1. Deterministic

Preparation must always produce identical outputs when executed on the same raw dataset.

No randomness is introduced during dataset preparation.

---

## 2. Dataset-Agnostic

The same preparation framework should operate on every benchmark dataset.

Dataset-specific handling is permitted only when required by differences in file format or target representation.

---

## 3. Information Preservation

Preparation standardizes the representation of the data while preserving its original information content.

No operation should intentionally alter the underlying statistical relationships among features.

---

## 4. Reproducibility

Every transformation performed during preparation must be deterministic, documented, and reproducible.

Hidden preprocessing steps are strictly avoided.

---

## 5. Separation of Responsibilities

Dataset preparation and experimental preprocessing are intentionally separated.

Dataset preparation creates a canonical dataset.

Experiments are responsible for all model-dependent transformations.

---

# Operations Performed

The preparation pipeline performs only deterministic, representation-level transformations.

## Column Names

Column names are standardized to improve consistency.

Typical operations include:

- removing leading and trailing whitespace
- converting names to `snake_case`
- ensuring unique column names

The semantic meaning of every feature is preserved.

---

## Missing Values

Known missing-value representations are converted to standard `NaN` values.

Examples include:

- `?`
- `NA`
- `N/A`
- `NULL`
- `None`
- empty strings

No imputation is performed.

---

## Infinite Values

Positive and negative infinity values are replaced with `NaN`.

Infinite values cannot be handled consistently by downstream learning algorithms and therefore require canonical representation.

---

## Target Labels

Target labels are converted into integer class identifiers.

The original label mapping is preserved within the processed metadata.

Example:

```text
BENIGN   → 0
DoS      → 1
Probe    → 2
...
```

---

## Data Types

Columns are converted to appropriate pandas data types whenever possible while preserving their semantic interpretation.

---

## Metadata

Each processed dataset contains metadata describing the preparation process.

Typical metadata include:

- dataset name
- target column
- feature names
- feature types
- target label mapping
- source dataset hash
- preparation version
- preparation timestamp

---

# Operations Explicitly NOT Performed

The preparation pipeline intentionally avoids any operation that modifies the statistical characteristics of the dataset.

Specifically, it does **not** perform:

- feature scaling
- normalization
- standardization
- PCA
- feature extraction
- feature selection
- correlation filtering
- duplicate feature removal
- zero-variance feature removal
- outlier removal
- oversampling
- undersampling
- class balancing
- train/validation/test splitting
- cross-validation
- hyperparameter tuning

These operations belong to the experimental stage.

---

# Duplicate Features

Duplicate feature columns identified during inspection are preserved.

Although redundant, they are part of the original published dataset.

Removing duplicate features changes the feature space and may influence the behaviour of feature-selection algorithms.

Their existence is documented, but the canonical processed dataset retains them unchanged.

---

# Zero-Variance Features

Zero-variance features are preserved.

Their existence is recorded during inspection.

Experiments may optionally remove them if required, but the canonical processed dataset always remains faithful to the original data.

---

# Missing Data

Missing values remain represented as `NaN`.

The strategy used to handle missing values is considered part of the experimental pipeline rather than dataset preparation.

Different models may require different imputation strategies, and these should be evaluated consistently during benchmarking.

---

# Categorical Features

Categorical predictors are preserved in their original semantic representation whenever possible.

Encoding methods such as:

- ordinal encoding
- one-hot encoding
- target encoding
- frequency encoding

are considered experimental preprocessing steps and are therefore not performed during dataset preparation.

---

# Binary Indicator Features

Binary indicator variables are preserved exactly as provided by the original dataset.

No transformation is applied.

---

# Experimental Responsibilities

Benchmark experiments are responsible for all model-specific preprocessing.

These include, but are not limited to:

- missing-value imputation
- categorical encoding
- feature scaling
- feature engineering
- feature selection
- train/validation/test splitting
- cross-validation
- class balancing
- hyperparameter optimisation

Applying these operations during experiments ensures that every feature-selection method is evaluated under identical preparation conditions.

---

# Reproducibility

Every processed dataset must remain fully traceable to:

- the original raw dataset,
- the inspection report,
- the preparation metadata,
- and the policy defined in this document.

This guarantees that every benchmark experiment is reproducible and that future datasets can be incorporated into the benchmark without changing the underlying preparation philosophy.