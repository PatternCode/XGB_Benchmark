# Canonical Dataset Preparation Policy

## Purpose

This document defines the canonical dataset preparation policy used throughout the **XGB Benchmark** project.

The primary objective of this repository is **not** to maximize predictive performance. Instead, it aims to rigorously evaluate the ability of **XGBoost's internal feature importance measures** to identify informative and discriminative features, and to compare their effectiveness with **SHAP-based feature selection** methods.

To ensure that this comparison is fair, reproducible, and scientifically meaningful, every experiment in this repository must operate on datasets produced by the same deterministic preparation pipeline.

The preparation pipeline performs only **deterministic, representation-level transformations** that preserve the original information content of each dataset while eliminating deterministic redundancies that cannot contribute to discriminative learning. It deliberately avoids transformations that modify the statistical properties of the data or introduce biases that could influence downstream feature-selection methods.

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

No operation should intentionally alter the underlying statistical relationships among features or between the features and the target variable.

---

## 4. Reproducibility

Every transformation performed during preparation must be deterministic, documented, and reproducible.

Hidden preprocessing steps are strictly avoided.

---

## 5. Separation of Responsibilities

Dataset preparation and experimental preprocessing are intentionally separated.

Dataset preparation creates a canonical processed dataset.

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

## Duplicate Features

Duplicate feature columns identified during inspection are consolidated by retaining a single representative from each group of identical features.

Since duplicate columns contain identical information, their removal does not alter the information content of the dataset. Removing duplicate features also eliminates unnecessary redundancy that could complicate the interpretation of feature-importance methods.

The retained feature preserves the original semantics of the dataset, while the removed duplicate columns will be documented in the preparation metadata to ensure full traceability.

---

## Zero-Variance Features

Zero-variance features are removed during dataset preparation.

Such features contain no discriminative information because they assume the same value for every observation. Their removal is deterministic, independent of class labels and learning algorithms, and therefore does not alter the informative content of the dataset.

Removed zero-variance features will be documented in the preparation metadata to maintain reproducibility and traceability.

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
- outlier removal
- oversampling
- undersampling
- class balancing
- train/validation/test splitting
- cross-validation
- hyperparameter tuning

These operations belong to the experimental stage.

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


----
# Canonical Dataset Preparation

## Purpose

The **XGB_Benchmark** project evaluates feature-selection methods across a diverse collection of publicly available tabular datasets. To ensure that every experiment is conducted under identical conditions, all datasets are first converted into a canonical representation using a deterministic preparation pipeline implemented in `prepare.py`.

The purpose of canonical preparation is **not** to optimize predictive performance or improve data quality. Instead, it provides a standardized dataset representation that preserves the information content of the original data while eliminating deterministic inconsistencies and redundancies that are irrelevant to downstream learning.

Every benchmark experiment operates exclusively on these canonical processed datasets.

---

# Processing Workflow

Each dataset follows the workflow shown below.

```text
Raw Dataset
      │
      ▼
inspect_raw.py
      │
      ▼
Inspection Reports
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

The responsibilities of each stage are intentionally separated.

- **Raw datasets** preserve the original published data.
- **inspect_raw.py** characterizes dataset structure and quality without modifying the data.
- **prepare.py** performs deterministic, representation-level transformations.
- **Benchmark experiments** perform all model-dependent preprocessing and evaluation.

---

# Design Principles

The preparation pipeline is governed by the following principles.

## Deterministic

Preparation is entirely deterministic.

Given the same raw dataset and the same preparation version, the generated processed dataset and metadata are guaranteed to be identical.

No randomness is introduced during preparation.

---

## Dataset-Agnostic

A single preparation framework is applied across all benchmark datasets.

Dataset-specific handling is introduced only when required to accommodate differences in published file formats or known representation issues.

---

## Information Preservation

Preparation standardizes dataset representation while preserving the complete information content of the original data.

Representation-level inconsistencies such as whitespace, naming conventions, duplicate feature columns, and zero-variance features may be removed because they do not contribute additional information.

No operation intentionally alters the statistical relationships among features or between the features and the target variable.

---

## Reproducibility

Every transformation performed during preparation is deterministic, documented, and fully reproducible.

Each processed dataset remains traceable to its corresponding raw dataset through metadata and cryptographic hashes.

---

## Separation of Responsibilities

Canonical dataset preparation and experimental preprocessing are intentionally separated.

Preparation creates a standardized dataset representation.

Experiments remain responsible for all preprocessing decisions that may influence model behaviour or predictive performance.

---

# Operations Performed

The preparation pipeline performs only deterministic representation-level transformations.

## Column Name Standardization

Feature names are converted into a consistent canonical representation.

Typical operations include:

- removing leading and trailing whitespace;
- converting names to `snake_case`;
- replacing non-alphanumeric characters;
- ensuring unique column names.

These transformations improve consistency without changing feature semantics.

---

## Missing Value Standardization

Known textual representations of missing values are converted into standard `NaN` values.

Typical examples include:

- `?`
- `NA`
- `N/A`
- `NULL`
- `None`
- empty strings

No missing-value imputation is performed.

---

## Infinite Values

Positive and negative infinity values are replaced with `NaN`.

This provides a consistent representation for values that cannot be processed reliably by most machine learning algorithms.

---

## Missing Target Values

Observations with missing target values are removed.

Since supervised learning requires a valid target label for every observation, such samples cannot participate in downstream experiments.

---

## Duplicate Features

Exactly identical feature columns are consolidated by retaining a single representative from each duplicate group.

Duplicate features contain identical information and therefore do not increase the information content of the dataset.

The removed columns are recorded in the preparation metadata to maintain complete traceability.

---

## Zero-Variance Features

Constant feature columns are removed.

Since these features assume the same value for every observation, they contain no information that can distinguish between samples and therefore cannot contribute to feature selection or predictive modelling.

All removed features are documented in the preparation metadata.

---

## Target Encoding

Target labels are converted into consecutive integer identifiers.

The mapping between the original labels and their encoded values is preserved in the dataset metadata.

Example:

```text
BENIGN                → 0
DoS Hulk              → 1
Web Attack - XSS      → 2
...
```

---

## Dataset-Specific Normalization

Minor dataset-specific corrections may be applied when necessary to preserve semantic consistency.

These corrections are limited to deterministic representation issues, such as repairing known encoding errors or accommodating published file-format differences.

No dataset-specific statistical preprocessing is performed.

---

## Data Types

Columns are stored using appropriate pandas data types whenever possible while preserving their semantic interpretation.

---

## Metadata Generation

Every processed dataset is accompanied by a metadata file documenting the preparation process.

Typical metadata include:

- dataset information;
- preparation version;
- source and processed file hashes;
- target column;
- target-label mapping;
- feature names;
- feature types;
- removed duplicate features;
- removed zero-variance features;
- numbers of missing and infinite values.

This metadata enables complete traceability and reproducibility.

---

# Operations Explicitly Not Performed

The preparation pipeline intentionally avoids any operation that modifies the statistical characteristics of the dataset.

In particular, it does **not** perform:

- missing-value imputation;
- categorical encoding;
- feature scaling;
- normalization;
- standardization;
- dimensionality reduction;
- feature extraction;
- feature selection;
- correlation filtering;
- outlier removal;
- class balancing;
- oversampling;
- undersampling;
- train/validation/test splitting;
- cross-validation;
- hyperparameter optimization.

These operations belong to the experimental stage because they may influence model behaviour or predictive performance.

---

# Experimental Responsibilities

Benchmark experiments operate exclusively on canonical processed datasets.

Experiments are responsible for any preprocessing that depends on the chosen learning algorithm, including:

- missing-value imputation;
- categorical encoding;
- feature scaling;
- feature engineering;
- train/validation/test splitting;
- cross-validation;
- class balancing;
- feature selection;
- hyperparameter optimization.

Separating deterministic dataset preparation from experimental preprocessing ensures that every feature-selection method is evaluated under identical initial conditions.

---

# Reproducibility

Every processed dataset can be traced back to:

- the original raw dataset;
- the corresponding inspection reports;
- the preparation metadata;
- the preparation policy defined in this document.

This guarantees that benchmark experiments are fully reproducible and allows new datasets to be incorporated into the benchmark without changing the underlying preparation philosophy.

---

# Summary

The canonical preparation stage provides a standardized, reproducible representation of every benchmark dataset while preserving its original information content.

By limiting preparation to deterministic representation-level transformations and delegating all model-dependent preprocessing to the experimental stage, the benchmark ensures that comparisons between feature-selection methods remain fair, transparent, and scientifically reproducible.