# Raw Dataset Inspection

## Overview

The `inspect_raw.py` utility provides a comprehensive inspection of all raw datasets used in the **XGB_Benchmark** project.

Its primary objective is to examine the quality and structure of every raw dataset **before** any preprocessing is performed. The script is purely diagnostic—it never modifies the original datasets.

The inspection results serve two purposes:

1. to verify that every dataset has been downloaded correctly and follows the expected format;
2. to guide the design and implementation of the preprocessing pipeline (`prepare.py`).

---

# Expected Raw Dataset Structure

Every dataset is expected to follow the standardized directory layout below:

```text
data/
└── raw/
    └── <dataset_name>/
        ├── data.csv
        └── metadata.json
```

Some manually downloaded datasets may additionally contain a `README.md`.

---

# Running the Inspection

## Quick inspection (default)

Inspects the first **10,000** rows of each dataset.

```bash
python src/data/inspect_raw.py
```

---

## Quick inspection with a custom number of rows

```bash
python src/data/inspect_raw.py --max-rows 50000
```

---

## Full inspection

Processes the entire dataset.

```bash
python src/data/inspect_raw.py --full
```

> **Note**
>
> Full inspection may require several minutes and substantial memory for very large datasets such as **HIGGS** and **CIC-IDS2017**.

---

# Generated Reports

Running the inspection creates two reports under

```text
data/reports/
```

```
inspection_report.json
inspection_summary.csv
```

These files are generated automatically and should **not** be committed to the repository.

---

# Information Collected

For every dataset, the inspection script records:

## Dataset information

- dataset key
- dataset name
- application domain
- machine learning task
- dataset size category

---

## Dataset integrity

- raw file existence
- target column
- target existence
- target data type
- SHA-256 hash of the raw dataset
- feature-name hash

---

## Dataset dimensions

- number of rows
- number of columns
- number of features

---

## Feature composition

- numerical features
- categorical features
- binary indicator features
- continuous numerical features

---

## Data quality

- missing values
- possible string-based missing tokens
- infinite values
- duplicate rows
- zero-variance columns
- duplicate feature columns

---

## Target information

- class labels
- class counts
- imbalance ratio

---

## Additional statistics

- memory usage
- categorical cardinality
- numerical summary statistics
  - minimum
  - maximum
  - mean
  - standard deviation

---

# Dataset-Specific Handling

## HIGGS

The HIGGS dataset does **not** contain a header row.

During inspection the script automatically assigns

```text
class
feature_1
feature_2
...
feature_28
```

before processing.

---

## CIC-IDS2017

Several columns contain leading or trailing whitespace.

The inspection automatically strips whitespace from column names immediately after loading the dataset.

---

# Current Inspection Results

The latest full inspection successfully loaded every registered dataset.

Important observations include:

| Dataset | Important observations |
|---------|------------------------|
| Adult Income | Missing values and duplicate rows |
| Bank Marketing | Large number of missing values |
| Dry Bean | Duplicate rows |
| Steel Plates Faults | Binary indicator features |
| Covertype | Large number of binary indicator features |
| Credit Card Fraud | Extremely imbalanced classes |
| UNSW-NB15 | Large number of duplicate rows |
| CIC-IDS2017 | Missing values, infinite values, duplicate rows, zero-variance features, duplicate feature columns |
| HIGGS | Clean numerical dataset with no missing values |

These observations provide valuable information for the preprocessing stage.

---

# Implications for `prepare.py`

The inspection phase defines the requirements for the preprocessing pipeline.

The following operations should be considered.

## 1. Column standardization

- strip whitespace from column names
- verify target column
- ensure consistent schema

---

## 2. Target preprocessing

- encode categorical targets
- preserve label mappings
- save target metadata

---

## 3. Missing values

- detect missing values
- apply a consistent imputation strategy
- record preprocessing decisions

---

## 4. Infinite values

Replace

```
+∞
-∞
```

with

```
NaN
```

before further processing.

---

## 5. Zero-variance features

Automatically remove constant feature columns.

---

## 6. Duplicate feature columns

Detect groups of identical features and retain only one representative column.

---

## 7. Duplicate rows

Duplicate rows should **not** necessarily be removed automatically.

Some datasets—particularly cybersecurity datasets—may intentionally contain repeated observations.

Duplicate-row removal should therefore remain configurable.

---

## 8. Binary indicator features

Binary indicator (one-hot) features should be detected automatically and treated separately from continuous numerical variables.

In particular, unnecessary scaling should be avoided.

---

## 9. Categorical features

Categorical features should be encoded consistently while preserving metadata describing the encoding process.

---

## 10. Continuous numerical features

Continuous numerical variables should be preprocessed independently of binary indicator variables.

---

## 11. Reproducibility

The preprocessing stage should preserve:

- SHA-256 hashes
- feature-name hashes
- preprocessing parameters
- removed columns
- encoding information

to ensure complete experiment reproducibility.

---

# Recommended Next Step

The next major component of the project is

```text
src/data/prepare.py
```

Its responsibility will be to transform

```text
data/raw/<dataset_name>/data.csv
```

into a standardized processed representation under

```text
data/processed/<dataset_name>/
```

The implementation of `prepare.py` should be driven by the findings of the inspection stage rather than assumptions about individual datasets.

---

# Summary

The inspection utility has become a core component of the **XGB_Benchmark** framework.

It provides:

- automated validation of raw datasets;
- comprehensive quality assessment;
- reproducibility through dataset hashing;
- detailed metadata for downstream preprocessing;
- a consistent foundation for designing the preprocessing pipeline.

With the inspection stage complete, the project is ready to proceed with the implementation of a unified and fully reproducible preprocessing framework.

# Current Inspection Results

The latest full inspection completed successfully for all registered datasets. The inspection identified several dataset-specific characteristics that should be considered during preprocessing.

| Dataset | Key findings | Implications for `prepare.py` |
|---------|--------------|-------------------------------|
| **Adult Income** | Contains missing values and duplicate rows. | Implement a consistent missing-value strategy and decide whether duplicate rows should be removed. |
| **Bank Marketing** | Contains a large number of missing values. | Missing-value handling will be a major preprocessing step for this dataset. |
| **Dry Bean** | Contains duplicate rows. | Evaluate whether duplicate rows should be retained or removed. |
| **Steel Plates Faults** | Contains **2 binary indicator features**. | Binary indicator features should be detected automatically and treated separately from continuous numerical features. |
| **Covertype** | Contains **44 binary indicator features** (one-hot encoded variables). | Avoid unnecessary scaling or transformation of binary indicator features. |
| **Credit Card Fraud** | Extremely imbalanced binary classification problem. | Preserve class distribution during train/test splitting and consider stratified sampling throughout the benchmarking pipeline. |
| **UNSW-NB15** | Contains a large number of duplicate rows. | Duplicate rows should not be removed blindly since repeated observations may be meaningful in cybersecurity datasets. |
| **CIC-IDS2017** | Contains missing values, infinite values, duplicate rows, zero-variance features, and duplicate feature columns. | This dataset requires the most extensive preprocessing, including handling infinities, removing constant features, and eliminating duplicate columns. |
| **HIGGS** | Large-scale, fully numerical dataset with no missing values, no duplicate feature columns, and a nearly balanced target distribution. | Serves as a clean baseline dataset requiring minimal preprocessing beyond standard validation. |

---

## Overall Observations

The inspection stage reveals that the benchmark datasets exhibit diverse characteristics and therefore require a flexible preprocessing framework rather than a one-size-fits-all solution.

The most important observations are:

- **Missing values** are present in multiple datasets and require a unified handling strategy.
- **Binary indicator (one-hot) features** appear in several datasets and should be distinguished from continuous numerical variables.
- **Duplicate rows** are present in multiple datasets; however, they should not be removed automatically without considering the semantics of each dataset.
- **Zero-variance** and **duplicate feature columns** occur in some datasets and should generally be removed during preprocessing.
- **Infinite values** occur in network traffic datasets and must be converted to valid missing values before further processing.
- **Class imbalance** varies considerably across datasets, ranging from nearly balanced (e.g., HIGGS) to highly imbalanced (e.g., Credit Card Fraud), which should be considered during model evaluation.
- The generated **SHA-256 hashes** and **feature-name hashes** provide an additional layer of reproducibility by allowing verification of both the raw data files and their schemas.

These observations form the design requirements for the next stage of the project: the implementation of `prepare.py`.



---
# Raw Dataset Inspection

## Purpose

The `inspect_raw.py` utility performs a comprehensive, read-only inspection of every raw dataset used in the **XGB_Benchmark** project.

Its primary objective is to characterize the structure, integrity, and quality of each dataset **before** any preprocessing is applied. The inspection stage never modifies the original data. Instead, it provides the evidence required to design, validate, and maintain a deterministic dataset preparation pipeline.

By separating inspection from preprocessing, the benchmark ensures that preparation decisions are driven by measured dataset characteristics rather than assumptions or dataset-specific heuristics.

---

# Inspection Workflow

Every raw dataset follows the inspection workflow shown below.

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
```

Inspection is therefore the first stage of the benchmark data pipeline and serves as the foundation for all subsequent preprocessing.

---

# Expected Dataset Structure

Every registered dataset is expected to follow the standardized directory structure below.

```text
data/
└── raw/
    └── <dataset_key>/
        ├── data.csv
        └── metadata.json
```

Some manually downloaded datasets may additionally include a `README.md` containing dataset-specific notes.

---

# Running the Inspection

## Quick inspection

Processes the first 10,000 rows of each dataset.

```bash
python src/data/inspect_raw.py
```

---

## Custom sample size

```bash
python src/data/inspect_raw.py --max-rows 50000
```

---

## Full inspection

Processes the complete dataset.

```bash
python src/data/inspect_raw.py --full
```

For very large datasets such as **HIGGS** and **CIC-IDS2017**, a full inspection may require substantial memory and execution time.

---

# Generated Reports

Running the inspection produces two reports under

```text
data/reports/
```

```
inspection_report.json
inspection_summary.csv
```

These reports summarize the structural and statistical properties of every inspected dataset.

Since they can be regenerated at any time, they generally should not be committed to version control.

---

# Information Collected

For every dataset, the inspection records the following information.

## Dataset metadata

- dataset key
- dataset name
- application domain
- machine learning task
- dataset size category

---

## Dataset integrity

- existence of required files
- target column
- target existence
- target data type
- SHA-256 hash of the raw dataset
- feature-name hash

---

## Dataset dimensions

- number of observations
- number of columns
- number of predictor variables

---

## Feature composition

- numerical features
- categorical features
- binary indicator features
- continuous numerical features

---

## Data quality

- missing values
- textual missing-value tokens
- infinite values
- duplicate rows
- zero-variance features
- duplicate feature columns

---

## Target information

- class labels
- class frequencies
- class imbalance

---

## Summary statistics

When applicable, numerical summary statistics are computed, including:

- minimum
- maximum
- mean
- standard deviation

The inspection also reports memory usage and categorical cardinalities where appropriate.

---

# Dataset-Specific Handling

The inspection framework is designed to be dataset-agnostic. Dataset-specific handling is introduced only when necessary to correctly interpret the published data.

Current exceptions include:

## HIGGS

The original HIGGS dataset does not contain a header row.

During inspection, the script automatically assigns the following column names:

```text
class
feature_1
feature_2
...
feature_28
```

---

## CIC-IDS2017

Several feature names contain leading or trailing whitespace.

The inspection standardizes column names immediately after loading the dataset to ensure consistent schema validation.

No other modifications are applied.

---

# Role in the Data Pipeline

The inspection stage is intentionally independent of dataset preparation.

Its responsibilities are limited to:

- validating raw datasets;
- measuring dataset characteristics;
- identifying potential data-quality issues;
- generating reproducible inspection reports.

The inspection stage does **not** modify datasets, remove observations, encode variables, or perform any preprocessing.

All dataset transformations are delegated to `prepare.py`.

---

# Reproducibility

Inspection results are fully deterministic.

Given the same raw dataset, the inspection utility always produces identical reports and dataset hashes.

This allows both datasets and inspection results to be independently verified and provides a reproducible foundation for the preparation stage.

---

# Summary

The inspection stage provides an objective characterization of every raw dataset before preprocessing begins.

Its outputs:

- validate dataset integrity;
- identify structural and statistical characteristics;
- document potential quality issues;
- support the design and verification of the preparation pipeline;
- improve the transparency and reproducibility of the benchmark.

Together with the canonical preparation stage, the inspection utility forms the foundation of the data management workflow used throughout the **XGB_Benchmark** project.