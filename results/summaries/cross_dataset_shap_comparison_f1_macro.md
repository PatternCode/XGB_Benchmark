# Cross-dataset native-method comparison against SHAP — Macro-F1

All paired differences are defined as **native method − SHAP** for exactly matched dataset/fold/k/downstream-model conditions.
For each dataset/model/k condition, the five fold-wise paired differences are averaged before assigning a practical outcome.
With equivalence margin **δ=0.010**, a native method is a **Win** when mean Δ > +δ, a **Tie** when −δ ≤ mean Δ ≤ +δ, and a **Loss** when mean Δ < −δ.
Win/tie/loss rates are calculated within each dataset first and then averaged across datasets, preventing datasets with more distinct k values from receiving greater weight.
The reported mean Δ values are descriptive paired differences, not statistical significance tests.

## Summary by downstream model

| Model | Method | Mean Δ vs SHAP (Macro-F1) | Wins (%) | Ties (%) | Losses (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| LR | Gain | +0.002 | 32.8 | 27.5 | 39.7 |
| LR | Weight | -0.074 | 6.5 | 24.5 | 69.0 |
| LR | Cover | -0.010 | 32.4 | 29.0 | 38.6 |
| XGBoost | Gain | -0.074 | 9.4 | 26.5 | 64.0 |
| XGBoost | Weight | -0.049 | 19.1 | 30.8 | 50.0 |
| XGBoost | Cover | -0.090 | 10.3 | 28.5 | 61.3 |
| DT3 | Gain | -0.009 | 21.7 | 40.9 | 37.5 |
| DT3 | Weight | -0.056 | 4.0 | 23.7 | 72.3 |
| DT3 | Cover | -0.019 | 26.7 | 40.9 | 32.5 |
| DT4 | Gain | -0.031 | 13.9 | 40.6 | 45.5 |
| DT4 | Weight | -0.063 | 13.1 | 29.9 | 57.0 |
| DT4 | Cover | -0.040 | 16.1 | 37.8 | 46.0 |
| DT5 | Gain | -0.042 | 13.9 | 34.1 | 52.0 |
| DT5 | Weight | -0.076 | 13.1 | 17.0 | 69.8 |
| DT5 | Cover | -0.054 | 12.9 | 32.4 | 54.7 |
| DT6 | Gain | -0.046 | 13.7 | 39.2 | 47.1 |
| DT6 | Weight | -0.071 | 12.8 | 25.7 | 61.5 |
| DT6 | Cover | -0.064 | 10.7 | 40.2 | 49.1 |

## Per-dataset mean paired differences

Values below report the mean native-minus-SHAP difference across all matched fold and subset-size comparisons for the indicated dataset and model. Positive values favour the native method.

### LR

| Dataset | Gain − SHAP | Weight − SHAP | Cover − SHAP |
| --- | ---: | ---: | ---: |
| Adult Income | +0.037 | -0.069 | +0.072 |
| Bank Marketing | +0.051 | -0.046 | +0.031 |
| Breast Cancer Wisconsin | -0.013 | -0.030 | -0.013 |
| CIC-IDS2017 | -0.019 | -0.071 | -0.048 |
| Covertype | -0.033 | -0.026 | -0.228 |
| Credit Card Fraud | +0.008 | -0.052 | +0.001 |
| Dry Bean | -0.045 | -0.237 | -0.001 |
| Steel Plates Faults | +0.034 | -0.094 | +0.077 |
| UNSW-NB15 | +0.001 | -0.045 | +0.014 |

### XGBoost

| Dataset | Gain − SHAP | Weight − SHAP | Cover − SHAP |
| --- | ---: | ---: | ---: |
| Adult Income | -0.046 | -0.085 | +0.012 |
| Bank Marketing | +0.009 | -0.067 | -0.012 |
| Breast Cancer Wisconsin | -0.010 | -0.025 | -0.011 |
| CIC-IDS2017 | -0.076 | +0.027 | -0.134 |
| Covertype | -0.245 | +0.005 | -0.454 |
| Credit Card Fraud | +0.011 | -0.031 | +0.004 |
| Dry Bean | -0.043 | -0.214 | -0.001 |
| Steel Plates Faults | -0.192 | -0.067 | -0.134 |
| UNSW-NB15 | -0.074 | +0.015 | -0.083 |

### DT3

| Dataset | Gain − SHAP | Weight − SHAP | Cover − SHAP |
| --- | ---: | ---: | ---: |
| Adult Income | +0.019 | -0.120 | +0.072 |
| Bank Marketing | +0.038 | -0.050 | +0.013 |
| Breast Cancer Wisconsin | -0.005 | -0.030 | -0.008 |
| CIC-IDS2017 | -0.034 | -0.057 | -0.044 |
| Covertype | -0.061 | -0.047 | -0.235 |
| Credit Card Fraud | +0.007 | -0.053 | +0.008 |
| Dry Bean | -0.006 | -0.046 | +0.028 |
| Steel Plates Faults | -0.018 | -0.074 | +0.015 |
| UNSW-NB15 | -0.018 | -0.023 | -0.022 |

### DT4

| Dataset | Gain − SHAP | Weight − SHAP | Cover − SHAP |
| --- | ---: | ---: | ---: |
| Adult Income | -0.004 | -0.097 | +0.050 |
| Bank Marketing | +0.037 | -0.044 | +0.012 |
| Breast Cancer Wisconsin | -0.009 | -0.030 | -0.011 |
| CIC-IDS2017 | -0.060 | -0.066 | -0.069 |
| Covertype | -0.064 | -0.019 | -0.232 |
| Credit Card Fraud | +0.004 | -0.058 | +0.009 |
| Dry Bean | -0.028 | -0.167 | +0.009 |
| Steel Plates Faults | -0.110 | -0.085 | -0.076 |
| UNSW-NB15 | -0.051 | -0.004 | -0.047 |

### DT5

| Dataset | Gain − SHAP | Weight − SHAP | Cover − SHAP |
| --- | ---: | ---: | ---: |
| Adult Income | -0.060 | -0.141 | -0.005 |
| Bank Marketing | +0.029 | -0.051 | +0.003 |
| Breast Cancer Wisconsin | -0.014 | -0.026 | -0.016 |
| CIC-IDS2017 | -0.066 | -0.071 | -0.093 |
| Covertype | -0.055 | -0.027 | -0.251 |
| Credit Card Fraud | +0.008 | -0.051 | +0.010 |
| Dry Bean | -0.038 | -0.236 | +0.002 |
| Steel Plates Faults | -0.138 | -0.069 | -0.089 |
| UNSW-NB15 | -0.048 | -0.015 | -0.048 |

### DT6

| Dataset | Gain − SHAP | Weight − SHAP | Cover − SHAP |
| --- | ---: | ---: | ---: |
| Adult Income | -0.054 | -0.120 | +0.002 |
| Bank Marketing | +0.028 | -0.051 | +0.002 |
| Breast Cancer Wisconsin | -0.017 | -0.032 | -0.015 |
| CIC-IDS2017 | -0.070 | -0.060 | -0.150 |
| Covertype | -0.068 | -0.017 | -0.283 |
| Credit Card Fraud | +0.007 | -0.056 | +0.007 |
| Dry Bean | -0.040 | -0.221 | +0.000 |
| Steel Plates Faults | -0.150 | -0.071 | -0.084 |
| UNSW-NB15 | -0.052 | -0.013 | -0.054 |

## Detailed condition-level paired differences

For reproducibility and fold-wise variability reporting, each entry below corresponds to one fixed dataset/model/k condition. Values are the mean ± sample SD of the five matched fold-wise differences, where Δ = native method − SHAP.

### LR

| Dataset | k | Method | Mean Δ | Fold SD | Outcome |
| --- | ---: | --- | ---: | ---: | --- |
| Adult Income | 1 | Cover | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Gain | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Weight | +0.000 | 0.000 | Tie |
| Adult Income | 2 | Cover | +0.160 | 0.006 | Win |
| Adult Income | 2 | Gain | -0.008 | 0.001 | Tie |
| Adult Income | 2 | Weight | +0.001 | 0.001 | Tie |
| Adult Income | 3 | Cover | +0.173 | 0.015 | Win |
| Adult Income | 3 | Gain | +0.166 | 0.039 | Win |
| Adult Income | 3 | Weight | -0.009 | 0.003 | Tie |
| Adult Income | 4 | Cover | +0.046 | 0.031 | Win |
| Adult Income | 4 | Gain | +0.046 | 0.031 | Win |
| Adult Income | 4 | Weight | -0.169 | 0.008 | Loss |
| Adult Income | 5 | Cover | -0.017 | 0.002 | Loss |
| Adult Income | 5 | Gain | -0.017 | 0.002 | Loss |
| Adult Income | 5 | Weight | -0.166 | 0.006 | Loss |
| Bank Marketing | 1 | Cover | +0.022 | 0.017 | Win |
| Bank Marketing | 1 | Gain | +0.022 | 0.017 | Win |
| Bank Marketing | 1 | Weight | -0.097 | 0.054 | Loss |
| Bank Marketing | 2 | Cover | +0.021 | 0.016 | Win |
| Bank Marketing | 2 | Gain | +0.079 | 0.007 | Win |
| Bank Marketing | 2 | Weight | -0.023 | 0.058 | Loss |
| Bank Marketing | 3 | Cover | +0.060 | 0.043 | Win |
| Bank Marketing | 3 | Gain | +0.081 | 0.003 | Win |
| Bank Marketing | 3 | Weight | -0.024 | 0.058 | Loss |
| Bank Marketing | 4 | Cover | +0.054 | 0.047 | Win |
| Bank Marketing | 4 | Gain | +0.076 | 0.010 | Win |
| Bank Marketing | 4 | Weight | -0.006 | 0.008 | Tie |
| Bank Marketing | 5 | Cover | +0.000 | 0.001 | Tie |
| Bank Marketing | 5 | Gain | +0.000 | 0.000 | Tie |
| Bank Marketing | 5 | Weight | -0.080 | 0.008 | Loss |
| Breast Cancer Wisconsin | 1 | Cover | -0.029 | 0.075 | Loss |
| Breast Cancer Wisconsin | 1 | Gain | -0.008 | 0.022 | Tie |
| Breast Cancer Wisconsin | 1 | Weight | -0.229 | 0.044 | Loss |
| Breast Cancer Wisconsin | 2 | Cover | +0.000 | 0.018 | Tie |
| Breast Cancer Wisconsin | 2 | Gain | -0.004 | 0.021 | Tie |
| Breast Cancer Wisconsin | 2 | Weight | -0.053 | 0.097 | Loss |
| Breast Cancer Wisconsin | 3 | Cover | -0.006 | 0.015 | Tie |
| Breast Cancer Wisconsin | 3 | Gain | -0.010 | 0.010 | Tie |
| Breast Cancer Wisconsin | 3 | Weight | -0.003 | 0.037 | Tie |
| Breast Cancer Wisconsin | 4 | Cover | +0.003 | 0.020 | Tie |
| Breast Cancer Wisconsin | 4 | Gain | -0.004 | 0.009 | Tie |
| Breast Cancer Wisconsin | 4 | Weight | +0.021 | 0.014 | Win |
| Breast Cancer Wisconsin | 5 | Cover | -0.017 | 0.023 | Loss |
| Breast Cancer Wisconsin | 5 | Gain | -0.019 | 0.027 | Loss |
| Breast Cancer Wisconsin | 5 | Weight | +0.004 | 0.017 | Tie |
| Breast Cancer Wisconsin | 6 | Cover | -0.016 | 0.023 | Loss |
| Breast Cancer Wisconsin | 6 | Gain | -0.012 | 0.023 | Loss |
| Breast Cancer Wisconsin | 6 | Weight | +0.013 | 0.008 | Win |
| Breast Cancer Wisconsin | 8 | Cover | -0.019 | 0.024 | Loss |
| Breast Cancer Wisconsin | 8 | Gain | -0.021 | 0.012 | Loss |
| Breast Cancer Wisconsin | 8 | Weight | -0.000 | 0.007 | Tie |
| Breast Cancer Wisconsin | 9 | Cover | -0.019 | 0.026 | Loss |
| Breast Cancer Wisconsin | 9 | Gain | -0.023 | 0.014 | Loss |
| Breast Cancer Wisconsin | 9 | Weight | +0.006 | 0.009 | Tie |
| CIC-IDS2017 | 2 | Cover | -0.003 | 0.007 | Tie |
| CIC-IDS2017 | 2 | Gain | +0.057 | 0.012 | Win |
| CIC-IDS2017 | 2 | Weight | -0.003 | 0.007 | Tie |
| CIC-IDS2017 | 4 | Cover | -0.031 | 0.006 | Loss |
| CIC-IDS2017 | 4 | Gain | -0.036 | 0.009 | Loss |
| CIC-IDS2017 | 4 | Weight | -0.106 | 0.006 | Loss |
| CIC-IDS2017 | 5 | Cover | -0.030 | 0.006 | Loss |
| CIC-IDS2017 | 5 | Gain | -0.034 | 0.013 | Loss |
| CIC-IDS2017 | 5 | Weight | -0.107 | 0.006 | Loss |
| CIC-IDS2017 | 7 | Cover | -0.028 | 0.039 | Loss |
| CIC-IDS2017 | 7 | Gain | -0.007 | 0.021 | Tie |
| CIC-IDS2017 | 7 | Weight | -0.074 | 0.027 | Loss |
| CIC-IDS2017 | 9 | Cover | -0.044 | 0.019 | Loss |
| CIC-IDS2017 | 9 | Gain | -0.038 | 0.014 | Loss |
| CIC-IDS2017 | 9 | Weight | -0.115 | 0.008 | Loss |
| CIC-IDS2017 | 10 | Cover | -0.055 | 0.019 | Loss |
| CIC-IDS2017 | 10 | Gain | -0.041 | 0.017 | Loss |
| CIC-IDS2017 | 10 | Weight | -0.088 | 0.017 | Loss |
| CIC-IDS2017 | 13 | Cover | -0.066 | 0.025 | Loss |
| CIC-IDS2017 | 13 | Gain | +0.000 | 0.033 | Tie |
| CIC-IDS2017 | 13 | Weight | -0.081 | 0.032 | Loss |
| CIC-IDS2017 | 17 | Cover | -0.090 | 0.025 | Loss |
| CIC-IDS2017 | 17 | Gain | -0.025 | 0.023 | Loss |
| CIC-IDS2017 | 17 | Weight | -0.019 | 0.032 | Loss |
| CIC-IDS2017 | 20 | Cover | -0.085 | 0.059 | Loss |
| CIC-IDS2017 | 20 | Gain | -0.042 | 0.037 | Loss |
| CIC-IDS2017 | 20 | Weight | -0.046 | 0.037 | Loss |
| Covertype | 2 | Cover | -0.247 | 0.001 | Loss |
| Covertype | 2 | Gain | -0.014 | 0.001 | Loss |
| Covertype | 2 | Weight | -0.180 | 0.002 | Loss |
| Covertype | 3 | Cover | -0.246 | 0.014 | Loss |
| Covertype | 3 | Gain | -0.014 | 0.006 | Loss |
| Covertype | 3 | Weight | -0.017 | 0.001 | Loss |
| Covertype | 5 | Cover | -0.244 | 0.014 | Loss |
| Covertype | 5 | Gain | -0.018 | 0.003 | Loss |
| Covertype | 5 | Weight | -0.018 | 0.004 | Loss |
| Covertype | 6 | Cover | -0.226 | 0.047 | Loss |
| Covertype | 6 | Gain | -0.030 | 0.012 | Loss |
| Covertype | 6 | Weight | -0.035 | 0.003 | Loss |
| Covertype | 7 | Cover | -0.216 | 0.048 | Loss |
| Covertype | 7 | Gain | -0.028 | 0.015 | Loss |
| Covertype | 7 | Weight | -0.010 | 0.006 | Loss |
| Covertype | 9 | Cover | -0.221 | 0.006 | Loss |
| Covertype | 9 | Gain | -0.054 | 0.013 | Loss |
| Covertype | 9 | Weight | -0.017 | 0.003 | Loss |
| Covertype | 11 | Cover | -0.218 | 0.006 | Loss |
| Covertype | 11 | Gain | -0.038 | 0.004 | Loss |
| Covertype | 11 | Weight | +0.022 | 0.002 | Win |
| Covertype | 14 | Cover | -0.229 | 0.006 | Loss |
| Covertype | 14 | Gain | -0.050 | 0.005 | Loss |
| Covertype | 14 | Weight | +0.006 | 0.003 | Tie |
| Covertype | 17 | Cover | -0.202 | 0.060 | Loss |
| Covertype | 17 | Gain | -0.055 | 0.006 | Loss |
| Covertype | 17 | Weight | +0.012 | 0.003 | Win |
| Credit Card Fraud | 1 | Cover | -0.036 | 0.036 | Loss |
| Credit Card Fraud | 1 | Gain | +0.000 | 0.000 | Tie |
| Credit Card Fraud | 1 | Weight | -0.206 | 0.026 | Loss |
| Credit Card Fraud | 2 | Cover | +0.035 | 0.010 | Win |
| Credit Card Fraud | 2 | Gain | +0.035 | 0.014 | Win |
| Credit Card Fraud | 2 | Weight | -0.124 | 0.113 | Loss |
| Credit Card Fraud | 3 | Cover | +0.008 | 0.010 | Tie |
| Credit Card Fraud | 3 | Gain | +0.012 | 0.007 | Win |
| Credit Card Fraud | 3 | Weight | -0.037 | 0.014 | Loss |
| Credit Card Fraud | 4 | Cover | +0.001 | 0.002 | Tie |
| Credit Card Fraud | 4 | Gain | +0.004 | 0.006 | Tie |
| Credit Card Fraud | 4 | Weight | -0.026 | 0.024 | Loss |
| Credit Card Fraud | 5 | Cover | -0.000 | 0.002 | Tie |
| Credit Card Fraud | 5 | Gain | -0.000 | 0.005 | Tie |
| Credit Card Fraud | 5 | Weight | -0.015 | 0.016 | Loss |
| Credit Card Fraud | 6 | Cover | +0.002 | 0.002 | Tie |
| Credit Card Fraud | 6 | Gain | +0.002 | 0.005 | Tie |
| Credit Card Fraud | 6 | Weight | -0.005 | 0.005 | Tie |
| Credit Card Fraud | 8 | Cover | +0.000 | 0.007 | Tie |
| Credit Card Fraud | 8 | Gain | +0.005 | 0.005 | Tie |
| Credit Card Fraud | 8 | Weight | -0.003 | 0.004 | Tie |
| Credit Card Fraud | 9 | Cover | -0.001 | 0.010 | Tie |
| Credit Card Fraud | 9 | Gain | +0.003 | 0.010 | Tie |
| Credit Card Fraud | 9 | Weight | +0.001 | 0.005 | Tie |
| Dry Bean | 1 | Cover | +0.132 | 0.006 | Win |
| Dry Bean | 1 | Gain | +0.052 | 0.071 | Win |
| Dry Bean | 1 | Weight | -0.207 | 0.009 | Loss |
| Dry Bean | 2 | Cover | -0.108 | 0.149 | Loss |
| Dry Bean | 2 | Gain | -0.246 | 0.226 | Loss |
| Dry Bean | 2 | Weight | -0.376 | 0.019 | Loss |
| Dry Bean | 3 | Cover | -0.001 | 0.001 | Tie |
| Dry Bean | 3 | Gain | -0.001 | 0.002 | Tie |
| Dry Bean | 3 | Weight | -0.303 | 0.010 | Loss |
| Dry Bean | 4 | Cover | +0.000 | 0.001 | Tie |
| Dry Bean | 4 | Gain | -0.000 | 0.001 | Tie |
| Dry Bean | 4 | Weight | -0.294 | 0.016 | Loss |
| Dry Bean | 5 | Cover | -0.028 | 0.015 | Loss |
| Dry Bean | 5 | Gain | -0.028 | 0.016 | Loss |
| Dry Bean | 5 | Weight | -0.006 | 0.017 | Tie |
| Steel Plates Faults | 1 | Cover | +0.115 | 0.090 | Win |
| Steel Plates Faults | 1 | Gain | +0.031 | 0.072 | Win |
| Steel Plates Faults | 1 | Weight | -0.080 | 0.003 | Loss |
| Steel Plates Faults | 2 | Cover | +0.178 | 0.022 | Win |
| Steel Plates Faults | 2 | Gain | +0.073 | 0.098 | Win |
| Steel Plates Faults | 2 | Weight | -0.098 | 0.016 | Loss |
| Steel Plates Faults | 3 | Cover | +0.160 | 0.034 | Win |
| Steel Plates Faults | 3 | Gain | +0.033 | 0.023 | Win |
| Steel Plates Faults | 3 | Weight | -0.094 | 0.016 | Loss |
| Steel Plates Faults | 4 | Cover | +0.113 | 0.047 | Win |
| Steel Plates Faults | 4 | Gain | +0.112 | 0.045 | Win |
| Steel Plates Faults | 4 | Weight | -0.056 | 0.053 | Loss |
| Steel Plates Faults | 5 | Cover | +0.064 | 0.021 | Win |
| Steel Plates Faults | 5 | Gain | +0.060 | 0.016 | Win |
| Steel Plates Faults | 5 | Weight | -0.076 | 0.025 | Loss |
| Steel Plates Faults | 6 | Cover | +0.037 | 0.031 | Win |
| Steel Plates Faults | 6 | Gain | +0.032 | 0.021 | Win |
| Steel Plates Faults | 6 | Weight | -0.108 | 0.018 | Loss |
| Steel Plates Faults | 7 | Cover | +0.007 | 0.045 | Tie |
| Steel Plates Faults | 7 | Gain | -0.011 | 0.035 | Loss |
| Steel Plates Faults | 7 | Weight | -0.135 | 0.036 | Loss |
| Steel Plates Faults | 9 | Cover | -0.056 | 0.039 | Loss |
| Steel Plates Faults | 9 | Gain | -0.060 | 0.045 | Loss |
| Steel Plates Faults | 9 | Weight | -0.104 | 0.041 | Loss |
| UNSW-NB15 | 1 | Cover | -0.026 | 0.036 | Loss |
| UNSW-NB15 | 1 | Gain | +0.101 | 0.037 | Win |
| UNSW-NB15 | 1 | Weight | +0.033 | 0.036 | Win |
| UNSW-NB15 | 2 | Cover | -0.006 | 0.063 | Tie |
| UNSW-NB15 | 2 | Gain | +0.038 | 0.004 | Win |
| UNSW-NB15 | 2 | Weight | -0.088 | 0.005 | Loss |
| UNSW-NB15 | 3 | Cover | +0.064 | 0.045 | Win |
| UNSW-NB15 | 3 | Gain | -0.028 | 0.008 | Loss |
| UNSW-NB15 | 3 | Weight | -0.089 | 0.034 | Loss |
| UNSW-NB15 | 4 | Cover | +0.060 | 0.034 | Win |
| UNSW-NB15 | 4 | Gain | -0.019 | 0.001 | Loss |
| UNSW-NB15 | 4 | Weight | -0.036 | 0.001 | Loss |
| UNSW-NB15 | 5 | Cover | -0.008 | 0.028 | Tie |
| UNSW-NB15 | 5 | Gain | -0.057 | 0.018 | Loss |
| UNSW-NB15 | 5 | Weight | -0.099 | 0.012 | Loss |
| UNSW-NB15 | 6 | Cover | -0.008 | 0.033 | Tie |
| UNSW-NB15 | 6 | Gain | -0.045 | 0.001 | Loss |
| UNSW-NB15 | 6 | Weight | -0.090 | 0.001 | Loss |
| UNSW-NB15 | 8 | Cover | +0.006 | 0.022 | Tie |
| UNSW-NB15 | 8 | Gain | -0.014 | 0.004 | Loss |
| UNSW-NB15 | 8 | Weight | -0.021 | 0.003 | Loss |
| UNSW-NB15 | 10 | Cover | +0.021 | 0.015 | Win |
| UNSW-NB15 | 10 | Gain | +0.015 | 0.033 | Win |
| UNSW-NB15 | 10 | Weight | -0.016 | 0.002 | Loss |
| UNSW-NB15 | 12 | Cover | +0.020 | 0.007 | Win |
| UNSW-NB15 | 12 | Gain | +0.017 | 0.014 | Win |
| UNSW-NB15 | 12 | Weight | +0.001 | 0.006 | Tie |

### XGBoost

| Dataset | k | Method | Mean Δ | Fold SD | Outcome |
| --- | ---: | --- | ---: | ---: | --- |
| Adult Income | 1 | Cover | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Gain | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Weight | +0.000 | 0.000 | Tie |
| Adult Income | 2 | Cover | -0.035 | 0.007 | Loss |
| Adult Income | 2 | Gain | -0.222 | 0.009 | Loss |
| Adult Income | 2 | Weight | -0.222 | 0.008 | Loss |
| Adult Income | 3 | Cover | +0.062 | 0.069 | Win |
| Adult Income | 3 | Gain | -0.038 | 0.043 | Loss |
| Adult Income | 3 | Weight | -0.030 | 0.006 | Loss |
| Adult Income | 4 | Cover | +0.026 | 0.003 | Win |
| Adult Income | 4 | Gain | +0.026 | 0.003 | Win |
| Adult Income | 4 | Weight | -0.078 | 0.005 | Loss |
| Adult Income | 5 | Cover | +0.006 | 0.003 | Tie |
| Adult Income | 5 | Gain | +0.006 | 0.003 | Tie |
| Adult Income | 5 | Weight | -0.094 | 0.007 | Loss |
| Bank Marketing | 1 | Cover | +0.004 | 0.020 | Tie |
| Bank Marketing | 1 | Gain | +0.004 | 0.020 | Tie |
| Bank Marketing | 1 | Weight | -0.112 | 0.063 | Loss |
| Bank Marketing | 2 | Cover | +0.012 | 0.022 | Win |
| Bank Marketing | 2 | Gain | +0.043 | 0.027 | Win |
| Bank Marketing | 2 | Weight | -0.035 | 0.052 | Loss |
| Bank Marketing | 3 | Cover | -0.009 | 0.036 | Tie |
| Bank Marketing | 3 | Gain | +0.007 | 0.011 | Tie |
| Bank Marketing | 3 | Weight | -0.074 | 0.053 | Loss |
| Bank Marketing | 4 | Cover | -0.026 | 0.041 | Loss |
| Bank Marketing | 4 | Gain | -0.011 | 0.012 | Loss |
| Bank Marketing | 4 | Weight | -0.070 | 0.016 | Loss |
| Bank Marketing | 5 | Cover | -0.043 | 0.007 | Loss |
| Bank Marketing | 5 | Gain | +0.003 | 0.001 | Tie |
| Bank Marketing | 5 | Weight | -0.042 | 0.008 | Loss |
| Breast Cancer Wisconsin | 1 | Cover | -0.018 | 0.042 | Loss |
| Breast Cancer Wisconsin | 1 | Gain | -0.008 | 0.039 | Tie |
| Breast Cancer Wisconsin | 1 | Weight | -0.210 | 0.094 | Loss |
| Breast Cancer Wisconsin | 2 | Cover | -0.000 | 0.007 | Tie |
| Breast Cancer Wisconsin | 2 | Gain | -0.008 | 0.011 | Tie |
| Breast Cancer Wisconsin | 2 | Weight | -0.036 | 0.112 | Loss |
| Breast Cancer Wisconsin | 3 | Cover | -0.010 | 0.025 | Tie |
| Breast Cancer Wisconsin | 3 | Gain | -0.010 | 0.012 | Tie |
| Breast Cancer Wisconsin | 3 | Weight | -0.005 | 0.039 | Tie |
| Breast Cancer Wisconsin | 4 | Cover | -0.008 | 0.020 | Tie |
| Breast Cancer Wisconsin | 4 | Gain | -0.012 | 0.026 | Loss |
| Breast Cancer Wisconsin | 4 | Weight | +0.019 | 0.017 | Win |
| Breast Cancer Wisconsin | 5 | Cover | -0.013 | 0.019 | Loss |
| Breast Cancer Wisconsin | 5 | Gain | -0.011 | 0.028 | Loss |
| Breast Cancer Wisconsin | 5 | Weight | +0.002 | 0.010 | Tie |
| Breast Cancer Wisconsin | 6 | Cover | -0.004 | 0.019 | Tie |
| Breast Cancer Wisconsin | 6 | Gain | -0.004 | 0.024 | Tie |
| Breast Cancer Wisconsin | 6 | Weight | +0.014 | 0.016 | Win |
| Breast Cancer Wisconsin | 8 | Cover | -0.014 | 0.028 | Loss |
| Breast Cancer Wisconsin | 8 | Gain | -0.008 | 0.012 | Tie |
| Breast Cancer Wisconsin | 8 | Weight | +0.007 | 0.008 | Tie |
| Breast Cancer Wisconsin | 9 | Cover | -0.023 | 0.017 | Loss |
| Breast Cancer Wisconsin | 9 | Gain | -0.017 | 0.027 | Loss |
| Breast Cancer Wisconsin | 9 | Weight | +0.008 | 0.014 | Tie |
| CIC-IDS2017 | 2 | Cover | -0.245 | 0.016 | Loss |
| CIC-IDS2017 | 2 | Gain | -0.107 | 0.035 | Loss |
| CIC-IDS2017 | 2 | Weight | -0.026 | 0.002 | Loss |
| CIC-IDS2017 | 4 | Cover | -0.152 | 0.028 | Loss |
| CIC-IDS2017 | 4 | Gain | -0.111 | 0.016 | Loss |
| CIC-IDS2017 | 4 | Weight | +0.011 | 0.029 | Win |
| CIC-IDS2017 | 5 | Cover | -0.148 | 0.054 | Loss |
| CIC-IDS2017 | 5 | Gain | -0.103 | 0.029 | Loss |
| CIC-IDS2017 | 5 | Weight | -0.015 | 0.027 | Loss |
| CIC-IDS2017 | 7 | Cover | -0.121 | 0.014 | Loss |
| CIC-IDS2017 | 7 | Gain | -0.047 | 0.043 | Loss |
| CIC-IDS2017 | 7 | Weight | +0.072 | 0.020 | Win |
| CIC-IDS2017 | 9 | Cover | -0.125 | 0.029 | Loss |
| CIC-IDS2017 | 9 | Gain | -0.063 | 0.026 | Loss |
| CIC-IDS2017 | 9 | Weight | +0.053 | 0.022 | Win |
| CIC-IDS2017 | 10 | Cover | -0.103 | 0.012 | Loss |
| CIC-IDS2017 | 10 | Gain | -0.048 | 0.026 | Loss |
| CIC-IDS2017 | 10 | Weight | +0.052 | 0.021 | Win |
| CIC-IDS2017 | 13 | Cover | -0.082 | 0.015 | Loss |
| CIC-IDS2017 | 13 | Gain | -0.047 | 0.009 | Loss |
| CIC-IDS2017 | 13 | Weight | +0.062 | 0.012 | Win |
| CIC-IDS2017 | 17 | Cover | -0.100 | 0.023 | Loss |
| CIC-IDS2017 | 17 | Gain | -0.063 | 0.008 | Loss |
| CIC-IDS2017 | 17 | Weight | +0.041 | 0.006 | Win |
| CIC-IDS2017 | 20 | Cover | -0.130 | 0.011 | Loss |
| CIC-IDS2017 | 20 | Gain | -0.097 | 0.016 | Loss |
| CIC-IDS2017 | 20 | Weight | -0.004 | 0.008 | Tie |
| Covertype | 2 | Cover | -0.235 | 0.004 | Loss |
| Covertype | 2 | Gain | -0.012 | 0.004 | Loss |
| Covertype | 2 | Weight | -0.165 | 0.004 | Loss |
| Covertype | 3 | Cover | -0.323 | 0.012 | Loss |
| Covertype | 3 | Gain | -0.100 | 0.005 | Loss |
| Covertype | 3 | Weight | +0.066 | 0.005 | Win |
| Covertype | 5 | Cover | -0.520 | 0.016 | Loss |
| Covertype | 5 | Gain | -0.296 | 0.009 | Loss |
| Covertype | 5 | Weight | +0.008 | 0.003 | Tie |
| Covertype | 6 | Cover | -0.465 | 0.047 | Loss |
| Covertype | 6 | Gain | -0.267 | 0.011 | Loss |
| Covertype | 6 | Weight | +0.076 | 0.003 | Win |
| Covertype | 7 | Cover | -0.467 | 0.047 | Loss |
| Covertype | 7 | Gain | -0.271 | 0.008 | Loss |
| Covertype | 7 | Weight | +0.078 | 0.004 | Win |
| Covertype | 9 | Cover | -0.532 | 0.005 | Loss |
| Covertype | 9 | Gain | -0.357 | 0.014 | Loss |
| Covertype | 9 | Weight | -0.010 | 0.003 | Tie |
| Covertype | 11 | Cover | -0.530 | 0.005 | Loss |
| Covertype | 11 | Gain | -0.336 | 0.001 | Loss |
| Covertype | 11 | Weight | +0.001 | 0.002 | Tie |
| Covertype | 14 | Cover | -0.526 | 0.007 | Loss |
| Covertype | 14 | Gain | -0.285 | 0.004 | Loss |
| Covertype | 14 | Weight | -0.004 | 0.003 | Tie |
| Covertype | 17 | Cover | -0.486 | 0.061 | Loss |
| Covertype | 17 | Gain | -0.279 | 0.005 | Loss |
| Covertype | 17 | Weight | -0.006 | 0.003 | Tie |
| Credit Card Fraud | 1 | Cover | +0.000 | 0.000 | Tie |
| Credit Card Fraud | 1 | Gain | +0.000 | 0.000 | Tie |
| Credit Card Fraud | 1 | Weight | +0.000 | 0.000 | Tie |
| Credit Card Fraud | 2 | Cover | +0.071 | 0.029 | Win |
| Credit Card Fraud | 2 | Gain | +0.126 | 0.021 | Win |
| Credit Card Fraud | 2 | Weight | -0.158 | 0.147 | Loss |
| Credit Card Fraud | 3 | Cover | -0.008 | 0.030 | Tie |
| Credit Card Fraud | 3 | Gain | +0.016 | 0.014 | Win |
| Credit Card Fraud | 3 | Weight | -0.052 | 0.033 | Loss |
| Credit Card Fraud | 4 | Cover | -0.008 | 0.012 | Tie |
| Credit Card Fraud | 4 | Gain | -0.012 | 0.005 | Loss |
| Credit Card Fraud | 4 | Weight | -0.042 | 0.050 | Loss |
| Credit Card Fraud | 5 | Cover | +0.004 | 0.006 | Tie |
| Credit Card Fraud | 5 | Gain | -0.007 | 0.007 | Tie |
| Credit Card Fraud | 5 | Weight | +0.001 | 0.007 | Tie |
| Credit Card Fraud | 6 | Cover | -0.004 | 0.015 | Tie |
| Credit Card Fraud | 6 | Gain | -0.011 | 0.018 | Loss |
| Credit Card Fraud | 6 | Weight | +0.002 | 0.012 | Tie |
| Credit Card Fraud | 8 | Cover | -0.013 | 0.008 | Loss |
| Credit Card Fraud | 8 | Gain | -0.012 | 0.010 | Loss |
| Credit Card Fraud | 8 | Weight | -0.003 | 0.004 | Tie |
| Credit Card Fraud | 9 | Cover | -0.007 | 0.009 | Tie |
| Credit Card Fraud | 9 | Gain | -0.012 | 0.006 | Loss |
| Credit Card Fraud | 9 | Weight | +0.001 | 0.004 | Tie |
| Dry Bean | 1 | Cover | +0.124 | 0.011 | Win |
| Dry Bean | 1 | Gain | +0.051 | 0.070 | Win |
| Dry Bean | 1 | Weight | -0.193 | 0.006 | Loss |
| Dry Bean | 2 | Cover | -0.108 | 0.151 | Loss |
| Dry Bean | 2 | Gain | -0.237 | 0.215 | Loss |
| Dry Bean | 2 | Weight | -0.359 | 0.008 | Loss |
| Dry Bean | 3 | Cover | +0.002 | 0.004 | Tie |
| Dry Bean | 3 | Gain | -0.001 | 0.004 | Tie |
| Dry Bean | 3 | Weight | -0.277 | 0.006 | Loss |
| Dry Bean | 4 | Cover | -0.001 | 0.002 | Tie |
| Dry Bean | 4 | Gain | -0.001 | 0.002 | Tie |
| Dry Bean | 4 | Weight | -0.247 | 0.013 | Loss |
| Dry Bean | 5 | Cover | -0.025 | 0.013 | Loss |
| Dry Bean | 5 | Gain | -0.027 | 0.013 | Loss |
| Dry Bean | 5 | Weight | +0.004 | 0.020 | Tie |
| Steel Plates Faults | 1 | Cover | -0.152 | 0.114 | Loss |
| Steel Plates Faults | 1 | Gain | -0.249 | 0.088 | Loss |
| Steel Plates Faults | 1 | Weight | -0.300 | 0.018 | Loss |
| Steel Plates Faults | 2 | Cover | -0.212 | 0.039 | Loss |
| Steel Plates Faults | 2 | Gain | -0.332 | 0.077 | Loss |
| Steel Plates Faults | 2 | Weight | -0.204 | 0.039 | Loss |
| Steel Plates Faults | 3 | Cover | -0.214 | 0.041 | Loss |
| Steel Plates Faults | 3 | Gain | -0.375 | 0.031 | Loss |
| Steel Plates Faults | 3 | Weight | -0.065 | 0.036 | Loss |
| Steel Plates Faults | 4 | Cover | -0.169 | 0.082 | Loss |
| Steel Plates Faults | 4 | Gain | -0.209 | 0.025 | Loss |
| Steel Plates Faults | 4 | Weight | +0.023 | 0.068 | Win |
| Steel Plates Faults | 5 | Cover | -0.085 | 0.028 | Loss |
| Steel Plates Faults | 5 | Gain | -0.133 | 0.065 | Loss |
| Steel Plates Faults | 5 | Weight | +0.016 | 0.018 | Win |
| Steel Plates Faults | 6 | Cover | -0.086 | 0.024 | Loss |
| Steel Plates Faults | 6 | Gain | -0.089 | 0.046 | Loss |
| Steel Plates Faults | 6 | Weight | -0.001 | 0.021 | Tie |
| Steel Plates Faults | 7 | Cover | -0.088 | 0.036 | Loss |
| Steel Plates Faults | 7 | Gain | -0.094 | 0.031 | Loss |
| Steel Plates Faults | 7 | Weight | -0.011 | 0.010 | Loss |
| Steel Plates Faults | 9 | Cover | -0.067 | 0.042 | Loss |
| Steel Plates Faults | 9 | Gain | -0.052 | 0.024 | Loss |
| Steel Plates Faults | 9 | Weight | +0.004 | 0.015 | Tie |
| UNSW-NB15 | 1 | Cover | -0.152 | 0.208 | Loss |
| UNSW-NB15 | 1 | Gain | -0.123 | 0.208 | Loss |
| UNSW-NB15 | 1 | Weight | +0.205 | 0.208 | Win |
| UNSW-NB15 | 2 | Cover | -0.234 | 0.079 | Loss |
| UNSW-NB15 | 2 | Gain | -0.175 | 0.002 | Loss |
| UNSW-NB15 | 2 | Weight | -0.025 | 0.002 | Loss |
| UNSW-NB15 | 3 | Cover | -0.179 | 0.036 | Loss |
| UNSW-NB15 | 3 | Gain | -0.212 | 0.002 | Loss |
| UNSW-NB15 | 3 | Weight | +0.020 | 0.003 | Win |
| UNSW-NB15 | 4 | Cover | -0.098 | 0.052 | Loss |
| UNSW-NB15 | 4 | Gain | -0.070 | 0.001 | Loss |
| UNSW-NB15 | 4 | Weight | -0.011 | 0.001 | Loss |
| UNSW-NB15 | 5 | Cover | -0.009 | 0.052 | Tie |
| UNSW-NB15 | 5 | Gain | -0.032 | 0.023 | Loss |
| UNSW-NB15 | 5 | Weight | -0.002 | 0.004 | Tie |
| UNSW-NB15 | 6 | Cover | -0.011 | 0.001 | Loss |
| UNSW-NB15 | 6 | Gain | -0.010 | 0.001 | Loss |
| UNSW-NB15 | 6 | Weight | -0.032 | 0.002 | Loss |
| UNSW-NB15 | 8 | Cover | -0.022 | 0.004 | Loss |
| UNSW-NB15 | 8 | Gain | -0.015 | 0.001 | Loss |
| UNSW-NB15 | 8 | Weight | -0.022 | 0.010 | Loss |
| UNSW-NB15 | 10 | Cover | -0.021 | 0.005 | Loss |
| UNSW-NB15 | 10 | Gain | -0.020 | 0.001 | Loss |
| UNSW-NB15 | 10 | Weight | +0.001 | 0.001 | Tie |
| UNSW-NB15 | 12 | Cover | -0.016 | 0.002 | Loss |
| UNSW-NB15 | 12 | Gain | -0.005 | 0.001 | Tie |
| UNSW-NB15 | 12 | Weight | +0.002 | 0.001 | Tie |

### DT3

| Dataset | k | Method | Mean Δ | Fold SD | Outcome |
| --- | ---: | --- | ---: | ---: | --- |
| Adult Income | 1 | Cover | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Gain | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Weight | +0.000 | 0.000 | Tie |
| Adult Income | 2 | Cover | +0.169 | 0.007 | Win |
| Adult Income | 2 | Gain | +0.000 | 0.000 | Tie |
| Adult Income | 2 | Weight | +0.000 | 0.000 | Tie |
| Adult Income | 3 | Cover | +0.152 | 0.168 | Win |
| Adult Income | 3 | Gain | +0.056 | 0.140 | Win |
| Adult Income | 3 | Weight | -0.135 | 0.123 | Loss |
| Adult Income | 4 | Cover | +0.038 | 0.002 | Win |
| Adult Income | 4 | Gain | +0.038 | 0.002 | Win |
| Adult Income | 4 | Weight | -0.235 | 0.092 | Loss |
| Adult Income | 5 | Cover | -0.000 | 0.000 | Tie |
| Adult Income | 5 | Gain | -0.000 | 0.000 | Tie |
| Adult Income | 5 | Weight | -0.233 | 0.004 | Loss |
| Bank Marketing | 1 | Cover | +0.003 | 0.019 | Tie |
| Bank Marketing | 1 | Gain | +0.003 | 0.019 | Tie |
| Bank Marketing | 1 | Weight | -0.111 | 0.063 | Loss |
| Bank Marketing | 2 | Cover | +0.003 | 0.019 | Tie |
| Bank Marketing | 2 | Gain | +0.089 | 0.006 | Win |
| Bank Marketing | 2 | Weight | -0.031 | 0.069 | Loss |
| Bank Marketing | 3 | Cover | +0.030 | 0.046 | Win |
| Bank Marketing | 3 | Gain | +0.049 | 0.004 | Win |
| Bank Marketing | 3 | Weight | -0.048 | 0.079 | Loss |
| Bank Marketing | 4 | Cover | +0.030 | 0.046 | Win |
| Bank Marketing | 4 | Gain | +0.049 | 0.004 | Win |
| Bank Marketing | 4 | Weight | -0.011 | 0.006 | Loss |
| Bank Marketing | 5 | Cover | +0.000 | 0.000 | Tie |
| Bank Marketing | 5 | Gain | +0.000 | 0.000 | Tie |
| Bank Marketing | 5 | Weight | -0.049 | 0.004 | Loss |
| Breast Cancer Wisconsin | 1 | Cover | -0.031 | 0.044 | Loss |
| Breast Cancer Wisconsin | 1 | Gain | -0.017 | 0.039 | Loss |
| Breast Cancer Wisconsin | 1 | Weight | -0.203 | 0.101 | Loss |
| Breast Cancer Wisconsin | 2 | Cover | -0.017 | 0.027 | Loss |
| Breast Cancer Wisconsin | 2 | Gain | -0.016 | 0.024 | Loss |
| Breast Cancer Wisconsin | 2 | Weight | -0.057 | 0.101 | Loss |
| Breast Cancer Wisconsin | 3 | Cover | -0.017 | 0.017 | Loss |
| Breast Cancer Wisconsin | 3 | Gain | -0.013 | 0.012 | Loss |
| Breast Cancer Wisconsin | 3 | Weight | -0.011 | 0.030 | Loss |
| Breast Cancer Wisconsin | 4 | Cover | -0.002 | 0.003 | Tie |
| Breast Cancer Wisconsin | 4 | Gain | -0.003 | 0.008 | Tie |
| Breast Cancer Wisconsin | 4 | Weight | +0.011 | 0.021 | Win |
| Breast Cancer Wisconsin | 5 | Cover | +0.005 | 0.009 | Tie |
| Breast Cancer Wisconsin | 5 | Gain | +0.007 | 0.010 | Tie |
| Breast Cancer Wisconsin | 5 | Weight | +0.008 | 0.008 | Tie |
| Breast Cancer Wisconsin | 6 | Cover | +0.007 | 0.015 | Tie |
| Breast Cancer Wisconsin | 6 | Gain | +0.011 | 0.015 | Win |
| Breast Cancer Wisconsin | 6 | Weight | +0.009 | 0.013 | Tie |
| Breast Cancer Wisconsin | 8 | Cover | -0.004 | 0.025 | Tie |
| Breast Cancer Wisconsin | 8 | Gain | -0.002 | 0.014 | Tie |
| Breast Cancer Wisconsin | 8 | Weight | +0.000 | 0.017 | Tie |
| Breast Cancer Wisconsin | 9 | Cover | -0.006 | 0.022 | Tie |
| Breast Cancer Wisconsin | 9 | Gain | -0.006 | 0.011 | Tie |
| Breast Cancer Wisconsin | 9 | Weight | +0.000 | 0.019 | Tie |
| CIC-IDS2017 | 2 | Cover | -0.057 | 0.001 | Loss |
| CIC-IDS2017 | 2 | Gain | -0.001 | 0.003 | Tie |
| CIC-IDS2017 | 2 | Weight | -0.045 | 0.001 | Loss |
| CIC-IDS2017 | 4 | Cover | -0.010 | 0.013 | Tie |
| CIC-IDS2017 | 4 | Gain | -0.051 | 0.000 | Loss |
| CIC-IDS2017 | 4 | Weight | -0.100 | 0.000 | Loss |
| CIC-IDS2017 | 5 | Cover | -0.013 | 0.010 | Loss |
| CIC-IDS2017 | 5 | Gain | -0.021 | 0.022 | Loss |
| CIC-IDS2017 | 5 | Weight | -0.105 | 0.006 | Loss |
| CIC-IDS2017 | 7 | Cover | -0.007 | 0.012 | Tie |
| CIC-IDS2017 | 7 | Gain | +0.008 | 0.006 | Tie |
| CIC-IDS2017 | 7 | Weight | -0.040 | 0.041 | Loss |
| CIC-IDS2017 | 9 | Cover | -0.022 | 0.011 | Loss |
| CIC-IDS2017 | 9 | Gain | -0.012 | 0.010 | Loss |
| CIC-IDS2017 | 9 | Weight | -0.023 | 0.012 | Loss |
| CIC-IDS2017 | 10 | Cover | -0.053 | 0.016 | Loss |
| CIC-IDS2017 | 10 | Gain | -0.047 | 0.008 | Loss |
| CIC-IDS2017 | 10 | Weight | -0.035 | 0.008 | Loss |
| CIC-IDS2017 | 13 | Cover | -0.094 | 0.015 | Loss |
| CIC-IDS2017 | 13 | Gain | -0.089 | 0.019 | Loss |
| CIC-IDS2017 | 13 | Weight | -0.077 | 0.034 | Loss |
| CIC-IDS2017 | 17 | Cover | -0.079 | 0.019 | Loss |
| CIC-IDS2017 | 17 | Gain | -0.054 | 0.000 | Loss |
| CIC-IDS2017 | 17 | Weight | -0.044 | 0.000 | Loss |
| CIC-IDS2017 | 20 | Cover | -0.064 | 0.032 | Loss |
| CIC-IDS2017 | 20 | Gain | -0.043 | 0.024 | Loss |
| CIC-IDS2017 | 20 | Weight | -0.044 | 0.000 | Loss |
| Covertype | 2 | Cover | -0.268 | 0.001 | Loss |
| Covertype | 2 | Gain | -0.032 | 0.037 | Loss |
| Covertype | 2 | Weight | -0.259 | 0.021 | Loss |
| Covertype | 3 | Cover | -0.261 | 0.013 | Loss |
| Covertype | 3 | Gain | -0.032 | 0.037 | Loss |
| Covertype | 3 | Weight | -0.033 | 0.000 | Loss |
| Covertype | 5 | Cover | -0.298 | 0.015 | Loss |
| Covertype | 5 | Gain | -0.087 | 0.039 | Loss |
| Covertype | 5 | Weight | -0.033 | 0.000 | Loss |
| Covertype | 6 | Cover | -0.257 | 0.048 | Loss |
| Covertype | 6 | Gain | -0.083 | 0.036 | Loss |
| Covertype | 6 | Weight | -0.033 | 0.000 | Loss |
| Covertype | 7 | Cover | -0.238 | 0.047 | Loss |
| Covertype | 7 | Gain | -0.081 | 0.037 | Loss |
| Covertype | 7 | Weight | -0.032 | 0.001 | Loss |
| Covertype | 9 | Cover | -0.204 | 0.002 | Loss |
| Covertype | 9 | Gain | -0.079 | 0.039 | Loss |
| Covertype | 9 | Weight | -0.032 | 0.001 | Loss |
| Covertype | 11 | Cover | -0.204 | 0.002 | Loss |
| Covertype | 11 | Gain | -0.048 | 0.002 | Loss |
| Covertype | 11 | Weight | +0.000 | 0.000 | Tie |
| Covertype | 14 | Cover | -0.210 | 0.003 | Loss |
| Covertype | 14 | Gain | -0.054 | 0.002 | Loss |
| Covertype | 14 | Weight | +0.000 | 0.000 | Tie |
| Covertype | 17 | Cover | -0.177 | 0.065 | Loss |
| Covertype | 17 | Gain | -0.055 | 0.002 | Loss |
| Covertype | 17 | Weight | +0.000 | 0.000 | Tie |
| Credit Card Fraud | 1 | Cover | +0.046 | 0.034 | Win |
| Credit Card Fraud | 1 | Gain | +0.000 | 0.000 | Tie |
| Credit Card Fraud | 1 | Weight | -0.298 | 0.035 | Loss |
| Credit Card Fraud | 2 | Cover | +0.013 | 0.016 | Win |
| Credit Card Fraud | 2 | Gain | +0.027 | 0.014 | Win |
| Credit Card Fraud | 2 | Weight | -0.116 | 0.104 | Loss |
| Credit Card Fraud | 3 | Cover | -0.003 | 0.012 | Tie |
| Credit Card Fraud | 3 | Gain | +0.014 | 0.013 | Win |
| Credit Card Fraud | 3 | Weight | -0.026 | 0.011 | Loss |
| Credit Card Fraud | 4 | Cover | -0.004 | 0.006 | Tie |
| Credit Card Fraud | 4 | Gain | +0.001 | 0.010 | Tie |
| Credit Card Fraud | 4 | Weight | -0.019 | 0.021 | Loss |
| Credit Card Fraud | 5 | Cover | +0.005 | 0.007 | Tie |
| Credit Card Fraud | 5 | Gain | +0.004 | 0.006 | Tie |
| Credit Card Fraud | 5 | Weight | +0.007 | 0.009 | Tie |
| Credit Card Fraud | 6 | Cover | +0.004 | 0.006 | Tie |
| Credit Card Fraud | 6 | Gain | +0.005 | 0.006 | Tie |
| Credit Card Fraud | 6 | Weight | +0.017 | 0.014 | Win |
| Credit Card Fraud | 8 | Cover | -0.000 | 0.001 | Tie |
| Credit Card Fraud | 8 | Gain | +0.002 | 0.004 | Tie |
| Credit Card Fraud | 8 | Weight | +0.003 | 0.008 | Tie |
| Credit Card Fraud | 9 | Cover | +0.001 | 0.002 | Tie |
| Credit Card Fraud | 9 | Gain | +0.002 | 0.004 | Tie |
| Credit Card Fraud | 9 | Weight | +0.003 | 0.005 | Tie |
| Dry Bean | 1 | Cover | +0.144 | 0.007 | Win |
| Dry Bean | 1 | Gain | +0.057 | 0.078 | Win |
| Dry Bean | 1 | Weight | -0.142 | 0.009 | Loss |
| Dry Bean | 2 | Cover | +0.013 | 0.014 | Win |
| Dry Bean | 2 | Gain | -0.067 | 0.068 | Loss |
| Dry Bean | 2 | Weight | -0.029 | 0.022 | Loss |
| Dry Bean | 3 | Cover | +0.002 | 0.004 | Tie |
| Dry Bean | 3 | Gain | +0.001 | 0.005 | Tie |
| Dry Bean | 3 | Weight | -0.003 | 0.010 | Tie |
| Dry Bean | 4 | Cover | -0.000 | 0.001 | Tie |
| Dry Bean | 4 | Gain | -0.000 | 0.001 | Tie |
| Dry Bean | 4 | Weight | -0.005 | 0.011 | Tie |
| Dry Bean | 5 | Cover | -0.020 | 0.012 | Loss |
| Dry Bean | 5 | Gain | -0.021 | 0.013 | Loss |
| Dry Bean | 5 | Weight | -0.050 | 0.014 | Loss |
| Steel Plates Faults | 1 | Cover | -0.046 | 0.096 | Loss |
| Steel Plates Faults | 1 | Gain | -0.129 | 0.078 | Loss |
| Steel Plates Faults | 1 | Weight | -0.176 | 0.014 | Loss |
| Steel Plates Faults | 2 | Cover | +0.015 | 0.054 | Win |
| Steel Plates Faults | 2 | Gain | -0.087 | 0.084 | Loss |
| Steel Plates Faults | 2 | Weight | -0.117 | 0.029 | Loss |
| Steel Plates Faults | 3 | Cover | +0.044 | 0.043 | Win |
| Steel Plates Faults | 3 | Gain | -0.031 | 0.027 | Loss |
| Steel Plates Faults | 3 | Weight | -0.033 | 0.057 | Loss |
| Steel Plates Faults | 4 | Cover | +0.058 | 0.039 | Win |
| Steel Plates Faults | 4 | Gain | +0.058 | 0.040 | Win |
| Steel Plates Faults | 4 | Weight | -0.002 | 0.032 | Tie |
| Steel Plates Faults | 5 | Cover | +0.014 | 0.012 | Win |
| Steel Plates Faults | 5 | Gain | +0.014 | 0.012 | Win |
| Steel Plates Faults | 5 | Weight | -0.041 | 0.053 | Loss |
| Steel Plates Faults | 6 | Cover | +0.014 | 0.012 | Win |
| Steel Plates Faults | 6 | Gain | +0.015 | 0.012 | Win |
| Steel Plates Faults | 6 | Weight | -0.063 | 0.043 | Loss |
| Steel Plates Faults | 7 | Cover | +0.010 | 0.014 | Win |
| Steel Plates Faults | 7 | Gain | +0.010 | 0.015 | Tie |
| Steel Plates Faults | 7 | Weight | -0.095 | 0.013 | Loss |
| Steel Plates Faults | 9 | Cover | +0.007 | 0.017 | Tie |
| Steel Plates Faults | 9 | Gain | +0.005 | 0.018 | Tie |
| Steel Plates Faults | 9 | Weight | -0.067 | 0.032 | Loss |
| UNSW-NB15 | 1 | Cover | -0.067 | 0.092 | Loss |
| UNSW-NB15 | 1 | Gain | -0.047 | 0.089 | Loss |
| UNSW-NB15 | 1 | Weight | +0.090 | 0.092 | Win |
| UNSW-NB15 | 2 | Cover | -0.085 | 0.040 | Loss |
| UNSW-NB15 | 2 | Gain | -0.056 | 0.002 | Loss |
| UNSW-NB15 | 2 | Weight | -0.053 | 0.001 | Loss |
| UNSW-NB15 | 3 | Cover | -0.032 | 0.022 | Loss |
| UNSW-NB15 | 3 | Gain | -0.055 | 0.002 | Loss |
| UNSW-NB15 | 3 | Weight | -0.044 | 0.011 | Loss |
| UNSW-NB15 | 4 | Cover | -0.010 | 0.003 | Loss |
| UNSW-NB15 | 4 | Gain | -0.011 | 0.001 | Loss |
| UNSW-NB15 | 4 | Weight | -0.032 | 0.001 | Loss |
| UNSW-NB15 | 5 | Cover | -0.009 | 0.025 | Tie |
| UNSW-NB15 | 5 | Gain | +0.003 | 0.002 | Tie |
| UNSW-NB15 | 5 | Weight | -0.041 | 0.010 | Loss |
| UNSW-NB15 | 6 | Cover | +0.002 | 0.000 | Tie |
| UNSW-NB15 | 6 | Gain | +0.002 | 0.000 | Tie |
| UNSW-NB15 | 6 | Weight | -0.044 | 0.002 | Loss |
| UNSW-NB15 | 8 | Cover | +0.002 | 0.001 | Tie |
| UNSW-NB15 | 8 | Gain | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 8 | Weight | -0.036 | 0.002 | Loss |
| UNSW-NB15 | 10 | Cover | +0.000 | 0.001 | Tie |
| UNSW-NB15 | 10 | Gain | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 10 | Weight | -0.023 | 0.001 | Loss |
| UNSW-NB15 | 12 | Cover | -0.001 | 0.001 | Tie |
| UNSW-NB15 | 12 | Gain | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 12 | Weight | -0.023 | 0.001 | Loss |

### DT4

| Dataset | k | Method | Mean Δ | Fold SD | Outcome |
| --- | ---: | --- | ---: | ---: | --- |
| Adult Income | 1 | Cover | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Gain | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Weight | +0.001 | 0.001 | Tie |
| Adult Income | 2 | Cover | +0.170 | 0.006 | Win |
| Adult Income | 2 | Gain | +0.000 | 0.000 | Tie |
| Adult Income | 2 | Weight | +0.000 | 0.000 | Tie |
| Adult Income | 3 | Cover | +0.026 | 0.065 | Win |
| Adult Income | 3 | Gain | -0.072 | 0.046 | Loss |
| Adult Income | 3 | Weight | -0.107 | 0.010 | Loss |
| Adult Income | 4 | Cover | +0.053 | 0.003 | Win |
| Adult Income | 4 | Gain | +0.053 | 0.003 | Win |
| Adult Income | 4 | Weight | -0.128 | 0.044 | Loss |
| Adult Income | 5 | Cover | +0.001 | 0.002 | Tie |
| Adult Income | 5 | Gain | +0.001 | 0.002 | Tie |
| Adult Income | 5 | Weight | -0.250 | 0.004 | Loss |
| Bank Marketing | 1 | Cover | +0.007 | 0.017 | Tie |
| Bank Marketing | 1 | Gain | +0.007 | 0.017 | Tie |
| Bank Marketing | 1 | Weight | -0.107 | 0.060 | Loss |
| Bank Marketing | 2 | Cover | +0.007 | 0.017 | Tie |
| Bank Marketing | 2 | Gain | +0.087 | 0.006 | Win |
| Bank Marketing | 2 | Weight | -0.029 | 0.067 | Loss |
| Bank Marketing | 3 | Cover | +0.026 | 0.041 | Win |
| Bank Marketing | 3 | Gain | +0.044 | 0.005 | Win |
| Bank Marketing | 3 | Weight | -0.044 | 0.082 | Loss |
| Bank Marketing | 4 | Cover | +0.030 | 0.052 | Win |
| Bank Marketing | 4 | Gain | +0.048 | 0.029 | Win |
| Bank Marketing | 4 | Weight | -0.004 | 0.024 | Tie |
| Bank Marketing | 5 | Cover | -0.008 | 0.007 | Tie |
| Bank Marketing | 5 | Gain | +0.000 | 0.000 | Tie |
| Bank Marketing | 5 | Weight | -0.036 | 0.014 | Loss |
| Breast Cancer Wisconsin | 1 | Cover | -0.034 | 0.051 | Loss |
| Breast Cancer Wisconsin | 1 | Gain | -0.021 | 0.049 | Loss |
| Breast Cancer Wisconsin | 1 | Weight | -0.244 | 0.096 | Loss |
| Breast Cancer Wisconsin | 2 | Cover | -0.023 | 0.030 | Loss |
| Breast Cancer Wisconsin | 2 | Gain | -0.025 | 0.030 | Loss |
| Breast Cancer Wisconsin | 2 | Weight | -0.045 | 0.122 | Loss |
| Breast Cancer Wisconsin | 3 | Cover | -0.006 | 0.023 | Tie |
| Breast Cancer Wisconsin | 3 | Gain | -0.003 | 0.008 | Tie |
| Breast Cancer Wisconsin | 3 | Weight | +0.006 | 0.049 | Tie |
| Breast Cancer Wisconsin | 4 | Cover | +0.003 | 0.010 | Tie |
| Breast Cancer Wisconsin | 4 | Gain | +0.003 | 0.006 | Tie |
| Breast Cancer Wisconsin | 4 | Weight | +0.017 | 0.016 | Win |
| Breast Cancer Wisconsin | 5 | Cover | -0.005 | 0.023 | Tie |
| Breast Cancer Wisconsin | 5 | Gain | -0.007 | 0.014 | Tie |
| Breast Cancer Wisconsin | 5 | Weight | +0.004 | 0.018 | Tie |
| Breast Cancer Wisconsin | 6 | Cover | +0.004 | 0.025 | Tie |
| Breast Cancer Wisconsin | 6 | Gain | +0.006 | 0.017 | Tie |
| Breast Cancer Wisconsin | 6 | Weight | +0.014 | 0.020 | Win |
| Breast Cancer Wisconsin | 8 | Cover | -0.013 | 0.023 | Loss |
| Breast Cancer Wisconsin | 8 | Gain | -0.009 | 0.018 | Tie |
| Breast Cancer Wisconsin | 8 | Weight | +0.003 | 0.011 | Tie |
| Breast Cancer Wisconsin | 9 | Cover | -0.014 | 0.028 | Loss |
| Breast Cancer Wisconsin | 9 | Gain | -0.013 | 0.019 | Loss |
| Breast Cancer Wisconsin | 9 | Weight | +0.007 | 0.007 | Tie |
| CIC-IDS2017 | 2 | Cover | -0.179 | 0.006 | Loss |
| CIC-IDS2017 | 2 | Gain | -0.050 | 0.015 | Loss |
| CIC-IDS2017 | 2 | Weight | -0.054 | 0.006 | Loss |
| CIC-IDS2017 | 4 | Cover | -0.048 | 0.002 | Loss |
| CIC-IDS2017 | 4 | Gain | -0.059 | 0.002 | Loss |
| CIC-IDS2017 | 4 | Weight | -0.110 | 0.001 | Loss |
| CIC-IDS2017 | 5 | Cover | -0.096 | 0.024 | Loss |
| CIC-IDS2017 | 5 | Gain | -0.105 | 0.035 | Loss |
| CIC-IDS2017 | 5 | Weight | -0.157 | 0.034 | Loss |
| CIC-IDS2017 | 7 | Cover | -0.102 | 0.012 | Loss |
| CIC-IDS2017 | 7 | Gain | -0.109 | 0.021 | Loss |
| CIC-IDS2017 | 7 | Weight | -0.146 | 0.027 | Loss |
| CIC-IDS2017 | 9 | Cover | -0.080 | 0.038 | Loss |
| CIC-IDS2017 | 9 | Gain | -0.091 | 0.038 | Loss |
| CIC-IDS2017 | 9 | Weight | -0.077 | 0.033 | Loss |
| CIC-IDS2017 | 10 | Cover | -0.024 | 0.013 | Loss |
| CIC-IDS2017 | 10 | Gain | -0.031 | 0.025 | Loss |
| CIC-IDS2017 | 10 | Weight | +0.006 | 0.013 | Tie |
| CIC-IDS2017 | 13 | Cover | -0.042 | 0.005 | Loss |
| CIC-IDS2017 | 13 | Gain | -0.047 | 0.022 | Loss |
| CIC-IDS2017 | 13 | Weight | -0.031 | 0.041 | Loss |
| CIC-IDS2017 | 17 | Cover | -0.034 | 0.013 | Loss |
| CIC-IDS2017 | 17 | Gain | -0.025 | 0.015 | Loss |
| CIC-IDS2017 | 17 | Weight | -0.011 | 0.001 | Loss |
| CIC-IDS2017 | 20 | Cover | -0.021 | 0.018 | Loss |
| CIC-IDS2017 | 20 | Gain | -0.025 | 0.015 | Loss |
| CIC-IDS2017 | 20 | Weight | -0.011 | 0.000 | Loss |
| Covertype | 2 | Cover | -0.256 | 0.001 | Loss |
| Covertype | 2 | Gain | -0.054 | 0.001 | Loss |
| Covertype | 2 | Weight | -0.179 | 0.011 | Loss |
| Covertype | 3 | Cover | -0.252 | 0.014 | Loss |
| Covertype | 3 | Gain | -0.048 | 0.013 | Loss |
| Covertype | 3 | Weight | -0.019 | 0.001 | Loss |
| Covertype | 5 | Cover | -0.312 | 0.013 | Loss |
| Covertype | 5 | Gain | -0.105 | 0.005 | Loss |
| Covertype | 5 | Weight | -0.007 | 0.001 | Tie |
| Covertype | 6 | Cover | -0.271 | 0.049 | Loss |
| Covertype | 6 | Gain | -0.097 | 0.005 | Loss |
| Covertype | 6 | Weight | -0.006 | 0.001 | Tie |
| Covertype | 7 | Cover | -0.233 | 0.057 | Loss |
| Covertype | 7 | Gain | -0.074 | 0.023 | Loss |
| Covertype | 7 | Weight | +0.015 | 0.022 | Win |
| Covertype | 9 | Cover | -0.194 | 0.024 | Loss |
| Covertype | 9 | Gain | -0.069 | 0.029 | Loss |
| Covertype | 9 | Weight | -0.004 | 0.022 | Tie |
| Covertype | 11 | Cover | -0.192 | 0.021 | Loss |
| Covertype | 11 | Gain | -0.048 | 0.022 | Loss |
| Covertype | 11 | Weight | -0.000 | 0.000 | Tie |
| Covertype | 14 | Cover | -0.201 | 0.021 | Loss |
| Covertype | 14 | Gain | -0.037 | 0.022 | Loss |
| Covertype | 14 | Weight | +0.000 | 0.000 | Tie |
| Covertype | 17 | Cover | -0.175 | 0.062 | Loss |
| Covertype | 17 | Gain | -0.041 | 0.023 | Loss |
| Covertype | 17 | Weight | +0.030 | 0.023 | Win |
| Credit Card Fraud | 1 | Cover | +0.042 | 0.039 | Win |
| Credit Card Fraud | 1 | Gain | +0.000 | 0.000 | Tie |
| Credit Card Fraud | 1 | Weight | -0.281 | 0.037 | Loss |
| Credit Card Fraud | 2 | Cover | +0.025 | 0.024 | Win |
| Credit Card Fraud | 2 | Gain | +0.032 | 0.016 | Win |
| Credit Card Fraud | 2 | Weight | -0.110 | 0.104 | Loss |
| Credit Card Fraud | 3 | Cover | +0.006 | 0.016 | Tie |
| Credit Card Fraud | 3 | Gain | +0.014 | 0.020 | Win |
| Credit Card Fraud | 3 | Weight | -0.036 | 0.014 | Loss |
| Credit Card Fraud | 4 | Cover | -0.004 | 0.013 | Tie |
| Credit Card Fraud | 4 | Gain | -0.002 | 0.007 | Tie |
| Credit Card Fraud | 4 | Weight | -0.025 | 0.025 | Loss |
| Credit Card Fraud | 5 | Cover | +0.004 | 0.005 | Tie |
| Credit Card Fraud | 5 | Gain | -0.004 | 0.007 | Tie |
| Credit Card Fraud | 5 | Weight | -0.001 | 0.020 | Tie |
| Credit Card Fraud | 6 | Cover | +0.002 | 0.006 | Tie |
| Credit Card Fraud | 6 | Gain | -0.000 | 0.005 | Tie |
| Credit Card Fraud | 6 | Weight | -0.002 | 0.016 | Tie |
| Credit Card Fraud | 8 | Cover | -0.002 | 0.003 | Tie |
| Credit Card Fraud | 8 | Gain | -0.003 | 0.009 | Tie |
| Credit Card Fraud | 8 | Weight | -0.007 | 0.006 | Tie |
| Credit Card Fraud | 9 | Cover | -0.000 | 0.002 | Tie |
| Credit Card Fraud | 9 | Gain | -0.002 | 0.006 | Tie |
| Credit Card Fraud | 9 | Weight | -0.003 | 0.005 | Tie |
| Dry Bean | 1 | Cover | +0.126 | 0.017 | Win |
| Dry Bean | 1 | Gain | +0.046 | 0.063 | Win |
| Dry Bean | 1 | Weight | -0.197 | 0.004 | Loss |
| Dry Bean | 2 | Cover | -0.078 | 0.085 | Loss |
| Dry Bean | 2 | Gain | -0.177 | 0.150 | Loss |
| Dry Bean | 2 | Weight | -0.258 | 0.020 | Loss |
| Dry Bean | 3 | Cover | -0.002 | 0.001 | Tie |
| Dry Bean | 3 | Gain | -0.003 | 0.001 | Tie |
| Dry Bean | 3 | Weight | -0.189 | 0.012 | Loss |
| Dry Bean | 4 | Cover | -0.003 | 0.002 | Tie |
| Dry Bean | 4 | Gain | -0.003 | 0.002 | Tie |
| Dry Bean | 4 | Weight | -0.189 | 0.013 | Loss |
| Dry Bean | 5 | Cover | -0.000 | 0.003 | Tie |
| Dry Bean | 5 | Gain | -0.003 | 0.002 | Tie |
| Dry Bean | 5 | Weight | -0.002 | 0.026 | Tie |
| Steel Plates Faults | 1 | Cover | -0.136 | 0.106 | Loss |
| Steel Plates Faults | 1 | Gain | -0.215 | 0.077 | Loss |
| Steel Plates Faults | 1 | Weight | -0.267 | 0.032 | Loss |
| Steel Plates Faults | 2 | Cover | -0.138 | 0.060 | Loss |
| Steel Plates Faults | 2 | Gain | -0.232 | 0.090 | Loss |
| Steel Plates Faults | 2 | Weight | -0.263 | 0.028 | Loss |
| Steel Plates Faults | 3 | Cover | -0.116 | 0.037 | Loss |
| Steel Plates Faults | 3 | Gain | -0.194 | 0.020 | Loss |
| Steel Plates Faults | 3 | Weight | -0.213 | 0.025 | Loss |
| Steel Plates Faults | 4 | Cover | -0.086 | 0.079 | Loss |
| Steel Plates Faults | 4 | Gain | -0.100 | 0.092 | Loss |
| Steel Plates Faults | 4 | Weight | -0.040 | 0.137 | Loss |
| Steel Plates Faults | 5 | Cover | -0.033 | 0.049 | Loss |
| Steel Plates Faults | 5 | Gain | -0.034 | 0.049 | Loss |
| Steel Plates Faults | 5 | Weight | +0.079 | 0.080 | Win |
| Steel Plates Faults | 6 | Cover | -0.035 | 0.048 | Loss |
| Steel Plates Faults | 6 | Gain | -0.035 | 0.047 | Loss |
| Steel Plates Faults | 6 | Weight | +0.034 | 0.075 | Win |
| Steel Plates Faults | 7 | Cover | -0.046 | 0.049 | Loss |
| Steel Plates Faults | 7 | Gain | -0.062 | 0.040 | Loss |
| Steel Plates Faults | 7 | Weight | -0.035 | 0.039 | Loss |
| Steel Plates Faults | 9 | Cover | -0.022 | 0.044 | Loss |
| Steel Plates Faults | 9 | Gain | -0.005 | 0.009 | Tie |
| Steel Plates Faults | 9 | Weight | +0.026 | 0.062 | Win |
| UNSW-NB15 | 1 | Cover | -0.101 | 0.138 | Loss |
| UNSW-NB15 | 1 | Gain | -0.071 | 0.138 | Loss |
| UNSW-NB15 | 1 | Weight | +0.136 | 0.139 | Win |
| UNSW-NB15 | 2 | Cover | -0.142 | 0.069 | Loss |
| UNSW-NB15 | 2 | Gain | -0.100 | 0.018 | Loss |
| UNSW-NB15 | 2 | Weight | -0.075 | 0.025 | Loss |
| UNSW-NB15 | 3 | Cover | -0.113 | 0.051 | Loss |
| UNSW-NB15 | 3 | Gain | -0.167 | 0.001 | Loss |
| UNSW-NB15 | 3 | Weight | -0.091 | 0.022 | Loss |
| UNSW-NB15 | 4 | Cover | -0.078 | 0.006 | Loss |
| UNSW-NB15 | 4 | Gain | -0.130 | 0.002 | Loss |
| UNSW-NB15 | 4 | Weight | -0.060 | 0.015 | Loss |
| UNSW-NB15 | 5 | Cover | +0.006 | 0.037 | Tie |
| UNSW-NB15 | 5 | Gain | +0.007 | 0.014 | Tie |
| UNSW-NB15 | 5 | Weight | +0.004 | 0.020 | Tie |
| UNSW-NB15 | 6 | Cover | +0.001 | 0.000 | Tie |
| UNSW-NB15 | 6 | Gain | +0.001 | 0.000 | Tie |
| UNSW-NB15 | 6 | Weight | -0.019 | 0.002 | Loss |
| UNSW-NB15 | 8 | Cover | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 8 | Gain | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 8 | Weight | -0.003 | 0.012 | Tie |
| UNSW-NB15 | 10 | Cover | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 10 | Gain | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 10 | Weight | +0.036 | 0.001 | Win |
| UNSW-NB15 | 12 | Cover | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 12 | Gain | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 12 | Weight | +0.036 | 0.001 | Win |

### DT5

| Dataset | k | Method | Mean Δ | Fold SD | Outcome |
| --- | ---: | --- | ---: | ---: | --- |
| Adult Income | 1 | Cover | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Gain | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Weight | +0.002 | 0.001 | Tie |
| Adult Income | 2 | Cover | -0.094 | 0.007 | Loss |
| Adult Income | 2 | Gain | -0.275 | 0.003 | Loss |
| Adult Income | 2 | Weight | -0.275 | 0.003 | Loss |
| Adult Income | 3 | Cover | +0.040 | 0.068 | Win |
| Adult Income | 3 | Gain | -0.054 | 0.053 | Loss |
| Adult Income | 3 | Weight | -0.114 | 0.015 | Loss |
| Adult Income | 4 | Cover | +0.010 | 0.005 | Tie |
| Adult Income | 4 | Gain | +0.010 | 0.005 | Tie |
| Adult Income | 4 | Weight | -0.138 | 0.054 | Loss |
| Adult Income | 5 | Cover | +0.018 | 0.003 | Win |
| Adult Income | 5 | Gain | +0.018 | 0.003 | Win |
| Adult Income | 5 | Weight | -0.181 | 0.059 | Loss |
| Bank Marketing | 1 | Cover | +0.005 | 0.017 | Tie |
| Bank Marketing | 1 | Gain | +0.005 | 0.017 | Tie |
| Bank Marketing | 1 | Weight | -0.108 | 0.062 | Loss |
| Bank Marketing | 2 | Cover | +0.005 | 0.017 | Tie |
| Bank Marketing | 2 | Gain | +0.082 | 0.003 | Win |
| Bank Marketing | 2 | Weight | -0.024 | 0.059 | Loss |
| Bank Marketing | 3 | Cover | +0.018 | 0.035 | Win |
| Bank Marketing | 3 | Gain | +0.036 | 0.008 | Win |
| Bank Marketing | 3 | Weight | -0.051 | 0.080 | Loss |
| Bank Marketing | 4 | Cover | +0.004 | 0.034 | Tie |
| Bank Marketing | 4 | Gain | +0.023 | 0.008 | Win |
| Bank Marketing | 4 | Weight | -0.028 | 0.008 | Loss |
| Bank Marketing | 5 | Cover | -0.017 | 0.005 | Loss |
| Bank Marketing | 5 | Gain | +0.000 | 0.000 | Tie |
| Bank Marketing | 5 | Weight | -0.046 | 0.022 | Loss |
| Breast Cancer Wisconsin | 1 | Cover | -0.036 | 0.061 | Loss |
| Breast Cancer Wisconsin | 1 | Gain | -0.021 | 0.056 | Loss |
| Breast Cancer Wisconsin | 1 | Weight | -0.217 | 0.120 | Loss |
| Breast Cancer Wisconsin | 2 | Cover | -0.029 | 0.038 | Loss |
| Breast Cancer Wisconsin | 2 | Gain | -0.030 | 0.034 | Loss |
| Breast Cancer Wisconsin | 2 | Weight | -0.055 | 0.103 | Loss |
| Breast Cancer Wisconsin | 3 | Cover | -0.010 | 0.028 | Loss |
| Breast Cancer Wisconsin | 3 | Gain | -0.002 | 0.017 | Tie |
| Breast Cancer Wisconsin | 3 | Weight | +0.006 | 0.048 | Tie |
| Breast Cancer Wisconsin | 4 | Cover | +0.000 | 0.020 | Tie |
| Breast Cancer Wisconsin | 4 | Gain | +0.002 | 0.010 | Tie |
| Breast Cancer Wisconsin | 4 | Weight | +0.035 | 0.028 | Win |
| Breast Cancer Wisconsin | 5 | Cover | -0.011 | 0.025 | Loss |
| Breast Cancer Wisconsin | 5 | Gain | -0.013 | 0.020 | Loss |
| Breast Cancer Wisconsin | 5 | Weight | +0.011 | 0.026 | Win |
| Breast Cancer Wisconsin | 6 | Cover | -0.011 | 0.029 | Loss |
| Breast Cancer Wisconsin | 6 | Gain | -0.011 | 0.016 | Loss |
| Breast Cancer Wisconsin | 6 | Weight | +0.009 | 0.012 | Tie |
| Breast Cancer Wisconsin | 8 | Cover | -0.015 | 0.020 | Loss |
| Breast Cancer Wisconsin | 8 | Gain | -0.018 | 0.012 | Loss |
| Breast Cancer Wisconsin | 8 | Weight | +0.003 | 0.019 | Tie |
| Breast Cancer Wisconsin | 9 | Cover | -0.017 | 0.024 | Loss |
| Breast Cancer Wisconsin | 9 | Gain | -0.017 | 0.012 | Loss |
| Breast Cancer Wisconsin | 9 | Weight | -0.000 | 0.015 | Tie |
| CIC-IDS2017 | 2 | Cover | -0.235 | 0.006 | Loss |
| CIC-IDS2017 | 2 | Gain | -0.050 | 0.065 | Loss |
| CIC-IDS2017 | 2 | Weight | -0.100 | 0.006 | Loss |
| CIC-IDS2017 | 4 | Cover | -0.038 | 0.001 | Loss |
| CIC-IDS2017 | 4 | Gain | -0.044 | 0.001 | Loss |
| CIC-IDS2017 | 4 | Weight | -0.078 | 0.002 | Loss |
| CIC-IDS2017 | 5 | Cover | -0.099 | 0.030 | Loss |
| CIC-IDS2017 | 5 | Gain | -0.088 | 0.008 | Loss |
| CIC-IDS2017 | 5 | Weight | -0.135 | 0.010 | Loss |
| CIC-IDS2017 | 7 | Cover | -0.106 | 0.024 | Loss |
| CIC-IDS2017 | 7 | Gain | -0.095 | 0.012 | Loss |
| CIC-IDS2017 | 7 | Weight | -0.122 | 0.015 | Loss |
| CIC-IDS2017 | 9 | Cover | -0.087 | 0.038 | Loss |
| CIC-IDS2017 | 9 | Gain | -0.092 | 0.019 | Loss |
| CIC-IDS2017 | 9 | Weight | -0.071 | 0.009 | Loss |
| CIC-IDS2017 | 10 | Cover | -0.076 | 0.029 | Loss |
| CIC-IDS2017 | 10 | Gain | -0.061 | 0.026 | Loss |
| CIC-IDS2017 | 10 | Weight | -0.059 | 0.029 | Loss |
| CIC-IDS2017 | 13 | Cover | -0.065 | 0.001 | Loss |
| CIC-IDS2017 | 13 | Gain | -0.050 | 0.005 | Loss |
| CIC-IDS2017 | 13 | Weight | -0.039 | 0.024 | Loss |
| CIC-IDS2017 | 17 | Cover | -0.072 | 0.005 | Loss |
| CIC-IDS2017 | 17 | Gain | -0.061 | 0.005 | Loss |
| CIC-IDS2017 | 17 | Weight | -0.026 | 0.008 | Loss |
| CIC-IDS2017 | 20 | Cover | -0.057 | 0.023 | Loss |
| CIC-IDS2017 | 20 | Gain | -0.050 | 0.023 | Loss |
| CIC-IDS2017 | 20 | Weight | -0.009 | 0.002 | Tie |
| Covertype | 2 | Cover | -0.259 | 0.001 | Loss |
| Covertype | 2 | Gain | -0.019 | 0.031 | Loss |
| Covertype | 2 | Weight | -0.171 | 0.004 | Loss |
| Covertype | 3 | Cover | -0.270 | 0.013 | Loss |
| Covertype | 3 | Gain | -0.029 | 0.025 | Loss |
| Covertype | 3 | Weight | -0.007 | 0.001 | Tie |
| Covertype | 5 | Cover | -0.318 | 0.014 | Loss |
| Covertype | 5 | Gain | -0.100 | 0.008 | Loss |
| Covertype | 5 | Weight | -0.026 | 0.002 | Loss |
| Covertype | 6 | Cover | -0.282 | 0.051 | Loss |
| Covertype | 6 | Gain | -0.099 | 0.012 | Loss |
| Covertype | 6 | Weight | -0.033 | 0.006 | Loss |
| Covertype | 7 | Cover | -0.254 | 0.047 | Loss |
| Covertype | 7 | Gain | -0.085 | 0.006 | Loss |
| Covertype | 7 | Weight | -0.018 | 0.001 | Loss |
| Covertype | 9 | Cover | -0.229 | 0.003 | Loss |
| Covertype | 9 | Gain | -0.086 | 0.012 | Loss |
| Covertype | 9 | Weight | -0.017 | 0.003 | Loss |
| Covertype | 11 | Cover | -0.218 | 0.003 | Loss |
| Covertype | 11 | Gain | -0.061 | 0.003 | Loss |
| Covertype | 11 | Weight | -0.002 | 0.004 | Tie |
| Covertype | 14 | Cover | -0.227 | 0.003 | Loss |
| Covertype | 14 | Gain | -0.011 | 0.005 | Loss |
| Covertype | 14 | Weight | +0.011 | 0.002 | Win |
| Covertype | 17 | Cover | -0.199 | 0.053 | Loss |
| Covertype | 17 | Gain | -0.006 | 0.005 | Tie |
| Covertype | 17 | Weight | +0.018 | 0.001 | Win |
| Credit Card Fraud | 1 | Cover | +0.056 | 0.035 | Win |
| Credit Card Fraud | 1 | Gain | +0.000 | 0.000 | Tie |
| Credit Card Fraud | 1 | Weight | -0.254 | 0.058 | Loss |
| Credit Card Fraud | 2 | Cover | +0.014 | 0.010 | Win |
| Credit Card Fraud | 2 | Gain | +0.046 | 0.012 | Win |
| Credit Card Fraud | 2 | Weight | -0.112 | 0.100 | Loss |
| Credit Card Fraud | 3 | Cover | -0.001 | 0.010 | Tie |
| Credit Card Fraud | 3 | Gain | +0.011 | 0.007 | Win |
| Credit Card Fraud | 3 | Weight | -0.028 | 0.014 | Loss |
| Credit Card Fraud | 4 | Cover | -0.004 | 0.005 | Tie |
| Credit Card Fraud | 4 | Gain | -0.008 | 0.009 | Tie |
| Credit Card Fraud | 4 | Weight | -0.020 | 0.024 | Loss |
| Credit Card Fraud | 5 | Cover | +0.007 | 0.005 | Tie |
| Credit Card Fraud | 5 | Gain | -0.003 | 0.014 | Tie |
| Credit Card Fraud | 5 | Weight | +0.000 | 0.014 | Tie |
| Credit Card Fraud | 6 | Cover | +0.004 | 0.010 | Tie |
| Credit Card Fraud | 6 | Gain | +0.005 | 0.012 | Tie |
| Credit Card Fraud | 6 | Weight | -0.003 | 0.016 | Tie |
| Credit Card Fraud | 8 | Cover | +0.003 | 0.015 | Tie |
| Credit Card Fraud | 8 | Gain | +0.007 | 0.007 | Tie |
| Credit Card Fraud | 8 | Weight | +0.002 | 0.007 | Tie |
| Credit Card Fraud | 9 | Cover | +0.001 | 0.011 | Tie |
| Credit Card Fraud | 9 | Gain | +0.007 | 0.007 | Tie |
| Credit Card Fraud | 9 | Weight | +0.007 | 0.005 | Tie |
| Dry Bean | 1 | Cover | +0.131 | 0.013 | Win |
| Dry Bean | 1 | Gain | +0.050 | 0.069 | Win |
| Dry Bean | 1 | Weight | -0.190 | 0.012 | Loss |
| Dry Bean | 2 | Cover | -0.117 | 0.163 | Loss |
| Dry Bean | 2 | Gain | -0.241 | 0.220 | Loss |
| Dry Bean | 2 | Weight | -0.358 | 0.011 | Loss |
| Dry Bean | 3 | Cover | +0.001 | 0.002 | Tie |
| Dry Bean | 3 | Gain | +0.000 | 0.002 | Tie |
| Dry Bean | 3 | Weight | -0.310 | 0.008 | Loss |
| Dry Bean | 4 | Cover | +0.001 | 0.002 | Tie |
| Dry Bean | 4 | Gain | +0.001 | 0.002 | Tie |
| Dry Bean | 4 | Weight | -0.292 | 0.012 | Loss |
| Dry Bean | 5 | Cover | -0.003 | 0.002 | Tie |
| Dry Bean | 5 | Gain | -0.002 | 0.003 | Tie |
| Dry Bean | 5 | Weight | -0.028 | 0.014 | Loss |
| Steel Plates Faults | 1 | Cover | -0.162 | 0.102 | Loss |
| Steel Plates Faults | 1 | Gain | -0.247 | 0.079 | Loss |
| Steel Plates Faults | 1 | Weight | -0.301 | 0.018 | Loss |
| Steel Plates Faults | 2 | Cover | -0.147 | 0.057 | Loss |
| Steel Plates Faults | 2 | Gain | -0.257 | 0.089 | Loss |
| Steel Plates Faults | 2 | Weight | -0.128 | 0.020 | Loss |
| Steel Plates Faults | 3 | Cover | -0.192 | 0.045 | Loss |
| Steel Plates Faults | 3 | Gain | -0.311 | 0.010 | Loss |
| Steel Plates Faults | 3 | Weight | -0.216 | 0.042 | Loss |
| Steel Plates Faults | 4 | Cover | -0.119 | 0.049 | Loss |
| Steel Plates Faults | 4 | Gain | -0.137 | 0.059 | Loss |
| Steel Plates Faults | 4 | Weight | -0.047 | 0.098 | Loss |
| Steel Plates Faults | 5 | Cover | -0.024 | 0.023 | Loss |
| Steel Plates Faults | 5 | Gain | -0.062 | 0.053 | Loss |
| Steel Plates Faults | 5 | Weight | +0.080 | 0.065 | Win |
| Steel Plates Faults | 6 | Cover | -0.025 | 0.037 | Loss |
| Steel Plates Faults | 6 | Gain | -0.031 | 0.050 | Loss |
| Steel Plates Faults | 6 | Weight | +0.034 | 0.082 | Win |
| Steel Plates Faults | 7 | Cover | -0.024 | 0.021 | Loss |
| Steel Plates Faults | 7 | Gain | -0.035 | 0.039 | Loss |
| Steel Plates Faults | 7 | Weight | -0.018 | 0.042 | Loss |
| Steel Plates Faults | 9 | Cover | -0.015 | 0.055 | Loss |
| Steel Plates Faults | 9 | Gain | -0.022 | 0.063 | Loss |
| Steel Plates Faults | 9 | Weight | +0.041 | 0.032 | Win |
| UNSW-NB15 | 1 | Cover | -0.117 | 0.161 | Loss |
| UNSW-NB15 | 1 | Gain | -0.088 | 0.160 | Loss |
| UNSW-NB15 | 1 | Weight | +0.152 | 0.160 | Win |
| UNSW-NB15 | 2 | Cover | -0.190 | 0.072 | Loss |
| UNSW-NB15 | 2 | Gain | -0.148 | 0.020 | Loss |
| UNSW-NB15 | 2 | Weight | -0.101 | 0.002 | Loss |
| UNSW-NB15 | 3 | Cover | -0.114 | 0.051 | Loss |
| UNSW-NB15 | 3 | Gain | -0.166 | 0.002 | Loss |
| UNSW-NB15 | 3 | Weight | -0.099 | 0.030 | Loss |
| UNSW-NB15 | 4 | Cover | -0.033 | 0.019 | Loss |
| UNSW-NB15 | 4 | Gain | -0.034 | 0.002 | Loss |
| UNSW-NB15 | 4 | Weight | -0.044 | 0.007 | Loss |
| UNSW-NB15 | 5 | Cover | +0.022 | 0.036 | Win |
| UNSW-NB15 | 5 | Gain | +0.006 | 0.023 | Tie |
| UNSW-NB15 | 5 | Weight | -0.016 | 0.007 | Loss |
| UNSW-NB15 | 6 | Cover | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 6 | Gain | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 6 | Weight | -0.061 | 0.001 | Loss |
| UNSW-NB15 | 8 | Cover | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 8 | Gain | -0.000 | 0.000 | Tie |
| UNSW-NB15 | 8 | Weight | -0.029 | 0.010 | Loss |
| UNSW-NB15 | 10 | Cover | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 10 | Gain | -0.000 | 0.000 | Tie |
| UNSW-NB15 | 10 | Weight | +0.030 | 0.000 | Win |
| UNSW-NB15 | 12 | Cover | -0.000 | 0.000 | Tie |
| UNSW-NB15 | 12 | Gain | +0.000 | 0.000 | Tie |
| UNSW-NB15 | 12 | Weight | +0.030 | 0.000 | Win |

### DT6

| Dataset | k | Method | Mean Δ | Fold SD | Outcome |
| --- | ---: | --- | ---: | ---: | --- |
| Adult Income | 1 | Cover | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Gain | +0.000 | 0.000 | Tie |
| Adult Income | 1 | Weight | +0.004 | 0.002 | Tie |
| Adult Income | 2 | Cover | -0.071 | 0.014 | Loss |
| Adult Income | 2 | Gain | -0.258 | 0.012 | Loss |
| Adult Income | 2 | Weight | -0.256 | 0.012 | Loss |
| Adult Income | 3 | Cover | +0.053 | 0.072 | Win |
| Adult Income | 3 | Gain | -0.041 | 0.048 | Loss |
| Adult Income | 3 | Weight | -0.092 | 0.036 | Loss |
| Adult Income | 4 | Cover | +0.009 | 0.004 | Tie |
| Adult Income | 4 | Gain | +0.009 | 0.004 | Tie |
| Adult Income | 4 | Weight | -0.107 | 0.007 | Loss |
| Adult Income | 5 | Cover | +0.019 | 0.003 | Win |
| Adult Income | 5 | Gain | +0.019 | 0.003 | Win |
| Adult Income | 5 | Weight | -0.148 | 0.032 | Loss |
| Bank Marketing | 1 | Cover | +0.008 | 0.016 | Tie |
| Bank Marketing | 1 | Gain | +0.008 | 0.016 | Tie |
| Bank Marketing | 1 | Weight | -0.106 | 0.060 | Loss |
| Bank Marketing | 2 | Cover | +0.006 | 0.017 | Tie |
| Bank Marketing | 2 | Gain | +0.083 | 0.008 | Win |
| Bank Marketing | 2 | Weight | -0.025 | 0.066 | Loss |
| Bank Marketing | 3 | Cover | +0.009 | 0.041 | Tie |
| Bank Marketing | 3 | Gain | +0.025 | 0.013 | Win |
| Bank Marketing | 3 | Weight | -0.055 | 0.074 | Loss |
| Bank Marketing | 4 | Cover | +0.008 | 0.035 | Tie |
| Bank Marketing | 4 | Gain | +0.025 | 0.021 | Win |
| Bank Marketing | 4 | Weight | -0.024 | 0.021 | Loss |
| Bank Marketing | 5 | Cover | -0.023 | 0.011 | Loss |
| Bank Marketing | 5 | Gain | +0.000 | 0.000 | Tie |
| Bank Marketing | 5 | Weight | -0.043 | 0.013 | Loss |
| Breast Cancer Wisconsin | 1 | Cover | -0.037 | 0.060 | Loss |
| Breast Cancer Wisconsin | 1 | Gain | -0.029 | 0.068 | Loss |
| Breast Cancer Wisconsin | 1 | Weight | -0.233 | 0.105 | Loss |
| Breast Cancer Wisconsin | 2 | Cover | -0.037 | 0.019 | Loss |
| Breast Cancer Wisconsin | 2 | Gain | -0.036 | 0.034 | Loss |
| Breast Cancer Wisconsin | 2 | Weight | -0.076 | 0.099 | Loss |
| Breast Cancer Wisconsin | 3 | Cover | +0.001 | 0.030 | Tie |
| Breast Cancer Wisconsin | 3 | Gain | -0.011 | 0.021 | Loss |
| Breast Cancer Wisconsin | 3 | Weight | +0.009 | 0.048 | Tie |
| Breast Cancer Wisconsin | 4 | Cover | +0.006 | 0.023 | Tie |
| Breast Cancer Wisconsin | 4 | Gain | +0.002 | 0.004 | Tie |
| Breast Cancer Wisconsin | 4 | Weight | +0.033 | 0.035 | Win |
| Breast Cancer Wisconsin | 5 | Cover | -0.003 | 0.028 | Tie |
| Breast Cancer Wisconsin | 5 | Gain | -0.013 | 0.015 | Loss |
| Breast Cancer Wisconsin | 5 | Weight | +0.010 | 0.022 | Win |
| Breast Cancer Wisconsin | 6 | Cover | -0.009 | 0.038 | Tie |
| Breast Cancer Wisconsin | 6 | Gain | -0.011 | 0.025 | Loss |
| Breast Cancer Wisconsin | 6 | Weight | +0.000 | 0.017 | Tie |
| Breast Cancer Wisconsin | 8 | Cover | -0.024 | 0.027 | Loss |
| Breast Cancer Wisconsin | 8 | Gain | -0.024 | 0.013 | Loss |
| Breast Cancer Wisconsin | 8 | Weight | -0.003 | 0.005 | Tie |
| Breast Cancer Wisconsin | 9 | Cover | -0.016 | 0.026 | Loss |
| Breast Cancer Wisconsin | 9 | Gain | -0.010 | 0.010 | Tie |
| Breast Cancer Wisconsin | 9 | Weight | +0.004 | 0.016 | Tie |
| CIC-IDS2017 | 2 | Cover | -0.238 | 0.006 | Loss |
| CIC-IDS2017 | 2 | Gain | +0.043 | 0.106 | Win |
| CIC-IDS2017 | 2 | Weight | -0.051 | 0.015 | Loss |
| CIC-IDS2017 | 4 | Cover | -0.132 | 0.013 | Loss |
| CIC-IDS2017 | 4 | Gain | -0.040 | 0.020 | Loss |
| CIC-IDS2017 | 4 | Weight | -0.102 | 0.013 | Loss |
| CIC-IDS2017 | 5 | Cover | -0.143 | 0.015 | Loss |
| CIC-IDS2017 | 5 | Gain | -0.118 | 0.032 | Loss |
| CIC-IDS2017 | 5 | Weight | -0.112 | 0.007 | Loss |
| CIC-IDS2017 | 7 | Cover | -0.171 | 0.013 | Loss |
| CIC-IDS2017 | 7 | Gain | -0.144 | 0.010 | Loss |
| CIC-IDS2017 | 7 | Weight | -0.136 | 0.020 | Loss |
| CIC-IDS2017 | 9 | Cover | -0.149 | 0.020 | Loss |
| CIC-IDS2017 | 9 | Gain | -0.139 | 0.009 | Loss |
| CIC-IDS2017 | 9 | Weight | -0.127 | 0.010 | Loss |
| CIC-IDS2017 | 10 | Cover | -0.159 | 0.025 | Loss |
| CIC-IDS2017 | 10 | Gain | -0.142 | 0.025 | Loss |
| CIC-IDS2017 | 10 | Weight | -0.051 | 0.014 | Loss |
| CIC-IDS2017 | 13 | Cover | -0.135 | 0.011 | Loss |
| CIC-IDS2017 | 13 | Gain | -0.082 | 0.029 | Loss |
| CIC-IDS2017 | 13 | Weight | -0.006 | 0.051 | Tie |
| CIC-IDS2017 | 17 | Cover | -0.126 | 0.032 | Loss |
| CIC-IDS2017 | 17 | Gain | -0.009 | 0.015 | Tie |
| CIC-IDS2017 | 17 | Weight | +0.030 | 0.016 | Win |
| CIC-IDS2017 | 20 | Cover | -0.097 | 0.031 | Loss |
| CIC-IDS2017 | 20 | Gain | -0.000 | 0.030 | Tie |
| CIC-IDS2017 | 20 | Weight | +0.020 | 0.011 | Win |
| Covertype | 2 | Cover | -0.262 | 0.004 | Loss |
| Covertype | 2 | Gain | -0.001 | 0.003 | Tie |
| Covertype | 2 | Weight | -0.167 | 0.003 | Loss |
| Covertype | 3 | Cover | -0.305 | 0.016 | Loss |
| Covertype | 3 | Gain | -0.052 | 0.009 | Loss |
| Covertype | 3 | Weight | +0.002 | 0.002 | Tie |
| Covertype | 5 | Cover | -0.371 | 0.014 | Loss |
| Covertype | 5 | Gain | -0.143 | 0.011 | Loss |
| Covertype | 5 | Weight | -0.015 | 0.003 | Loss |
| Covertype | 6 | Cover | -0.318 | 0.045 | Loss |
| Covertype | 6 | Gain | -0.122 | 0.011 | Loss |
| Covertype | 6 | Weight | +0.010 | 0.010 | Win |
| Covertype | 7 | Cover | -0.296 | 0.045 | Loss |
| Covertype | 7 | Gain | -0.111 | 0.012 | Loss |
| Covertype | 7 | Weight | +0.013 | 0.008 | Win |
| Covertype | 9 | Cover | -0.273 | 0.009 | Loss |
| Covertype | 9 | Gain | -0.102 | 0.024 | Loss |
| Covertype | 9 | Weight | -0.006 | 0.007 | Tie |
| Covertype | 11 | Cover | -0.267 | 0.008 | Loss |
| Covertype | 11 | Gain | -0.077 | 0.012 | Loss |
| Covertype | 11 | Weight | +0.005 | 0.005 | Tie |
| Covertype | 14 | Cover | -0.246 | 0.011 | Loss |
| Covertype | 14 | Gain | -0.002 | 0.012 | Tie |
| Covertype | 14 | Weight | -0.002 | 0.002 | Tie |
| Covertype | 17 | Cover | -0.210 | 0.054 | Loss |
| Covertype | 17 | Gain | -0.000 | 0.014 | Tie |
| Covertype | 17 | Weight | +0.006 | 0.001 | Tie |
| Credit Card Fraud | 1 | Cover | +0.052 | 0.044 | Win |
| Credit Card Fraud | 1 | Gain | +0.000 | 0.000 | Tie |
| Credit Card Fraud | 1 | Weight | -0.254 | 0.059 | Loss |
| Credit Card Fraud | 2 | Cover | +0.020 | 0.014 | Win |
| Credit Card Fraud | 2 | Gain | +0.058 | 0.013 | Win |
| Credit Card Fraud | 2 | Weight | -0.099 | 0.095 | Loss |
| Credit Card Fraud | 3 | Cover | -0.001 | 0.013 | Tie |
| Credit Card Fraud | 3 | Gain | +0.006 | 0.011 | Tie |
| Credit Card Fraud | 3 | Weight | -0.041 | 0.018 | Loss |
| Credit Card Fraud | 4 | Cover | -0.012 | 0.016 | Loss |
| Credit Card Fraud | 4 | Gain | -0.009 | 0.014 | Tie |
| Credit Card Fraud | 4 | Weight | -0.030 | 0.029 | Loss |
| Credit Card Fraud | 5 | Cover | +0.005 | 0.006 | Tie |
| Credit Card Fraud | 5 | Gain | -0.001 | 0.011 | Tie |
| Credit Card Fraud | 5 | Weight | -0.011 | 0.020 | Loss |
| Credit Card Fraud | 6 | Cover | -0.002 | 0.006 | Tie |
| Credit Card Fraud | 6 | Gain | +0.000 | 0.010 | Tie |
| Credit Card Fraud | 6 | Weight | -0.010 | 0.016 | Tie |
| Credit Card Fraud | 8 | Cover | -0.004 | 0.005 | Tie |
| Credit Card Fraud | 8 | Gain | +0.002 | 0.011 | Tie |
| Credit Card Fraud | 8 | Weight | -0.006 | 0.009 | Tie |
| Credit Card Fraud | 9 | Cover | -0.005 | 0.012 | Tie |
| Credit Card Fraud | 9 | Gain | +0.001 | 0.012 | Tie |
| Credit Card Fraud | 9 | Weight | +0.001 | 0.008 | Tie |
| Dry Bean | 1 | Cover | +0.118 | 0.017 | Win |
| Dry Bean | 1 | Gain | +0.046 | 0.063 | Win |
| Dry Bean | 1 | Weight | -0.192 | 0.008 | Loss |
| Dry Bean | 2 | Cover | -0.113 | 0.151 | Loss |
| Dry Bean | 2 | Gain | -0.240 | 0.215 | Loss |
| Dry Bean | 2 | Weight | -0.354 | 0.006 | Loss |
| Dry Bean | 3 | Cover | -0.000 | 0.001 | Tie |
| Dry Bean | 3 | Gain | -0.000 | 0.001 | Tie |
| Dry Bean | 3 | Weight | -0.288 | 0.007 | Loss |
| Dry Bean | 4 | Cover | -0.000 | 0.001 | Tie |
| Dry Bean | 4 | Gain | -0.000 | 0.001 | Tie |
| Dry Bean | 4 | Weight | -0.270 | 0.009 | Loss |
| Dry Bean | 5 | Cover | -0.004 | 0.003 | Tie |
| Dry Bean | 5 | Gain | -0.004 | 0.004 | Tie |
| Dry Bean | 5 | Weight | -0.000 | 0.018 | Tie |
| Steel Plates Faults | 1 | Cover | -0.159 | 0.108 | Loss |
| Steel Plates Faults | 1 | Gain | -0.248 | 0.081 | Loss |
| Steel Plates Faults | 1 | Weight | -0.310 | 0.024 | Loss |
| Steel Plates Faults | 2 | Cover | -0.175 | 0.049 | Loss |
| Steel Plates Faults | 2 | Gain | -0.303 | 0.084 | Loss |
| Steel Plates Faults | 2 | Weight | -0.121 | 0.044 | Loss |
| Steel Plates Faults | 3 | Cover | -0.192 | 0.048 | Loss |
| Steel Plates Faults | 3 | Gain | -0.364 | 0.026 | Loss |
| Steel Plates Faults | 3 | Weight | -0.172 | 0.048 | Loss |
| Steel Plates Faults | 4 | Cover | -0.085 | 0.053 | Loss |
| Steel Plates Faults | 4 | Gain | -0.097 | 0.030 | Loss |
| Steel Plates Faults | 4 | Weight | +0.002 | 0.080 | Tie |
| Steel Plates Faults | 5 | Cover | -0.020 | 0.047 | Loss |
| Steel Plates Faults | 5 | Gain | -0.062 | 0.055 | Loss |
| Steel Plates Faults | 5 | Weight | +0.064 | 0.035 | Win |
| Steel Plates Faults | 6 | Cover | -0.008 | 0.035 | Tie |
| Steel Plates Faults | 6 | Gain | -0.046 | 0.038 | Loss |
| Steel Plates Faults | 6 | Weight | +0.005 | 0.036 | Tie |
| Steel Plates Faults | 7 | Cover | -0.003 | 0.032 | Tie |
| Steel Plates Faults | 7 | Gain | -0.041 | 0.059 | Loss |
| Steel Plates Faults | 7 | Weight | -0.026 | 0.056 | Loss |
| Steel Plates Faults | 9 | Cover | -0.028 | 0.054 | Loss |
| Steel Plates Faults | 9 | Gain | -0.039 | 0.065 | Loss |
| Steel Plates Faults | 9 | Weight | -0.007 | 0.058 | Tie |
| UNSW-NB15 | 1 | Cover | -0.130 | 0.178 | Loss |
| UNSW-NB15 | 1 | Gain | -0.101 | 0.178 | Loss |
| UNSW-NB15 | 1 | Weight | +0.183 | 0.178 | Win |
| UNSW-NB15 | 2 | Cover | -0.191 | 0.073 | Loss |
| UNSW-NB15 | 2 | Gain | -0.147 | 0.019 | Loss |
| UNSW-NB15 | 2 | Weight | -0.074 | 0.002 | Loss |
| UNSW-NB15 | 3 | Cover | -0.125 | 0.056 | Loss |
| UNSW-NB15 | 3 | Gain | -0.147 | 0.001 | Loss |
| UNSW-NB15 | 3 | Weight | -0.100 | 0.024 | Loss |
| UNSW-NB15 | 4 | Cover | -0.034 | 0.030 | Loss |
| UNSW-NB15 | 4 | Gain | -0.052 | 0.001 | Loss |
| UNSW-NB15 | 4 | Weight | -0.041 | 0.007 | Loss |
| UNSW-NB15 | 5 | Cover | +0.012 | 0.032 | Win |
| UNSW-NB15 | 5 | Gain | -0.012 | 0.027 | Loss |
| UNSW-NB15 | 5 | Weight | -0.031 | 0.010 | Loss |
| UNSW-NB15 | 6 | Cover | -0.009 | 0.001 | Tie |
| UNSW-NB15 | 6 | Gain | -0.009 | 0.002 | Tie |
| UNSW-NB15 | 6 | Weight | -0.071 | 0.001 | Loss |
| UNSW-NB15 | 8 | Cover | -0.007 | 0.004 | Tie |
| UNSW-NB15 | 8 | Gain | +0.002 | 0.000 | Tie |
| UNSW-NB15 | 8 | Weight | -0.034 | 0.014 | Loss |
| UNSW-NB15 | 10 | Cover | -0.001 | 0.004 | Tie |
| UNSW-NB15 | 10 | Gain | +0.002 | 0.000 | Tie |
| UNSW-NB15 | 10 | Weight | +0.026 | 0.000 | Win |
| UNSW-NB15 | 12 | Cover | +0.001 | 0.001 | Tie |
| UNSW-NB15 | 12 | Gain | -0.000 | 0.000 | Tie |
| UNSW-NB15 | 12 | Weight | +0.025 | 0.001 | Win |
