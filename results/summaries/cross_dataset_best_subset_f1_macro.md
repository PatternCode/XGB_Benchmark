# Cross-dataset compact-subset comparison — Macro-F1

Each cell reports **mean Macro-F1 (k)**.
For Gain, Weight, Cover, SHAP, and Random, k is selected using a descriptive one-standard-error rule: choose the smallest subset whose mean performance lies within one standard error of that method's best observed subset performance for the same dataset and downstream model.
For Random, the 20 repetitions are averaged within each fold/k/model condition before the fold-level values are summarized.
The **All** column reports the mean performance using the complete feature set.

## LR

| Dataset | Gain | Weight | Cover | SHAP | Random | All |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult Income | 0.720 (k=4) | 0.537 (k=5) | 0.720 (k=4) | 0.704 (k=5) | 0.574 (k=5) | 0.721 (k=14) |
| Bank Marketing | 0.678 (k=4) | 0.596 (k=4) | 0.678 (k=5) | 0.678 (k=5) | 0.548 (k=5) | 0.679 (k=16) |
| Breast Cancer Wisconsin | 0.935 (k=4) | 0.970 (k=9) | 0.943 (k=4) | 0.957 (k=5) | 0.949 (k=8) | 0.971 (k=30) |
| CIC-IDS2017 | 0.345 (k=20) | 0.332 (k=17) | 0.301 (k=20) | 0.387 (k=20) | 0.380 (k=20) | 0.601 (k=65) |
| Covertype | 0.432 (k=17) | 0.500 (k=17) | 0.286 (k=17) | 0.487 (k=17) | 0.297 (k=17) | 0.533 (k=54) |
| Credit Card Fraud | 0.849 (k=3) | 0.844 (k=6) | 0.846 (k=4) | 0.845 (k=4) | 0.816 (k=9) | 0.861 (k=30) |
| Dry Bean | 0.888 (k=3) | 0.910 (k=5) | 0.888 (k=3) | 0.916 (k=5) | 0.903 (k=5) | 0.936 (k=16) |
| Steel Plates Faults | 0.558 (k=9) | 0.514 (k=9) | 0.562 (k=9) | 0.618 (k=9) | 0.546 (k=9) | 0.716 (k=27) |
| UNSW-NB15 | 0.690 (k=12) | 0.675 (k=12) | 0.693 (k=12) | 0.674 (k=12) | 0.624 (k=12) | 0.785 (k=39) |

## XGBoost

| Dataset | Gain | Weight | Cover | SHAP | Random | All |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult Income | 0.787 (k=5) | 0.688 (k=5) | 0.788 (k=5) | 0.781 (k=5) | 0.713 (k=5) | 0.811 (k=14) |
| Bank Marketing | 0.737 (k=5) | 0.692 (k=5) | 0.692 (k=5) | 0.734 (k=5) | 0.602 (k=5) | 0.736 (k=16) |
| Breast Cancer Wisconsin | 0.935 (k=6) | 0.964 (k=9) | 0.924 (k=4) | 0.951 (k=8) | 0.942 (k=8) | 0.956 (k=30) |
| CIC-IDS2017 | 0.715 (k=20) | 0.807 (k=20) | 0.682 (k=20) | 0.812 (k=20) | 0.744 (k=20) | 0.813 (k=65) |
| Covertype | 0.498 (k=17) | 0.770 (k=14) | 0.290 (k=17) | 0.777 (k=17) | 0.435 (k=17) | 0.793 (k=54) |
| Credit Card Fraud | 0.917 (k=8) | 0.925 (k=8) | 0.922 (k=9) | 0.929 (k=8) | 0.900 (k=9) | 0.929 (k=30) |
| Dry Bean | 0.887 (k=3) | 0.919 (k=5) | 0.890 (k=3) | 0.914 (k=5) | 0.907 (k=5) | 0.940 (k=16) |
| Steel Plates Faults | 0.747 (k=9) | 0.803 (k=9) | 0.733 (k=9) | 0.799 (k=9) | 0.730 (k=9) | 0.822 (k=27) |
| UNSW-NB15 | 0.966 (k=12) | 0.972 (k=10) | 0.955 (k=12) | 0.971 (k=10) | 0.893 (k=12) | 0.983 (k=39) |
