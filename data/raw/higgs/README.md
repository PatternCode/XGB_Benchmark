# HIGGS

This dataset requires manual download.

## Source

- **Provider:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/ml/datasets/HIGGS

## Download

Download the compressed dataset (`HIGGS.csv.gz`) from the official repository and extract it.

## Raw Data Standardization

This project assumes that every dataset in `data/raw/` contains a single data file named:

```
data.csv
```

After downloading:

1. Extract `HIGGS.csv.gz`.
2. Rename `HIGGS.csv` to `data.csv`.
3. Place `data.csv` in this directory.
4. Do **not** commit dataset files to Git.

The directory should finally look like:

```
data/raw/higgs/
├── data.csv
├── metadata.json
└── README.md
```

## Notes

- The original downloaded file is named `HIGGS.csv`.
- The preprocessing pipeline assumes that the raw dataset has already been standardized into this format.
- This is a very large dataset (approximately 11 million samples), so downloading and preprocessing may take considerable time.