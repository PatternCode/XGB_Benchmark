# CIC-IDS2017

This dataset requires manual download.

## Source

- **Provider:** Canadian Institute for Cybersecurity
- **URL:** https://www.unb.ca/cic/datasets/ids-2017.html

## Download

Download all CSV files from the official dataset repository.

## Raw Data Standardization

This project assumes that every dataset in `data/raw/` contains a single data file named:

```
data.csv
```

After downloading:

1. Place all original CSV files in this directory.
2. Merge the CSV files into a single file named `data.csv`.
3. Once `data.csv` has been verified, the original CSV files may be removed.
4. Do **not** commit any dataset files to Git.

The directory should finally look like:

```
data/raw/cic_ids2017/
├── data.csv
├── metadata.json
└── README.md
```

## Notes

- The merged `data.csv` contains all records from the original daily capture files.
- The preprocessing pipeline assumes that the raw dataset has already been standardized into this format.