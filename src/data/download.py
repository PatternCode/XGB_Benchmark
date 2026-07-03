from pathlib import Path
import json

import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, fetch_covtype
from ucimlrepo import fetch_ucirepo

from registry import DATASETS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def save_dataset(dataset_key: str, X: pd.DataFrame, y: pd.Series, metadata: dict) -> None:
    out_dir = RAW_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    target = metadata["target"]

    df = X.copy()
    df[target] = y

    df.to_csv(out_dir / "data.csv", index=False)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[OK] {dataset_key}: {df.shape[0]} samples, {df.shape[1] - 1} features")


def build_metadata(dataset_key: str, config: dict, source_name: str, samples: int, features: int) -> dict:
    return {
        "key": dataset_key,
        "name": config["name"],
        "domain": config["domain"],
        "source": source_name,
        "target": config["target"],
        "task": config["task"],
        "size": config["size"],
        "samples": int(samples),
        "features": int(features),
    }


def download_openml(dataset_key: str, config: dict) -> None:
    source = config["source"]

    dataset = fetch_openml(name=source["name"], version=source.get("version"),as_frame=True)
    X = dataset.data
    y = dataset.target

    metadata = build_metadata(
        dataset_key,
        config,
        source_name="OpenML",
        samples=X.shape[0],
        features=X.shape[1],
    )

    save_dataset(dataset_key, X, y, metadata)


def download_sklearn(dataset_key: str, config: dict) -> None:
    source = config["source"]

    if source["loader"] == "load_breast_cancer":
        dataset = load_breast_cancer(as_frame=True)
    elif source["loader"] == "fetch_covtype":
        dataset = fetch_covtype(as_frame=True)
    else:
        raise ValueError(f"Unknown sklearn loader: {source['loader']}")

    X = dataset.data
    y = dataset.target

    metadata = build_metadata(
        dataset_key,
        config,
        source_name="scikit-learn",
        samples=X.shape[0],
        features=X.shape[1],
    )

    save_dataset(dataset_key, X, y, metadata)


def download_uci(dataset_key: str, config: dict) -> None:
    source = config["source"]

    dataset = fetch_ucirepo(id=source["id"])

    X = dataset.data.features
    y_df = dataset.data.targets

    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
        target_name = y_df.columns[0]
    else:
        y = y_df.idxmax(axis=1)
        target_name = config["target"]

    metadata = build_metadata(
        dataset_key,
        config,
        source_name="UCI Machine Learning Repository",
        samples=X.shape[0],
        features=X.shape[1],
    )

    metadata["uci_id"] = source["id"]
    metadata["target"] = target_name

    save_dataset(dataset_key, X, y, metadata)


def download_kaggle(dataset_key: str, config: dict) -> None:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for Kaggle datasets. "
            "Install it with: pip install kagglehub[pandas-datasets]"
        ) from exc

    source = config["source"]

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        source["slug"],
        source.get("file_path") or "",
    )

    target = config["target"]
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in {dataset_key}. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    metadata = build_metadata(
        dataset_key,
        config,
        source_name="Kaggle",
        samples=X.shape[0],
        features=X.shape[1],
    )

    metadata["kaggle_slug"] = source["slug"]
    metadata["requires_auth"] = source.get("requires_auth", True)

    save_dataset(dataset_key, X, y, metadata)


def write_manual_download_note(dataset_key: str, config: dict) -> None:
    source = config["source"]

    out_dir = RAW_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_files = "\n".join(
        f"- `{file}`" for file in source.get("expected_files", [])
    )

    readme = f"""# {config["name"]}

This dataset requires manual download.

## Source

- Provider: {source.get("provider", "Unknown")}
- URL: {source.get("url", "Not specified")}

## Expected files

{expected_files if expected_files else "- Not specified"}

## Instructions

1. Download the dataset from the source above.
2. Place the expected files in this folder:

   `data/raw/{dataset_key}/`

3. Do not commit dataset files to Git.

## Notes

{config.get("notes", "Manual download required.")}
"""

    metadata = {
        "key": dataset_key,
        "name": config["name"],
        "domain": config["domain"],
        "source": source.get("provider", "manual"),
        "url": source.get("url"),
        "target": config["target"],
        "task": config["task"],
        "size": config["size"],
        "download": "manual",
        "expected_files": source.get("expected_files", []),
        "notes": config.get("notes", ""),
    }

    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[MANUAL] {dataset_key}: see {out_dir / 'README.md'}")


def main() -> None:
    for dataset_key, config in DATASETS.items():
        source_type = config["source"]["type"]

        print(f"Processing {dataset_key}...")

        if source_type == "openml":
            download_openml(dataset_key, config)
        elif source_type == "sklearn":
            download_sklearn(dataset_key, config)
        elif source_type == "uci":
            download_uci(dataset_key, config)
        elif source_type == "kaggle":
            download_kaggle(dataset_key, config)
        elif source_type == "manual":
            write_manual_download_note(dataset_key, config)
        else:
            raise ValueError(f"Unknown source type: {source_type}")


if __name__ == "__main__":
    main()