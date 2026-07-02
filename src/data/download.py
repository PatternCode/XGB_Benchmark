from pathlib import Path
import json

import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, fetch_covtype
from ucimlrepo import fetch_ucirepo

from dataset_registry import DATASETS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def save_dataset(name: str, X: pd.DataFrame, y: pd.Series, metadata: dict) -> None:
    out_dir = RAW_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = X.copy()
    df[metadata["target"]] = y

    df.to_csv(out_dir / "data.csv", index=False)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[OK] {name}: {df.shape[0]} samples, {df.shape[1] - 1} features")


def download_openml(name: str, config: dict) -> None:
    dataset = fetch_openml(name=config["openml_name"], as_frame=True)
    X = dataset.data
    y = dataset.target

    metadata = {
        "name": name,
        "source": "OpenML",
        "target": config["target"],
        "task": config["task"],
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
    }

    save_dataset(name, X, y, metadata)


def download_sklearn(name: str, config: dict) -> None:
    if config["loader"] == "load_breast_cancer":
        dataset = load_breast_cancer(as_frame=True)
        X = dataset.data
        y = dataset.target

    elif config["loader"] == "fetch_covtype":
        dataset = fetch_covtype(as_frame=True)
        X = dataset.data
        y = dataset.target

    else:
        raise ValueError(f"Unknown sklearn loader: {config['loader']}")

    metadata = {
        "name": name,
        "source": "scikit-learn",
        "target": config["target"],
        "task": config["task"],
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
    }

    save_dataset(name, X, y, metadata)


def download_uci(name: str, config: dict) -> None:
    dataset = fetch_ucirepo(id=config["uci_id"])

    X = dataset.data.features
    y_df = dataset.data.targets

    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
        target_name = y_df.columns[0]
    else:
        # Steel Plates Faults has multiple one-hot fault columns.
        # Convert one-hot fault indicators into a single multiclass target.
        y = y_df.idxmax(axis=1)
        target_name = "fault_type"

    metadata = {
        "name": name,
        "source": "UCI Machine Learning Repository",
        "uci_id": config["uci_id"],
        "target": target_name,
        "task": config["task"],
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
    }

    metadata["target"] = target_name
    save_dataset(name, X, y, metadata)


def main() -> None:
    for name, config in DATASETS.items():
        print(f"Downloading {name}...")

        if config["source"] == "openml":
            download_openml(name, config)
        elif config["source"] == "sklearn":
            download_sklearn(name, config)
        elif config["source"] == "uci":
            download_uci(name, config)
        else:
            raise ValueError(f"Unknown source: {config['source']}")


if __name__ == "__main__":
    main()
