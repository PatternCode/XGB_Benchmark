"""Coordinate leakage-free XGB Benchmark experiments."""

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from benchmark.datasets import load_dataset
from benchmark.importance import compute_feature_importance
from benchmark.metrics import calculate_metrics
from benchmark.models import predict_model, train_model
from benchmark.selection import (
    get_unique_feature_counts,
    select_random_features,
    select_top_k,
)


class ExperimentError(Exception):
    """Raised when an experiment cannot be completed."""


def _get_enabled_datasets(
    config: dict[str, Any],
) -> list[str]:
    """Return the names of enabled datasets."""
    datasets = config["datasets"]

    enabled_datasets = [
        dataset_name
        for dataset_name, settings in datasets.items()
        if settings.get("enabled", False)
    ]

    if not enabled_datasets:
        raise ExperimentError(
            "At least one dataset must be enabled."
        )

    return enabled_datasets


def _get_model_variants(
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Expand enabled model configurations into trainable variants."""
    variants: list[tuple[str, dict[str, Any]]] = []

    for model_name, model_config in config["models"].items():
        if not model_config.get("enabled", False):
            continue

        if model_name == "decision_tree":
            max_depths = model_config.get("max_depths")

            if not isinstance(max_depths, list) or not max_depths:
                raise ExperimentError(
                    "models.decision_tree.max_depths must be "
                    "a non-empty list."
                )

            for max_depth in max_depths:
                variants.append(
                    (
                        model_name,
                        {
                            "max_depth": max_depth,
                        },
                    )
                )

            continue

        variant_config = {
            key: value
            for key, value in model_config.items()
            if key != "enabled"
        }

        variants.append(
            (
                model_name,
                variant_config,
            )
        )

    if not variants:
        raise ExperimentError(
            "At least one downstream model must be enabled."
        )

    return variants


def _derive_random_seed(
    base_seed: int,
    dataset_index: int,
    outer_fold: int,
    subset_index: int = 0,
    repetition: int = 0,
) -> int:
    """Create a deterministic seed for one experimental operation."""
    return (
        base_seed
        + dataset_index * 100_000
        + outer_fold * 10_000
        + subset_index * 100
        + repetition
    )


def _model_parameter_fields(
    model_name: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Return flat model-parameter fields for one result row."""
    return {
        "max_depth": (
            model_config.get("max_depth")
            if model_name == "decision_tree"
            else None
        ),
        "n_neighbors": (
            model_config.get("n_neighbors")
            if model_name == "knn"
            else None
        ),
        "C": (
            model_config.get("C")
            if model_name == "rbf_svm"
            else None
        ),
        "gamma": (
            model_config.get("gamma")
            if model_name == "rbf_svm"
            else None
        ),
    }


def _evaluate_feature_subset(
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_features: list[str],
    model_name: str,
    model_config: dict[str, Any],
    metric_names: list[str],
    random_seed: int,
) -> dict[str, float]:
    """Train and evaluate one model on one feature subset."""
    X_train_selected = X_train.loc[:, selected_features]
    X_test_selected = X_test.loc[:, selected_features]

    model = train_model(
        X_train=X_train_selected,
        y_train=y_train,
        model_name=model_name,
        model_config=model_config,
        random_seed=random_seed,
    )

    y_pred, y_prob = predict_model(
        model=model,
        X=X_test_selected,
        n_classes=int(y_train.nunique()),
    )

    return calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        metric_names=metric_names,
    )


def _build_result_row(
    *,
    experiment_name: str,
    dataset_name: str,
    outer_fold: int,
    selection_method: str,
    requested_percentage: float | None,
    actual_percentage: float,
    n_selected_features: int,
    random_repetition: int | None,
    model_name: str,
    model_config: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Create one flat experiment-result record."""
    result: dict[str, Any] = {
        "experiment": experiment_name,
        "dataset": dataset_name,
        "outer_fold": outer_fold,
        "selection_method": selection_method,
        "requested_percentage": requested_percentage,
        "actual_percentage": actual_percentage,
        "n_selected_features": n_selected_features,
        "random_repetition": random_repetition,
        "model": model_name,
    }

    result.update(
        _model_parameter_fields(
            model_name=model_name,
            model_config=model_config,
        )
    )

    result.update(metrics)

    return result


def run_experiment(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run the complete leakage-free benchmark experiment.

    The function performs stratified outer cross-validation. Feature
    ranking, feature selection, preprocessing, and model fitting are all
    performed using outer-training data only. The outer test fold is used
    only for prediction and final metric calculation.

    Parameters
    ----------
    config
        Validated benchmark configuration.

    Returns
    -------
    list[dict[str, Any]]
        One flat result record for every evaluated combination of dataset,
        outer fold, feature-selection strategy, subset size, and downstream
        model.

    Raises
    ------
    ExperimentError
        If no dataset or downstream model is enabled, or if the experiment
        configuration is inconsistent.
    """
    experiment_name = config["experiment"]["name"]
    base_seed = config["experiment"]["random_seed"]

    enabled_datasets = _get_enabled_datasets(config)
    model_variants = _get_model_variants(config)

    ranking_methods = config["feature_selection"][
        "ranking_methods"
    ]
    feature_percentages = config["feature_selection"][
        "feature_percentages"
    ]
    random_repetitions = config["feature_selection"][
        "random_repetitions"
    ]
    metric_names = config["metrics"]

    cross_validation = config["cross_validation"]

    splitter = StratifiedKFold(
        n_splits=cross_validation["n_splits"],
        shuffle=cross_validation["shuffle"],
        random_state=(
            base_seed
            if cross_validation["shuffle"]
            else None
        ),
    )

    results: list[dict[str, Any]] = []

    for dataset_index, dataset_name in enumerate(
        enabled_datasets
    ):
        dataset = load_dataset(dataset_name)

        subset_sizes = get_unique_feature_counts(
            percentages=feature_percentages,
            n_features=dataset.n_features,
        )

        for outer_fold, (
            train_indices,
            test_indices,
        ) in enumerate(
            splitter.split(dataset.X, dataset.y),
            start=1,
        ):
            X_train = dataset.X.iloc[train_indices].copy()
            X_test = dataset.X.iloc[test_indices].copy()
            y_train = dataset.y.iloc[train_indices].copy()
            y_test = dataset.y.iloc[test_indices].copy()

            fold_seed = _derive_random_seed(
                base_seed=base_seed,
                dataset_index=dataset_index,
                outer_fold=outer_fold,
            )

            _, rankings = compute_feature_importance(
                X_train=X_train,
                y_train=y_train,
                xgboost_config=config["ranking_xgboost"],
                shap_config=config["shap"],
                ranking_methods=ranking_methods,
                random_seed=fold_seed,
            )

            for method in ranking_methods:
                ranking = rankings[method]

                for subset_index, subset_info in enumerate(
                    subset_sizes
                ):
                    selected_features = select_top_k(
                        ranking=ranking,
                        k=int(
                            subset_info[
                                "n_selected_features"
                            ]
                        ),
                    )

                    for model_name, model_config in model_variants:
                        model_seed = _derive_random_seed(
                            base_seed=base_seed,
                            dataset_index=dataset_index,
                            outer_fold=outer_fold,
                            subset_index=subset_index,
                        )

                        metrics = _evaluate_feature_subset(
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            selected_features=selected_features,
                            model_name=model_name,
                            model_config=model_config,
                            metric_names=metric_names,
                            random_seed=model_seed,
                        )

                        results.append(
                            _build_result_row(
                                experiment_name=experiment_name,
                                dataset_name=dataset_name,
                                outer_fold=outer_fold,
                                selection_method=method,
                                requested_percentage=float(
                                    subset_info[
                                        "requested_percentage"
                                    ]
                                ),
                                actual_percentage=float(
                                    subset_info[
                                        "actual_percentage"
                                    ]
                                ),
                                n_selected_features=len(
                                    selected_features
                                ),
                                random_repetition=None,
                                model_name=model_name,
                                model_config=model_config,
                                metrics=metrics,
                            )
                        )

            for subset_index, subset_info in enumerate(
                subset_sizes
            ):
                k = int(
                    subset_info["n_selected_features"]
                )

                for repetition in range(
                    random_repetitions
                ):
                    random_seed = _derive_random_seed(
                        base_seed=base_seed,
                        dataset_index=dataset_index,
                        outer_fold=outer_fold,
                        subset_index=subset_index,
                        repetition=repetition + 1,
                    )

                    selected_features = select_random_features(
                        feature_names=X_train.columns.tolist(),
                        k=k,
                        random_seed=random_seed,
                    )

                    for model_name, model_config in model_variants:
                        metrics = _evaluate_feature_subset(
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            selected_features=selected_features,
                            model_name=model_name,
                            model_config=model_config,
                            metric_names=metric_names,
                            random_seed=random_seed,
                        )

                        results.append(
                            _build_result_row(
                                experiment_name=experiment_name,
                                dataset_name=dataset_name,
                                outer_fold=outer_fold,
                                selection_method="random",
                                requested_percentage=float(
                                    subset_info[
                                        "requested_percentage"
                                    ]
                                ),
                                actual_percentage=float(
                                    subset_info[
                                        "actual_percentage"
                                    ]
                                ),
                                n_selected_features=len(
                                    selected_features
                                ),
                                random_repetition=repetition + 1,
                                model_name=model_name,
                                model_config=model_config,
                                metrics=metrics,
                            )
                        )

            all_features = X_train.columns.tolist()

            for model_name, model_config in model_variants:
                metrics = _evaluate_feature_subset(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    selected_features=all_features,
                    model_name=model_name,
                    model_config=model_config,
                    metric_names=metric_names,
                    random_seed=fold_seed,
                )

                results.append(
                    _build_result_row(
                        experiment_name=experiment_name,
                        dataset_name=dataset_name,
                        outer_fold=outer_fold,
                        selection_method="all_features",
                        requested_percentage=None,
                        actual_percentage=100.0,
                        n_selected_features=len(all_features),
                        random_repetition=None,
                        model_name=model_name,
                        model_config=model_config,
                        metrics=metrics,
                    )
                )

    return results