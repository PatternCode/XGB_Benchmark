"""Coordinate leakage-free XGB Benchmark experiments."""

from time import perf_counter
from typing import Any, Callable

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from benchmark.datasets import load_dataset
from benchmark.importance import compute_feature_importance
from benchmark.metrics import calculate_metrics
from benchmark.models import get_model_complexity, predict_model, train_model
from benchmark.selection import (
    get_unique_feature_counts,
    select_random_features,
    select_top_k,
)

ExperimentOutput = dict[str, list[dict[str, Any]]]
DatasetCompleteCallback = Callable[[str, ExperimentOutput], None]


class ExperimentError(Exception):
    """Raised when an experiment cannot be completed."""


def _format_duration(seconds: float) -> str:
    """Return a human-readable duration."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    if hours:
        return f"{hours} h {minutes} min {remaining_seconds} s"
    if minutes:
        return f"{minutes} min {remaining_seconds} s"
    return f"{remaining_seconds} s"


def _get_enabled_datasets(config: dict[str, Any]) -> list[str]:
    """Return the names of enabled datasets."""
    enabled_datasets = [
        dataset_name
        for dataset_name, settings in config["datasets"].items()
        if settings.get("enabled", False)
    ]
    if not enabled_datasets:
        raise ExperimentError("At least one dataset must be enabled.")
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
                    "models.decision_tree.max_depths must be a non-empty list."
                )
            for max_depth in max_depths:
                variants.append((model_name, {"max_depth": max_depth}))
            continue

        variants.append(
            (
                model_name,
                {
                    key: value
                    for key, value in model_config.items()
                    if key != "enabled"
                },
            )
        )

    if not variants:
        raise ExperimentError("At least one downstream model must be enabled.")
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
        "max_depth": model_config.get("max_depth")
        if model_name == "decision_tree"
        else None,
        "n_neighbors": model_config.get("n_neighbors")
        if model_name == "knn"
        else None,
        "C": model_config.get("C") if model_name == "rbf_svm" else None,
        "gamma": model_config.get("gamma")
        if model_name == "rbf_svm"
        else None,
    }


def _evaluate_feature_subset(
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str,
    model_config: dict[str, Any],
    metric_names: list[str],
    random_seed: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, int | None]]:
    """Train and evaluate one model on one feature subset."""
    X_train_selected = X_train.loc[:, selected_features]
    X_test_selected = X_test.loc[:, selected_features]
    selected_numeric_features = [
        feature for feature in selected_features if feature in numeric_features
    ]
    selected_categorical_features = [
        feature
        for feature in selected_features
        if feature in categorical_features
    ]

    training_start = perf_counter()
    model = train_model(
        X_train=X_train_selected,
        y_train=y_train,
        model_name=model_name,
        model_config=model_config,
        random_seed=random_seed,
        numeric_features=selected_numeric_features,
        categorical_features=selected_categorical_features,
    )
    model_training_time = perf_counter() - training_start

    prediction_start = perf_counter()
    y_pred, y_prob = predict_model(
        model=model,
        X=X_test_selected,
        n_classes=int(y_train.nunique()),
    )
    prediction_time = perf_counter() - prediction_start

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        metric_names=metric_names,
    )
    timings = {
        "model_training_time_seconds": model_training_time,
        "prediction_time_seconds": prediction_time,
    }
    return metrics, timings, get_model_complexity(model)


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
    ranking_time_seconds: float | None,
    timings: dict[str, float],
    complexity: dict[str, int | None],
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
        "ranking_time_seconds": ranking_time_seconds,
    }
    result.update(_model_parameter_fields(model_name, model_config))
    result.update(timings)
    result.update(complexity)
    result.update(metrics)
    return result


def _build_ranking_rows(
    *,
    experiment_name: str,
    dataset_name: str,
    outer_fold: int,
    ranking_method: str,
    ranking: pd.Series,
) -> list[dict[str, Any]]:
    """Create one record for every feature in an importance ranking."""
    return [
        {
            "experiment": experiment_name,
            "dataset": dataset_name,
            "outer_fold": outer_fold,
            "ranking_method": ranking_method,
            "feature": str(feature),
            "rank": rank,
            "importance_score": float(score),
        }
        for rank, (feature, score) in enumerate(ranking.items(), start=1)
    ]


def _build_selected_feature_rows(
    *,
    experiment_name: str,
    dataset_name: str,
    outer_fold: int,
    selection_method: str,
    requested_percentage: float | None,
    actual_percentage: float,
    random_repetition: int | None,
    selected_features: list[str],
    ranking: pd.Series | None,
) -> list[dict[str, Any]]:
    """Create one record for every selected feature."""
    rows: list[dict[str, Any]] = []
    for selection_rank, feature in enumerate(selected_features, start=1):
        rows.append(
            {
                "experiment": experiment_name,
                "dataset": dataset_name,
                "outer_fold": outer_fold,
                "selection_method": selection_method,
                "requested_percentage": requested_percentage,
                "actual_percentage": actual_percentage,
                "random_repetition": random_repetition,
                "feature": feature,
                "selection_rank": selection_rank,
                "importance_score": (
                    float(ranking.loc[feature]) if ranking is not None else None
                ),
            }
        )
    return rows


def _build_experiment_output(
    *,
    results: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    selected_feature_rows: list[dict[str, Any]],
) -> ExperimentOutput:
    """Build the cumulative experiment output structure."""
    return {
        "results": results,
        "rankings": ranking_rows,
        "selected_features": selected_feature_rows,
    }


def run_experiment(
    config: dict[str, Any],
    on_dataset_complete: DatasetCompleteCallback | None = None,
) -> ExperimentOutput:
    """Run the complete leakage-free benchmark experiment."""
    experiment_name = config["experiment"]["name"]
    base_seed = config["experiment"]["random_seed"]
    enabled_datasets = _get_enabled_datasets(config)
    model_variants = _get_model_variants(config)
    ranking_methods = config["feature_selection"]["ranking_methods"]
    feature_percentages = config["feature_selection"]["feature_percentages"]
    random_repetitions = config["feature_selection"]["random_repetitions"]
    metric_names = config["metrics"]
    cross_validation = config["cross_validation"]

    n_datasets = len(enabled_datasets)
    n_folds = cross_validation["n_splits"]
    experiment_start = perf_counter()

    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Datasets: {n_datasets}")
    print(f"Outer folds: {n_folds}")
    print("=" * 60)

    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=cross_validation["shuffle"],
        random_state=base_seed if cross_validation["shuffle"] else None,
    )

    results: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    selected_feature_rows: list[dict[str, Any]] = []

    for dataset_index, dataset_name in enumerate(enabled_datasets):
        dataset_start = perf_counter()
        print()
        print(f"[Dataset {dataset_index + 1}/{n_datasets}] {dataset_name}")

        dataset = load_dataset(dataset_name)
        print(
            f"  Samples: {dataset.X.shape[0]:,} | "
            f"Features: {dataset.n_features}"
        )

        subset_sizes = get_unique_feature_counts(
            percentages=feature_percentages,
            n_features=dataset.n_features,
        )

        for outer_fold, (train_indices, test_indices) in enumerate(
            splitter.split(dataset.X, dataset.y),
            start=1,
        ):
            fold_start = perf_counter()
            print(f"  [Fold {outer_fold}/{n_folds}] Starting")

            X_train = dataset.X.iloc[train_indices].copy()
            X_test = dataset.X.iloc[test_indices].copy()
            y_train = dataset.y.iloc[train_indices].copy()
            y_test = dataset.y.iloc[test_indices].copy()

            fold_seed = _derive_random_seed(
                base_seed=base_seed,
                dataset_index=dataset_index,
                outer_fold=outer_fold,
            )

            print("    Computing feature-importance rankings")
            ranking_start = perf_counter()
            _, rankings = compute_feature_importance(
                X_train=X_train,
                y_train=y_train,
                xgboost_config=config["ranking_xgboost"],
                shap_config=config["shap"],
                ranking_methods=ranking_methods,
                random_seed=fold_seed,
            )
            ranking_time = perf_counter() - ranking_start
            print(
                "    Rankings completed in "
                f"{_format_duration(ranking_time)}"
            )

            for method, ranking in rankings.items():
                ranking_rows.extend(
                    _build_ranking_rows(
                        experiment_name=experiment_name,
                        dataset_name=dataset_name,
                        outer_fold=outer_fold,
                        ranking_method=method,
                        ranking=ranking,
                    )
                )

            for method in ranking_methods:
                print(f"    Selection method: {method}")
                ranking = rankings[method]

                for subset_index, subset_info in enumerate(subset_sizes):
                    requested_percentage = float(
                        subset_info["requested_percentage"]
                    )
                    n_selected_features = int(
                        subset_info["n_selected_features"]
                    )
                    print(
                        f"      {requested_percentage:g}% "
                        f"({n_selected_features} features)"
                    )

                    selected_features = select_top_k(
                        ranking=ranking,
                        k=n_selected_features,
                    )
                    selected_feature_rows.extend(
                        _build_selected_feature_rows(
                            experiment_name=experiment_name,
                            dataset_name=dataset_name,
                            outer_fold=outer_fold,
                            selection_method=method,
                            requested_percentage=requested_percentage,
                            actual_percentage=float(
                                subset_info["actual_percentage"]
                            ),
                            random_repetition=None,
                            selected_features=selected_features,
                            ranking=ranking,
                        )
                    )

                    for model_name, model_config in model_variants:
                        model_seed = _derive_random_seed(
                            base_seed=base_seed,
                            dataset_index=dataset_index,
                            outer_fold=outer_fold,
                            subset_index=subset_index,
                        )
                        metrics, timings, complexity = _evaluate_feature_subset(
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            selected_features=selected_features,
                            numeric_features=dataset.numeric_features,
                            categorical_features=dataset.categorical_features,
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
                                requested_percentage=requested_percentage,
                                actual_percentage=float(
                                    subset_info["actual_percentage"]
                                ),
                                n_selected_features=len(selected_features),
                                random_repetition=None,
                                model_name=model_name,
                                model_config=model_config,
                                ranking_time_seconds=ranking_time,
                                timings=timings,
                                complexity=complexity,
                                metrics=metrics,
                            )
                        )

            print("    Selection method: random")
            for subset_index, subset_info in enumerate(subset_sizes):
                k = int(subset_info["n_selected_features"])
                requested_percentage = float(
                    subset_info["requested_percentage"]
                )
                print(f"      {requested_percentage:g}% ({k} features)")

                for repetition in range(random_repetitions):
                    print(
                        f"        Repetition {repetition + 1}/"
                        f"{random_repetitions}"
                    )
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
                    selected_feature_rows.extend(
                        _build_selected_feature_rows(
                            experiment_name=experiment_name,
                            dataset_name=dataset_name,
                            outer_fold=outer_fold,
                            selection_method="random",
                            requested_percentage=requested_percentage,
                            actual_percentage=float(
                                subset_info["actual_percentage"]
                            ),
                            random_repetition=repetition + 1,
                            selected_features=selected_features,
                            ranking=None,
                        )
                    )

                    for model_name, model_config in model_variants:
                        metrics, timings, complexity = _evaluate_feature_subset(
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            selected_features=selected_features,
                            numeric_features=dataset.numeric_features,
                            categorical_features=dataset.categorical_features,
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
                                requested_percentage=requested_percentage,
                                actual_percentage=float(
                                    subset_info["actual_percentage"]
                                ),
                                n_selected_features=len(selected_features),
                                random_repetition=repetition + 1,
                                model_name=model_name,
                                model_config=model_config,
                                ranking_time_seconds=None,
                                timings=timings,
                                complexity=complexity,
                                metrics=metrics,
                            )
                        )

            print("    Selection method: all_features")
            all_features = X_train.columns.tolist()
            selected_feature_rows.extend(
                _build_selected_feature_rows(
                    experiment_name=experiment_name,
                    dataset_name=dataset_name,
                    outer_fold=outer_fold,
                    selection_method="all_features",
                    requested_percentage=None,
                    actual_percentage=100.0,
                    random_repetition=None,
                    selected_features=all_features,
                    ranking=None,
                )
            )

            for model_name, model_config in model_variants:
                metrics, timings, complexity = _evaluate_feature_subset(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    selected_features=all_features,
                    numeric_features=dataset.numeric_features,
                    categorical_features=dataset.categorical_features,
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
                        ranking_time_seconds=None,
                        timings=timings,
                        complexity=complexity,
                        metrics=metrics,
                    )
                )

            fold_elapsed = perf_counter() - fold_start
            print(
                f"  [Fold {outer_fold}/{n_folds}] Completed in "
                f"{_format_duration(fold_elapsed)}"
            )

        dataset_elapsed = perf_counter() - dataset_start
        print(
            f"[Dataset {dataset_index + 1}/{n_datasets}] "
            f"Completed in {_format_duration(dataset_elapsed)}"
        )

        experiment_output = _build_experiment_output(
            results=results,
            ranking_rows=ranking_rows,
            selected_feature_rows=selected_feature_rows,
        )
        if on_dataset_complete is not None:
            on_dataset_complete(dataset_name, experiment_output)

    experiment_elapsed = perf_counter() - experiment_start
    print()
    print("=" * 60)
    print(
        "Experiment computations completed in "
        f"{_format_duration(experiment_elapsed)}"
    )
    print("=" * 60)

    return _build_experiment_output(
        results=results,
        ranking_rows=ranking_rows,
        selected_feature_rows=selected_feature_rows,
    )