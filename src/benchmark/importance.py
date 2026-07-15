"""Compute XGBoost and SHAP feature-importance rankings."""

from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


SUPPORTED_RANKING_METHODS = {
    "gain",
    "weight",
    "cover",
    "shap",
}


class ImportanceError(Exception):
    """Raised when feature-importance computation fails."""


def _validate_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Validate training data used for feature-importance computation."""
    if not isinstance(X_train, pd.DataFrame):
        raise ImportanceError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise ImportanceError("y_train must be a pandas Series.")

    if X_train.empty:
        raise ImportanceError("X_train must not be empty.")

    if y_train.empty:
        raise ImportanceError("y_train must not be empty.")

    if len(X_train) != len(y_train):
        raise ImportanceError(
            "X_train and y_train must contain the same number of samples."
        )

    if X_train.shape[1] == 0:
        raise ImportanceError(
            "X_train must contain at least one feature."
        )

    if not X_train.columns.is_unique:
        raise ImportanceError(
            "X_train must contain unique feature names."
        )

    if y_train.isna().any():
        raise ImportanceError(
            "y_train must not contain missing values."
        )


def _validate_ranking_methods(
    ranking_methods: list[str],
) -> None:
    """Validate the requested feature-ranking methods."""
    if not isinstance(ranking_methods, list):
        raise ImportanceError(
            "ranking_methods must be provided as a list."
        )

    if not ranking_methods:
        raise ImportanceError(
            "ranking_methods must not be empty."
        )

    unknown_methods = (
        set(ranking_methods) - SUPPORTED_RANKING_METHODS
    )

    if unknown_methods:
        raise ImportanceError(
            "Unsupported ranking methods: "
            f"{sorted(unknown_methods)}."
        )

    if len(ranking_methods) != len(set(ranking_methods)):
        raise ImportanceError(
            "ranking_methods must not contain duplicates."
        )


def _resolve_xgboost_parameters(
    y_train: pd.Series,
    xgboost_config: dict[str, Any],
    random_seed: int,
) -> tuple[dict[str, Any], int]:
    """Create task-appropriate native XGBoost parameters."""
    parameters = dict(xgboost_config)
    num_boost_round = parameters.pop("num_boost_round", None)

    if not isinstance(num_boost_round, int) or num_boost_round < 1:
        raise ImportanceError(
            "ranking_xgboost.num_boost_round must be "
            "a positive integer."
        )

    objective = parameters.pop("objective", "auto")
    eval_metric = parameters.pop("eval_metric", "auto")

    classes = sorted(y_train.unique().tolist())
    n_classes = len(classes)

    expected_classes = list(range(n_classes))

    if classes != expected_classes:
        raise ImportanceError(
            "Target labels must be integer encoded from 0 to "
            f"{n_classes - 1}, but received {classes}."
        )

    if n_classes < 2:
        raise ImportanceError(
            "Feature-importance computation requires "
            "at least two target classes."
        )

    if objective == "auto":
        if n_classes == 2:
            parameters["objective"] = "binary:logistic"
        else:
            parameters["objective"] = "multi:softprob"
            parameters["num_class"] = n_classes
    else:
        parameters["objective"] = objective

    if eval_metric == "auto":
        parameters["eval_metric"] = (
            "logloss" if n_classes == 2 else "mlogloss"
        )
    else:
        parameters["eval_metric"] = eval_metric

    parameters["seed"] = random_seed

    return parameters, num_boost_round


def _sort_importance(scores: pd.Series) -> pd.Series:
    """Sort importance scores by value and then by feature name."""
    ranking = (
        scores.rename_axis("feature")
        .reset_index(name="importance")
        .sort_values(
            by=["importance", "feature"],
            ascending=[False, True],
        )
    )

    return (
        ranking.set_index("feature")["importance"]
        .rename(scores.name)
    )


def _extract_xgboost_importance(
    booster: xgb.Booster,
    feature_names: list[str],
    method: str,
) -> pd.Series:
    """Extract one complete XGBoost internal importance ranking."""
    reported_scores = booster.get_score(
        importance_type=method
    )

    scores = pd.Series(
        0.0,
        index=feature_names,
        dtype=float,
        name=method,
    )

    unknown_features = (
        set(reported_scores) - set(feature_names)
    )

    if unknown_features:
        raise ImportanceError(
            "XGBoost returned unknown feature names: "
            f"{sorted(unknown_features)}."
        )

    for feature, value in reported_scores.items():
        scores.loc[feature] = float(value)

    return _sort_importance(scores)


def _select_shap_samples(
    X_train: pd.DataFrame,
    shap_config: dict[str, Any],
    random_seed: int,
) -> pd.DataFrame:
    """Select training samples used to calculate SHAP importance."""
    max_samples = shap_config.get("max_samples")

    if max_samples is None:
        return X_train

    if not isinstance(max_samples, int) or max_samples < 1:
        raise ImportanceError(
            "shap.max_samples must be null or a positive integer."
        )

    if len(X_train) <= max_samples:
        return X_train

    return X_train.sample(
        n=max_samples,
        random_state=random_seed,
    )


def _aggregate_shap_values(
    shap_values: Any,
    n_features: int,
) -> np.ndarray:
    """Aggregate absolute SHAP values across samples and classes."""
    if isinstance(shap_values, list):
        arrays = [np.asarray(values) for values in shap_values]

        if not arrays:
            raise ImportanceError(
                "SHAP returned an empty list of values."
            )

        for values in arrays:
            if values.ndim != 2 or values.shape[1] != n_features:
                raise ImportanceError(
                    "Unexpected SHAP value shape: "
                    f"{values.shape}."
                )

        stacked_values = np.stack(arrays, axis=0)

        return np.abs(stacked_values).mean(axis=(0, 1))

    values = np.asarray(shap_values)

    if values.ndim == 2:
        if values.shape[1] != n_features:
            raise ImportanceError(
                "Unexpected SHAP value shape: "
                f"{values.shape}."
            )

        return np.abs(values).mean(axis=0)

    if values.ndim == 3:
        if values.shape[1] == n_features:
            return np.abs(values).mean(axis=(0, 2))

        if values.shape[2] == n_features:
            return np.abs(values).mean(axis=(0, 1))

    raise ImportanceError(
        f"Unexpected SHAP value shape: {values.shape}."
    )


def _compute_shap_importance(
    booster: xgb.Booster,
    X_train: pd.DataFrame,
    shap_config: dict[str, Any],
    random_seed: int,
) -> pd.Series:
    """Calculate global mean absolute SHAP feature importance."""
    explainer_name = shap_config.get("explainer", "tree")

    if explainer_name != "tree":
        raise ImportanceError(
            "Only the 'tree' SHAP explainer is currently supported."
        )

    X_shap = _select_shap_samples(
        X_train=X_train,
        shap_config=shap_config,
        random_seed=random_seed,
    )

    try:
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(
            X_shap,
            check_additivity=False,
        )
    except Exception as error:
        raise ImportanceError(
            f"Could not calculate SHAP values: {error}"
        ) from error

    importance_values = _aggregate_shap_values(
        shap_values=shap_values,
        n_features=X_train.shape[1],
    )

    scores = pd.Series(
        importance_values,
        index=X_train.columns,
        dtype=float,
        name="shap",
    )

    return _sort_importance(scores)


def compute_feature_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    xgboost_config: dict[str, Any],
    shap_config: dict[str, Any],
    ranking_methods: list[str],
    random_seed: int,
) -> tuple[xgb.Booster, dict[str, pd.Series]]:
    """Train one XGBoost model and compute requested feature rankings.

    The function must receive only data from an outer training fold.
    It does not split data and has no access to the outer test fold.

    Parameters
    ----------
    X_train
        Outer-fold training features.
    y_train
        Outer-fold training labels.
    xgboost_config
        Configuration for the ranking XGBoost model.
    shap_config
        Configuration for SHAP importance.
    ranking_methods
        Requested methods chosen from gain, weight, cover, and shap.
    random_seed
        Seed used for XGBoost training and optional SHAP sampling.

    Returns
    -------
    tuple[xgb.Booster, dict[str, pd.Series]]
        Trained ranking booster and one sorted importance Series
        for every requested ranking method.

    Raises
    ------
    ImportanceError
        If the inputs, configuration, model training, or importance
        computation are invalid.
    """
    _validate_training_data(X_train, y_train)
    _validate_ranking_methods(ranking_methods)

    parameters, num_boost_round = _resolve_xgboost_parameters(
        y_train=y_train,
        xgboost_config=xgboost_config,
        random_seed=random_seed,
    )

    training_matrix = xgb.DMatrix(
        data=X_train,
        label=y_train,
        feature_names=X_train.columns.tolist(),
    )

    try:
        booster = xgb.train(
            params=parameters,
            dtrain=training_matrix,
            num_boost_round=num_boost_round,
        )
    except xgb.core.XGBoostError as error:
        raise ImportanceError(
            f"Could not train the ranking XGBoost model: {error}"
        ) from error

    rankings: dict[str, pd.Series] = {}

    for method in ranking_methods:
        if method in {"gain", "weight", "cover"}:
            rankings[method] = _extract_xgboost_importance(
                booster=booster,
                feature_names=X_train.columns.tolist(),
                method=method,
            )
        elif method == "shap":
            rankings[method] = _compute_shap_importance(
                booster=booster,
                X_train=X_train,
                shap_config=shap_config,
                random_seed=random_seed,
            )

    return booster, rankings