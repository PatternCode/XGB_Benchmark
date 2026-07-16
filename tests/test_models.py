"""Tests for downstream model training and prediction."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from benchmark.models import (
    ModelError,
    get_model_complexity,
    predict_model,
    train_model,
)


@pytest.fixture
def binary_training_data(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return small binary-classification training and test data."""
    X_train = pd.DataFrame(
        {
            "feature_a": [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
            ],
            "feature_b": [
                1.5,
                1.4,
                1.3,
                1.2,
                1.1,
                1.0,
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
                0.0,
            ],
            "feature_c": [
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
            ],
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_a": [0.15, 0.45, 1.05, 1.35],
            "feature_b": [1.35, 1.05, 0.45, 0.15],
            "feature_c": [0.0, 1.0, 0.0, 1.0],
        }
    )

    y_train = pd.Series(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        name="target",
    )

    return X_train, X_test, y_train


@pytest.fixture
def multiclass_training_data(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return small three-class training and test data."""
    X_train = pd.DataFrame(
        {
            "feature_a": [
                0.0,
                0.1,
                0.2,
                1.0,
                1.1,
                1.2,
                2.0,
                2.1,
                2.2,
            ],
            "feature_b": [
                0.2,
                0.1,
                0.0,
                1.2,
                1.1,
                1.0,
                2.2,
                2.1,
                2.0,
            ],
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_a": [0.15, 1.15, 2.15],
            "feature_b": [0.05, 1.05, 2.05],
        }
    )

    y_train = pd.Series(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        name="target",
    )

    return X_train, X_test, y_train


@pytest.fixture
def model_configs() -> dict[str, dict[str, Any]]:
    """Return compact configurations for all supported models."""
    return {
        "decision_tree": {
            "max_depth": 3,
        },
        "knn": {
            "n_neighbors": 3,
        },
        "logistic_regression": {
            "max_iter": 500,
        },
        "rbf_svm": {
            "C": 1.0,
            "gamma": "scale",
        },
        "xgboost": {
            "booster": "gbtree",
            "objective": "auto",
            "eval_metric": "auto",
            "num_boost_round": 5,
            "max_depth": 2,
            "eta": 0.2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
        },
    }


def assert_valid_predictions(
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_samples: int,
    n_classes: int,
) -> None:
    """Check common properties of labels and probabilities."""
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_prob, np.ndarray)

    assert y_pred.shape == (n_samples,)
    assert y_prob.shape == (n_samples, n_classes)

    assert np.issubdtype(y_pred.dtype, np.integer)
    assert np.isfinite(y_prob).all()
    assert (y_prob >= 0).all()
    assert (y_prob <= 1).all()

    np.testing.assert_allclose(
        y_prob.sum(axis=1),
        np.ones(n_samples),
        atol=1e-6,
    )

    np.testing.assert_array_equal(
        y_pred,
        y_prob.argmax(axis=1),
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "decision_tree",
        "knn",
        "logistic_regression",
        "rbf_svm",
        "xgboost",
    ],
)
def test_train_and_predict_binary_models(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
    model_configs: dict[str, dict[str, Any]],
    model_name: str,
) -> None:
    """Train every supported model for binary classification."""
    X_train, X_test, y_train = binary_training_data

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name,
        model_config=model_configs[model_name],
        random_seed=42,
    )

    y_pred, y_prob = predict_model(
        model=model,
        X=X_test,
        n_classes=2,
    )

    assert_valid_predictions(
        y_pred=y_pred,
        y_prob=y_prob,
        n_samples=len(X_test),
        n_classes=2,
    )


def test_train_decision_tree_returns_fitted_tree(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    """Return a fitted decision-tree classifier."""
    X_train, _, y_train = binary_training_data

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name="decision_tree",
        model_config={"max_depth": 2},
        random_seed=42,
    )

    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 2
    assert hasattr(model, "tree_")


@pytest.mark.parametrize(
    "model_name",
    [
        "knn",
        "logistic_regression",
        "rbf_svm",
    ],
)
def test_scaled_models_return_fitted_pipelines(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
    model_configs: dict[str, dict[str, Any]],
    model_name: str,
) -> None:
    """Use fitted scaling pipelines for scale-sensitive models."""
    X_train, _, y_train = binary_training_data

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name,
        model_config=model_configs[model_name],
        random_seed=42,
    )

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["scaler", "model"]
    assert hasattr(model.named_steps["scaler"], "mean_")


def test_train_xgboost_returns_booster(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
    model_configs: dict[str, dict[str, Any]],
) -> None:
    """Return a fitted native XGBoost Booster."""
    X_train, _, y_train = binary_training_data

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name="xgboost",
        model_config=model_configs["xgboost"],
        random_seed=42,
    )

    assert isinstance(model, xgb.Booster)


def test_train_and_predict_multiclass_xgboost(
    multiclass_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
    model_configs: dict[str, dict[str, Any]],
) -> None:
    """Train native XGBoost for multiclass classification."""
    X_train, X_test, y_train = multiclass_training_data

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name="xgboost",
        model_config=model_configs["xgboost"],
        random_seed=42,
    )

    y_pred, y_prob = predict_model(
        model=model,
        X=X_test,
        n_classes=3,
    )

    assert_valid_predictions(
        y_pred=y_pred,
        y_prob=y_prob,
        n_samples=len(X_test),
        n_classes=3,
    )


def test_model_training_is_reproducible(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    """Produce identical tree predictions with the same seed."""
    X_train, X_test, y_train = binary_training_data

    first_model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name="decision_tree",
        model_config={"max_depth": 3},
        random_seed=42,
    )

    second_model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name="decision_tree",
        model_config={"max_depth": 3},
        random_seed=42,
    )

    first_predictions, first_probabilities = predict_model(
        model=first_model,
        X=X_test,
        n_classes=2,
    )

    second_predictions, second_probabilities = predict_model(
        model=second_model,
        X=X_test,
        n_classes=2,
    )

    np.testing.assert_array_equal(
        first_predictions,
        second_predictions,
    )

    np.testing.assert_allclose(
        first_probabilities,
        second_probabilities,
    )


def test_train_model_rejects_unsupported_model(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    """Reject unsupported model names."""
    X_train, _, y_train = binary_training_data

    with pytest.raises(
        ModelError,
        match="Unsupported model",
    ):
        train_model(
            X_train=X_train,
            y_train=y_train,
            model_name="unknown_model",
            model_config={},
            random_seed=42,
        )


def test_train_model_rejects_length_mismatch(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    """Reject feature and label objects with different lengths."""
    X_train, _, y_train = binary_training_data

    with pytest.raises(
        ModelError,
        match="same number of samples",
    ):
        train_model(
            X_train=X_train,
            y_train=y_train.iloc[:-1],
            model_name="decision_tree",
            model_config={"max_depth": 2},
            random_seed=42,
        )


def test_train_model_rejects_non_contiguous_labels(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    """Require target classes to be encoded from zero consecutively."""
    X_train, _, y_train = binary_training_data
    invalid_y = y_train.replace({1: 2})

    with pytest.raises(
        ModelError,
        match="integer encoded from 0",
    ):
        train_model(
            X_train=X_train,
            y_train=invalid_y,
            model_name="decision_tree",
            model_config={"max_depth": 2},
            random_seed=42,
        )


@pytest.mark.parametrize(
    "max_depth",
    [
        0,
        -1,
        1.5,
        "3",
        True,
    ],
)
def test_train_decision_tree_rejects_invalid_depth(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
    max_depth: object,
) -> None:
    """Reject invalid decision-tree depths."""
    X_train, _, y_train = binary_training_data

    with pytest.raises(
        ModelError,
        match="decision_tree.max_depth",
    ):
        train_model(
            X_train=X_train,
            y_train=y_train,
            model_name="decision_tree",
            model_config={
                "max_depth": max_depth,
            },
            random_seed=42,
        )


@pytest.mark.parametrize(
    "n_neighbors",
    [
        0,
        -1,
        1.5,
        "3",
        True,
    ],
)
def test_train_knn_rejects_invalid_neighbor_count(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
    n_neighbors: object,
) -> None:
    """Reject invalid KNN neighbor counts."""
    X_train, _, y_train = binary_training_data

    with pytest.raises(
        ModelError,
        match="knn.n_neighbors",
    ):
        train_model(
            X_train=X_train,
            y_train=y_train,
            model_name="knn",
            model_config={
                "n_neighbors": n_neighbors,
            },
            random_seed=42,
        )


@pytest.mark.parametrize(
    "C",
    [
        0,
        -1,
        "1.0",
        True,
    ],
)
def test_train_svm_rejects_invalid_c(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
    C: object,
) -> None:
    """Reject invalid RBF-SVM regularisation values."""
    X_train, _, y_train = binary_training_data

    with pytest.raises(
        ModelError,
        match="rbf_svm.C",
    ):
        train_model(
            X_train=X_train,
            y_train=y_train,
            model_name="rbf_svm",
            model_config={
                "C": C,
                "gamma": "scale",
            },
            random_seed=42,
        )


def test_train_xgboost_rejects_invalid_boost_rounds(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
    model_configs: dict[str, dict[str, Any]],
) -> None:
    """Require a positive number of XGBoost boosting rounds."""
    X_train, _, y_train = binary_training_data
    invalid_config = dict(model_configs["xgboost"])
    invalid_config["num_boost_round"] = 0

    with pytest.raises(
        ModelError,
        match="num_boost_round must be a positive integer",
    ):
        train_model(
            X_train=X_train,
            y_train=y_train,
            model_name="xgboost",
            model_config=invalid_config,
            random_seed=42,
        )


@pytest.mark.parametrize(
    "n_classes",
    [
        0,
        1,
        2.5,
        "2",
        True,
    ],
)
def test_predict_model_rejects_invalid_class_count(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
    n_classes: object,
) -> None:
    """Reject invalid numbers of target classes."""
    X_train, X_test, y_train = binary_training_data

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name="decision_tree",
        model_config={"max_depth": 2},
        random_seed=42,
    )

    with pytest.raises(
        ModelError,
        match="n_classes must be an integer of at least 2",
    ):
        predict_model(
            model=model,
            X=X_test,
            n_classes=n_classes,  # type: ignore[arg-type]
        )

def test_get_model_complexity_returns_tree_statistics(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    """Return realised complexity for a fitted decision tree."""
    X_train, _, y_train = binary_training_data

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name="decision_tree",
        model_config={"max_depth": 3},
        random_seed=42,
    )

    complexity = get_model_complexity(model)

    assert complexity["actual_tree_depth"] is not None
    assert complexity["n_tree_leaves"] is not None
    assert complexity["n_tree_nodes"] is not None
    assert complexity["n_tree_features_used"] is not None

    assert 1 <= complexity["actual_tree_depth"] <= 3
    assert complexity["n_tree_leaves"] >= 2
    assert complexity["n_tree_nodes"] >= 3
    assert complexity["n_tree_features_used"] >= 1

    assert (
        complexity["n_tree_nodes"]
        == 2 * complexity["n_tree_leaves"] - 1
    )

    assert (
        complexity["n_tree_features_used"]
        <= X_train.shape[1]
    )


def test_get_model_complexity_returns_none_for_non_tree_model(
    binary_training_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    """Return empty tree-complexity fields for a non-tree model."""
    X_train, _, y_train = binary_training_data

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name="logistic_regression",
        model_config={"max_iter": 500},
        random_seed=42,
    )

    complexity = get_model_complexity(model)

    assert complexity == {
        "actual_tree_depth": None,
        "n_tree_leaves": None,
        "n_tree_nodes": None,
        "n_tree_features_used": None,
    }