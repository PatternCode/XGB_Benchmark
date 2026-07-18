"""Tests for downstream model training, prediction, and complexity."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from benchmark.models import (
    ModelError,
    get_model_complexity,
    predict_model,
    train_model,
)


@pytest.fixture
def binary_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return a small binary-classification training dataset."""
    X = pd.DataFrame(
        {
            "numeric_feature_1": [
                0.1,
                0.2,
                0.3,
                0.4,
                1.1,
                1.2,
                1.3,
                1.4,
            ],
            "numeric_feature_2": [
                1.0,
                1.2,
                0.8,
                1.1,
                3.0,
                3.2,
                2.8,
                3.1,
            ],
            "categorical_feature": [
                0,
                0,
                1,
                1,
                2,
                2,
                1,
                2,
            ],
        }
    )

    y = pd.Series(
        [0, 0, 0, 0, 1, 1, 1, 1],
        name="target",
    )

    return X, y


@pytest.fixture
def binary_data_with_missing_values(
) -> tuple[pd.DataFrame, pd.Series]:
    """Return binary training data containing missing predictors."""
    X = pd.DataFrame(
        {
            "numeric_feature": [
                1.0,
                np.nan,
                2.0,
                2.5,
                8.0,
                8.5,
                np.nan,
                9.0,
            ],
            "categorical_feature": [
                0,
                0,
                np.nan,
                1,
                2,
                np.nan,
                2,
                2,
            ],
        }
    )

    y = pd.Series(
        [0, 0, 0, 0, 1, 1, 1, 1],
        name="target",
    )

    return X, y


@pytest.fixture
def multiclass_training_data(
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a small multiclass training dataset."""
    X = pd.DataFrame(
        {
            "numeric_feature_1": [
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
            "numeric_feature_2": [
                0.2,
                0.0,
                0.1,
                1.2,
                1.0,
                1.1,
                2.2,
                2.0,
                2.1,
            ],
            "categorical_feature": [
                0,
                0,
                0,
                1,
                1,
                1,
                2,
                2,
                2,
            ],
        }
    )

    y = pd.Series(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        name="target",
    )

    return X, y


def _feature_types(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return feature-type lists for the standard test datasets."""
    numeric_features = [
        feature
        for feature in X.columns
        if feature.startswith("numeric_")
    ]

    categorical_features = [
        feature
        for feature in X.columns
        if feature.startswith("categorical_")
    ]

    return numeric_features, categorical_features


def _train(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model_config: dict,
    random_seed: int = 42,
):
    """Train a model using feature types inferred from test columns."""
    numeric_features, categorical_features = _feature_types(X)

    return train_model(
        X_train=X,
        y_train=y,
        model_name=model_name,
        model_config=model_config,
        random_seed=random_seed,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


@pytest.mark.parametrize(
    ("model_name", "model_config"),
    [
        (
            "decision_tree",
            {
                "max_depth": 3,
            },
        ),
        (
            "knn",
            {
                "n_neighbors": 3,
            },
        ),
        (
            "logistic_regression",
            {
                "max_iter": 500,
            },
        ),
        (
            "rbf_svm",
            {
                "C": 1.0,
                "gamma": "scale",
            },
        ),
    ],
)
def test_train_sklearn_models_returns_pipeline(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
    model_name: str,
    model_config: dict,
) -> None:
    """Every scikit-learn model should include preprocessing."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name=model_name,
        model_config=model_config,
    )

    assert isinstance(model, Pipeline)

    if model_name == "decision_tree":
        assert list(model.named_steps) == [
            "preprocessor",
            "model",
        ]
    else:
        assert list(model.named_steps) == [
            "preprocessor",
            "scaler",
            "model",
        ]


def test_decision_tree_pipeline_contains_expected_model(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Decision Tree should be the final pipeline estimator."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name="decision_tree",
        model_config={
            "max_depth": 2,
        },
    )

    assert isinstance(
        model.named_steps["preprocessor"],
        ColumnTransformer,
    )

    assert isinstance(
        model.named_steps["model"],
        DecisionTreeClassifier,
    )

    assert model.named_steps["model"].max_depth == 2
    assert model.named_steps["model"].random_state == 42


@pytest.mark.parametrize(
    "model_name",
    [
        "knn",
        "logistic_regression",
        "rbf_svm",
    ],
)
def test_scaled_models_contain_standard_scaler(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
    model_name: str,
) -> None:
    """Distance- and coefficient-based models should be scaled."""
    X, y = binary_training_data

    model_configs = {
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
    }

    model = _train(
        X=X,
        y=y,
        model_name=model_name,
        model_config=model_configs[model_name],
    )

    assert isinstance(
        model.named_steps["preprocessor"],
        ColumnTransformer,
    )

    assert isinstance(
        model.named_steps["scaler"],
        StandardScaler,
    )


def test_preprocessor_uses_correct_imputation_strategies(
    binary_data_with_missing_values: tuple[
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    """Numeric and categorical features need different imputers."""
    X, y = binary_data_with_missing_values

    model = _train(
        X=X,
        y=y,
        model_name="logistic_regression",
        model_config={
            "max_iter": 500,
        },
    )

    preprocessor = model.named_steps["preprocessor"]

    assert isinstance(preprocessor, ColumnTransformer)

    numeric_imputer = preprocessor.named_transformers_[
        "numeric_imputer"
    ]
    categorical_imputer = preprocessor.named_transformers_[
        "categorical_imputer"
    ]

    assert isinstance(numeric_imputer, SimpleImputer)
    assert numeric_imputer.strategy == "median"

    assert isinstance(categorical_imputer, SimpleImputer)
    assert categorical_imputer.strategy == "most_frequent"


@pytest.mark.parametrize(
    ("model_name", "model_config"),
    [
        (
            "decision_tree",
            {
                "max_depth": 3,
            },
        ),
        (
            "knn",
            {
                "n_neighbors": 3,
            },
        ),
        (
            "logistic_regression",
            {
                "max_iter": 500,
            },
        ),
        (
            "rbf_svm",
            {
                "C": 1.0,
                "gamma": "scale",
            },
        ),
    ],
)
def test_sklearn_models_train_with_missing_values(
    binary_data_with_missing_values: tuple[
        pd.DataFrame,
        pd.Series,
    ],
    model_name: str,
    model_config: dict,
) -> None:
    """Fold-local preprocessing should allow training with NaNs."""
    X, y = binary_data_with_missing_values

    model = _train(
        X=X,
        y=y,
        model_name=model_name,
        model_config=model_config,
    )

    predictions, probabilities = predict_model(
        model=model,
        X=X,
        n_classes=2,
    )

    assert predictions.shape == (len(X),)
    assert probabilities.shape == (len(X), 2)
    assert not np.isnan(probabilities).any()


def test_prediction_pipeline_handles_new_missing_values(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The fitted training imputers should transform test NaNs."""
    X_train, y_train = binary_training_data

    model = _train(
        X=X_train,
        y=y_train,
        model_name="logistic_regression",
        model_config={
            "max_iter": 500,
        },
    )

    X_test = pd.DataFrame(
        {
            "numeric_feature_1": [np.nan, 1.25],
            "numeric_feature_2": [1.05, np.nan],
            "categorical_feature": [np.nan, 2],
        }
    )

    predictions, probabilities = predict_model(
        model=model,
        X=X_test,
        n_classes=2,
    )

    assert predictions.shape == (2,)
    assert probabilities.shape == (2, 2)
    assert not np.isnan(probabilities).any()


def test_imputation_statistics_are_learned_from_training_data_only(
) -> None:
    """Test values must not influence fitted imputation statistics."""
    X_train = pd.DataFrame(
        {
            "numeric_feature": [
                1.0,
                2.0,
                np.nan,
                100.0,
            ],
            "categorical_feature": [
                0,
                0,
                1,
                np.nan,
            ],
        }
    )

    y_train = pd.Series(
        [0, 0, 1, 1],
        name="target",
    )

    model = _train(
        X=X_train,
        y=y_train,
        model_name="decision_tree",
        model_config={
            "max_depth": 2,
        },
    )

    preprocessor = model.named_steps["preprocessor"]

    numeric_imputer = preprocessor.named_transformers_[
        "numeric_imputer"
    ]
    categorical_imputer = preprocessor.named_transformers_[
        "categorical_imputer"
    ]

    assert numeric_imputer.statistics_[0] == pytest.approx(2.0)
    assert categorical_imputer.statistics_[0] == pytest.approx(0.0)

    X_test = pd.DataFrame(
        {
            "numeric_feature": [
                np.nan,
                1_000_000.0,
            ],
            "categorical_feature": [
                np.nan,
                9,
            ],
        }
    )

    predict_model(
        model=model,
        X=X_test,
        n_classes=2,
    )

    assert numeric_imputer.statistics_[0] == pytest.approx(2.0)
    assert categorical_imputer.statistics_[0] == pytest.approx(0.0)


def test_model_trains_with_numeric_features_only() -> None:
    """A subset containing no categorical features should work."""
    X = pd.DataFrame(
        {
            "numeric_feature_1": [
                1.0,
                np.nan,
                2.0,
                8.0,
                9.0,
                np.nan,
            ],
            "numeric_feature_2": [
                0.5,
                0.7,
                np.nan,
                4.0,
                4.2,
                4.5,
            ],
        }
    )

    y = pd.Series(
        [0, 0, 0, 1, 1, 1],
        name="target",
    )

    model = train_model(
        X_train=X,
        y_train=y,
        model_name="logistic_regression",
        model_config={
            "max_iter": 500,
        },
        random_seed=42,
        numeric_features=X.columns.tolist(),
        categorical_features=[],
    )

    preprocessor = model.named_steps["preprocessor"]

    assert "numeric_imputer" in preprocessor.named_transformers_
    assert (
        "categorical_imputer"
        not in preprocessor.named_transformers_
    )

    predictions, probabilities = predict_model(
        model=model,
        X=X,
        n_classes=2,
    )

    assert predictions.shape == (len(X),)
    assert probabilities.shape == (len(X), 2)


def test_model_trains_with_categorical_features_only() -> None:
    """A subset containing no numeric features should work."""
    X = pd.DataFrame(
        {
            "categorical_feature_1": [
                0,
                np.nan,
                0,
                2,
                2,
                np.nan,
            ],
            "categorical_feature_2": [
                1,
                1,
                np.nan,
                3,
                3,
                3,
            ],
        }
    )

    y = pd.Series(
        [0, 0, 0, 1, 1, 1],
        name="target",
    )

    model = train_model(
        X_train=X,
        y_train=y,
        model_name="decision_tree",
        model_config={
            "max_depth": 2,
        },
        random_seed=42,
        numeric_features=[],
        categorical_features=X.columns.tolist(),
    )

    preprocessor = model.named_steps["preprocessor"]

    assert (
        "numeric_imputer"
        not in preprocessor.named_transformers_
    )
    assert (
        "categorical_imputer"
        in preprocessor.named_transformers_
    )

    predictions, probabilities = predict_model(
        model=model,
        X=X,
        n_classes=2,
    )

    assert predictions.shape == (len(X),)
    assert probabilities.shape == (len(X), 2)


def test_train_binary_xgboost_returns_booster(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Binary XGBoost training should return a native Booster."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name="xgboost",
        model_config={
            "objective": "auto",
            "eval_metric": "auto",
            "max_depth": 2,
            "eta": 0.3,
            "num_boost_round": 5,
        },
    )

    assert isinstance(model, xgb.Booster)


def test_xgboost_trains_with_missing_values(
    binary_data_with_missing_values: tuple[
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    """Native XGBoost should receive and handle NaNs directly."""
    X, y = binary_data_with_missing_values

    model = _train(
        X=X,
        y=y,
        model_name="xgboost",
        model_config={
            "objective": "auto",
            "eval_metric": "auto",
            "max_depth": 2,
            "eta": 0.3,
            "num_boost_round": 5,
        },
    )

    assert isinstance(model, xgb.Booster)

    predictions, probabilities = predict_model(
        model=model,
        X=X,
        n_classes=2,
    )

    assert predictions.shape == (len(X),)
    assert probabilities.shape == (len(X), 2)
    assert not np.isnan(probabilities).any()


def test_predict_binary_xgboost_returns_two_probability_columns(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Binary Booster probabilities should be converted to two columns."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name="xgboost",
        model_config={
            "objective": "auto",
            "eval_metric": "auto",
            "max_depth": 2,
            "eta": 0.3,
            "num_boost_round": 5,
        },
    )

    predictions, probabilities = predict_model(
        model=model,
        X=X,
        n_classes=2,
    )

    assert predictions.shape == (len(X),)
    assert probabilities.shape == (len(X), 2)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X)),
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_array_equal(
        predictions,
        probabilities.argmax(axis=1),
    )


def test_predict_multiclass_xgboost(
    multiclass_training_data: tuple[
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    """Multiclass Booster should return one column per class."""
    X, y = multiclass_training_data

    model = _train(
        X=X,
        y=y,
        model_name="xgboost",
        model_config={
            "objective": "auto",
            "eval_metric": "auto",
            "max_depth": 2,
            "eta": 0.3,
            "num_boost_round": 5,
        },
    )

    predictions, probabilities = predict_model(
        model=model,
        X=X,
        n_classes=3,
    )

    assert predictions.shape == (len(X),)
    assert probabilities.shape == (len(X), 3)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X)),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("model_name", "model_config"),
    [
        (
            "decision_tree",
            {
                "max_depth": 3,
            },
        ),
        (
            "knn",
            {
                "n_neighbors": 3,
            },
        ),
        (
            "logistic_regression",
            {
                "max_iter": 500,
            },
        ),
        (
            "rbf_svm",
            {
                "C": 1.0,
                "gamma": "scale",
            },
        ),
    ],
)
def test_predict_sklearn_models(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
    model_name: str,
    model_config: dict,
) -> None:
    """All supported sklearn models should predict probabilities."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name=model_name,
        model_config=model_config,
    )

    predictions, probabilities = predict_model(
        model=model,
        X=X,
        n_classes=2,
    )

    assert predictions.shape == (len(X),)
    assert probabilities.shape == (len(X), 2)

    assert set(np.unique(predictions)).issubset({0, 1})

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_get_model_complexity_reads_decision_tree_pipeline(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Complexity should be extracted from the pipeline model step."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name="decision_tree",
        model_config={
            "max_depth": 2,
        },
    )

    complexity = get_model_complexity(model)

    assert complexity["actual_tree_depth"] is not None
    assert complexity["actual_tree_depth"] <= 2

    assert complexity["n_tree_leaves"] is not None
    assert complexity["n_tree_leaves"] >= 2

    assert complexity["n_tree_nodes"] is not None
    assert complexity["n_tree_nodes"] >= 3

    assert complexity["n_tree_features_used"] is not None
    assert complexity["n_tree_features_used"] >= 1


def test_get_model_complexity_supports_bare_decision_tree(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The helper should remain compatible with a bare tree."""
    X, y = binary_training_data

    model = DecisionTreeClassifier(
        max_depth=2,
        random_state=42,
    )

    model.fit(X, y)

    complexity = get_model_complexity(model)

    assert complexity["actual_tree_depth"] is not None
    assert complexity["n_tree_leaves"] is not None
    assert complexity["n_tree_nodes"] is not None
    assert complexity["n_tree_features_used"] is not None


def test_get_model_complexity_returns_nulls_for_non_tree_model(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Tree complexity fields should not apply to Logistic Regression."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name="logistic_regression",
        model_config={
            "max_iter": 500,
        },
    )

    assert get_model_complexity(model) == {
        "actual_tree_depth": None,
        "n_tree_leaves": None,
        "n_tree_nodes": None,
        "n_tree_features_used": None,
    }


def test_get_model_complexity_returns_nulls_for_xgboost(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Decision-tree complexity fields do not represent XGBoost."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name="xgboost",
        model_config={
            "objective": "auto",
            "eval_metric": "auto",
            "max_depth": 2,
            "eta": 0.3,
            "num_boost_round": 3,
        },
    )

    assert get_model_complexity(model) == {
        "actual_tree_depth": None,
        "n_tree_leaves": None,
        "n_tree_nodes": None,
        "n_tree_features_used": None,
    }


def test_train_model_rejects_non_dataframe_features() -> None:
    """X_train must retain named pandas columns."""
    with pytest.raises(
        ModelError,
        match="X_train must be a pandas DataFrame",
    ):
        train_model(
            X_train=np.asarray([[1.0], [2.0]]),
            y_train=pd.Series([0, 1]),
            model_name="decision_tree",
            model_config={
                "max_depth": 2,
            },
            random_seed=42,
            numeric_features=["feature"],
            categorical_features=[],
        )


def test_train_model_rejects_non_series_target(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """y_train must be a pandas Series."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match="y_train must be a pandas Series",
    ):
        train_model(
            X_train=X,
            y_train=y.to_numpy(),
            model_name="decision_tree",
            model_config={
                "max_depth": 2,
            },
            random_seed=42,
            numeric_features=[
                "numeric_feature_1",
                "numeric_feature_2",
            ],
            categorical_features=[
                "categorical_feature",
            ],
        )


def test_train_model_rejects_missing_target_values(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Target labels must not contain missing values."""
    X, y = binary_training_data
    y = y.astype(float)
    y.iloc[0] = np.nan

    with pytest.raises(
        ModelError,
        match="y_train must not contain missing values",
    ):
        _train(
            X=X,
            y=y,
            model_name="decision_tree",
            model_config={
                "max_depth": 2,
            },
        )


def test_train_model_rejects_unknown_model(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Only explicitly supported classifiers may be trained."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match="Unsupported model",
    ):
        _train(
            X=X,
            y=y,
            model_name="unknown_model",
            model_config={},
        )


def test_train_model_rejects_non_dictionary_config(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Model configuration must be a dictionary."""
    X, y = binary_training_data
    numeric_features, categorical_features = _feature_types(X)

    with pytest.raises(
        ModelError,
        match="model_config must be a dictionary",
    ):
        train_model(
            X_train=X,
            y_train=y,
            model_name="decision_tree",
            model_config=None,
            random_seed=42,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )


def test_train_model_rejects_non_integer_seed(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The model seed must be an integer."""
    X, y = binary_training_data
    numeric_features, categorical_features = _feature_types(X)

    with pytest.raises(
        ModelError,
        match="random_seed must be an integer",
    ):
        train_model(
            X_train=X,
            y_train=y,
            model_name="decision_tree",
            model_config={
                "max_depth": 2,
            },
            random_seed=42.5,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )


def test_train_model_rejects_single_target_class(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Classification training requires at least two classes."""
    X, _ = binary_training_data
    y = pd.Series(
        np.zeros(len(X), dtype=int),
        name="target",
    )

    with pytest.raises(
        ModelError,
        match="requires at least two target classes",
    ):
        _train(
            X=X,
            y=y,
            model_name="decision_tree",
            model_config={
                "max_depth": 2,
            },
        )


def test_train_model_rejects_non_consecutive_target_labels(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Labels must be encoded consecutively starting from zero."""
    X, y = binary_training_data
    y = y.replace({0: 1, 1: 2})

    with pytest.raises(
        ModelError,
        match="Target labels must be integer encoded from 0",
    ):
        _train(
            X=X,
            y=y,
            model_name="decision_tree",
            model_config={
                "max_depth": 2,
            },
        )


def test_train_model_rejects_unclassified_feature(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Every selected feature must have exactly one feature type."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match="do not classify all columns",
    ):
        train_model(
            X_train=X,
            y_train=y,
            model_name="decision_tree",
            model_config={
                "max_depth": 2,
            },
            random_seed=42,
            numeric_features=["numeric_feature_1"],
            categorical_features=["categorical_feature"],
        )


def test_train_model_rejects_unknown_typed_feature(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Feature-type lists must not reference absent columns."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match="not present in X_train",
    ):
        train_model(
            X_train=X,
            y_train=y,
            model_name="decision_tree",
            model_config={
                "max_depth": 2,
            },
            random_seed=42,
            numeric_features=[
                "numeric_feature_1",
                "numeric_feature_2",
                "unknown_feature",
            ],
            categorical_features=["categorical_feature"],
        )


def test_train_model_rejects_overlapping_feature_types(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """A selected feature cannot be numeric and categorical."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match="numeric_features and categorical_features overlap",
    ):
        train_model(
            X_train=X,
            y_train=y,
            model_name="decision_tree",
            model_config={
                "max_depth": 2,
            },
            random_seed=42,
            numeric_features=[
                "numeric_feature_1",
                "numeric_feature_2",
                "categorical_feature",
            ],
            categorical_features=["categorical_feature"],
        )


def test_train_model_rejects_invalid_tree_depth(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Decision-tree depth must be null or positive."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match="decision_tree.max_depth",
    ):
        _train(
            X=X,
            y=y,
            model_name="decision_tree",
            model_config={
                "max_depth": 0,
            },
        )


def test_train_model_accepts_unlimited_tree_depth(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """None should retain sklearn's unlimited-depth behavior."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name="decision_tree",
        model_config={
            "max_depth": None,
        },
    )

    assert model.named_steps["model"].max_depth is None


def test_train_model_rejects_invalid_knn_neighbors(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """KNN requires a positive integer neighbour count."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match="knn.n_neighbors",
    ):
        _train(
            X=X,
            y=y,
            model_name="knn",
            model_config={
                "n_neighbors": 0,
            },
        )


def test_train_model_rejects_invalid_logistic_max_iter(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Logistic Regression requires positive max_iter."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match="logistic_regression.max_iter",
    ):
        _train(
            X=X,
            y=y,
            model_name="logistic_regression",
            model_config={
                "max_iter": 0,
            },
        )


@pytest.mark.parametrize(
    "invalid_C",
    [
        0,
        -1.0,
        "1.0",
        True,
    ],
)
def test_train_model_rejects_invalid_svm_C(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
    invalid_C: object,
) -> None:
    """RBF SVM C must be a positive numeric value."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match=r"rbf_svm\.C",
    ):
        _train(
            X=X,
            y=y,
            model_name="rbf_svm",
            model_config={
                "C": invalid_C,
                "gamma": "scale",
            },
        )


@pytest.mark.parametrize(
    "invalid_gamma",
    [
        0,
        -0.5,
        "invalid",
        True,
    ],
)
def test_train_model_rejects_invalid_svm_gamma(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
    invalid_gamma: object,
) -> None:
    """RBF SVM gamma must use an accepted positive value."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match=r"rbf_svm\.gamma",
    ):
        _train(
            X=X,
            y=y,
            model_name="rbf_svm",
            model_config={
                "C": 1.0,
                "gamma": invalid_gamma,
            },
        )


def test_train_model_rejects_invalid_boost_round_count(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """XGBoost requires a positive number of boosting rounds."""
    X, y = binary_training_data

    with pytest.raises(
        ModelError,
        match="xgboost.num_boost_round",
    ):
        _train(
            X=X,
            y=y,
            model_name="xgboost",
            model_config={
                "objective": "auto",
                "eval_metric": "auto",
                "num_boost_round": 0,
            },
        )


def test_predict_model_rejects_non_dataframe_input(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Prediction data must preserve named pandas columns."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name="decision_tree",
        model_config={
            "max_depth": 2,
        },
    )

    with pytest.raises(
        ModelError,
        match="X must be a pandas DataFrame",
    ):
        predict_model(
            model=model,
            X=X.to_numpy(),
            n_classes=2,
        )


@pytest.mark.parametrize(
    "invalid_n_classes",
    [
        1,
        0,
        -1,
        2.5,
        True,
    ],
)
def test_predict_model_rejects_invalid_class_count(
    binary_training_data: tuple[pd.DataFrame, pd.Series],
    invalid_n_classes: object,
) -> None:
    """Prediction requires an integer class count of at least two."""
    X, y = binary_training_data

    model = _train(
        X=X,
        y=y,
        model_name="decision_tree",
        model_config={
            "max_depth": 2,
        },
    )

    with pytest.raises(
        ModelError,
        match="n_classes must be an integer of at least 2",
    ):
        predict_model(
            model=model,
            X=X,
            n_classes=invalid_n_classes,
        )