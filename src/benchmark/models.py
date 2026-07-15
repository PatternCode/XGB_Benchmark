"""Train downstream classifiers and generate predictions."""

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SUPPORTED_MODELS = {
    "decision_tree",
    "knn",
    "logistic_regression",
    "rbf_svm",
    "xgboost",
}


class ModelError(Exception):
    """Raised when downstream model training or prediction fails."""


def _validate_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Validate downstream-model training data."""
    if not isinstance(X_train, pd.DataFrame):
        raise ModelError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise ModelError("y_train must be a pandas Series.")

    if X_train.empty:
        raise ModelError("X_train must not be empty.")

    if y_train.empty:
        raise ModelError("y_train must not be empty.")

    if len(X_train) != len(y_train):
        raise ModelError(
            "X_train and y_train must contain the same number of samples."
        )

    if X_train.shape[1] == 0:
        raise ModelError(
            "X_train must contain at least one feature."
        )

    if not X_train.columns.is_unique:
        raise ModelError(
            "X_train must contain unique feature names."
        )

    if y_train.isna().any():
        raise ModelError(
            "y_train must not contain missing values."
        )


def _validate_prediction_data(X: pd.DataFrame) -> None:
    """Validate data supplied for prediction."""
    if not isinstance(X, pd.DataFrame):
        raise ModelError("X must be a pandas DataFrame.")

    if X.empty:
        raise ModelError("X must not be empty.")

    if X.shape[1] == 0:
        raise ModelError(
            "X must contain at least one feature."
        )

    if not X.columns.is_unique:
        raise ModelError(
            "X must contain unique feature names."
        )


def _validate_target_labels(y_train: pd.Series) -> int:
    """Validate class labels and return the number of classes."""
    classes = sorted(y_train.unique().tolist())
    n_classes = len(classes)

    if n_classes < 2:
        raise ModelError(
            "Model training requires at least two target classes."
        )

    expected_classes = list(range(n_classes))

    if classes != expected_classes:
        raise ModelError(
            "Target labels must be integer encoded from 0 to "
            f"{n_classes - 1}, but received {classes}."
        )

    return n_classes


def _positive_integer(
    config: dict[str, Any],
    key: str,
    model_name: str,
) -> int:
    """Read and validate a positive integer model parameter."""
    value = config.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelError(
            f"{model_name}.{key} must be an integer."
        )

    if value < 1:
        raise ModelError(
            f"{model_name}.{key} must be at least 1."
        )

    return value


def _build_sklearn_model(
    model_name: str,
    model_config: dict[str, Any],
    random_seed: int,
) -> BaseEstimator:
    """Construct one unfitted scikit-learn estimator or pipeline."""
    if model_name == "decision_tree":
        max_depth = model_config.get("max_depth")

        if (
            max_depth is not None
            and (
                isinstance(max_depth, bool)
                or not isinstance(max_depth, int)
                or max_depth < 1
            )
        ):
            raise ModelError(
                "decision_tree.max_depth must be null or "
                "a positive integer."
            )

        return DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_seed,
        )

    if model_name == "knn":
        n_neighbors = _positive_integer(
            config=model_config,
            key="n_neighbors",
            model_name="knn",
        )

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                    ),
                ),
            ]
        )

    if model_name == "logistic_regression":
        max_iter = _positive_integer(
            config=model_config,
            key="max_iter",
            model_name="logistic_regression",
        )

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=max_iter,
                        random_state=random_seed,
                    ),
                ),
            ]
        )

    if model_name == "rbf_svm":
        C = model_config.get("C")
        gamma = model_config.get("gamma")

        if (
            isinstance(C, bool)
            or not isinstance(C, (int, float))
            or C <= 0
        ):
            raise ModelError(
                "rbf_svm.C must be a positive number."
            )

        if not (
            gamma in {"scale", "auto"}
            or (
                isinstance(gamma, (int, float))
                and not isinstance(gamma, bool)
                and gamma > 0
            )
        ):
            raise ModelError(
                "rbf_svm.gamma must be 'scale', 'auto', "
                "or a positive number."
            )

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=float(C),
                        gamma=gamma,
                        kernel="rbf",
                        probability=True,
                        random_state=random_seed,
                    ),
                ),
            ]
        )

    raise ModelError(
        f"Unsupported scikit-learn model: '{model_name}'."
    )


def _resolve_xgboost_parameters(
    y_train: pd.Series,
    model_config: dict[str, Any],
    random_seed: int,
) -> tuple[dict[str, Any], int]:
    """Create task-appropriate native XGBoost parameters."""
    parameters = dict(model_config)
    parameters.pop("enabled", None)

    num_boost_round = parameters.pop(
        "num_boost_round",
        None,
    )

    if (
        isinstance(num_boost_round, bool)
        or not isinstance(num_boost_round, int)
        or num_boost_round < 1
    ):
        raise ModelError(
            "xgboost.num_boost_round must be a positive integer."
        )

    objective = parameters.pop("objective", "auto")
    eval_metric = parameters.pop("eval_metric", "auto")

    n_classes = _validate_target_labels(y_train)

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


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    model_config: dict[str, Any],
    random_seed: int,
) -> BaseEstimator | xgb.Booster:
    """Train one downstream classification model.

    Parameters
    ----------
    X_train
        Outer-fold training features restricted to the selected subset.
    y_train
        Outer-fold training labels.
    model_name
        Name of the downstream classifier.
    model_config
        Configuration for the requested model.
    random_seed
        Seed used by stochastic models.

    Returns
    -------
    BaseEstimator | xgb.Booster
        Fitted scikit-learn estimator, fitted pipeline, or native
        XGBoost Booster.

    Raises
    ------
    ModelError
        If the inputs, model name, configuration, or training fails.
    """
    _validate_training_data(X_train, y_train)

    if not isinstance(model_name, str):
        raise ModelError("model_name must be a string.")

    if model_name not in SUPPORTED_MODELS:
        raise ModelError(
            f"Unsupported model: '{model_name}'."
        )

    if not isinstance(model_config, dict):
        raise ModelError(
            "model_config must be a dictionary."
        )

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ModelError(
            "random_seed must be an integer."
        )

    _validate_target_labels(y_train)

    if model_name == "xgboost":
        parameters, num_boost_round = (
            _resolve_xgboost_parameters(
                y_train=y_train,
                model_config=model_config,
                random_seed=random_seed,
            )
        )

        training_matrix = xgb.DMatrix(
            data=X_train,
            label=y_train,
            feature_names=X_train.columns.tolist(),
        )

        try:
            return xgb.train(
                params=parameters,
                dtrain=training_matrix,
                num_boost_round=num_boost_round,
            )
        except xgb.core.XGBoostError as error:
            raise ModelError(
                f"Could not train XGBoost model: {error}"
            ) from error

    model = _build_sklearn_model(
        model_name=model_name,
        model_config=model_config,
        random_seed=random_seed,
    )

    try:
        model.fit(X_train, y_train)
    except (TypeError, ValueError) as error:
        raise ModelError(
            f"Could not train '{model_name}': {error}"
        ) from error

    return model


def predict_model(
    model: BaseEstimator | xgb.Booster,
    X: pd.DataFrame,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predicted labels and class probabilities.

    Probabilities always have shape ``(n_samples, n_classes)``. For binary
    native XGBoost models, the one-dimensional positive-class probability
    is converted into two columns: class 0 and class 1.

    Parameters
    ----------
    model
        Fitted downstream classifier.
    X
        Feature matrix to classify.
    n_classes
        Number of target classes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Predicted labels and class-probability matrix.

    Raises
    ------
    ModelError
        If prediction inputs or outputs are invalid.
    """
    _validate_prediction_data(X)

    if (
        isinstance(n_classes, bool)
        or not isinstance(n_classes, int)
        or n_classes < 2
    ):
        raise ModelError(
            "n_classes must be an integer of at least 2."
        )

    try:
        if isinstance(model, xgb.Booster):
            prediction_matrix = xgb.DMatrix(
                data=X,
                feature_names=X.columns.tolist(),
            )

            raw_probabilities = np.asarray(
                model.predict(prediction_matrix)
            )

            if n_classes == 2:
                if raw_probabilities.ndim != 1:
                    raise ModelError(
                        "Binary XGBoost predictions must be "
                        "one-dimensional."
                    )

                probabilities = np.column_stack(
                    (
                        1.0 - raw_probabilities,
                        raw_probabilities,
                    )
                )
            else:
                probabilities = raw_probabilities

                if probabilities.shape != (
                    len(X),
                    n_classes,
                ):
                    raise ModelError(
                        "Unexpected multiclass XGBoost "
                        f"probability shape: {probabilities.shape}."
                    )

            predictions = probabilities.argmax(axis=1)

            return predictions.astype(int), probabilities

        predictions = np.asarray(model.predict(X))

        if not hasattr(model, "predict_proba"):
            raise ModelError(
                "The fitted model does not provide predict_proba()."
            )

        probabilities = np.asarray(
            model.predict_proba(X)
        )

    except ModelError:
        raise
    except (TypeError, ValueError, xgb.core.XGBoostError) as error:
        raise ModelError(
            f"Could not generate model predictions: {error}"
        ) from error

    if predictions.shape != (len(X),):
        raise ModelError(
            f"Unexpected prediction shape: {predictions.shape}."
        )

    if probabilities.shape != (len(X), n_classes):
        raise ModelError(
            "Unexpected probability shape: "
            f"{probabilities.shape}; expected "
            f"({len(X)}, {n_classes})."
        )

    return predictions.astype(int), probabilities